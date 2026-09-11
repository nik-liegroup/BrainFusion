import numpy as np

from brainfusion.match_contours import interpolate_contour, align_contours, boundary_match_contours
from brainfusion.average_contours import find_average_contour
from brainfusion.transform_2Dmap import extend_grid, transform_grid2contour
from brainfusion.interpolation import nearest_neighbour_interp, fit_coordinates_gmm
from brainfusion.utils import regular_grid_on_bbox
from brainfusion.sample import Sample, replace


def brain_fusion(samples: list[Sample], contour_template="average", contour_interp_n=200, clustering='Mean',
                 outline_averaging='star_domain', smooth='auto', curvature=0.5, fit_routine='ellipse', **kwargs) -> dict:
    """
    Align, deform and interpolate a list of samples onto a common template shape, then average them.

    Parameters
    ----------
    samples : list of Sample
        One entry per measurement. The sample used as template index (see `fuse_boundaries`) provides the
        reference shape everything else is warped to.
    contour_template : str, default="average"
        "average": use the shape-averaged contour of all samples as template.
        "first_element": use `samples[0]` as template (excluded from the averaged result).
    contour_interp_n : int, default=200
        Number of points each contour is resampled to before alignment.
    clustering : str, default='Mean'
        How measurements are combined onto the shared interpolation grid: "Mean", "Median", "Sum" or "GMM".
    outline_averaging : str, default='star_domain'
        Method used to average contours when `contour_template="average"`.
    smooth : float, 'auto' or 'weighted', default='auto'
        RBF interpolation smoothing factor for the non-rigid grid transformation.
    curvature : float, default=0.5
        DTW curvature-mismatch penalty used for boundary matching.
    fit_routine : str, default='ellipse'
        Initial affine contour alignment method: "ellipse" or "bbox".

    Returns
    -------
    structured_data : dict
        All intermediate and final results (aligned/matched contours, original and transformed grids, warped
        datasets, the averaged dataset on the common grid, affine transformation matrices, background images).
    """
    print("Starting brainfusion analysis.")

    template_sample, measurement_samples, template_contours = fuse_boundaries(
        contour_template, samples, contour_interp_n=contour_interp_n, outline_averaging=outline_averaging,
        curvature=curvature, fit_routine=fit_routine)

    ext_grid, ext_grid_shape = extend_grid([s.grid for s in measurement_samples], 0.05, 0.05)
    trafo_data_maps, trafo_grids, trafo_ver_grids, verification_grids, trafo_contours = fuse_grids(
        measurement_samples, ext_grid, template_contours, smooth=smooth)

    avg_data = fuse_measurement_datasets(measurement_samples, trafo_data_maps, trafo_grids, clustering=clustering)

    structured_data = _structured_data(template_sample, measurement_samples, template_contours, trafo_contours,
                                       verification_grids, trafo_grids, trafo_ver_grids, trafo_data_maps)
    structured_data.update({
        'measurement_interpolated_grid': ext_grid,
        'measurement_interpolated_grid_shape': ext_grid_shape,
        'measurement_interpolated_dataset': avg_data,
    })
    return structured_data


def brain_fusion_correlation(samples: list[Sample], contour_template="average", contour_interp_n=200,
                             clustering='Mean', outline_averaging='star_domain', smooth='auto', curvature=0.5,
                             fit_routine='ellipse', **kwargs) -> dict:
    """
    Same as `brain_fusion`, but each sample is warped onto its own interpolation grid instead of a shared one
    (no cross-sample averaging). Used to correlate individual samples rather than fuse them into one map.
    """
    print("Starting correlation analysis.")

    template_sample, measurement_samples, template_contours = fuse_boundaries(
        contour_template, samples, contour_interp_n=contour_interp_n, outline_averaging=outline_averaging,
        curvature=curvature, fit_routine=fit_routine)

    ext_grids, ext_grids_shape = [], []
    trafo_data_maps, trafo_grids, trafo_ver_grids, verification_grids, trafo_contours = [], [], [], [], []
    for index, sample in enumerate(measurement_samples):
        ext_grid, ext_grid_shape = extend_grid([sample.grid], 0.05, 0.05)
        maps, grids, ver_grids, ver_grids_raw, contours = fuse_grids(
            [sample], ext_grid, [template_contours[index]], smooth=smooth)

        ext_grids.append(ext_grid)
        ext_grids_shape.append(np.array(ext_grid_shape))
        trafo_data_maps.extend(maps)
        trafo_grids.extend(grids)
        trafo_ver_grids.extend(ver_grids)
        verification_grids.extend(ver_grids_raw)
        trafo_contours.extend(contours)

    structured_data = _structured_data(template_sample, measurement_samples, template_contours, trafo_contours,
                                       verification_grids, trafo_grids, trafo_ver_grids, trafo_data_maps)
    structured_data.update({
        'measurement_interpolated_grid': ext_grids,
        'measurement_interpolated_grid_shape': ext_grids_shape,
    })
    return structured_data


def fuse_boundaries(template, samples: list[Sample], contour_interp_n=200, outline_averaging='star_domain',
                    curvature=0.5, fit_routine='ellipse'):
    """
    Affinely align all sample contours to a common template, then refine the match with boundary DTW.

    The template is always returned as its own `Sample` (index 0 of the conceptual list) so that every other
    per-sample list produced downstream (grids, datasets, scales, affine matrices, background images, ...) lines
    up with `measurement_samples` regardless of whether the template was one of the inputs or a synthetic average.

    Returns
    -------
    template_sample : Sample
    measurement_samples : list of Sample
        Every sample other than the template, with its contour replaced by the DTW-matched version.
    measurement_template_contours : list of np.ndarray
        The template contour warped to match each measurement sample (needed since DTW produces a different
        template correspondence for every sample).
    """
    template_index = 0  # Fixed: samples[0] is either the actual template or becomes one after averaging

    contours_interp = [interpolate_contour(_close_contour(s.contour), contour_interp_n) for s in samples]

    aligned_contours, aligned_grids, affine_matrices = align_contours(
        contours_interp, [s.grid for s in samples], landmarks_list=[s.landmarks for s in samples],
        template_index=template_index, fit_routine=fit_routine)

    aligned_samples = [replace(sample, contour=contour, grid=grid, affine=affine)
                      for sample, contour, grid, affine in zip(samples, aligned_contours, aligned_grids,
                                                               affine_matrices)]

    if template == 'average':
        template_contour, _ = find_average_contour(aligned_contours, average=outline_averaging,
                                                    star_bins=contour_interp_n, error_metric='frechet')
        aligned_samples.insert(template_index, Sample(contour=template_contour, filename='template_average'))
    elif template == 'first_element':
        pass
    else:
        raise ValueError(f'Choice of template: {template} is not implemented!')

    dtw_contours, dtw_template_contours = boundary_match_contours(
        [s.contour for s in aligned_samples], template_index=template_index, curvature=curvature)

    template_sample = replace(aligned_samples[template_index], contour=dtw_contours[template_index])
    measurement_samples = [replace(sample, contour=contour)
                          for sample, contour in zip(aligned_samples[1:], dtw_contours[1:])]
    measurement_template_contours = dtw_template_contours[1:]

    return template_sample, measurement_samples, measurement_template_contours


def fuse_grids(measurement_samples: list[Sample], ext_grid, template_contours, smooth='auto'):
    """
    Warp each measurement's grid into the template's coordinate space and interpolate its data onto `ext_grid`.
    """
    verification_grids = [regular_grid_on_bbox(s.contour) for s in measurement_samples]

    trafo_data_maps, trafo_grids, trafo_contours, trafo_ver_grids = [], [], [], []
    for index, sample in enumerate(measurement_samples):
        trafo_grid, trafo_ver_grid, trafo_contour = transform_grid2contour(
            sample.contour, template_contours[index], sample.grid, verification_grids[index], smooth=smooth,
            progress=f' {index + 1} out of {len(measurement_samples)}')

        trafo_data = {
            key: nearest_neighbour_interp(trafo_grid, data_map.ravel(), ext_grid, unique=False)
            for key, data_map in sample.dataset.items()
        }

        trafo_data_maps.append(trafo_data)
        trafo_grids.append(trafo_grid)
        trafo_ver_grids.append(trafo_ver_grid)
        trafo_contours.append(trafo_contour)

    return trafo_data_maps, trafo_grids, trafo_ver_grids, verification_grids, trafo_contours


def fuse_measurement_datasets(measurement_samples: list[Sample], trafo_data_maps, trafo_grids, clustering='Mean'):
    """Combine the per-sample datasets, already warped onto a shared grid, into a single averaged dataset."""
    if clustering == 'GMM':
        # Cluster data points using a Gaussian Mixture Model and average using the median
        datasets = [s.dataset for s in measurement_samples]
        _, _, avg_data = fit_coordinates_gmm(trafo_grids, datasets, trafo_data_maps, same_maps=True,
                                             num_components='mean')
    elif clustering in ("Sum", "Median", "Mean"):
        projection = {"Sum": np.nansum, "Median": np.nanmedian, "Mean": np.nanmean}[clustering]
        keys = trafo_data_maps[0].keys()
        avg_data = {key: projection(np.array([d[key] for d in trafo_data_maps]), axis=0) for key in keys}
    else:
        raise ValueError(f'Clustering option {clustering} is not implemented!')

    return avg_data


def _close_contour(contour: np.ndarray) -> np.ndarray:
    """Append the first point to the end of a contour if it isn't already closed."""
    if np.array_equal(contour[0], contour[-1]):
        return contour
    return np.vstack([contour, contour[0]])


def _structured_data(template_sample, measurement_samples, template_contours, trafo_contours, verification_grids,
                     trafo_grids, trafo_ver_grids, trafo_data_maps) -> dict:
    """Assemble the result dict shared by `brain_fusion` and `brain_fusion_correlation`."""
    return {
        'affine_matrices': [template_sample.affine] + [s.affine for s in measurement_samples],
        'scale_matrices': [template_sample.scale] + [s.scale for s in measurement_samples],
        'template_contours': template_contours,
        'measurement_contours': [s.contour for s in measurement_samples],
        'measurement_trafo_contours': trafo_contours,
        'template_grid': template_sample.grid,
        'measurement_grids': [s.grid for s in measurement_samples],
        'measurement_grids_shape': [s.grid_shape for s in measurement_samples],
        'verification_grids': verification_grids,
        'measurement_trafo_grids': trafo_grids,
        'verification_trafo_grids': trafo_ver_grids,
        'template_dataset': template_sample.dataset,
        'measurement_datasets': [s.dataset for s in measurement_samples],
        'measurement_trafo_datasets': trafo_data_maps,
        'measurement_filenames': [s.filename for s in measurement_samples],
        'measurement_metadata': [s.metadata for s in measurement_samples],
        'background_image': [s.bg_image for s in measurement_samples],
    }
