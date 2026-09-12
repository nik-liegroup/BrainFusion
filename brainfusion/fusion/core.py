import os

import numpy as np

from brainfusion.fusion.match_contours import interpolate_contour, align_contours, boundary_match_contours
from brainfusion.fusion.average_contours import find_average_contour
from brainfusion.fusion.transform_2Dmap import extend_grid, transform_grid2contour
from brainfusion.fusion.interpolation import nearest_neighbour_interp, fit_coordinates_gmm
from brainfusion.io import check_parameters, export_analysis, import_analysis
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
        How measurements are combined: "Mean", "Median" and "Sum" resample every sample onto one shared
        regular grid first, then reduce across samples at each grid point. "GMM" instead pools every
        sample's own warped points and clusters them by density with a Gaussian Mixture Model, so its
        result lives on its own data-driven set of points rather than that shared regular grid.
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

    # "GMM" clusters each sample's own warped points directly and builds its own grid from them, so it's the
    # only option that doesn't need samples pre-resampled onto one shared regular grid first.
    if clustering == 'GMM':
        ext_grid, ext_grid_shape = None, None
    else:
        ext_grid, ext_grid_shape = extend_grid([s.grid for s in measurement_samples], 0.05, 0.05)

    trafo_data_maps, trafo_grids, trafo_ver_grids, verification_grids, trafo_contours = fuse_grids(
        measurement_samples, template_contours, ext_grid=ext_grid, smooth=smooth)

    avg_data, avg_grid = fuse_measurement_datasets(measurement_samples, trafo_data_maps, trafo_grids, ext_grid,
                                                   clustering=clustering)

    structured_data = _structured_data(template_sample, measurement_samples, template_contours, trafo_contours,
                                       verification_grids, trafo_grids, trafo_ver_grids, trafo_data_maps)
    structured_data.update({
        'measurement_interpolated_grid': avg_grid,
        'measurement_interpolated_grid_shape': ext_grid_shape,
        'measurement_interpolated_dataset': avg_data,
    })
    return structured_data


def run_fusion(samples: list[Sample], fusion_kwargs: dict, results_path: str, overwrite: bool = False) -> dict:
    """
    Run `brain_fusion` on `samples` and cache the result at `results_path`. If a cached result already
    exists and `overwrite` is False, it is loaded instead and its stored parameters are compared against
    `fusion_kwargs`.
    """
    if overwrite or not os.path.exists(results_path):
        analysis = brain_fusion(samples, **fusion_kwargs)
        export_analysis(results_path, analysis, fusion_kwargs)
    else:
        analysis, loaded_kwargs = import_analysis(results_path)
        check_parameters(fusion_kwargs, loaded_kwargs)

    return analysis


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


def fuse_grids(measurement_samples: list[Sample], template_contours, ext_grid=None, smooth='auto'):
    """
    Warp each measurement's grid into the template's coordinate space.

    If `ext_grid` is given, each sample's data is also resampled onto it (needed by the "Mean"/"Median"/
    "Sum" options in `fuse_measurement_datasets`); pass `None` to skip that when it isn't needed (e.g. for
    "GMM", which builds its own grid instead).
    """
    verification_grids = [regular_grid_on_bbox(s.contour) for s in measurement_samples]

    trafo_data_maps, trafo_grids, trafo_contours, trafo_ver_grids = [], [], [], []
    for index, sample in enumerate(measurement_samples):
        trafo_grid, trafo_ver_grid, trafo_contour = transform_grid2contour(
            sample.contour, template_contours[index], sample.grid, verification_grids[index], smooth=smooth,
            progress=f' {index + 1} out of {len(measurement_samples)}')

        trafo_data = None
        if ext_grid is not None:
            trafo_data = {
                key: nearest_neighbour_interp(trafo_grid, data_map.ravel(), ext_grid, unique=False)
                for key, data_map in sample.dataset.items()
            }

        trafo_data_maps.append(trafo_data)
        trafo_grids.append(trafo_grid)
        trafo_ver_grids.append(trafo_ver_grid)
        trafo_contours.append(trafo_contour)

    return trafo_data_maps, trafo_grids, trafo_ver_grids, verification_grids, trafo_contours


def fuse_measurement_datasets(measurement_samples: list[Sample], trafo_data_maps, trafo_grids, ext_grid,
                              clustering='Mean'):
    """
    Combine the per-sample datasets into a single averaged dataset, together with the grid it lives on.

    "Mean"/"Median"/"Sum" reduce `trafo_data_maps` (already resampled onto `ext_grid`) across samples at
    each grid point, so the result lives on `ext_grid`. "GMM" instead pools every sample's own warped
    points (`trafo_grids`) and clusters them by density, so the result lives on a new set of data-driven
    coordinates unrelated to `ext_grid` (which can be `None` in that case).
    """
    if clustering == 'GMM':
        datasets = [s.dataset for s in measurement_samples]
        avg_grid, avg_data = fit_coordinates_gmm(trafo_grids, datasets, num_components='mean')
    elif clustering in ("Sum", "Median", "Mean"):
        projection = {"Sum": np.nansum, "Median": np.nanmedian, "Mean": np.nanmean}[clustering]
        keys = trafo_data_maps[0].keys()
        avg_data = {key: projection(np.array([d[key] for d in trafo_data_maps]), axis=0) for key in keys}
        avg_grid = ext_grid
    else:
        raise ValueError(f'Clustering option {clustering} is not implemented!')

    return avg_data, avg_grid


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
