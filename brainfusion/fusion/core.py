import hashlib
import os
import warnings

import numpy as np

from brainfusion.fusion.match_contours import interpolate_contour, align_contours, boundary_match_contours
from brainfusion.fusion.average_contours import find_average_contour
from brainfusion.fusion.transform_2Dmap import extend_grid, transform_grid2contour
from brainfusion.fusion.interpolation import nearest_neighbour_interp, gmm_average, gmm_reduce, typical_spacing
from brainfusion.fusion.grouping import group_average_entry
from brainfusion.io import check_parameters, export_analysis, import_analysis
from brainfusion.utils import regular_grid_on_bbox
from brainfusion.sample import Sample, replace


def brain_fusion(samples: list[Sample], contour_template="average", contour_interp_n=200, clustering='Mean',
                 outline_averaging='star_domain', smooth='auto', curvature=0.5, fit_routine='ellipse',
                 landmark_fit='similarity', group_field=None, max_gap=None, canonical_axes=True, **kwargs) -> dict:
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
        A `Sample` or an (N, 2) array: a template outline you give (e.g. `average_outline` of a larger set of samples,
        so several fusions share one outline; add `landmarks` to the Sample to align by landmarks). Every sample is
        aligned and warped onto it; it is not part of the averages.
    contour_interp_n : int, default=200
        Number of points each contour is resampled to before alignment.
    clustering : str or list of str, default='Mean'
        How measurements are combined: "Mean", "Median" and "Sum" resample every sample onto one shared
        regular grid first, then reduce across samples at each grid point. "GMM" instead pools every
        sample's own warped points and clusters them by density with a Gaussian Mixture Model, so its
        result lives on its own data-driven set of points rather than that shared regular grid.
        A list (e.g. ["Median", "GMM"]) computes all of them from the same alignment and warp; each result is
        stored in `averaged[method]` (grid, dataset, n, spread - see `average_across_samples`), and the FIRST
        method is also stored under the `measurement_interpolated_*` keys (and is the one used for groups).
    outline_averaging : str, default='star_domain'
        Method used to average contours when `contour_template="average"`.
    smooth : float, 'auto' or 'weighted', default='auto'
        RBF interpolation smoothing factor for the non-rigid grid transformation.
    curvature : float, default=0.5
        DTW curvature-mismatch penalty used for boundary matching.
    landmark_fit : 'similarity' or 'affine', default='similarity'
        How samples with landmarks are aligned (see `match_contour_with_landmarks`). Similarity (rotation, uniform scale,
        translation) is the stable choice for round outlines and for landmarks on a line.
    fit_routine : str, default='ellipse'
        Initial affine contour alignment method: "ellipse" or "bbox".
    canonical_axes : bool, default=True
        With the ellipse fit: put the first outline's fitted ellipse with its major axis along y and its minor axis along x
        (instead of keeping the orientation it happens to have) and match the others to it, so every fusion has the same
        orientation (see `canonical_ellipse_trafos`).
    max_gap : float, optional
        Without it (None), every point of the shared averaging grid takes the value of the nearest measured point
        of each sample, however far away - the maps are filled in beyond where anything was measured. With it,
        a grid point further than `max_gap` times the sample's own point spacing from every one of its (warped)
        measurement points stays empty (NaN) for that sample, so it neither fills in nor enters the average.
        Typical: 1.5-2 (the spacing is the median distance of a point to its nearest neighbour). Not used with
        'GMM'.
    group_field : str, optional
        If given, additionally split the fused samples by `.metadata[group_field]` (e.g. 'condition') and
        average each group separately, with EVERY method of `clustering`, right here at fusion time. The result is
        stored as `group_field` and `group_averaged[method][group]` - one entry per group with the same content as
        `averaged[method]` (grid, dataset, n, spread, standard_error; a 'GMM' group lives on the components of the one clustering of all samples, so all groups share one grid) - so it is
        cached, reused by whatever asks for a group's map afterwards (`plot_average_map`, `average_statistic`,
        `correlate_group_maps`, ...), and persisted through `export_analysis`/`import_analysis`.
        For the first of the methods that live on the shared grid (Mean, Median, Sum) the per-group datasets are also
        stored under `group_datasets[group]` (the form `pairwise_correlate_groups` takes).

    Returns
    -------
    structured_data : dict
        All intermediate and final results (aligned/matched contours, original and transformed grids, warped
        datasets, the averaged dataset on the common grid, affine transformation matrices, background images,
        and, if `group_field` was given, `group_field` and `group_datasets`).
    """
    print("Starting brainfusion analysis.")
    methods = [clustering] if isinstance(clustering, str) else list(clustering)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError(f"clustering must be one method or a list of different methods, got {clustering!r}")

    template_sample, measurement_samples, template_contours = fuse_boundaries(
        contour_template, samples, contour_interp_n=contour_interp_n, outline_averaging=outline_averaging,
        curvature=curvature, fit_routine=fit_routine, landmark_fit=landmark_fit, canonical_axes=canonical_axes)

    # "GMM" clusters each sample's own warped points directly and builds its own grid from them, so it's the
    # only option that doesn't need samples pre-resampled onto one shared regular grid first.
    if methods == ['GMM']:
        ext_grid, ext_grid_shape = None, None
    else:
        ext_grid, ext_grid_shape = extend_grid([s.grid for s in measurement_samples], 0.05, 0.05)

    trafo_data_maps, trafo_grids, trafo_ver_grids, verification_grids, trafo_contours = fuse_grids(
        measurement_samples, template_contours, ext_grid=ext_grid, smooth=smooth, max_gap=max_gap)

    averaged = {method: average_across_samples(method, measurement_samples, trafo_data_maps, trafo_grids, ext_grid)
                for method in methods if method != 'GMM'}
    gmm_pooled = None
    if 'GMM' in methods:   # one clustering of all samples; its components are the grid of every GMM average, groups included
        gmm_pooled = gmm_average(trafo_grids, [s.dataset for s in measurement_samples], num_components='mean')
        averaged['GMM'] = {name: gmm_pooled[name] for name in ('grid', 'dataset', 'n', 'spread', 'standard_error')}
    avg_data, avg_grid = averaged[methods[0]]['dataset'], averaged[methods[0]]['grid']

    structured_data = _structured_data(template_sample, measurement_samples, template_contours, trafo_contours,
                                       verification_grids, trafo_grids, trafo_ver_grids, trafo_data_maps)
    structured_data.update({
        'measurement_interpolated_grid': avg_grid,
        'measurement_interpolated_grid_shape': None if methods[0] == 'GMM' else ext_grid_shape,
        'measurement_interpolated_dataset': avg_data,
        'averaged_methods': methods,
        'averaged': averaged,
    })

    if group_field is not None:
        structured_data['group_field'] = group_field
        structured_data['group_averaged'] = _average_groups(methods, group_field, measurement_samples, trafo_data_maps,
                                                            trafo_grids, ext_grid, gmm_pooled)
        shared = [m for m in methods if m != 'GMM']
        if shared:
            structured_data['group_datasets'] = {
                group: entry['dataset'] for group, entry in structured_data['group_averaged'][shared[0]].items()}

    return structured_data


def _average_groups(methods, group_field, measurement_samples, trafo_data_maps, trafo_grids, ext_grid, gmm_pooled=None):
    """{method: {group: `average_across_samples` result of the samples with that metadata[group_field]}}."""
    values = np.array([str(s.metadata[group_field]) for s in measurement_samples])
    groups = sorted(set(values))
    averaged = {}
    for method in methods:
        averaged[method] = {}
        for group in groups:
            index = np.flatnonzero(values == group)
            if method == 'GMM':   # the components of the clustering of ALL samples, so every group lives on the same grid
                counts = [len(trafo_grids[i]) for i in range(len(measurement_samples))]
                offsets = np.concatenate([[0], np.cumsum(counts)])
                labels = np.concatenate([gmm_pooled['labels'][offsets[i]:offsets[i + 1]] for i in index])
                result = gmm_reduce(gmm_pooled['grid'], labels, [counts[i] for i in index],
                                    [measurement_samples[i].dataset for i in index])
                averaged[method][group] = {name: result[name] for name in ('grid', 'dataset', 'n', 'spread', 'standard_error')}
                continue
            averaged[method][group] = average_across_samples(
                method, [measurement_samples[i] for i in index],
                None if trafo_data_maps[0] is None else [trafo_data_maps[i] for i in index],
                [trafo_grids[i] for i in index], ext_grid)
    return averaged


def run_fusion(samples: list[Sample], fusion_kwargs: dict, results_path: str, overwrite: bool = False) -> dict:
    """
    Run `brain_fusion` on `samples` and cache the result at `results_path`. If a cached result already
    exists and `overwrite` is False, it is loaded instead and its stored parameters are compared against
    `fusion_kwargs`. The cache also stores a fingerprint of the samples' outlines: if they changed since (e.g. an
    outline drawn again, or a repeat added to an averaged outline), loading the cache raises a ValueError - the cached
    fusion was made with other outlines (caches written before the fingerprint existed are not checked).
    """
    template = fusion_kwargs.get('contour_template')
    stored = dict(fusion_kwargs)
    if isinstance(template, (Sample, np.ndarray)):   # stored as a fingerprint of the outline
        contour = template.contour if isinstance(template, Sample) else template
        stored['contour_template'] = 'given:' + hashlib.md5(np.round(np.asarray(contour, dtype=float), 6).tobytes()).hexdigest()
    outlines = _fingerprint([s.contour for s in samples])
    if overwrite or not os.path.exists(results_path):
        analysis = brain_fusion(samples, **fusion_kwargs)
        export_analysis(results_path, analysis, {**stored, 'input_outlines': outlines})
    else:
        analysis, loaded_kwargs = import_analysis(results_path)
        check_parameters(stored, {k: v for k, v in loaded_kwargs.items() if k != 'input_outlines'})
        cached = loaded_kwargs.get('input_outlines')
        if cached is not None and cached != outlines:
            raise ValueError(f"{results_path} was made with other outlines than the ones loaded now (an outline was "
                             f"drawn again or a repeat added/removed): recompute it (overwrite=True)")

    return analysis


def _fingerprint(contours):
    """md5 of the outlines (rounded relative to their size, so that it does not depend on the unit)."""
    digest = hashlib.md5()
    for contour in contours:
        contour = np.asarray(contour, dtype=float)
        size = float(np.ptp(contour, axis=0).max()) or 1.0
        digest.update(np.round(contour / size, 6).tobytes())
    return digest.hexdigest()


def fuse_boundaries(template, samples: list[Sample], contour_interp_n=200, outline_averaging='star_domain',
                    curvature=0.5, fit_routine='ellipse', landmark_fit='similarity', canonical_axes=True):
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
    if isinstance(template, (Sample, np.ndarray)):   # a given outline becomes the first element, the others follow
        given = template if isinstance(template, Sample) else Sample(contour=np.asarray(template, dtype=float),
                                                                     filename='template_given')
        samples, template = [given] + list(samples), 'first_element'

    contours_interp = [interpolate_contour(_close_contour(s.contour), contour_interp_n) for s in samples]

    aligned_contours, aligned_grids, affine_matrices = align_contours(
        contours_interp, [s.grid for s in samples], landmarks_list=[s.landmarks for s in samples],
        template_index=template_index, fit_routine=fit_routine, landmark_fit=landmark_fit, canonical_axes=canonical_axes)

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


def average_outline(samples: list[Sample], outline_averaging='median', contour_interp_n=200, fit_routine='ellipse',
                    landmark_fit='similarity', canonical_axes=True) -> Sample:
    """
    The average outline of `samples` as a template `Sample` (no data, no landmarks), for `brain_fusion(contour_template=...)`.
    The outlines are aligned to the first sample as in a fusion with `contour_template="average"` and averaged; fusions
    run separately (e.g. one per modality) with this template share the outline but each makes its own grid.
    """
    contours = [interpolate_contour(_close_contour(s.contour), contour_interp_n) for s in samples]
    aligned, _, _ = align_contours(contours, [s.grid for s in samples], landmarks_list=[s.landmarks for s in samples],
                                   template_index=0, fit_routine=fit_routine, landmark_fit=landmark_fit,
                                   canonical_axes=canonical_axes)
    contour, _ = find_average_contour(aligned, average=outline_averaging, star_bins=contour_interp_n, error_metric='frechet')
    return Sample(contour=contour, filename='template_average')


def fuse_grids(measurement_samples: list[Sample], template_contours, ext_grid=None, smooth='auto', max_gap=None):
    """
    Warp each measurement's grid into the template's coordinate space.

    If `ext_grid` is given, each sample's data is also resampled onto it (needed by the "Mean"/"Median"/
    "Sum" options in `fuse_measurement_datasets`); pass `None` to skip that when it isn't needed (e.g. for
    "GMM", which builds its own grid instead). `max_gap`: see `brain_fusion`.
    """
    verification_grids = [regular_grid_on_bbox(s.contour) for s in measurement_samples]

    trafo_data_maps, trafo_grids, trafo_contours, trafo_ver_grids = [], [], [], []
    for index, sample in enumerate(measurement_samples):
        trafo_grid, trafo_ver_grid, trafo_contour = transform_grid2contour(
            sample.contour, template_contours[index], sample.grid, verification_grids[index], smooth=smooth,
            progress=f' {index + 1} out of {len(measurement_samples)}')

        trafo_data = None
        if ext_grid is not None:
            max_distance = None if max_gap is None else max_gap * typical_spacing(trafo_grid)
            trafo_data = {
                key: nearest_neighbour_interp(trafo_grid, data_map.ravel(), ext_grid, unique=False,
                                              max_distance=max_distance)
                for key, data_map in sample.dataset.items()
            }

        trafo_data_maps.append(trafo_data)
        trafo_grids.append(trafo_grid)
        trafo_ver_grids.append(trafo_ver_grid)
        trafo_contours.append(trafo_contour)

    return trafo_data_maps, trafo_grids, trafo_ver_grids, verification_grids, trafo_contours


def average_across_samples(clustering, measurement_samples: list[Sample], trafo_data_maps, trafo_grids, ext_grid):
    """
    Combine the per-sample datasets with one method, together with the grid it lives on and how well it is known.

    "Mean"/"Median"/"Sum" reduce `trafo_data_maps` (already resampled onto `ext_grid`) across samples at
    each grid point, so the result lives on `ext_grid`. "GMM" instead pools every sample's own warped
    points (`trafo_grids`) and clusters them by density, so the result lives on a new set of data-driven
    coordinates unrelated to `ext_grid` (which can be `None` in that case).

    Returns a dict with 'grid', 'dataset' ({key: average}), 'n' ({key: number of samples contributing at each
    point}), 'spread' ({key: spread between samples at each point}) and 'standard_error' ({key: standard error of
    the average}). The spread is the standard deviation for "Mean"/"Sum" and the robust standard deviation
    (1.4826 x median absolute deviation) for "Median" and "GMM" (for GMM: of the pooled values of a component).
    The standard error is spread / sqrt(n) for "Mean" and 1.2533 x spread / sqrt(n) for "Median" and "GMM" (the
    large-sample value for normal data); NaN for "Sum". Both assume independent samples and ignore alignment errors.
    """
    if clustering == 'GMM':
        result = gmm_average(trafo_grids, [s.dataset for s in measurement_samples], num_components='mean')
        return {name: result[name] for name in ('grid', 'dataset', 'n', 'spread', 'standard_error')}
    if clustering not in ("Sum", "Median", "Mean"):
        raise ValueError(f'Clustering option {clustering} is not implemented!')

    projection = {"Sum": np.nansum, "Median": np.nanmedian, "Mean": np.nanmean}[clustering]
    dataset, n_samples, spread, standard_error = {}, {}, {}, {}
    for key in trafo_data_maps[0].keys():
        values = np.array([d[key] for d in trafo_data_maps], dtype=float)
        n_samples[key] = np.isfinite(values).sum(axis=0)
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            dataset[key] = projection(values, axis=0)
            if clustering == 'Median':
                spread[key] = 1.4826 * np.nanmedian(np.abs(values - np.nanmedian(values, axis=0)), axis=0)
            else:
                spread[key] = np.where(n_samples[key] > 1, np.nanstd(values, axis=0, ddof=1), np.nan)
            factor = {'Mean': 1.0, 'Median': 1.2533, 'Sum': np.nan}[clustering]
            standard_error[key] = factor * spread[key] / np.sqrt(n_samples[key])
    return {'grid': ext_grid, 'dataset': dataset, 'n': n_samples, 'spread': spread, 'standard_error': standard_error}


def fuse_measurement_datasets(measurement_samples: list[Sample], trafo_data_maps, trafo_grids, ext_grid,
                              clustering='Mean'):
    """
    Combine the per-sample datasets into a single averaged dataset, together with the grid it lives on
    (see `average_across_samples`, which also returns how many samples contribute and their spread).
    """
    result = average_across_samples(clustering, measurement_samples, trafo_data_maps, trafo_grids, ext_grid)
    return result['dataset'], result['grid']


STATISTICS = ('n', 'spread', 'relative_spread', 'standard_error', 'relative_standard_error')


def average_statistic(analysis, key, method=None, statistic='n', group=None):
    """
    One statistic of an averaged map of a `brain_fusion` result, as (grid, values) on the grid of that method.

    method: 'Mean', 'Median', 'GMM' or 'Sum' (default: the first one of the fusion).
    group: one group of a fusion made with `group_field` (default: the average of all samples).
    statistic:
        'n'                       samples contributing at each point
        'spread'                  spread between samples (see `average_across_samples`)
        'relative_spread'         spread as % of the average
        'standard_error'          standard error of the average (see `average_across_samples`; not defined for 'Sum')
        'relative_standard_error' standard error as % of the average
    """
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}, not {statistic!r}")
    method = method or analysis['averaged_methods'][0]
    entry = analysis['averaged'][method] if group is None else group_average_entry(analysis, group, method)
    name = statistic.replace('relative_', '')
    if name == 'standard_error' and method == 'Sum':
        raise ValueError("A standard error is not defined for the sum of the samples.")
    values = np.asarray(entry[name][key], dtype=float)
    if statistic.startswith('relative_'):
        with np.errstate(all='ignore'):
            values = 100 * values / np.asarray(entry['dataset'][key], dtype=float)
    return entry['grid'], values


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
