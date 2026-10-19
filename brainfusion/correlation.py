from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

from brainfusion.utils import mask_contour
from brainfusion.fusion.grouping import extract_groups, extract_group_native_data, group_average_entry
from brainfusion.fusion.interpolation import resample_points, typical_spacing


def correlate_on_shared_grid(data_a, data_b, grid, contour, name_a='A', name_b='B'):
    """
    Pearson-correlate two datasets that already live on the exact same grid, point-by-point.

    Both `brainfusion.fusion.grouping.group_average_on_shared_grid` (splitting one pooled fusion into
    per-group averages) and `load_fused_analysis` (loading one or more previously-fused analyses back in as
    `Sample`s, then re-fusing them together - e.g. two separately-fused modalities) land their outputs on a
    shared grid, so this is the one function that does the actual correlation once you have two such
    datasets.

    Restricts the correlation itself to points inside `contour` with valid (non-NaN) data in both datasets,
    but returns the two FULL datasets unmasked - apply 'valid_mask' yourself at whichever point you actually
    need the masked values (e.g. `data[result['valid_mask']]` right before plotting).

    Returns
    -------
    dict with keys 'pearson_correlation', 'pearson_p_value', 'spearman_correlation', 'spearman_p_value', 'n_points',
    'valid_mask', name_a and name_b (the two full, unmasked datasets as given). The p-values assume independent points;
    neighbouring grid points of a map are not, so treat them as descriptive.
    """
    data_a, data_b = np.asarray(data_a, dtype=float), np.asarray(data_b, dtype=float)
    inside = mask_contour(contour, grid)
    valid = inside & ~np.isnan(data_a) & ~np.isnan(data_b)

    if valid.sum() < 2:
        raise ValueError(f"Not enough valid overlapping points to correlate ({int(valid.sum())} found).")

    correlation, p_value = pearsonr(data_a[valid], data_b[valid])
    rank_correlation, rank_p_value = spearmanr(data_a[valid], data_b[valid])
    return {
        'pearson_correlation': correlation,
        'pearson_p_value': p_value,
        'spearman_correlation': rank_correlation,
        'spearman_p_value': rank_p_value,
        'n_points': int(valid.sum()),
        'valid_mask': valid,
        name_a: data_a,
        name_b: data_b,
    }


def compute_max_radius(grid):
    """Non-overlapping radius for each point in `grid`: half the distance to its nearest neighbor."""
    tree = cKDTree(grid)
    distances, _ = tree.query(grid, k=2)  # k=2: first match is the point itself
    return distances[:, 1] / 2


def average_within_radius(reference_grid, other_grid, other_data, radius='max', average_func=np.nanmean,
                          exclusive=True):
    """
    For each point in `reference_grid`, average every `other_data` value whose `other_grid` position falls
    within `radius` of it.

    `radius` is
      - 'max' (default): `compute_max_radius(reference_grid)` - the largest radius around each reference point
        that still doesn't overlap its neighbors' circles;
      - one number: the same radius everywhere;
      - a list/array with one radius per reference point, e.g. the footprint of each measurement (the contact
        radius of an indentation, the focal spot of a point measurement, ...).
    Radii are in the units of the grids.

    `exclusive=True` (default): each `other_grid` point is used by at most one reference point (whichever it falls
    within first), so overlapping circles never double-count a value. `exclusive=False`: every reference point
    averages everything inside its own circle, so a value may enter several averages - what you want when the
    circles are physical footprints that may overlap.

    Returns
    -------
    avg_values : (len(reference_grid),) array
        `average_func` of every matched `other_data` value; NaN where nothing fell within radius.
    radius : (len(reference_grid),) array
        The radius actually used at each reference point.
    """
    if isinstance(radius, str):
        if radius != 'max':
            raise ValueError(f"radius must be 'max', a number or one number per reference point, not {radius!r}")
        radius = compute_max_radius(reference_grid)
    else:
        radius = np.broadcast_to(np.asarray(radius, dtype=float), (len(reference_grid),)).copy()

    other_data = np.asarray(other_data, dtype=float)
    tree = cKDTree(other_grid)
    avg_values = np.full(len(reference_grid), np.nan)
    assigned_mask = np.zeros(len(other_grid), dtype=bool)

    for i, point in enumerate(reference_grid):
        indices = tree.query_ball_point(point, radius[i])
        valid_indices = [idx for idx in indices if not (exclusive and assigned_mask[idx])]
        if valid_indices:
            avg_values[i] = average_func(other_data[valid_indices])
            assigned_mask[valid_indices] = True

    return avg_values, radius


def correlate_around_reference_grid(reference_data, reference_grid, other_data, other_grid, contour,
                                    radius='max', average_func=np.nanmean, name_a='reference', name_b='other',
                                    exclusive=True):
    """
    Pearson-correlate two datasets that live on DIFFERENT grids of noticeably different density (e.g. sparse
    AFM indentation points vs a dense per-pixel Brillouin/myelin image), by picking one of them as the
    reference grid and locally averaging the other's values within a non-overlapping radius around each
    reference point - instead of resampling both onto a third, in-between shared grid the way
    `correlate_on_shared_grid`/`group_average_on_shared_grid` do.

    Prefer this over the shared-grid route whenever the two densities differ a lot: forcing both onto one
    intermediate regular grid (`group_average_on_shared_grid`'s `extend_grid`) ends up upsampling the sparse
    side (nearest-neighbour duplicates the same sparse value across several new grid points) and aliasing
    the dense side (picks one nearest raw value per new grid point instead of properly averaging the many
    real ones nearby) whenever the grids' native spacings are far apart. Radius-averaging the dense side
    down onto the sparse side's own real points avoids both problems, at the cost of not living on a nice
    regular grid afterward (so it isn't directly usable with `plot_average_map`'s image mode, and
    `pairwise_correlate_groups` doesn't apply - each pair needs the two grids passed to this function
    directly, once per pair).

    IMPORTANT: pass the SPARSER dataset as the reference (`reference_data`/`reference_grid`), not the denser
    one. With `radius='max'` the search radius is sized off the reference grid's OWN nearest-neighbour
    spacing - a sparse reference gets a generously large radius that comfortably reaches many nearby dense
    points, but a dense reference gets a tiny radius that can miss the sparser side entirely (in the worst
    case, matching 0 points and raising the "not enough points" error below). Also make sure both grids are
    already warped into the SAME template coordinate space (e.g. both are `measurement_trafo_grids`/
    `measurement_datasets` entries from one shared `run_fusion` call) - two samples' own raw native grids
    aren't spatially comparable at all before that.

    Only reference points inside `contour` are used, but - same as `correlate_on_shared_grid` - the
    correlation itself is restricted internally while the RETURNED data stays full (here: every
    inside-contour reference point, including ones with no radius match); apply 'valid_mask' yourself at
    whichever point you actually need the masked values, e.g. `result[name_a][result['valid_mask']]`. See
    `plot_correlation_with_radii` to sanity-check the chosen radii visually before trusting the correlation.

    Returns
    -------
    dict with keys 'pearson_correlation', 'pearson_p_value', 'spearman_correlation', 'spearman_p_value', 'n_points',
    'valid_mask', 'radii',
    'reference_grid' (restricted to points inside `contour`, matching 'radii'/name_a/name_b 1:1), name_a
    (the reference data, inside-contour only) and name_b (the radius-averaged 'other' value at each
    reference point, NaN where nothing fell within radius).
    """
    inside = mask_contour(contour, reference_grid)
    reference_grid = reference_grid[inside]
    reference_data = np.asarray(reference_data, dtype=float)[inside]

    if not isinstance(radius, str) and np.ndim(radius) > 0:   # one radius per reference point: keep the inside ones
        radius = np.asarray(radius, dtype=float)[inside]
    avg_other, radii = average_within_radius(reference_grid, other_grid, np.asarray(other_data, dtype=float),
                                             radius, average_func, exclusive)
    valid = ~np.isnan(avg_other)
    if valid.sum() < 2:
        raise ValueError(f"Not enough valid overlapping points to correlate ({int(valid.sum())} found).")

    correlation, p_value = pearsonr(reference_data[valid], avg_other[valid])
    rank_correlation, rank_p_value = spearmanr(reference_data[valid], avg_other[valid])
    return {
        'pearson_correlation': correlation,
        'pearson_p_value': p_value,
        'spearman_correlation': rank_correlation,
        'spearman_p_value': rank_p_value,
        'n_points': int(valid.sum()),
        'valid_mask': valid,
        'radii': radii,
        'reference_grid': reference_grid,
        name_a: reference_data,
        name_b: avg_other,
    }


def correlate_group_maps(analysis, key_quant, group_a, group_b, method=None, resample='gaussian', sigma='auto',
                         radius=None, target='auto', min_weight=0.25):
    """
    Correlate the averaged maps of two groups of a fusion made with `group_field`, whether or not they live on the same
    points. With a method on the shared grid (Mean, Median, Sum) the two maps share their points and are correlated
    directly - also with "GMM", whose groups all live on the components of one clustering of all samples. Only maps on
    different grids (e.g. from separate fusions) are resampled: the finer of the two maps onto the points of the coarser one (`target='auto'`; or 'a' / 'b'), by `resample` ('gaussian', 'radius' or 'nearest', see
    `resample_points`; `sigma`, `radius`, `min_weight` as there), and the correlation is taken over the points of the
    target inside the outline that have a value in both.

    method: which clustering of the fusion (default: its first). Returns what `correlate_on_shared_grid` returns, with the
    two groups' values (only the points used) under their group names, plus 'grid' (those points), 'contour' and 'resample'
    (what was done).
    """
    entry_a, entry_b = (group_average_entry(analysis, g, method) for g in (group_a, group_b))
    contour = analysis['template_contours'][0]
    grid_a, grid_b = np.asarray(entry_a['grid'], dtype=float), np.asarray(entry_b['grid'], dtype=float)
    data_a = np.asarray(entry_a['dataset'][key_quant], dtype=float)
    data_b = np.asarray(entry_b['dataset'][key_quant], dtype=float)

    if grid_a.shape == grid_b.shape and np.array_equal(grid_a, grid_b):
        result = correlate_on_shared_grid(data_a, data_b, grid_a, contour, name_a=group_a, name_b=group_b)
        keep = result['valid_mask']
        result.update({group_a: data_a[keep], group_b: data_b[keep], 'grid': grid_a[keep], 'contour': contour,
                       'resample': 'none: shared grid'})
        return result

    if target not in ('auto', 'a', 'b'):
        raise ValueError(f"target must be 'auto', 'a' or 'b', not {target!r}")
    if target == 'auto':
        target = 'a' if typical_spacing(grid_a) >= typical_spacing(grid_b) else 'b'
    if target == 'a':
        reference_grid, reference_data, other_grid, other_data = grid_a, data_a, grid_b, data_b
    else:
        reference_grid, reference_data, other_grid, other_data = grid_b, data_b, grid_a, data_a
    other_on_reference = resample_points(other_grid, other_data, reference_grid, method=resample, sigma=sigma,
                                         radius=radius, min_weight=min_weight)
    values_a, values_b = (reference_data, other_on_reference) if target == 'a' else (other_on_reference, reference_data)
    result = correlate_on_shared_grid(values_a, values_b, reference_grid, contour, name_a=group_a, name_b=group_b)
    keep = result['valid_mask']
    result.update({group_a: values_a[keep], group_b: values_b[keep], 'grid': reference_grid[keep], 'contour': contour,
                   'resample': f"{resample} onto the points of group {group_a if target == 'a' else group_b}"})
    return result


def pairwise_correlate_groups(group_maps, grid, contour, key_quant, groups=None):
    """
    Pearson-correlate every pair of groups in `group_maps` (as returned by `group_average_on_shared_grid`)
    for one dataset key - with 2 groups that's a single pair, with N groups it's every unordered pair
    (e.g. 'Control', 'CS', 'Treated' gives ('Control','CS'), ('Control','Treated'), ('CS','Treated')), so
    nothing needs to be picked out or hardcoded by name.

    Parameters
    ----------
    group_maps : dict {group: {key: (P,) array}}
        As returned by `group_average_on_shared_grid`.
    grid, contour : arrays
        As returned by `group_average_on_shared_grid`.
    key_quant : str
        Which dataset key to correlate (`group_maps` may hold several).
    groups : list, optional
        Which groups to include, and in what order. Defaults to every group in `group_maps`.

    Returns
    -------
    dict {(group_a, group_b): result}
        One `correlate_on_shared_grid` result per unordered group pair.
    """
    groups = list(group_maps.keys()) if groups is None else groups
    return {
        (group_a, group_b): correlate_on_shared_grid(
            group_maps[group_a][key_quant], group_maps[group_b][key_quant], grid, contour,
            name_a=group_a, name_b=group_b)
        for group_a, group_b in combinations(groups, 2)
    }


def correlate_groups(analysis, key_quant, groups=None):
    """
    Pearson-correlate every pair of an analysis's groups on their shared grid - the `analysis`-level
    counterpart of `pairwise_correlate_groups`, so callers only need the analysis itself plus which key to
    correlate. Requires `analysis` to have been fused with `run_fusion(..., group_field=...)`, so
    `group_datasets` is already baked in (no separate extraction step).

    Parameters
    ----------
    analysis : dict
        A `run_fusion`/`brain_fusion` result, fused with `group_field=...`.
    key_quant : str
        Dataset key to correlate.
    groups : list, optional
        Which groups to include, and in what order. Defaults to `extract_groups(analysis)`.

    Returns
    -------
    dict {(group_a, group_b): result}
        Each `pairwise_correlate_groups` result, plus 'grid' and 'contour' (the same shared grid/contour
        every pair was correlated on) so plotting (`plot_correlation_masks`, `plot_norm_corr`) can pull
        everything it needs straight off the result instead of the caller re-fetching them from `analysis`.
    """
    groups = extract_groups(analysis) if groups is None else groups
    grid = analysis['measurement_interpolated_grid']
    contour = analysis['template_contours'][0]
    results = pairwise_correlate_groups(analysis['group_datasets'], grid, contour, key_quant, groups=groups)
    for result in results.values():
        result['grid'] = grid
        result['contour'] = contour
    return results


def correlate_groups_by_density(analysis, group_field, key_quant, groups=None, priority=None, radius='max',
                                average_func=np.nanmean):
    """
    Pearson-correlate an analysis's groups whose native point densities differ too much to share one grid -
    the `analysis`-level counterpart of `pairwise_correlate_by_density`, so callers only need the analysis
    itself plus which metadata field to split by. Splits `analysis` by `group_field` at NATIVE
    (non-resampled) resolution via `extract_group_native_data`, then radius-matches every pair.

    Parameters
    ----------
    analysis : dict
        A `run_fusion`/`brain_fusion` result covering every group to compare - they must already be warped
        into the same template coordinate space via one shared `run_fusion` call. `group_field` does NOT
        need to be the field `run_fusion` was fused with, if any (e.g. group by 'modality' post-hoc even if
        `run_fusion` itself used `group_field='condition'`).
    group_field : str
        Key into each sample's `.metadata` dict to group by, e.g. 'modality'.
    key_quant : str
        Dataset key to correlate.
    groups, priority, radius, average_func :
        Passed through to `extract_group_native_data`/`pairwise_correlate_by_density`.

    Returns
    -------
    dict {(sparse_name, dense_name): result}
        Each `pairwise_correlate_by_density` result, plus 'contour' and 'dense_grid'/'dense_data' (the denser
        side's own raw grid/data, before radius-averaging) so plotting (`plot_correlation_with_radii`,
        `plot_norm_corr`) can pull everything it needs straight off the result instead of the caller
        re-deriving it.
    """
    datasets = extract_group_native_data(analysis, group_field, key_quant, groups=groups)
    contour = analysis['template_contours'][0]
    results = pairwise_correlate_by_density(datasets, contour, priority=priority, radius=radius,
                                            average_func=average_func)
    for (sparse_name, dense_name), result in results.items():
        result['contour'] = contour
        result['dense_grid'], result['dense_data'] = datasets[dense_name]
    return results


def pairwise_correlate_by_density(datasets, contour, priority=None, radius='max', average_func=np.nanmean):
    """
    Pearson-correlate every pair in `datasets` whose point densities differ too much to share one grid -
    generalizes `correlate_around_reference_grid` (built for exactly 2 datasets) to N, e.g. 3+ modalities or
    the groups returned by `brainfusion.fusion.grouping.extract_group_native_data`.

    For each pair, the SPARSER dataset (fewer points) is used as the reference grid and the denser one is
    radius-averaged down onto it - see `correlate_around_reference_grid`'s docstring for why this direction
    matters. With only 2 datasets, comparing point counts always picks the right one automatically; with 3+,
    pairwise point counts can disagree with the "true" overall density ordering (e.g. A happens to have more
    points than B in this particular scan, even though B's imaging modality is intrinsically denser) - pass
    `priority` (every name in `datasets`, SPARSEST first) to fix the ordering explicitly instead of trusting
    point counts pair-by-pair.

    Parameters
    ----------
    datasets : dict {name: (grid, data)}
        E.g. from `extract_group_native_data`, or built by hand from separate analyses' own
        `measurement_interpolated_grid`/`measurement_interpolated_dataset` (each already warped into the
        same template coordinate space via one shared `run_fusion` call).
    contour : array
        Template contour shared by every dataset.
    priority : list of str, optional
        Every key of `datasets`, sparsest first. Defaults to auto-picking the sparser side of each pair by
        point count.
    radius, average_func :
        Passed through to `correlate_around_reference_grid`.

    Returns
    -------
    dict {(sparse_name, dense_name): result}
        One `correlate_around_reference_grid` result per unordered pair, keyed sparse-name first.
    """
    names = list(datasets.keys())
    rank = {name: priority.index(name) for name in names} if priority is not None else None

    results = {}
    for name_a, name_b in combinations(names, 2):
        if rank is not None:
            sparse_name, dense_name = (name_a, name_b) if rank[name_a] < rank[name_b] else (name_b, name_a)
        else:
            sparse_name, dense_name = (name_a, name_b) if len(datasets[name_a][0]) <= len(datasets[name_b][0]) \
                else (name_b, name_a)

        sparse_grid, sparse_data = datasets[sparse_name]
        dense_grid, dense_data = datasets[dense_name]
        results[(sparse_name, dense_name)] = correlate_around_reference_grid(
            sparse_data, sparse_grid, dense_data, dense_grid, contour, radius=radius, average_func=average_func,
            name_a=sparse_name, name_b=dense_name)

    return results
