from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, fisher_exact

from brainfusion.utils import mask_contour


def list_groups(analysis, group_field):
    """List the distinct values found under `metadata[group_field]` across an analysis's samples."""
    return sorted({str(m[group_field]) for m in analysis['measurement_metadata']})


def group_average_on_shared_grid(analysis, group_field, groups=None):
    """
    Split a pooled `run_fusion`/`brain_fusion` result into per-group averaged maps that all live on the
    analysis's own shared grid, using each sample's `.metadata[group_field]` (e.g. `group_field='condition'`
    to split a pooled Control+CS fusion back into a 'Control' and a 'CS' average).

    Every dataset key present on the samples (e.g. 'modulus' and, if loaded, 'beta_pyforce') is averaged in
    one pass - there is no per-quantity cost to computing them all, so there's no reason to ask for just
    one up front. Pull out whichever key you need downstream, e.g. `group_maps['Control']['modulus']`.

    Because every sample was already resampled onto the same `measurement_interpolated_grid` inside
    `fuse_grids` (stored as `measurement_trafo_datasets`), the returned group averages land on identical
    grid points with no extra spatial matching needed - pass them straight to `correlate_on_shared_grid` or
    `pairwise_correlate_groups`.

    Only works when `clustering` was 'Mean', 'Median' or 'Sum': 'GMM' clustering builds its own
    data-driven grid instead of resampling every sample onto one shared grid, so there is nothing here to
    split by group.

    Parameters
    ----------
    analysis : dict
        A `run_fusion`/`brain_fusion` result (pooled across every group you want to compare).
    group_field : str
        Key into each sample's `.metadata` dict to group by, e.g. 'condition'.
    groups : list, optional
        Which group values to compute, and in what order. Defaults to every unique value found
        (see `list_groups`).

    Returns
    -------
    grid : (P, 2) array
        The shared grid every group average lives on.
    contour : (N, 2) array
        The analysis's template contour.
    group_maps : dict {group_value: {key: (P,) array}}
        `nanmean` of every dataset key across each group's samples.
    """
    per_sample = analysis.get('measurement_trafo_datasets')
    if not per_sample or per_sample[0] is None:
        raise ValueError("This analysis has no per-sample data resampled onto a shared grid - re-run "
                         "run_fusion/brain_fusion with clustering='Mean', 'Median' or 'Sum' (not 'GMM').")

    values = np.array([m[group_field] for m in analysis['measurement_metadata']])
    groups = list_groups(analysis, group_field) if groups is None else groups
    keys = per_sample[0].keys()

    group_maps = {}
    for group in groups:
        sample_idx = np.where(values == group)[0]
        if len(sample_idx) == 0:
            raise ValueError(f"No samples found with metadata['{group_field}'] == '{group}'.")
        group_maps[group] = {key: np.nanmean(np.array([per_sample[i][key] for i in sample_idx]), axis=0)
                             for key in keys}

    return analysis['measurement_interpolated_grid'], analysis['template_contours'][0], group_maps


def correlate_on_shared_grid(data_a, data_b, grid, contour, name_a='A', name_b='B'):
    """
    Pearson-correlate two datasets that already live on the exact same grid, point-by-point.

    Both `group_average_on_shared_grid` (splitting one pooled fusion into per-group averages) and
    `load_fused_analysis` (loading one or more previously-fused analyses back in as `Sample`s, then
    re-fusing them together - e.g. two separately-fused modalities) land their outputs on a shared grid,
    so this is the one function that does the actual correlation once you have two such datasets.

    Restricts the comparison to points inside `contour` with valid (non-NaN) data in both datasets.

    Returns
    -------
    dict with keys 'pearson_correlation', 'pearson_p_value', 'n_points', 'valid_mask',
    f'{name_a}_valid' and f'{name_b}_valid' (the two datasets, already indexed down to the valid points -
    ready to hand to a scatter/plotting function).
    """
    data_a, data_b = np.asarray(data_a, dtype=float), np.asarray(data_b, dtype=float)
    inside = mask_contour(contour, grid)
    valid = inside & ~np.isnan(data_a) & ~np.isnan(data_b)

    if valid.sum() < 2:
        raise ValueError(f"Not enough valid overlapping points to correlate ({int(valid.sum())} found).")

    correlation, p_value = pearsonr(data_a[valid], data_b[valid])
    return {
        'pearson_correlation': correlation,
        'pearson_p_value': p_value,
        'n_points': int(valid.sum()),
        'valid_mask': valid,
        f'{name_a}_valid': data_a[valid],
        f'{name_b}_valid': data_b[valid],
    }


def compute_max_radius(grid):
    """Non-overlapping radius for each point in `grid`: half the distance to its nearest neighbor."""
    tree = cKDTree(grid)
    distances, _ = tree.query(grid, k=2)  # k=2: first match is the point itself
    return distances[:, 1] / 2


def average_within_radius(reference_grid, other_grid, other_data, radius='max', average_func=np.nanmean):
    """
    For each point in `reference_grid`, average every `other_data` value whose `other_grid` position falls
    within `radius` of it. Each `other_grid` point is used by at most one reference point (whichever it
    falls within first), so overlapping radii never double-count a value.

    `radius='max'` (default) uses `compute_max_radius(reference_grid)` - the largest radius around each
    reference point that still doesn't overlap its neighbors' circles. Pass a single number instead to use
    the same radius everywhere.

    Returns
    -------
    avg_values : (len(reference_grid),) array
        `average_func` of every matched `other_data` value; NaN where nothing fell within radius.
    radius : (len(reference_grid),) array
        The radius actually used at each reference point.
    """
    if radius == 'max':
        radius = compute_max_radius(reference_grid)
    elif isinstance(radius, (float, int)):
        radius = np.full(len(reference_grid), radius)

    tree = cKDTree(other_grid)
    avg_values = np.full(len(reference_grid), np.nan)
    assigned_mask = np.zeros(len(other_grid), dtype=bool)

    for i, point in enumerate(reference_grid):
        indices = tree.query_ball_point(point, radius[i])
        valid_indices = [idx for idx in indices if not assigned_mask[idx]]
        if valid_indices:
            avg_values[i] = average_func(other_data[valid_indices])
            assigned_mask[valid_indices] = True

    return avg_values, radius


def correlate_around_reference_grid(reference_data, reference_grid, other_data, other_grid, contour,
                                    radius='max', average_func=np.nanmean, name_a='reference', name_b='other'):
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

    Only reference points inside `contour` are used. See `plot_correlation_with_radii` to sanity-check the
    chosen radii visually before trusting the correlation.

    Returns
    -------
    dict with keys 'pearson_correlation', 'pearson_p_value', 'n_points', 'valid_mask', 'radii',
    'reference_grid' (already restricted to points inside `contour`), and f'{name_a}_valid'/f'{name_b}_valid'
    (the two datasets at the valid reference points, ready for a scatter/plotting function).
    """
    inside = mask_contour(contour, reference_grid)
    reference_grid = reference_grid[inside]
    reference_data = np.asarray(reference_data, dtype=float)[inside]

    avg_other, radii = average_within_radius(reference_grid, other_grid, np.asarray(other_data, dtype=float),
                                             radius, average_func)
    valid = ~np.isnan(avg_other)
    if valid.sum() < 2:
        raise ValueError(f"Not enough valid overlapping points to correlate ({int(valid.sum())} found).")

    correlation, p_value = pearsonr(reference_data[valid], avg_other[valid])
    return {
        'pearson_correlation': correlation,
        'pearson_p_value': p_value,
        'n_points': int(valid.sum()),
        'valid_mask': valid,
        'radii': radii,
        'reference_grid': reference_grid,
        f'{name_a}_valid': reference_data[valid],
        f'{name_b}_valid': avg_other[valid],
    }


def analyse_correlation_percentile(data_a, data_b, percentile_a=50, percentile_b=50, name_a="A", name_b="B"):
    """
    Beyond a plain Pearson correlation, also test co-occurrence: threshold each dataset at its own
    percentile (default: median) to call each point "present"/"absent" in that dataset, then run a Fisher
    exact test on the resulting 2x2 contingency table - e.g. "are high-stiffness points more likely than
    chance to also be high-myelin points", independent of the linear Pearson relationship.

    `data_a`/`data_b` must already be paired point-by-point - e.g. `correlate_around_reference_grid`'s or
    `correlate_on_shared_grid`'s `f'{name_a}_valid'`/`f'{name_b}_valid'` outputs.
    """
    data_a, data_b = np.asarray(data_a), np.asarray(data_b)
    if len(data_a) < 2:
        raise ValueError("Not enough values to correlate.")

    correlation, p_value = pearsonr(data_a, data_b)
    a_present = data_a >= np.percentile(data_a, percentile_a)
    b_present = data_b >= np.percentile(data_b, percentile_b)
    cond_prop_table, stats_results = conditional_probability_table(a_present, b_present, name_a, name_b)

    results = {
        "pearson_correlation": correlation,
        "pearson_p_value": p_value,
        f"present_percentile_{name_a}": percentile_a,
        f"present_percentile_{name_b}": percentile_b,
        "probability_table": cond_prop_table,
    }
    return results, stats_results, a_present, b_present


def conditional_probability_table(a_present, b_present, name_a="A", name_b="B"):
    """The 2x2 contingency table + Fisher exact test behind `analyse_correlation_percentile`."""
    assert len(a_present) == len(b_present)
    a_absent, b_absent = ~a_present, ~b_present

    def safe_div(numerator, denominator):
        return numerator / denominator if denominator > 0 else np.nan

    n11 = np.sum(a_present & b_present)
    n12 = np.sum(a_present & b_absent)
    n21 = np.sum(a_absent & b_present)
    n22 = np.sum(a_absent & b_absent)

    cond_prop_table = pd.DataFrame({
        f"{name_b} absent": [safe_div(n22, np.sum(b_absent)), safe_div(n12, np.sum(b_absent))],
        f"{name_b} present": [safe_div(n21, np.sum(b_present)), safe_div(n11, np.sum(b_present))],
    }, index=[f"{name_a} absent", f"{name_a} present"])

    contingency_table = np.array([[n22, n21], [n12, n11]])
    oddsratio, p_value = fisher_exact(contingency_table)

    # Confidence interval for the odds ratio (Woolf's method): log(OR) +/- 1.96 * SE(log(OR))
    with np.errstate(divide='ignore', invalid='ignore'):
        log_or = np.log(oddsratio)
        se_log_or = np.sqrt(1 / contingency_table[0, 0] + 1 / contingency_table[0, 1] +
                            1 / contingency_table[1, 0] + 1 / contingency_table[1, 1])
        ci_low, ci_high = np.exp(log_or - 1.96 * se_log_or), np.exp(log_or + 1.96 * se_log_or)

    stats_results = {
        "contingency_table": contingency_table,
        "odds_ratio": oddsratio,
        "fisher_p_value": p_value,
        "odds_ratio_CI_95": (ci_low, ci_high),
    }
    return cond_prop_table, stats_results


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
