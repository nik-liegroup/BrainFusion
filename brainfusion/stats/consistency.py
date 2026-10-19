"""How alike are the maps of the animals of one fusion, and how reproducible is their average map?

All functions work on the per-animal maps of a finished fusion on its shared grid (`animal_maps`), inside the template outline.
"""

import warnings
from itertools import combinations

import numpy as np
from scipy.stats import rankdata

from brainfusion.utils import mask_contour

AVERAGES = {'mean': np.nanmean, 'median': np.nanmedian}


def animal_maps(analysis, key, group=None, group_field=None):
    """
    The maps of the samples (animals) of a fusion on its shared grid: (maps (n samples, n grid points), grid (n points, 2),
    inside (n points: inside the template outline), metadata list). `group`: only the samples with
    metadata[group_field] == group (group_field default: the one the fusion was made with). Needs a fusion with a clustering
    on the shared grid (Mean, Median or Sum).
    """
    per_sample = analysis.get('measurement_trafo_datasets')
    if not per_sample or per_sample[0] is None:
        raise ValueError("This analysis has no per-sample maps on a shared grid - fuse with clustering 'Mean', 'Median' or "
                         "'Sum' (a fusion with 'GMM' only has none).")
    grid = np.asarray(analysis['measurement_interpolated_grid'], dtype=float).reshape(-1, 2)
    metadata = list(analysis['measurement_metadata'])
    index = np.arange(len(per_sample))
    if group is not None:
        field = group_field or analysis.get('group_field')
        if field is None:
            raise ValueError("A group was given but the analysis has no group_field and none was passed.")
        index = np.array([i for i in index if str(metadata[i][field]) == str(group)], dtype=int)
        if len(index) == 0:
            raise ValueError(f"No samples with metadata['{field}'] == '{group}'.")
    maps = np.array([np.asarray(per_sample[i][key], dtype=float).ravel() for i in index])
    inside = mask_contour(analysis['template_contours'][0], grid)
    return maps, grid, inside, [metadata[i] for i in index]


def average_map(maps, how='median'):
    """The mean or median over the animals at every grid point (NaN values ignored; NaN where there is none)."""
    if how not in AVERAGES:
        raise ValueError(f"average must be one of {list(AVERAGES)}, not {how!r}")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='(All-NaN slice|Mean of empty slice)')
        return AVERAGES[how](maps, axis=0)


def _pearson(a, b, valid, min_overlap):
    if valid.sum() < min_overlap:
        return np.nan
    a, b = a[valid], b[valid]
    return np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan


def _spearman(a, b, valid, min_overlap):
    if valid.sum() < min_overlap:
        return np.nan
    return _pearson(rankdata(a[valid]), rankdata(b[valid]), np.ones(valid.sum(), dtype=bool), 1)


def pairwise_map_correlations(maps, inside, min_overlap=30):
    """Pearson and Spearman correlation of the maps of every pair of animals over the points both have, as two arrays."""
    valid = np.isfinite(maps) & inside
    pairs = list(combinations(range(len(maps)), 2))
    pearson = np.array([_pearson(maps[i], maps[j], valid[i] & valid[j], min_overlap) for i, j in pairs])
    spearman = np.array([_spearman(maps[i], maps[j], valid[i] & valid[j], min_overlap) for i, j in pairs])
    return pearson, spearman


def split_half_reliability(maps, inside, average='median', n_splits=200, seed=0, min_overlap=30, splits=None):
    """
    Reproducibility of the average map: the animals are split at random into two halves, each half is averaged, and the two
    average maps are correlated; the correlation is corrected with the Spearman-Brown formula to the reliability of the
    average of ALL animals. Mean over `n_splits` random splits. Returns {'r': mean, 'sd': sd, 'values': array}.
    `splits`: the animal orders to use (a list of permutations), to compare several averages on the same splits.
    """
    n = len(maps)
    if n < 4:
        return {'r': np.nan, 'sd': np.nan, 'values': np.array([])}
    if splits is None:
        rng = np.random.default_rng(seed)
        splits = [rng.permutation(n) for _ in range(n_splits)]
    values = []
    for order in splits:
        a = average_map(maps[order[: n // 2]], average)
        b = average_map(maps[order[n // 2: 2 * (n // 2)]], average)
        r = _pearson(a, b, np.isfinite(a) & np.isfinite(b) & inside, min_overlap)
        values.append(2 * r / (1 + r) if np.isfinite(r) else np.nan)
    values = np.array(values)
    return {'r': float(np.nanmean(values)), 'sd': float(np.nanstd(values)), 'values': values}


def map_consistency(maps, inside, n_splits=200, seed=0, min_overlap=30, min_animals_point=3):
    """
    How consistent are the maps of the animals with each other (maps: n animals x n grid points, inside: which grid points
    count). Returns a nested dict:
      'n_animals', 'animal_values' (the mean of every animal's map), 'value_mean', 'value_sd', 'value_cv_pct'
      'pairwise'       Pearson r of every pair of animals: mean, median, p25, p75, and the mean Spearman rho
      'leave_one_out'  for the 'mean' and the 'median' average map: the correlation of every animal's map with the average of
                       all the others - mean, min, max over animals
      'split_half'     for the 'mean' and the 'median' average map: reproducibility of the average map (see
                       `split_half_reliability`; the same random splits for both)
      'pointwise'      between-animal spread at a grid point (median over points; SD, and as % of the mean) and the signal
                       ratio: variance of the average map over the grid / mean variance between animals at a grid point
                       (> 1: the spatial pattern stands out of the animal-to-animal scatter); for the median map with the
                       robust variance (1.4826 x MAD)
    """
    n = len(maps)
    valid_maps = np.isfinite(maps) & inside
    values = np.array([maps[i][valid_maps[i]].mean() if valid_maps[i].any() else np.nan for i in range(n)])
    result = {'n_animals': n, 'animal_values': values, 'value_mean': float(np.nanmean(values)),
              'value_sd': float(np.nanstd(values, ddof=1)), 'value_cv_pct': float(100 * np.nanstd(values, ddof=1) / np.nanmean(values))}

    pearson, spearman = pairwise_map_correlations(maps, inside, min_overlap)
    result['pairwise'] = {'n_pairs': int(np.isfinite(pearson).sum()), 'pearson_mean': float(np.nanmean(pearson)),
                          'pearson_median': float(np.nanmedian(pearson)), 'pearson_p25': float(np.nanpercentile(pearson, 25)),
                          'pearson_p75': float(np.nanpercentile(pearson, 75)), 'spearman_mean': float(np.nanmean(spearman)),
                          'pearson': pearson}

    result['leave_one_out'] = {}
    for how in AVERAGES:
        loo = []
        for i in range(n):
            others = average_map(np.delete(maps, i, axis=0), how)
            loo.append(_pearson(maps[i], others, valid_maps[i] & np.isfinite(others) & inside, min_overlap))
        loo = np.array(loo)
        result['leave_one_out'][how] = {'r_mean': float(np.nanmean(loo)), 'r_min': float(np.nanmin(loo)),
                                        'r_max': float(np.nanmax(loo)), 'r': loo}

    rng = np.random.default_rng(seed)
    splits = [rng.permutation(n) for _ in range(n_splits)] if n >= 4 else []
    result['split_half'] = {how: split_half_reliability(maps, inside, how, min_overlap=min_overlap, splits=splits)
                            for how in AVERAGES}

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='(All-NaN slice|Mean of empty slice|Degrees of freedom)')
        masked = np.where(valid_maps, maps, np.nan)
        enough = valid_maps.sum(axis=0) >= min_animals_point
        sd = np.nanstd(masked, axis=0, ddof=1)
        mean = np.nanmean(masked, axis=0)
        middle = np.nanmedian(masked, axis=0)
        robust_sd = 1.4826 * np.nanmedian(np.abs(masked - middle), axis=0)
        result['pointwise'] = {
            'sd_median': float(np.nanmedian(sd[enough])), 'cv_pct_median': float(100 * np.nanmedian((sd / mean)[enough])),
            'signal_ratio_mean_map': float(np.nanvar(mean[enough]) / np.nanmean(sd[enough] ** 2)),
            'signal_ratio_median_map': float(np.nanvar(middle[enough]) / np.nanmean(robust_sd[enough] ** 2))}
    return result
