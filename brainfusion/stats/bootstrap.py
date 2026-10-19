"""Uncertainty of the correlation of two average maps: resample the animals."""

import numpy as np
from scipy.stats import norm, rankdata

from brainfusion.stats.consistency import animal_maps, average_map


def _correlations(x, y, inside, min_points):
    valid = inside & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < min_points:
        return np.nan, np.nan
    x, y = x[valid], y[valid]
    if x.std() == 0 or y.std() == 0:
        return np.nan, np.nan
    return np.corrcoef(x, y)[0, 1], np.corrcoef(rankdata(x), rankdata(y))[0, 1]


def bootstrap_map_correlation(maps_a, maps_b, inside, n_boot=1000, average='median', confidence=0.95, seed=0, paired=False,
                              min_points=10):
    """
    Correlation of the average map of group A with that of group B, and its uncertainty from the animals: in every bootstrap
    draw the animals of each group are resampled with replacement, the average maps are recomputed and correlated.
    `paired=True` resamples the same animals in both groups (the maps are then of the same animals, in the same order); the
    default treats the two groups as different animals. The interval describes how much the animals one happened to
    measure matter - not the point-wise noise within a map, and not the spatial autocorrelation (see `spatial_null_test`
    for that). A resampled set repeats animals and so averages less noise away than the original: the draws sit below the
    estimate (clearly so for noisy maps and few animals). The interval 'pearson_ci' therefore uses only the spread of the draws:
    the estimate +/- z x SD of the Fisher-transformed draws, back-transformed. The percentile interval of the draws is given
    as 'pearson_ci_percentile' for comparison; it is shifted down by that bias.

    maps_a, maps_b: (animals, grid points) on the same grid; inside: which grid points count.
    Returns a dict: 'pearson', 'spearman' (of the average maps of all animals), 'pearson_ci', 'spearman_ci' (low, high),
    'pearson_ci_percentile', 'spearman_ci_percentile', 'pearson_boot', 'spearman_boot' (the draws), 'n_points', 'n_a', 'n_b', 'n_boot', 'average', 'confidence'.
    """
    maps_a, maps_b = np.asarray(maps_a, dtype=float), np.asarray(maps_b, dtype=float)
    if paired and len(maps_a) != len(maps_b):
        raise ValueError("paired=True needs the same number of animals in both groups")
    pearson, spearman = _correlations(average_map(maps_a, average), average_map(maps_b, average), inside, min_points)
    valid = inside & np.isfinite(average_map(maps_a, average)) & np.isfinite(average_map(maps_b, average))

    rng = np.random.default_rng(seed)
    pearson_boot, spearman_boot = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        index_a = rng.integers(0, len(maps_a), len(maps_a))
        index_b = index_a if paired else rng.integers(0, len(maps_b), len(maps_b))
        pearson_boot[i], spearman_boot[i] = _correlations(average_map(maps_a[index_a], average),
                                                          average_map(maps_b[index_b], average), inside, min_points)
    low, high = 100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2
    critical = norm.ppf(0.5 + confidence / 2)

    def normal_interval(estimate, draws):
        se = np.nanstd(np.arctanh(np.clip(draws, -0.999999, 0.999999)))
        centre = np.arctanh(np.clip(estimate, -0.999999, 0.999999))
        return np.tanh([centre - critical * se, centre + critical * se])

    return {'pearson': pearson, 'spearman': spearman,
            'pearson_ci': normal_interval(pearson, pearson_boot), 'spearman_ci': normal_interval(spearman, spearman_boot),
            'pearson_ci_percentile': np.nanpercentile(pearson_boot, [low, high]),
            'spearman_ci_percentile': np.nanpercentile(spearman_boot, [low, high]),
            'pearson_boot': pearson_boot, 'spearman_boot': spearman_boot, 'n_points': int(valid.sum()), 'n_a': len(maps_a),
            'n_b': len(maps_b), 'n_boot': n_boot, 'average': average, 'confidence': confidence}


def bootstrap_group_correlation(analysis, key, group_a, group_b, **kwargs):
    """`bootstrap_map_correlation` for two groups of a fusion made with `group_field` on the shared grid (Mean, Median or
    Sum); `kwargs` as there. The two groups are different animals (use `paired=True` for the same animals under two
    group labels)."""
    maps_a, _, inside, _ = animal_maps(analysis, key, group=group_a)
    maps_b, _, _, _ = animal_maps(analysis, key, group=group_b)
    return bootstrap_map_correlation(maps_a, maps_b, inside, **kwargs)


def correlation_vs_animals(maps_a, maps_b, inside, sizes, n_draws=300, average='median', seed=0, min_points=10):
    """
    How the correlation of two average maps grows with the number of animals averaged: for every size k, k animals are drawn
    (the same ones in both groups, so the animals must be the same and in the same order) without replacement, averaged,
    and correlated; mean and SD over `n_draws` draws (one draw for k = all animals). With k = 1 this is the correlation of
    single animals, and the curve shows how far noise in single maps holds it down.
    Returns {k: {'mean': .., 'sd': ..}}.
    """
    maps_a, maps_b = np.asarray(maps_a, dtype=float), np.asarray(maps_b, dtype=float)
    if len(maps_a) != len(maps_b):
        raise ValueError("the same animals are needed in both groups")
    rng = np.random.default_rng(seed)
    curve = {}
    for k in sizes:
        draws = 1 if k >= len(maps_a) else n_draws
        r = [_correlations(average_map(maps_a[pick], average), average_map(maps_b[pick], average), inside, min_points)[0]
             for pick in (rng.choice(len(maps_a), min(k, len(maps_a)), replace=False) for _ in range(draws))]
        curve[int(k)] = {'mean': float(np.nanmean(r)), 'sd': float(np.nanstd(r))}
    return curve


def bootstrap_map_scalar(maps, inside, statistic='median', average='median', n_boot=1000, confidence=0.95, seed=0):
    """
    One number of the average map - the median (or mean) of its values over the grid points inside the outline - and its
    uncertainty from the animals: in every draw the animals are resampled with replacement, the average map recomputed
    and the number taken again. Returns {'value', 'ci' (percentile interval), 'sd' (of the draws), 'n_animals', 'n_boot',
    'statistic', 'average', 'confidence'}. maps: (animals, grid points); inside: which grid points count.
    """
    functions = {'median': np.nanmedian, 'mean': np.nanmean}
    if statistic not in functions:
        raise ValueError(f"statistic must be one of {list(functions)}, not {statistic!r}")
    maps = np.asarray(maps, dtype=float)

    def scalar(selected):
        values = average_map(selected, average)[inside]
        return functions[statistic](values[np.isfinite(values)])

    rng = np.random.default_rng(seed)
    draws = np.array([scalar(maps[rng.integers(0, len(maps), len(maps))]) for _ in range(n_boot)])
    low, high = 100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2
    return {'value': float(scalar(maps)), 'ci': np.percentile(draws, [low, high]), 'sd': float(draws.std()),
            'n_animals': len(maps), 'n_boot': n_boot, 'statistic': statistic, 'average': average, 'confidence': confidence}
