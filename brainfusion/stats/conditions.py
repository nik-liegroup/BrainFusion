"""Comparing two conditions (e.g. Control and CS) animal by animal: one number per animal, or the whole map."""

import warnings

import numpy as np


def _average(maps, how):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='(All-NaN slice|Mean of empty slice)')
        return np.nanmedian(maps, axis=0) if how == 'median' else np.nanmean(maps, axis=0)


def permutation_scalar_difference(values_a, values_b, statistic='median', n_perm=10000, n_boot=5000, confidence=0.95, seed=0):
    """
    Difference of one number per animal between two conditions (a - b): the difference of the medians (or means),
    its percentile interval over bootstrapped animals (each condition resampled on its own), and a two-sided
    permutation p-value (the condition labels of the animals are shuffled). Also Cliff's delta, the probability that a
    random animal of a is above one of b minus the reverse (-1 to 1), as an effect size that needs no distribution.
    """
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Need at least two animals per condition.")
    reduce = np.median if statistic == 'median' else np.mean
    rng = np.random.default_rng(seed)
    observed = reduce(a) - reduce(b)
    pooled = np.concatenate([a, b])
    perm = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(pooled)
        perm[i] = reduce(shuffled[:len(a)]) - reduce(shuffled[len(a):])
    boot = np.array([reduce(rng.choice(a, len(a))) - reduce(rng.choice(b, len(b))) for _ in range(n_boot)])
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(boot, [tail, 100 - tail])
    greater, smaller = (a[:, None] > b[None, :]).sum(), (a[:, None] < b[None, :]).sum()
    return {'n_a': len(a), 'n_b': len(b), 'statistic': statistic, 'difference': float(observed), 'ci': (float(low), float(high)),
            'p_value': float((1 + (np.abs(perm) >= abs(observed) - 1e-12).sum()) / (n_perm + 1)),
            'cliffs_delta': float((greater - smaller) / (len(a) * len(b)))}


def _welch_t(a, b):
    """Welch t at every column of two (animals, points) arrays, NaN animals ignored (0 where a point has no spread)."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='(Degrees of freedom|Mean of empty slice|invalid value)')
        n_a, n_b = (~np.isnan(a)).sum(0), (~np.isnan(b)).sum(0)
        error = np.sqrt(np.nanvar(a, axis=0, ddof=1) / n_a + np.nanvar(b, axis=0, ddof=1) / n_b)
        t = (np.nanmean(a, axis=0) - np.nanmean(b, axis=0)) / error
    return np.where(np.isfinite(t), t, 0.0)


def permutation_map_difference(maps_a, maps_b, inside, average='median', n_perm=2000, min_animals=3, seed=0):
    """
    Where do two conditions differ? `maps_a`, `maps_b`: (animals, points) maps of the animals of each condition on the
    same grid (`animal_maps`), `inside`: the points inside the outline. The map of the difference (a - b) is that of the
    average maps (`average`). It is tested with a Welch t statistic at every point (it is standardised by the animals' own
    spread inside each condition, so a real difference in one region does not inflate the spread elsewhere), and the
    p-value of a point is corrected for looking at all points at once: the share of permutations (condition labels of the
    animals shuffled) whose LARGEST |t| anywhere on the map is as big as the point's - family-wise error control without any
    assumption about the spatial correlation. Points with fewer than `min_animals` animals per condition are not tested.
    `global_p` uses the same permutations on the mean t^2 over the map: do the conditions differ anywhere at all?

    Returns {'difference', 't', 'p_corrected', 'tested', 'global_p', 'n_a', 'n_b'} (arrays over the points).
    """
    a, b = np.asarray(maps_a, dtype=float), np.asarray(maps_b, dtype=float)
    inside = np.asarray(inside, dtype=bool)
    n_a, n_b = len(a), len(b)
    both = np.concatenate([a, b])
    tested = inside & ((~np.isnan(a)).sum(0) >= min_animals) & ((~np.isnan(b)).sum(0) >= min_animals)
    rng = np.random.default_rng(seed)
    t_obs = _welch_t(a[:, tested], b[:, tested])
    max_perm, msq_perm = np.empty(n_perm), np.empty(n_perm)
    for i in range(n_perm):
        order = rng.permutation(n_a + n_b)
        t = _welch_t(both[order[:n_a]][:, tested], both[order[n_a:]][:, tested])
        max_perm[i], msq_perm[i] = np.abs(t).max(), np.mean(t ** 2)
    p_point = (1 + (max_perm[:, None] >= np.abs(t_obs)[None, :] - 1e-12).sum(0)) / (n_perm + 1)
    t_map, p_map = np.full(len(inside), np.nan), np.full(len(inside), np.nan)
    t_map[tested], p_map[tested] = t_obs, p_point
    return {'difference': _average(a, average) - _average(b, average), 't': t_map, 'p_corrected': p_map, 'tested': tested,
            'n_a': n_a, 'n_b': n_b, 'global_p': float((1 + (msq_perm >= np.mean(t_obs ** 2) - 1e-12).sum()) / (n_perm + 1))}
