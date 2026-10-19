"""Spatial null model: how large is a correlation of two maps by chance, given that neighbouring points are alike?

Two maps that vary smoothly in space correlate by chance far more often than independent points would. The null distribution
here comes from surrogate maps that keep the spatial autocorrelation (the variogram) and the value distribution of the
original map but are otherwise random (Burt et al. 2020, NeuroImage 220:117038, "Generative modeling of brain maps with
spatial autocorrelation", the BrainSMASH method): shuffle the values, smooth them with a Gaussian kernel whose width is chosen
so that the variogram fits the original one, and add white noise for the rest of the variance. The measured correlation is
compared with the correlations between a surrogate of one map and the other map.
"""

import numpy as np
from scipy.optimize import nnls
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from brainfusion.utils import mask_contour


class _Variogram:
    """Variogram bins over the pairs of points: pair indices per bin, so that it can be evaluated quickly many times."""

    def __init__(self, distances, n_bins, max_lag_percentile):
        upper = np.triu_indices(len(distances), k=1)
        d = distances[upper]
        edges = np.linspace(0, np.percentile(d, max_lag_percentile), n_bins + 1)
        which = np.digitize(d, edges) - 1
        keep = (which >= 0) & (which < n_bins)
        self.i, self.j, self.bin = upper[0][keep], upper[1][keep], which[keep]
        self.count = np.bincount(self.bin, minlength=n_bins)
        self.used = self.count > 0

    def __call__(self, values):
        half_squares = 0.5 * (values[self.i] - values[self.j]) ** 2
        return (np.bincount(self.bin, half_squares, minlength=len(self.count))[self.used] / self.count[self.used])


def spatial_surrogates(values, coords, n_surrogates=1000, seed=0, n_bins=25, max_lag_percentile=50, n_bandwidths=10,
                       preserve_distribution=True):
    """
    Surrogate maps of `values` at `coords` with (about) the same variogram and, with `preserve_distribution`, exactly the same
    set of values (the surrogate's values are rank-matched to the original ones). Returns (surrogates (n_surrogates, n points),
    info dict with the chosen kernel width 'bandwidth', the variogram fit 'alpha' and 'beta', and 'fit_error').
    The kernel width is chosen among `n_bandwidths` widths between the 1st and the 35th percentile of the point distances as
    the one whose smoothed shuffles match the variogram of `values` best (up to the lag `max_lag_percentile` of the distances).
    """
    values, coords = np.asarray(values, dtype=float), np.asarray(coords, dtype=float)
    if len(values) < 20:
        raise ValueError("Too few points for spatial surrogates (need at least 20).")
    distances = cdist(coords, coords)
    variogram = _Variogram(distances, n_bins, max_lag_percentile)
    target = variogram(values)
    rng = np.random.default_rng(seed)
    nonzero = distances[distances > 0]
    bandwidths = np.percentile(nonzero, np.linspace(1, 35, n_bandwidths))

    def smoother(bandwidth):
        kernel = np.exp(-distances ** 2 / (2 * bandwidth ** 2))
        return kernel / kernel.sum(axis=1, keepdims=True)

    best = None
    for bandwidth in bandwidths:
        weights = smoother(bandwidth)
        smooth = [variogram(weights @ rng.permutation(values)) for _ in range(3)]
        design = np.column_stack([np.mean(smooth, axis=0), np.ones(len(target))])
        coefficients, error = nnls(design, target)   # target ~ alpha * smoothed variogram + beta, both >= 0
        if best is None or error < best[0]:
            best = (error, bandwidth, coefficients, weights)
    error, bandwidth, (alpha, beta), weights = best

    surrogates = np.empty((n_surrogates, len(values)))
    sorted_values = np.sort(values)
    for s in range(n_surrogates):
        smoothed = weights @ rng.permutation(values)
        surrogate = np.sqrt(alpha) * (smoothed - smoothed.mean()) + values.mean() + np.sqrt(beta) * rng.standard_normal(len(values))
        surrogates[s] = sorted_values[rankdata(surrogate, method='ordinal').astype(int) - 1] if preserve_distribution else surrogate
    return surrogates, {'bandwidth': float(bandwidth), 'alpha': float(alpha), 'beta': float(beta), 'fit_error': float(error)}


def _pearson_to_many(surrogates, other):
    a = surrogates - surrogates.mean(axis=1, keepdims=True)
    b = other - other.mean()
    return (a @ b) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum())


def spatial_null_test(map_a, map_b, grid, contour, n_surrogates=1000, seed=0, **surrogate_options):
    """
    Is the correlation of two maps on the same points larger than expected from their spatial autocorrelation alone?
    Surrogate maps of A (same variogram and values) are correlated with B, and surrogates of B with A; the p value is the
    share of surrogates (+1) whose correlation is at least as large in absolute value as the measured one, taken as the
    larger of the two (a surrogate of either map may be the more conservative). Done for Pearson and for Spearman.

    map_a, map_b: values on `grid` (n points, 2); only points inside `contour` with a value in both are used.
    Returns a dict: 'pearson', 'spearman', 'p_pearson', 'p_spearman', the p values from the A and the B surrogates
    ('p_pearson_a', ...), 'n_points', 'n_surrogates', 'null_pearson_a', 'null_pearson_b' (the surrogate correlations) and
    'surrogate_info_a', 'surrogate_info_b'.
    """
    map_a, map_b, grid = (np.asarray(x, dtype=float) for x in (map_a, map_b, grid))
    valid = mask_contour(contour, grid) & np.isfinite(map_a) & np.isfinite(map_b)
    a, b, coords = map_a[valid], map_b[valid], grid[valid]
    ranks_a, ranks_b = rankdata(a), rankdata(b)
    result = {'pearson': float(np.corrcoef(a, b)[0, 1]), 'spearman': float(np.corrcoef(ranks_a, ranks_b)[0, 1]),
              'n_points': int(valid.sum()), 'n_surrogates': n_surrogates}

    for name, source, other, ranks_source, ranks_other in (('a', a, b, ranks_a, ranks_b), ('b', b, a, ranks_b, ranks_a)):
        surrogates, info = spatial_surrogates(source, coords, n_surrogates=n_surrogates, seed=seed, **surrogate_options)
        null = _pearson_to_many(surrogates, other)
        null_rank = _pearson_to_many(np.apply_along_axis(rankdata, 1, surrogates), ranks_other)
        result[f'null_pearson_{name}'] = null
        result[f'null_spearman_{name}'] = null_rank
        result[f'surrogate_info_{name}'] = info
        result[f'p_pearson_{name}'] = float((1 + np.sum(np.abs(null) >= abs(result['pearson']))) / (1 + n_surrogates))
        result[f'p_spearman_{name}'] = float((1 + np.sum(np.abs(null_rank) >= abs(result['spearman']))) / (1 + n_surrogates))
    result['p_pearson'] = max(result['p_pearson_a'], result['p_pearson_b'])
    result['p_spearman'] = max(result['p_spearman_a'], result['p_spearman_b'])
    return result
