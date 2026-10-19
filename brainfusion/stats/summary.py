"""Summaries over animals, and the correction of a correlation for noise in the maps."""

import numpy as np
from scipy import stats


def summarise_correlations(r, confidence=0.95):
    """
    Summary of one correlation per animal (the animal is the unit): the Fisher-z mean back-transformed to r, its t-interval
    over animals, the t test of the mean z against zero, the median and the number of animals with r > 0.
    Returns a dict: 'n', 'mean_r', 'ci' (low, high), 'p' (two-sided), 'median_r', 'sd_r', 'n_positive'.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        raise ValueError("Need at least three animals for a summary.")
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    mean, se = z.mean(), z.std(ddof=1) / np.sqrt(len(z))
    half = stats.t.ppf(0.5 + confidence / 2, len(z) - 1) * se
    return {'n': int(len(r)), 'mean_r': float(np.tanh(mean)), 'ci': np.tanh([mean - half, mean + half]),
            'p': float(2 * stats.t.sf(abs(mean / se), len(z) - 1)) if se > 0 else np.nan, 'median_r': float(np.median(r)),
            'sd_r': float(r.std(ddof=1)), 'n_positive': int((r > 0).sum())}


def disattenuate(r, reliability_a, reliability_b):
    """
    The correlation corrected for noise in the two maps: r / sqrt(reliability_a * reliability_b), what r would be if both maps
    were measured without noise (the reliabilities are those of the maps correlated, e.g. `split_half_reliability` of the
    average maps; a value above 1 only says the reliabilities are too low an estimate). Both reliabilities must be positive.
    """
    if reliability_a <= 0 or reliability_b <= 0:
        raise ValueError("Both reliabilities must be positive.")
    return float(r / np.sqrt(reliability_a * reliability_b))
