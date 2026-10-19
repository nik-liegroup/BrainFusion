"""Co-occurrence of high values in two paired maps (Fisher exact test on the thresholded points)."""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pearsonr


def analyse_correlation_percentile(data_a, data_b, percentile_a=50, percentile_b=50, name_a="A", name_b="B"):
    """
    Beyond a plain Pearson correlation, also test co-occurrence: threshold each dataset at its own
    percentile (default: median) to call each point "present"/"absent" in that dataset, then run a Fisher
    exact test on the resulting 2x2 contingency table - e.g. "are high-stiffness points more likely than
    chance to also be high-myelin points", independent of the linear Pearson relationship.

    `data_a`/`data_b` must already be paired point-by-point - e.g. `correlate_around_reference_grid`'s or
    `correlate_on_shared_grid`'s name_a/name_b outputs, masked by their own 'valid_mask'.
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
