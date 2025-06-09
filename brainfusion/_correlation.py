import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, fisher_exact
import pandas as pd
from brainfusion._plot_maps import plot_correlation_with_radii
from brainfusion._utils import mask_contour


def correlate_dense_around_sparse(sparse_data, sparse_grid, sparse_perc, dense_data, dense_grid, dense_perc,
                                  radius='max', average_func=np.nanmean):
    """
    Computes the correlation between a sparse data map and a dense data map within a given radius.
    Only considers data points inside the sparse contour.
    """
    # Compute averaged dense values
    avg_dense_values, radii = average_within_radius(sparse_grid, dense_grid, dense_data,
                                                    radius, average_func)

    # Correlate
    result, stats_results, sparse_present, dense_present = analyse_correlation_percentile(sparse_data,
                                                                                          avg_dense_values,
                                                                                          sparse_perc=sparse_perc,
                                                                                          dense_perc=dense_perc)

    return avg_dense_values, radii, result, stats_results, sparse_present, dense_present


def average_within_radius(sparse_grid, dense_grid, dense_data, radius='max', average_func=np.nanmean):
    """
    Computes the average of dense gridded values within a given radius for each sparse measurement point,
    ensuring that no dense point is used more than once.
    """
    if radius == 'max':
        radius = compute_max_radius(sparse_grid)  # Compute adaptive radii
    elif isinstance(radius, (float, int)):
        radius = np.full(len(sparse_grid), radius)  # Use the same radius for all sparse points

    # Build KDTree for fast spatial queries
    tree = cKDTree(dense_grid)

    # Initialize output array
    avg_values = np.full(len(sparse_grid), np.nan)

    # Track which dense points have been assigned (to prevent duplicate use)
    assigned_mask = np.zeros(len(dense_grid), dtype=bool)

    # Query the KDTree for points within the radius
    for i, sparse_point in enumerate(sparse_grid):
        search_radius = radius[i]
        indices = tree.query_ball_point(sparse_point, search_radius)

        # Filter out already assigned dense points
        valid_indices = [idx for idx in indices if not assigned_mask[idx]]

        if valid_indices:  # If valid dense points are found
            avg_values[i] = average_func(dense_data[valid_indices])

            # Mark these dense points as assigned, so they are not reused
            assigned_mask[valid_indices] = True

    return avg_values, radius


def compute_max_radius(grid):
    """
    Computes the maximal non-overlapping radius for each grid point.
    The radius is set as half of the distance to the nearest neighbor.
    """
    tree = cKDTree(grid)
    distances, _ = tree.query(grid, k=2)  # k=2 because first match is the point itself
    max_radii = distances[:, 1] / 2  # Take half of the nearest neighbor distance
    return max_radii


def analyse_correlation_percentile(sparse_data, dense_data, sparse_perc=50, dense_perc=50, a_name="A", b_name="B"):
    # Ensure arrays are numpy
    sparse_data = np.asarray(sparse_data)
    dense_data = np.asarray(dense_data)

    if len(sparse_data) > 1:
        correlation, p_value = pearsonr(sparse_data, dense_data)

        # Percentile thresholds
        threshold_sparse = np.percentile(sparse_data, sparse_perc)
        threshold_dense = np.percentile(dense_data, dense_perc)

        # Define present / absent by percentile
        sparse_present = sparse_data >= threshold_sparse
        dense_present = dense_data >= threshold_dense

        cond_prop_table, stats_results = conditional_probability_table(sparse_present, dense_present, a_name=a_name,
                                                                       b_name=b_name)

    else:
        raise Exception("Given datasets do not contain enough values!")

    results = {
        "pearson_correlation": correlation,
        "pearson__value": p_value,
        f"present_percentile_{a_name}": sparse_perc,
        f"present_percentile_{b_name}": dense_perc,
        "probability_table": cond_prop_table,
    }

    return results, stats_results, sparse_present, dense_present


def conditional_probability_table(a_present, b_present, a_name="A", b_name="B"):
    assert len(a_present) == len(b_present)
    a_absent = ~a_present
    b_absent = ~b_present

    def safe_div(numerator, denominator):
        return numerator / denominator if denominator > 0 else np.nan

    # Conditional probability table
    n11 = np.sum(a_present & b_present)
    n12 = np.sum(a_present & b_absent)
    n21 = np.sum(a_absent & b_present)
    n22 = np.sum(a_absent & b_absent)

    cond_prop_table = pd.DataFrame({
        f"{b_name} absent": [
            safe_div(n22, np.sum(b_absent)),
            safe_div(n12, np.sum(b_absent))

        ],
        f"{b_name} present": [
            safe_div(n21, np.sum(b_present)),
            safe_div(n11, np.sum(b_present))

        ]
    }, index=[f"{a_name} absent", f"{a_name} present"])

    # Contingency table
    contingency_table = np.array([
        [n22, n21],
        [n12, n11]
    ])

    # Fisher exact test
    stats_results = {}
    oddsratio, p_value = fisher_exact(contingency_table)

    # Optional: Confidence interval for odds ratio (approximate, using Woolf method)
    # log(OR) ± 1.96 * SE(log(OR))
    with np.errstate(divide='ignore', invalid='ignore'):
        log_or = np.log(oddsratio)
        se_log_or = np.sqrt(
            1 / contingency_table[0, 0] + 1 / contingency_table[0, 1] +
            1 / contingency_table[1, 0] + 1 / contingency_table[1, 1]
        )
        ci_low = np.exp(log_or - 1.96 * se_log_or)
        ci_high = np.exp(log_or + 1.96 * se_log_or)

    # Add to result
    stats_results["contingency_table"] = contingency_table
    stats_results["odds_ratio"] = oddsratio
    stats_results["fisher_p_value"] = p_value
    stats_results["odds_ratio_CI_95"] = (ci_low, ci_high)

    return cond_prop_table, stats_results


def correlate_afm_myelin(afm_analysis, radius='max', afm_metric='modulus', average_func=np.nanmean, verify_corr=False):
    """
    Computes the correlation between AFM data and averaged myelin data within a given radius.
    Only considers data points inside the AFM contour.
    """
    afm_contour = afm_analysis['template_contours'][0]
    afm_data = afm_analysis['template_dataset'][afm_metric]
    afm_grid = afm_analysis['template_grid']

    # Insert interpolated dataset at index 0
    myelin_datasets = [afm_analysis['measurement_interpolated_dataset']] + afm_analysis['measurement_datasets']
    myelin_grids = [afm_analysis['measurement_interpolated_grid']] + afm_analysis['measurement_trafo_grids']
    myelin_filenames = ['Interpolated'] + afm_analysis['measurement_filenames']

    results = []

    # Mask AFM points inside contour
    afm_mask = mask_contour(afm_contour, afm_grid)
    afm_grid_filtered = afm_grid[afm_mask]
    afm_data_filtered = afm_data[afm_mask]

    # Calculate correlations between AFM data and each myelin dataset
    for idx, myelin_name in enumerate(myelin_filenames):
        myelin_grid = myelin_grids[idx]
        myelin_dataset = myelin_datasets[idx]

        if verify_corr:
            myelin_grid = afm_analysis['verification_trafo_grids'][idx-1] if idx > 0 else afm_analysis['verification_grids'][0]
            myelin_dataset = np.random.choice(np.linspace(1, 10, 10), size=myelin_grid.shape[0])

        # Mask myelin points inside contour
        myelin_mask = mask_contour(afm_contour, myelin_grid)
        myelin_grid_filtered = myelin_grid[myelin_mask]
        myelin_data_filtered = myelin_dataset[myelin_mask]

        # Compute averaged myelin values
        avg_myelin_values, radii = average_within_radius(myelin_grid_filtered, myelin_data_filtered, afm_grid_filtered,
                                                         radius,
                                                         average_func)

        # Remove NaNs before correlation
        valid_mask = ~np.isnan(avg_myelin_values)
        afm_data_valid = afm_data_filtered[valid_mask]
        avg_myelin_values_valid = avg_myelin_values[valid_mask]

        # Ensure there are enough points for correlation
        if len(afm_data_valid) > 1:
            correlation, p_value = pearsonr(afm_data_valid, avg_myelin_values_valid)
        else:
            correlation, p_value = np.nan, np.nan  # Not enough data

        # Store result
        results.append({
            "correlation_pair": f"AFM_{afm_metric} vs {myelin_name}",
            "correlation_value": correlation,
            "p_value": p_value
        })

        plot_correlation_with_radii(afm_grid_filtered, myelin_grid_filtered, afm_contour, radii,
                                    title=results[-1]["correlation_pair"])
        print(results[-1]["correlation_pair"] + ": " + f"{correlation}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df
