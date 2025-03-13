import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
import pandas as pd

from brainfusion._match_contours import align_contours
from brainfusion._transform_2Dmap import transform_grid2contour
from brainfusion._plot_maps import plot_contours, plot_corr_maps, plot_correlation_with_radii
from brainfusion._utils import mask_contour


def fit_coordinates_gmm(grids, data_list, trafo_data=None, same_maps=True, num_components='mean'):
    assert num_components in ['mean', 'min'], 'Please provide a valid metric for estimating the cluster numbers!'
    if num_components == 'mean':
        num_components = max(1, int(np.mean([len(grid) for grid in grids])))
    elif num_components == 'min':
        num_components = int(min(([len(grid) for grid in grids])))
    all_coords = np.vstack(grids)
    gmm = GaussianMixture(n_components=num_components, random_state=42)
    gmm.fit(all_coords)

    # Get the cluster labels for all points
    labels = gmm.predict(all_coords)

    # Get the cluster centers
    representative_coords = gmm.means_

    if same_maps:
        avg_dict = {}
        gmm_dict = {}
        for key, _ in data_list[0].items():
            # Average interpolated datasets
            if trafo_data is not None:
                avg_dict = average_regular_dicts(data_list, trafo_data)
                data_interp = [d[key + '_trafo'] for d in trafo_data]
                avg_dict[f'{key}'] = np.mean(data_interp, axis=0)

            data_list_tmp = [d[key] for d in data_list]
            all_values = np.concatenate(data_list_tmp)

            # Calculate the median values for each cluster
            median_values = []
            for i in range(num_components):
                # Find indices of points in the current cluster
                cluster_indices = np.where(labels == i)[0]

                # Get the corresponding values for these indices
                cluster_values = np.array([all_values[j] for j in cluster_indices]) if len(
                    cluster_indices) > 0 else np.array(
                    [])

                # Calculate the median of the values
                median_value = np.nanmedian(cluster_values) if len(cluster_values) > 0 else np.nan
                median_values.append(median_value)

            gmm_dict[f'{key}'] = median_values

    else:
        avg_dict, gmm_dict = None, None
    return representative_coords, gmm_dict, avg_dict


def average_regular_dicts(data_list, trafo_data):
    avg_dict = {}
    for key, _ in data_list[0].items():
        # Average interpolated datasets
        data_interp = [d[key + '_trafo'] for d in trafo_data]
        avg_dict[f'{key}'] = np.mean(data_interp, axis=0)

    return avg_dict


def bin_and_correlate(data_map1, data_map2, grid_points, bin_size=4, map1_fit_limits=None, map2_fit_limits=None):
    # Ensure both data maps are the same shape
    assert data_map1.shape == data_map2.shape, "Data maps must have the same shape."

    # Reshape the grid points and data maps for processing
    grid_points = grid_points.reshape(-1, 2)  # Assuming grid_points is Nx2
    data_map1 = data_map1.flatten()
    data_map2 = data_map2.flatten()

    # Create a KD-tree for nearest neighbor searching
    tree = cKDTree(grid_points)

    # Create bins by finding the closest points (4 points to one median)
    indices = tree.query(grid_points, k=bin_size)[1]  # Find the indices of closest points

    # Initialize lists to hold median values
    median_map1 = []
    median_map2 = []

    # Calculate the median for each bin
    for i in range(len(grid_points)):
        if i % bin_size == 0:  # Take the first point of each bin
            # Extract the indices for the current bin
            current_indices = indices[i:i + bin_size]

            # Calculate the median values for both data maps
            median1 = np.nanmedian(data_map1[current_indices])
            median2 = np.nanmedian(data_map2[current_indices])

            median_map1.append(median1)
            median_map2.append(median2)

    # Convert lists to arrays
    median_map1 = np.array(median_map1)
    median_map2 = np.array(median_map2)

    mask = ~np.isnan(median_map1) & ~np.isnan(median_map2)

    # Filter out the NaNs using the mask
    median_map1 = median_map1[mask]
    median_map2 = median_map2[mask]

    # Normalize both heatmaps
    heatmap1_norm = normalize_heatmap(median_map1)
    heatmap2_norm = normalize_heatmap(median_map2)

    # Filter boundaries
    mask1 = mask2 = np.full(heatmap1_norm.shape, True, dtype=bool)
    if map1_fit_limits:
        mask1 = (heatmap1_norm > map1_fit_limits[0]) & (heatmap1_norm < map1_fit_limits[1])

    if map2_fit_limits:
        mask2 = (heatmap2_norm > map2_fit_limits[0]) & (heatmap2_norm < map2_fit_limits[1])

    # Apply the combined mask
    combined_mask = mask1 & mask2
    heatmap1_norm_f, heatmap2_norm_f = heatmap1_norm[combined_mask], heatmap2_norm[combined_mask]

    # Calculate the Pearson correlation coefficient
    if len(heatmap1_norm_f) > 1 and len(heatmap2_norm_f) > 1:
        correlation, p_value = pearsonr(heatmap1_norm_f, heatmap2_norm_f)
        print("Correlation after binning and taking medians:", correlation)
    else:
        print("Not enough valid data points for correlation.")
        correlation = None

    return {'map1': heatmap1_norm, 'map2': heatmap2_norm, 'pearson': correlation, 'p_value': p_value}


def normalize_heatmap(heatmap):
    scaler = MinMaxScaler()
    return scaler.fit_transform(heatmap.reshape(-1, 1)).flatten()


def afm_brillouin_transformation(afm_analysis, afm_params, br_analysis, br_params, results_folder):
    # Extract data variables
    afm_data = afm_analysis['average_data']
    afm_grid = afm_analysis['average_grid']
    afm_contour = afm_analysis['average_contour']
    afm_key = f'{afm_params['raw_data_key']}_median'

    br_data = br_analysis['average_data']
    br_grid = br_analysis['average_grid']
    br_contour = br_analysis['average_contour']
    br_key = f'{br_params['raw_data_key']}_median'

    # Centre coordinate grids around origin
    afm_centre = np.mean(afm_grid, axis=0)
    afm_grid = afm_grid - afm_centre
    afm_contour = afm_contour - afm_centre

    br_centre = np.mean(br_grid, axis=0)
    br_grid = br_grid - br_centre
    br_contour = br_contour - br_centre

    # Create contours list
    contours_list = [afm_contour, br_contour]

    # Calculate average contour
    avg_contour, contours_list, template_contour, matched_contour_list, error_list = align_contours(
        contours_list=contours_list)

    # Plot the average contour
    fig = plot_contours(avg_contour, template_contour, matched_contour_list)
    output_path = os.path.join(results_folder, 'matched_average_contours.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Transform original AFM grid to coordinate system of average contour
    afm_trafo_grid, afm_trafo_contour = transform_grid2contour(contours_list[0],
                                                               avg_contour,
                                                               afm_grid)

    # Transform original Brillouin grid to coordinate system of average contour
    br_trafo_grid, br_trafo_contour = transform_grid2contour(contours_list[1],
                                                             avg_contour,
                                                             br_grid)

    # Find new coordinates to compare both grids
    trafo_grid_list = [afm_trafo_grid, br_trafo_grid]
    data_list = [afm_data, br_data]
    grid_avg, _, _ = fit_coordinates_gmm(trafo_grid_list, data_list, same_maps=False, num_components='min')

    # Transform data to new grid
    afm_data_interp, br_data_interp = {}, {}
    for key, value in afm_data.items():
        afm_data_interp[f'{key}'] = griddata(afm_trafo_grid, value, grid_avg, method='linear')
    for key, value in br_data.items():
        br_data_interp[f'{key}'] = griddata(br_trafo_grid, value, grid_avg, method='linear')

    # Plot transformed heatmaps side by side
    fig = plot_corr_maps(avg_contour,
                         afm_map=afm_data_interp[afm_key],
                         brillouin_map=br_data_interp[br_key],
                         grid=grid_avg,
                         mask=True,
                         marker_size=150,
                         vmin=None,
                         vmax=None)
    output_path = os.path.join(results_folder, f'MapComparison.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return afm_data_interp, br_data_interp, grid_avg, avg_contour


def compute_max_radius(afm_grid):
    """
    Computes the maximal non-overlapping radius for each AFM point.
    The radius is set as half of the distance to the nearest AFM neighbor.
    """
    tree = cKDTree(afm_grid)
    distances, _ = tree.query(afm_grid, k=2)  # k=2 because first match is the point itself
    max_radii = distances[:, 1] / 2  # Take half of the nearest neighbor distance
    return max_radii


def average_within_radius(myelin_grid, myelin_data, afm_grid, radius, average_func=np.nanmean):
    """
    Computes the average of myelin values within a given radius for each AFM measurement point,
    ensuring that no myelin point is used more than once.
    """
    if radius == 'max':
        radius = compute_max_radius(afm_grid)  # Compute adaptive radii
    elif isinstance(radius, (float, int)):
        radius = np.full(len(afm_grid), radius)  # Use the same radius for all AFM points

    # Build KDTree for fast spatial queries
    tree = cKDTree(myelin_grid)

    # Initialize output array
    avg_values = np.full(len(afm_grid), np.nan)

    # Track which myelin points have been assigned (to prevent duplicate use)
    assigned_mask = np.zeros(len(myelin_grid), dtype=bool)

    # Query the KDTree for points within the radius
    for i, afm_point in enumerate(afm_grid):
        search_radius = radius[i]
        indices = tree.query_ball_point(afm_point, search_radius)

        # Filter out already assigned myelin points
        valid_indices = [idx for idx in indices if not assigned_mask[idx]]

        if valid_indices:  # If valid myelin points are found
            avg_values[i] = average_func(myelin_data[valid_indices])

            # Mark these myelin points as assigned so they are not reused
            assigned_mask[valid_indices] = True

    return avg_values, radius


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
