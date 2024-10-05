import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from _find_average_contour import match_contours, calculate_median_contour
from _transform_2Dmap import transform_grid2contour
from _plot_maps import plot_contours, plot_corr_maps, plot_norm_corr
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler


def fit_coordinates_gmm(grid_list, data_list, same_maps=True, num_components='mean'):
    assert num_components in ['mean', 'min'], print('Please provide a valid metric for estimating the cluster numbers!')
    if num_components == 'mean':
        num_components = max(1, int(np.mean([len(grid) for grid in grid_list])))
    elif num_components == 'min':
        num_components = int(min(([len(grid) for grid in grid_list])))
    all_coords = np.vstack(grid_list)
    gmm = GaussianMixture(n_components=num_components, random_state=42)
    gmm.fit(all_coords)

    # Get the cluster labels for all points
    labels = gmm.predict(all_coords)

    # Get the cluster centers
    representative_coords = gmm.means_

    avg_dict = {}
    if same_maps:
        for key, _ in data_list[0].items():
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

            avg_dict[f'{key}_median'] = median_values

    else:
        avg_dict = None
    return representative_coords, avg_dict


def bin_and_correlate(data_map1, data_map2, grid_points, bin_size=4):
    # Ensure both data maps are the same shape
    assert data_map1.shape == data_map2.shape, "Data maps must have the same shape."

    # Reshape the grid points and data maps for processing
    grid_points = grid_points.reshape(-1, 2)  # Assuming grid_points is Nx2
    data_map1 = data_map1.flatten()
    data_map2 = data_map2.flatten()

    # Create a KD-tree for nearest neighbor searching
    tree = cKDTree(grid_points)

    # Create bins by finding the closest points (4 points to one median)
    indices = tree.query(grid_points, k=bin_size)[1]  # Find the indices of 4 closest points

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

    # Calculate the Pearson correlation coefficient
    if len(median_map1) > 1 and len(median_map1) > 1:
        correlation, _ = pearsonr(median_map1, median_map2)
        print("Correlation after binning and taking medians:", correlation)
    else:
        print("Not enough valid data points for correlation.")
        correlation = None

    # Normalize both heatmaps
    heatmap1_norm = normalize_heatmap(median_map1)
    heatmap2_norm = normalize_heatmap(median_map2)

    return correlation, heatmap1_norm, heatmap2_norm


# Function to normalize heatmaps
def normalize_heatmap(heatmap):
    scaler = MinMaxScaler()
    return scaler.fit_transform(heatmap.reshape(-1, 1)).flatten()


def afm_brillouin_corr(afm_analysis, afm_params, br_analysis, br_params, results_folder):
    # Extract data variables
    afm_data = afm_analysis['average_data']
    afm_grid = afm_analysis['average_grid']
    afm_contour = afm_analysis['median_contour']
    afm_key = f'{afm_params['raw_data_key']}_median'
    br_data = br_analysis['average_data']
    br_grid = br_analysis['average_grid']
    br_contour = br_analysis['median_contour']
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

    # Circularly align and match contours using ellipse fitting
    matched_contour_list, template_contour = match_contours(contours_list, template_index=0)

    # Calculate the median contour
    median_contour = calculate_median_contour(matched_contour_list)

    fig = plot_contours(median_contour, template_contour, matched_contour_list)
    output_path = os.path.join(results_folder, 'matched_mask_contours.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Transform original AFM grid to coordinate system of median contour
    afm_trafo_grid, afm_trafo_contour = transform_grid2contour(contours_list[0],
                                                               median_contour,
                                                               afm_grid)

    # Transform original Brillouin grid to coordinate system of median contour
    br_trafo_grid, br_trafo_contour = transform_grid2contour(contours_list[1],
                                                             median_contour,
                                                             br_grid)

    # Find new coordinates to compare both grids
    trafo_grid_list = [afm_trafo_grid, br_trafo_grid]
    data_list = [afm_data, br_data]
    grid_avg, _ = fit_coordinates_gmm(trafo_grid_list, data_list, same_maps=False, num_components='min')

    # Transform data to new grid
    afm_data_interp, br_data_interp = {}, {}
    for key, value in afm_data.items():
        afm_data_interp[f'{key}'] = griddata(afm_trafo_grid, value, grid_avg, method='linear')
    for key, value in br_data.items():
        br_data_interp[f'{key}'] = griddata(br_trafo_grid, value, grid_avg, method='linear')

    # Plot transformed heatmaps side by side
    fig = plot_corr_maps(median_contour,
                         afm_map=afm_data_interp[afm_key],
                         brillouin_map=br_data_interp[br_key],
                         grid=grid_avg,
                         mask=True,
                         marker_size=150,
                         vmin=None,
                         vmax=None)
    output_path = os.path.join(results_folder, f'MapsComparison.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate correlation between maps
    correlation, afm_norm, br_norm, = bin_and_correlate(afm_data_interp[afm_key],
                                                        br_data_interp[br_key],
                                                        grid_avg,
                                                        bin_size=1)

    # Plot normalized correlation
    fig = plot_norm_corr(afm_norm, br_norm, correlation)
    output_path = os.path.join(results_folder, f'BrillouinAFMCorrelationMap.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return correlation, median_contour, afm_data_interp, br_data_interp, grid_avg
