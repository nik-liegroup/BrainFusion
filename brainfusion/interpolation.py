import numpy as np
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture


def nearest_neighbour_interp(org_points: np.ndarray, values: np.ndarray, target_points: np.ndarray, unique: bool = True) -> np.ndarray:
    """
    Interpolate values at target_points from org_points using nearest-neighbour assignment.

    Parameters:
    - org_points: (N, 2) array of source coordinates
    - values: (N,) array of values at each source point
    - target_points: (M, 2) array of coordinates to interpolate onto
    - unique: if True, assigns each source point to its nearest target (many-to-one).
              Each target point may only receive a value from one source. If multiple
              target points share the same coordinates, only one will be assigned a value;
              the others will remain NaN.
              If False, each target gets its nearest source value (one-to-one).

    Returns:
    - interpolated_values: (M,) array of interpolated values
    """

    if len(org_points) != len(values):
        raise ValueError("Length of org_points and values must match.")
    if len(org_points) == 0:
        raise ValueError("Length of org_points is 0.")

    if not unique:
        # griddata-like nearest
        tree_src = cKDTree(org_points)
        _, nearest_src_idx = tree_src.query(target_points, k=1)
        return values[nearest_src_idx]
    else:
        # unique nearest: source -> nearest target, assign only closest source per target
        tree_tgt = cKDTree(target_points)
        distances, nearest_tgt_idx = tree_tgt.query(org_points, k=1)

        M = len(target_points)
        interpolated_values = np.full(M, np.nan)
        min_distances = np.full(M, np.inf)

        for src_idx, tgt_idx in enumerate(nearest_tgt_idx):
            dist = distances[src_idx]
            if dist < min_distances[tgt_idx]:
                min_distances[tgt_idx] = dist
                interpolated_values[tgt_idx] = values[src_idx]

    return interpolated_values


def fit_coordinates_gmm(grids, data_list, trafo_data=None, same_maps=True, num_components='mean'):
    # ToDo: Check function and add docstring + make more general
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
