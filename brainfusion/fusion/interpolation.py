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


def fit_coordinates_gmm(grids: list, data_list: list, num_components: str = 'mean'):
    """
    Cluster pooled measurement points from all samples into Gaussian components, then reduce each dataset
    key to the median of the values assigned to each component.

    Unlike resampling everything onto a shared regular grid, this doesn't require the samples to already
    share common point locations: it pools every sample's own (x, y) coordinates and lets a Gaussian
    Mixture Model pick representative locations based on where points are actually dense, instead of a
    fixed spacing.

    Parameters
    ----------
    grids : list of np.ndarray
        Per-sample (N_i, 2) point coordinates, already warped into the shared template space.
    data_list : list of dict
        Per-sample datasets - one dict of {key: (N_i,) array} per sample, matching that sample's own entry
        in `grids` by position (same points, same order).
    num_components : 'mean' or 'min', default='mean'
        Number of Gaussian components to fit: the mean (or minimum) point count across samples.

    Returns
    -------
    representative_coords : np.ndarray of shape (num_components, 2)
        The fitted Gaussian components' centers - the new grid the returned dataset lives on.
    avg_data : dict
        {key: (num_components,) array}, the median of every pooled point assigned to each component.
    """
    assert num_components in ('mean', 'min'), "num_components must be 'mean' or 'min'"
    counts = [len(grid) for grid in grids]
    n_components = max(1, int(np.mean(counts))) if num_components == 'mean' else min(counts)

    all_coords = np.vstack(grids)
    # Standardize before fitting: GaussianMixture's default `reg_covar` (1e-6) is an absolute value, so on
    # coordinates whose real spread is far below that (e.g. metre-scale stage positions, ~1e-4) it swamps the
    # actual variance and collapses every component onto nearly the same point instead of spreading out.
    coord_mean, coord_std = all_coords.mean(axis=0), all_coords.std(axis=0)
    coord_std[coord_std == 0] = 1
    normalized_coords = (all_coords - coord_mean) / coord_std

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(normalized_coords)
    representative_coords = gmm.means_ * coord_std + coord_mean

    avg_data = {}
    for key in data_list[0].keys():
        all_values = np.concatenate([d[key] for d in data_list])
        avg_data[key] = np.array([
            np.nanmedian(all_values[labels == i]) if np.any(labels == i) else np.nan
            for i in range(n_components)
        ])

    return representative_coords, avg_data
