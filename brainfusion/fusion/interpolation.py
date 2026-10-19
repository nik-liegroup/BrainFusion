import numpy as np
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture


def typical_spacing(points: np.ndarray) -> float:
    """Median distance of each point to its nearest neighbour: the grid spacing of a point cloud."""
    if len(points) < 2:
        raise ValueError("Need at least two points to estimate a spacing.")
    distances, _ = cKDTree(points).query(points, k=2)
    return float(np.median(distances[:, 1]))


def nearest_neighbour_interp(org_points: np.ndarray, values: np.ndarray, target_points: np.ndarray, unique: bool = True,
                             max_distance: float = None) -> np.ndarray:
    """
    Interpolate values at target_points from org_points using nearest-neighbour assignment.

    Parameters:
    - org_points: (N, 2) array of source coordinates
    - values: (N,) array of values at each source point
    - target_points: (M, 2) array of coordinates to interpolate onto
    - max_distance: if given, a target further than this from every source point gets NaN instead of the value
              of the nearest one (no filling-in beyond the measured area). None: every target gets a value.
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
        distances, nearest_src_idx = tree_src.query(target_points, k=1)
        result = np.asarray(values)[nearest_src_idx]
        if max_distance is not None:
            result = np.where(distances <= max_distance, result.astype(float), np.nan)
        return result
    else:
        # unique nearest: source -> nearest target, assign only closest source per target
        tree_tgt = cKDTree(target_points)
        distances, nearest_tgt_idx = tree_tgt.query(org_points, k=1)

        M = len(target_points)
        interpolated_values = np.full(M, np.nan)
        min_distances = np.full(M, np.inf)

        for src_idx, tgt_idx in enumerate(nearest_tgt_idx):
            dist = distances[src_idx]
            if max_distance is not None and dist > max_distance:
                continue
            if dist < min_distances[tgt_idx]:
                min_distances[tgt_idx] = dist
                interpolated_values[tgt_idx] = values[src_idx]

    return interpolated_values


def gmm_average(grids: list, data_list: list, num_components: str = 'mean'):
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
    dict with
        'grid'    : (num_components, 2) the fitted Gaussian components' centers - the grid the dataset lives on.
        'dataset' : {key: (num_components,) array}, the median of every pooled point assigned to each component.
        'n'       : {key: (num_components,) array}, number of different samples with a finite value in the component.
        'spread'  : {key: (num_components,) array}, robust standard deviation (1.4826 x median absolute deviation)
                    of the pooled values of the component.
        'standard_error' : {key: (num_components,) array}, standard error of the component's median:
                    1.2533 x spread / sqrt(n), with n the number of different samples (the pooled points of one sample
                    are not independent). Assumes independent samples.
        'labels'  : (total points,) component of every pooled point, in the order of the concatenated `grids`.
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
    return gmm_reduce(representative_coords, labels, counts, data_list)


def gmm_reduce(centres, labels, counts, data_list):
    """
    Reduce pooled points to one value per component of an existing clustering: `centres` (K, 2), `labels` (total points,) the
    component of every pooled point, `counts` the number of points per sample and `data_list` the per-sample datasets, both in the
    order of the pooled points. Same return as `gmm_average`. Used with the labels of ONE clustering of all samples to average
    any subset of the samples (e.g. a condition) on the same components, so all averages of a fusion share one grid.
    """
    n_components = len(centres)
    owner = np.concatenate([np.full(count, i) for i, count in enumerate(counts)])

    dataset, n_samples, spread = {}, {}, {}
    for key in data_list[0].keys():
        all_values = np.concatenate([d[key] for d in data_list]).astype(float)
        finite = np.isfinite(all_values)
        dataset[key] = np.full(n_components, np.nan)
        spread[key] = np.full(n_components, np.nan)
        for i in range(n_components):
            values = all_values[(labels == i) & finite]
            if len(values):
                centre = np.median(values)
                dataset[key][i] = centre
                spread[key][i] = 1.4826 * np.median(np.abs(values - centre))
        pairs = np.unique(labels[finite] * len(counts) + owner[finite])   # one entry per (component, sample)
        n_samples[key] = np.bincount(pairs // len(counts), minlength=n_components)

    with np.errstate(all='ignore'):
        standard_error = {key: 1.2533 * spread[key] / np.sqrt(n_samples[key]) for key in spread}
    return {'grid': np.asarray(centres), 'dataset': dataset, 'n': n_samples, 'spread': spread,
            'standard_error': standard_error, 'labels': labels}


def fit_coordinates_gmm(grids: list, data_list: list, num_components: str = 'mean'):
    """`gmm_average` reduced to what it returned originally: (component centers, {key: median per component})."""
    result = gmm_average(grids, data_list, num_components)
    return result['grid'], result['dataset']


def resample_points(source_grid, source_values, target_grid, method='gaussian', sigma='auto', radius=None, min_weight=0.25,
                    truncate=3.0):
    """
    Values of a point set at other points: what to do when two maps do not live on the same points (e.g. group maps
    of "GMM" clusterings, each with its own component centres, or a sparse and a dense measurement).

    method
        'gaussian'  weighted mean of the source points around each target point, weight exp(-d^2 / (2 sigma^2)),
                    points further than `truncate` x sigma ignored. A soft version of 'radius': no hard edge, nothing is
                    thrown away. NaN where the weights sum to less than `min_weight` (a source point right on the target
                    has weight 1): the target is then too far from everything measured.
        'radius'    mean of the source points within `radius` of the target point (NaN if there is none).
        'nearest'   value of the nearest source point, NaN if it is further away than `radius` (None: no limit).
    sigma: 'auto' = half the typical spacing of the source points, or a number (in the units of the grids).
    Source values that are NaN are ignored. Returns an array with one value per target point.
    """
    source_grid, target_grid = np.asarray(source_grid, dtype=float), np.asarray(target_grid, dtype=float)
    values = np.asarray(source_values, dtype=float)
    keep = np.isfinite(values)
    source_grid, values = source_grid[keep], values[keep]
    if len(values) == 0:
        return np.full(len(target_grid), np.nan)
    tree = cKDTree(source_grid)

    if method == 'nearest':
        distance, index = tree.query(target_grid, distance_upper_bound=np.inf if radius is None else radius)
        found = np.isfinite(distance)
        out = np.full(len(target_grid), np.nan)
        out[found] = values[index[found]]
        return out
    if method == 'radius':
        if radius is None:
            raise ValueError("method 'radius' needs a radius")
        lists = tree.query_ball_point(target_grid, radius)
        return np.array([values[i].mean() if len(i) else np.nan for i in lists])
    if method != 'gaussian':
        raise ValueError(f"method must be 'gaussian', 'radius' or 'nearest', not {method!r}")

    if isinstance(sigma, str):
        if sigma != 'auto':
            raise ValueError(f"sigma must be a number or 'auto', not {sigma!r}")
        sigma = 0.5 * typical_spacing(source_grid)
    out = np.full(len(target_grid), np.nan)
    for i, index in enumerate(tree.query_ball_point(target_grid, truncate * sigma)):
        if len(index) == 0:
            continue
        weights = np.exp(-((source_grid[index] - target_grid[i]) ** 2).sum(axis=1) / (2 * sigma ** 2))
        if weights.sum() >= min_weight:
            out[i] = (weights * values[index]).sum() / weights.sum()
    return out
