import numpy as np
from scipy.interpolate import Rbf
from scipy.interpolate import RBFInterpolator
from tqdm import tqdm
from scipy.spatial import cKDTree, distance_matrix
import numbers
from typing import Union, Callable, Tuple, List


def transform_grid2contour(original_contour: np.ndarray, deformed_contour: np.ndarray, original_grid: np.ndarray,
                           test_grid: np.ndarray, smooth: Union[float, str] = 'auto', progress: str = '') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform a measurement grid and test grid from the original contour space to the deformed contour space
    using Radial Basis Function (RBF) interpolation.

    Parameters
    ----------
    original_contour : np.ndarray of shape (N, 2)
        2D coordinates defining the original contour (reference shape).
    deformed_contour : np.ndarray of shape (N, 2)
        2D coordinates defining the deformed contour (target shape).
    original_grid : np.ndarray of shape (M, 2)
        Grid points inside or around the original contour to be transformed.
    test_grid : np.ndarray of shape (K, 2)
        Additional grid points to be transformed (e.g., for validation).
    smooth : float, 'auto' or 'weighted', default='auto'
        Smoothing parameter for RBF interpolation. If 'auto', a default heuristic is used.
    progress : str, optional
        Suffix to append to tqdm progress bar description.

    Returns
    -------
    trafo_grid : np.ndarray of shape (M, 2)
        Transformed `original_grid` in the deformed contour space.
    trafo_ver_grid : np.ndarray of shape (K, 2)
        Transformed `test_grid` in the deformed contour space.
    trafo_contour : np.ndarray of shape (N, 2)
        Transformed original contour to check RBF accuracy.
    """
    for name, arr in zip(
        ["original_contour", "deformed_contour", "original_grid", "test_grid"],
        [original_contour, deformed_contour, original_grid, test_grid]
    ):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{name} must have shape (N, 2). Got {arr.shape}.")

    if isinstance(smooth, str) and smooth not in ['auto', 'weighted']:
        raise ValueError("`smooth` must be a float, 'auto' or 'weighted'.")

    # Create RBF interpolators
    rbf_x, rbf_y = create_rbf_interpolators(original_contour=original_contour,
                                            deformed_contour=deformed_contour,
                                            smooth=smooth)

    # Apply transformation to original contour to check whether deformation works as expected
    trafo_contour = np.array([
        [rbf_x(x, y), rbf_y(x, y)] for x, y in original_contour
    ])

    # Deform original grid to coordinate system of deformed contour
    trafo_grid = np.array([
        [rbf_x(x, y), rbf_y(x, y)] for x, y in tqdm(original_grid, desc="Transforming grid" + progress)
    ])

    trafo_ver_grid = np.array([
        [rbf_x(x, y), rbf_y(x, y)] for x, y in test_grid
    ])

    return trafo_grid, trafo_ver_grid, trafo_contour


def create_rbf_interpolators(original_contour: np.ndarray, deformed_contour: np.ndarray, function: str = 'linear',
                             smooth: Union[float, str] = 'auto',
) -> Tuple[Callable, Callable]:
    """
    Create RBF interpolators to map coordinates from original to deformed contour.

    Parameters
    ----------
    original_contour : (N, 2) array
        Points on the original contour (X, Y).
    deformed_contour : (N, 2) array
        Corresponding points on the deformed contour (X', Y').
    function : str
        Kernel type: {'multiquadric','inverse','gaussian','linear','cubic','quintic','thin_plate'}.
        Mapped to RBFInterpolator's kernels.
    smooth : float, 'auto' or 'weighted'
        Global smoothing; 'auto' picks a small value from point spacing.
        If 'weigthed', interpreted as per-point smoothing (larger devations = higher influence).

    Returns
    -------
    rbf_x, rbf_y : Callable
        Callables f(x, y) -> x' and f(x, y) -> y'.
    """
    # ToDo: Implement smooth bijective transformation to quickly invert for faster interpolation to regular grid
    if not (isinstance(original_contour, np.ndarray) and isinstance(deformed_contour, np.ndarray)):
        raise TypeError("Contours must be numpy arrays.")
    if original_contour.shape != deformed_contour.shape or original_contour.shape[1] != 2:
        raise ValueError("Contours must be of shape (N, 2) and match in size.")
    if not isinstance(function, str):
        raise TypeError("`function` must be a string.")

    valid_functions = {'multiquadric', 'inverse_multiquadric', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate_spline'}
    if function not in valid_functions:
        raise ValueError(f"`function` must be one of: {valid_functions}. Got '{function}'.")

    P = np.asarray(original_contour, dtype=float)
    Q = np.asarray(deformed_contour, dtype=float)
    N = P.shape[0]

    # Nearest-neighbour spacing (for auto smooth/epsilon)
    D = distance_matrix(P, P)
    np.fill_diagonal(D, np.inf)
    nn = np.min(D, axis=1)
    typical_spacing = float(np.mean(nn))
    median_spacing = float(np.median(nn))

    base_smooth = 1e-1
    if isinstance(smooth, str):
        if smooth == "auto":
            smoothing = base_smooth * typical_spacing
        elif smooth == "weighted":
            U = Q - P
            mag = np.linalg.norm(U, axis=1)
            med = np.median(mag) if np.any(mag > 0) else 1.0
            p = 4
            weight = (mag / (med + 1e-12)) ** p
            smoothing = (base_smooth * typical_spacing) / (1.0 + weight)
        else:
            raise TypeError('"smooth" must be a number, "auto", or "weighted".')
    else:
        smoothing = float(smooth)

    # --- epsilon for Gaussian/MQ: epsilon = 1/length_scale ---
    needs_epsilon = function in {"gaussian", "multiquadric", "inverse_multiquadric"}
    if needs_epsilon:
        length_scale = 1 * median_spacing  # ToDO: Implement length_scale_factor
        epsilon = 1.0 / max(length_scale, 1e-12)
    else:
        epsilon = None

        # Single 2D-valued interpolator
        rbf = RBFInterpolator(
            P, Q,
            kernel=function,
            smoothing=smoothing,
            epsilon=epsilon,
        )

    # Vectorised wrappers
    def _eval_xy(xq, yq):
        xq_arr = np.asarray(xq, dtype=float)
        yq_arr = np.asarray(yq, dtype=float)
        if xq_arr.shape != yq_arr.shape:
            raise ValueError("x and y must have the same shape.")
        pts = np.column_stack([xq_arr.ravel(), yq_arr.ravel()])
        out = rbf(pts).reshape(xq_arr.shape + (2,))
        return out

    def rbf_x(xq, yq):
        out = _eval_xy(xq, yq)[..., 0]
        return float(out) if np.isscalar(xq) and np.isscalar(yq) else out

    def rbf_y(xq, yq):
        out = _eval_xy(xq, yq)[..., 1]
        return float(out) if np.isscalar(xq) and np.isscalar(yq) else out

    return rbf_x, rbf_y


def create_rbf_interpolators_legacy(original_contour: np.ndarray, deformed_contour: np.ndarray,
                                    function: str = 'linear',
                                    smooth: Union[float, str] = 'auto') -> Tuple[Callable, Callable]:
    """
   Create RBF interpolators to map coordinates from original to deformed contour.

   Parameters
   ----------
   original_contour : np.ndarray of shape (N, 2)
       Points on the original contour (X, Y).
   deformed_contour : np.ndarray of shape (N, 2)
       Corresponding points on the deformed contour (X', Y').
   function : str
       Type of RBF function (e.g. 'linear', 'multiquadric', 'gaussian').
   smooth : float or 'auto'
       Smoothing parameter. If 'auto', a default based on average spacing is used.

   Returns
   -------
   rbf_x : Callable
       Interpolator for x-coordinates.
   rbf_y : Callable
       Interpolator for y-coordinates.
   """
    # ToDo: Implement new interpolation function scipy.interpolate.RBFInterpolator
    # ToDo: Implement smooth bijective transformation to quickly invert for faster interpolation to regular grid
    if not (isinstance(original_contour, np.ndarray) and isinstance(deformed_contour, np.ndarray)):
        raise TypeError("Contours must be numpy arrays.")

    if original_contour.shape != deformed_contour.shape or original_contour.shape[1] != 2:
        raise ValueError("Contours must be of shape (N, 2) and match in size.")

    if not isinstance(function, str):
        raise TypeError("`function` must be a string.")

    valid_functions = {'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'}
    if function not in valid_functions:
        raise ValueError(f"`function` must be one of: {valid_functions}. Got '{function}'.")

    if smooth == "auto":
        D = distance_matrix(original_contour, original_contour)
        np.fill_diagonal(D, np.inf)
        typical_spacing = np.min(D, axis=1).mean()
        smooth = 0.1 * typical_spacing
    elif isinstance(smooth, numbers.Number) is False:
        raise TypeError('`smooth` must be a number or "auto".')

    x_orig, y_orig = original_contour[:, 0], original_contour[:, 1]
    x_deform, y_deform = deformed_contour[:, 0], deformed_contour[:, 1]

    # Define forward mapping from definition space to deformation space
    rbf_x = Rbf(x_orig, y_orig, x_deform, function=function, smooth=smooth)
    rbf_y = Rbf(x_orig, y_orig, y_deform, function=function, smooth=smooth)

    return rbf_x, rbf_y


def extend_grid(measurement_grids: List[np.ndarray], x_extend: float, y_extend: float) -> Tuple[np.ndarray, List[int]]:
    """
    Generate an extended regular grid that encompasses all provided measurement grids.

    This function computes a global 2D bounding box across multiple input grids,
    estimates the typical local spacing in x and y directions using nearest-neighbour distances,
    and returns a regularly spaced grid that extends beyond the original bounding box
    by a user-defined percentage in each direction.

    Parameters
    ----------
    measurement_grids : list of np.ndarray
        A list of 2D arrays of shape (N, 2), each representing (x, y) coordinates of a measurement grid.
    x_extend : float
        Fraction by which to extend the grid in the x-direction, relative to the total x-span.
        For example, `x_extend=0.1` adds 10% to both the left and right of the x-range.
    y_extend : float
        Same as `x_extend`, but in the y-direction.

    Returns
    -------
    extended_grid_stack : np.ndarray of shape (P, 2)
        Flattened list of (x, y) coordinates of the extended grid.
    extended_grid_shape : list of int
        Shape of the regular grid as [n_rows, n_cols], useful for reshaping.

    Notes
    -----
    When estimating spacing, each grid uses distances to up to 4 nearest neighbours (excluding itself).
    If a grid contains fewer than 5 points, the number of neighbours is automatically reduced
    to avoid indexing errors and ensure robust spacing estimation.
    """
    if not isinstance(measurement_grids, list):
        raise TypeError("measurement_grids must be a list of 2D numpy arrays.")
    if not all(isinstance(grid, np.ndarray) and grid.ndim == 2 and grid.shape[1] == 2 for grid in measurement_grids):
        raise ValueError("Each grid must be a 2D NumPy array of shape (N, 2).")

    # Find global min/max across all grids
    all_points = np.vstack(measurement_grids)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)

    # Compute spacing separately for each grid
    x_spacings = []
    y_spacings = []

    # Compute the median of unique absolute x and y distances to the 4 nearest neighbours for a set of 2D coordinates
    for grid in measurement_grids:
        n_points = len(grid)
        k_query = min(5, n_points)
        tree = cKDTree(grid)
        dists, idxs = tree.query(grid, k=k_query)

        seen_pairs = set()
        dx_list = []
        dy_list = []

        for i in range(len(grid)):
            for j_idx in range(1, k_query):  # skip self (index 0)
                j = idxs[i, j_idx]
                pair = tuple(sorted((i, j)))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    dx = abs(grid[i, 0] - grid[j, 0])
                    dy = abs(grid[i, 1] - grid[j, 1])
                    dx_list.append(dx)
                    dy_list.append(dy)

        median_dx = np.median(dx_list)
        median_dy = np.median(dy_list)

        x_spacings.append(median_dx)
        y_spacings.append(median_dy)

    # Take the median across all grids to determine global spacing
    x_median_spacing = np.median(x_spacings)
    y_median_spacing = np.median(y_spacings)
    spac = np.mean([x_median_spacing, y_median_spacing])

    # Compute extension factor
    x_extend_factor = x_extend * (x_max - x_min)
    y_extend_factor = y_extend * (y_max - y_min)

    # Generate extended coordinate range
    x_new = np.arange(x_min - x_extend_factor, x_max + x_extend_factor + x_median_spacing, x_median_spacing)
    y_new = np.arange(y_min - y_extend_factor, y_max + y_extend_factor + y_median_spacing, y_median_spacing)

    # Create the extended meshgrid
    x, y = np.meshgrid(x_new, y_new)
    extended_grid_regular = np.stack([x, y], axis=-1)
    extended_grid_stack = np.vstack([extended_grid_regular[:, :, 0].ravel(), extended_grid_regular[:, :, 1].ravel()]).T
    extended_grid_shape = list(x.shape)

    return extended_grid_stack, extended_grid_shape
