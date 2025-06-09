import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import directed_hausdorff
from shapely.geometry import Polygon, Point, LineString
from brainfusion._match_contours import get_contour_orientation


def find_average_contour(contours_list: List[np.ndarray], average: str = 'star_domain', star_bins: int = 360,
                         error_metric: str = 'jaccard') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute an average contour and corresponding errors for a list of input contours.

    Parameters
    ----------
    contours_list : list of np.ndarray
        List of (N_i, 2) arrays representing input contours centred around coordinate centre [0, 0].
    average : str, optional
        Method for computing the average contour ('star_domain', 'median', 'mean').
    star_bins : int, optional
        Number of angular bins to use if using star-domain averaging.
    error_metric : str, optional
        Metric to compute errors between the input contours and the average contour ('jaccard', 'frechet', 'hausdorff').

    Returns
    -------
    avg_contour : np.ndarray of shape (N, 2)
        The computed average contour.
    errors : np.ndarray of shape (len(contours_list),)
        Array of errors between each input contour and the average contour.
    """
    if not contours_list:
        raise ValueError("contours_list must not be empty")
    if not all(isinstance(c, np.ndarray) and c.ndim == 2 and c.shape[1] == 2 for c in contours_list):
        raise ValueError("Each contour must be a (N, 2) np.ndarray")

    # Calculate the average contour
    avg_contour = calculate_average_contour(contours_list, average=average, star_bins=star_bins)

    # Calculate error
    errors = calculate_error_distances(contours_list, avg_contour, metric=error_metric)

    return avg_contour, errors


def calculate_average_contour(contours: List[np.ndarray], average: str = 'star_domain', star_bins: int = 360) -> np.ndarray:
    """
    Calculate an average contour from a list of input contours.

    Parameters
    ----------
    contours : list of np.ndarray
        List of (N_i, 2) arrays of (x, y) coordinates representing input contours.
        For 'mean' and 'median' averaging, all contours must have the same number of points.
    average : str
        Averaging method to use:
            - 'star_domain' : polar averaging using the geometric centre of each contour.
            - 'median'      : median of corresponding (x, y) coordinates (requires same number of points).
            - 'mean'        : mean of corresponding (x, y) coordinates (requires same number of points).
    star_bins : int
        Number of angular bins to use for 'star_domain' averaging.

    Returns
    -------
    contour : np.ndarray of shape (N, 2)
        The average contour. The contour is returned closed (last point equals first point).

    Notes
    -----
    For 'star_domain' averaging, each contour must define a star domain with respect to its own geometric centre.
    The function closes the returned contour explicitly.
    """
    if not contours:
        raise ValueError("contours must not be empty")
    if not all(isinstance(c, np.ndarray) and c.ndim == 2 and c.shape[1] == 2 for c in contours):
        raise ValueError("Each contour must be a (N, 2) np.ndarray")

    if average == 'star_domain':
        # Check if all contours are star domains with respect to the given centre
        assert all(is_star_domain(contour, [0, 0]) for contour in contours), ('Not all contours define star '
                                                                              'domains with respect to their geometric '
                                                                              'centre!')

        # Convert contours to polar coordinates
        angles = []
        radii = []
        init_angles = []
        for contour in contours:
            x, y = contour[:, 0], contour[:, 1]
            theta = np.arctan2(y, x)  # Compute angles
            r = np.sqrt(x ** 2 + y ** 2)
            angles.append(theta)
            radii.append(r)
            init_angles.append(theta[0])

        # Compute the mean starting angle using circular statistics
        mean_angle = np.angle(np.mean(np.exp(1j * np.array(init_angles))))

        # Create a uniform grid of angles starting at the mean angle
        angle_bins = np.linspace(mean_angle, mean_angle + 2 * np.pi, star_bins, endpoint=False)

        # Interpolate radii for each contour onto the uniform grid
        interpolated_radii = np.array([
            np.interp(angle_bins, np.sort(theta), r[np.argsort(theta)], period=2 * np.pi)
            for theta, r in zip(angles, radii)
        ])

        # Compute the mean radius at each angle bin
        avg_radii = np.mean(interpolated_radii, axis=0)

        # Convert mean polar coordinates back to Cartesian
        avg_x = avg_radii * np.cos(angle_bins)
        avg_y = avg_radii * np.sin(angle_bins)
        avg_contour = np.column_stack((avg_x, avg_y))

    elif average == 'median':
        # Require same number of points
        n_points = [c.shape[0] for c in contours]
        if len(set(n_points)) != 1:
            raise ValueError("For 'median' averaging, all contours must have the same number of points.")
        avg_contour = np.median(np.array(contours), axis=0)

    elif average == 'mean':
        n_points = [c.shape[0] for c in contours]
        if len(set(n_points)) != 1:
            raise ValueError("For 'mean' averaging, all contours must have the same number of points.")
        avg_contour = np.mean(np.array(contours), axis=0)

    else:
        raise Exception(f'{average} is not defined for contour averaging.')

    # Topologically orient contour
    if get_contour_orientation(contours[0]) != get_contour_orientation(avg_contour):
        avg_contour = avg_contour[::-1]  # Reverse the order

    # Close contour
    avg_contour[-1, :] = avg_contour[0, :]
    return avg_contour


def is_star_domain(contour: np.ndarray, centre: tuple[float, float], tol_factor: float = 1e-3) -> bool:
    """
    Check whether a given polygon contour defines a star domain with respect to a centre point,
    using a tolerance scaled to the contour size.

    Parameters
    ----------
    contour : np.ndarray of shape (N, 2)
        Contour coordinates of the polygon.
    centre : tuple of float
        (x, y) coordinates of the centre point.
    tol_factor : float
        Tolerance factor relative to the contour's bounding box diagonal (default 1e-3).

    Returns
    -------
    bool
        True if the polygon is star-shaped with respect to the centre, False otherwise.
    """
    if not (isinstance(contour, np.ndarray) and contour.ndim == 2 and contour.shape[1] == 2):
        raise ValueError(f"Contour must be a numpy array of shape (N, 2), got shape {getattr(contour, 'shape', None)}")

    polygon = Polygon(contour)
    center_point = Point(centre)

    # Check centre inside polygon
    if not polygon.contains(center_point):
        return False

    # Calculate contour bounding box diagonal length for tolerance scaling
    minx, miny, maxx, maxy = polygon.bounds
    diag_len = np.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)
    atol = tol_factor * diag_len

    # Check visibility of all boundary points
    for point in contour:
        target_point = Point(point)
        line_of_sight = LineString([centre, point])

        intersection = polygon.boundary.intersection(line_of_sight)

        # Convert intersection to list
        if isinstance(intersection, Point):
            intersections = [intersection]
        elif hasattr(intersection, "geoms"):
            intersections = list(intersection.geoms)
        else:
            raise RuntimeError("Unexpected intersection type")

        # Check if at least one intersection is close enough to target point
        tst = [target_point.distance(inter) for inter in intersections]
        if any(target_point.distance(inter) > atol for inter in intersections):
            return False

    return True


def calculate_error_distances(contours: List[np.ndarray], master_contour: np.ndarray, metric: str = 'jaccard') -> np.ndarray:
    """
    Compute distances between a list of contours and a master contour.

    Parameters
    ----------
    contours : list of np.ndarray
        List of (N_i, 2) arrays representing individual contours.
    master_contour : np.ndarray of shape (N, 2)
        The reference contour to compare against.
    metric : str
        Distance metric to use: 'jaccard', 'frechet', or 'hausdorff'.

    Returns
    -------
    np.ndarray of shape (len(contours),)
        Array of distance values.
    """
    metric_funcs = {
        'jaccard': jaccard_distance,
        'frechet': frechet_distance,
        'hausdorff': hausdorff_distance
    }

    if metric not in metric_funcs:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(metric_funcs.keys())}.")

    distance_func = metric_funcs[metric]
    distances = [distance_func(contour, master_contour) for contour in contours]
    return np.array(distances)


def jaccard_distance(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    """
    Compute the Jaccard distance between two 2D polygonal curves.

    Parameters
    ----------
    curve_a : np.ndarray of shape (N, 2)
        First polygon contour.
    curve_b : np.ndarray of shape (M, 2)
        Second polygon contour.

    Returns
    -------
    float
        Jaccard distance = 1 - (area of intersection / area of union)
    """
    for name, curve in zip(["curve_a", "curve_b"], [curve_a, curve_b]):
        if not isinstance(curve, np.ndarray) or curve.ndim != 2 or curve.shape[1] != 2:
            raise ValueError(f"{name} must be a NumPy array of shape (N, 2)")
        if len(np.unique(curve, axis=0)) < 3:
            raise ValueError(f"{name} must have at least 3 distinct points to define a polygon")
        if not np.allclose(curve[0], curve[-1]):
            raise ValueError(f"{name} must be closed (first point must equal last point)")

    # Create Polygon objects
    poly_a = Polygon(curve_a)
    poly_b = Polygon(curve_b)

    if not poly_a.is_valid or not poly_b.is_valid:
        raise ValueError("One of the input curves does not form a valid polygon")

    # Calculate the intersection and union of the polygons
    intersection_area = poly_a.intersection(poly_b).area
    union_area = poly_a.union(poly_b).area

    if union_area == 0:
        raise ValueError(f"{union_area} must not be 0")

    # Calculate the Jaccard Index based on areas
    jaccard_index = intersection_area / union_area

    # Calculate the Jaccard Distance
    jaccard_dist = 1 - jaccard_index
    return jaccard_dist


def frechet_distance(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    """
    Compute the discrete Fréchet distance between two curves.

    Parameters
    ----------
    curve_a : np.ndarray of shape (N, 2)
        First curve as a sequence of (x, y) points.
    curve_b : np.ndarray of shape (N, 2)
        Second curve as a sequence of (x, y) points.

    Returns
    -------
    float
        The Fréchet distance between the two curves. Returns np.nan if calculation fails.

    Notes
    -----
    This implementation uses the *discrete* Fréchet distance, which depends on the order and number
    of points in each curve. Curves must have the same number of points. Use interpolation beforehand
    if matching geometries with different sampling densities.
    """
    curve_a = np.asarray(curve_a)
    curve_b = np.asarray(curve_b)

    for name, curve in zip(['curve_a', 'curve_b'], [curve_a, curve_b]):
        if curve.ndim != 2 or curve.shape[1] != 2:
            raise ValueError(f"{name} must be a NumPy array of shape (N, 2)")

    if len(curve_a) != len(curve_b):
        raise ValueError("Curves must have the same number of points for discrete Fréchet distance")

    try:
        n = len(curve_a)
        ca = -np.ones((n, n))

        def c(i, j):
            if ca[i, j] > -1:
                return ca[i, j]
            d = np.linalg.norm(curve_a[i] - curve_b[j])
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i > 0 and j == 0:
                ca[i, j] = max(c(i - 1, 0), d)
            elif i == 0 and j > 0:
                ca[i, j] = max(c(0, j - 1), d)
            else:
                ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), d)
            return ca[i, j]

        return c(n - 1, n - 1)

    except (RecursionError, Exception) as e:
        print(f"Fréchet distance calculation failed: {e}")
        return np.nan


def hausdorff_distance(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    """
    Compute the symmetric Hausdorff distance between two 2D point sets.

    Parameters
    ----------
    curve_a : np.ndarray of shape (N, 2)
        First sequence of (x, y) points.
    curve_b : np.ndarray of shape (M, 2)
        Second sequence of (x, y) points.

    Returns
    -------
    float
        The Hausdorff distance: the greatest of all the distances from a point in one set to the closest point in the
        other set.
    """
    for name, curve in zip(["curve_a", "curve_b"], [curve_a, curve_b]):
        if not isinstance(curve, np.ndarray) or curve.ndim != 2 or curve.shape[1] != 2:
            raise ValueError(f"{name} must be a NumPy array of shape (N, 2)")

    return max(
        directed_hausdorff(curve_a, curve_b)[0],
        directed_hausdorff(curve_b, curve_a)[0]
    )
