import numpy as np
import cv2
from scipy.interpolate import interp1d
import scipy.ndimage as ndi
from skimage.transform import AffineTransform
from typing import List, Tuple
from collections import defaultdict
from numpy.linalg import svd
from brainfusion._dtw import dtw_with_curvature_penalty, segmented_contour_dtw


def interpolate_contour(contour: np.ndarray, num_points: int) -> np.ndarray:
    """
    Resample a 2D contour to have exactly `num_points` points, equally spaced along its arc length.

    Parameters:
    - contour: (N, 2) numpy array of (x, y) points
    - num_points: number of points in the resampled contour

    Returns:
    - interpolated_contour: (num_points, 2) array of equally spaced points
    """
    if not isinstance(contour, np.ndarray):
        raise ValueError("contour must be a numpy array")
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be a 2D array of shape (N, 2)")

    # Arc length and interpolation
    dists = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    cumulative_dists = np.concatenate(([0], np.cumsum(dists)))
    interp_x = interp1d(cumulative_dists, contour[:, 0], kind='linear')
    interp_y = interp1d(cumulative_dists, contour[:, 1], kind='linear')
    new_dists = np.linspace(0, cumulative_dists[-1], num_points)
    return np.vstack((interp_x(new_dists), interp_y(new_dists))).T


def align_contours(contour_list, grid_list, rot_axes=None, init_points=None, template_index=0, fit_routine='ellipse'):
    """
    Align a set of contours to a template contour specified by template index in the contours list. If template_index is
    None, calculate average contour and align to it.
    """
    assert 0 <= template_index < len(contour_list), print("Contour template index out of range!")

    # Match contours using ellipse fit
    matched_contours, matched_grids, matched_points, affine_matrices = [], [], [], []
    for i, contour in enumerate(contour_list):
        ang = angle_between_lines(rot_axes[i], rot_axes[template_index]) if rot_axes[i] is not None else None

        # Find rigid affine transformations
        if fit_routine == 'ellipse':
            contour_trafo, _ = match_contour_with_ellipse(contour, contour_list[template_index], rot_ang=ang)
        elif fit_routine == 'bbox':
            contour_trafo, _ = match_contour_with_bbox(contour, contour_list[template_index], rot_ang=ang)
        else:
            raise ValueError(f'The specified fit routine: {fit_routine}, is not implemented!')

        # Apply transformations to contours and grids
        matched_contour = contour_trafo(contour)
        matched_grid = contour_trafo(grid_list[i]) if grid_list[i] is not None else None
        matched_point = contour_trafo(init_points[i]) if init_points[i] is not None else None

        # Store transformed arrays
        matched_contours.append(matched_contour)
        matched_grids.append(matched_grid)
        matched_points.append(matched_point)
        affine_matrices.append(np.linalg.inv(contour_trafo.params))  # Save inverted 3x3 affine matrix

    # Circularly shift contours to match template
    template_contour = matched_contours[template_index]
    shifted_contours = circularly_shift_contours(matched_contours,
                                                 template_contour=template_contour,
                                                 init_points=matched_points)

    return shifted_contours, matched_grids, affine_matrices


def angle_between_lines(source_axis: np.ndarray, target_axis: np.ndarray) -> float:
    """
    Calculate the signed angle (in radians) from source_axis to target_axis in 2D.

    Parameters:
    - source_axis: (2, 2) array defining the first line (2 points, 2D)
    - target_axis: (2, 2) array defining the second line (2 points, 2D)

    Returns:
    - signed_angle: float, angle in radians in range ]-π, π[
    """
    if not isinstance(source_axis, np.ndarray) or source_axis.shape != (2, 2):
        raise ValueError("source_axis must be a numpy array of shape (2, 2)")
    if not isinstance(target_axis, np.ndarray) or target_axis.shape != (2, 2):
        raise ValueError("target_axis must be a numpy array of shape (2, 2)")

    # Compute direction vectors
    v1 = source_axis[1] - source_axis[0]  # Vector of first line
    v2 = target_axis[1] - target_axis[0]  # Vector of second line

    # Compute angle using dot product (magnitude)
    dot_product = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)

    if norm == 0:
        raise ValueError("One or both line segments have zero length")

    cos_theta = dot_product / norm
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)  # Always 0 to π

    # Compute the sign using the 2D cross product
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    signed_angle = np.sign(cross_z) * angle

    return signed_angle


def match_contour_with_ellipse(a: np.ndarray, b: np.ndarray, rot_ang: float | None = None) ->(
        tuple)[AffineTransform,AffineTransform]:
    """
    Compute affine transformations that align contour `a` to contour `b` using ellipse fitting.

    Parameters
    ----------
    a : np.ndarray of shape (N, 2)
        First contour to be aligned (source). Must be a 2D NumPy array of points.
    b : np.ndarray of shape (M, 2)
        Second contour to be matched against (target). Must be a 2D NumPy array of points.
    rot_ang : float or None
        Optional external rotation angle (in radians). If `None`, the angle is computed from the
        difference in fitted ellipse orientations (shortest path within ±90°).

    Returns
    -------
    affine_transformation_a : AffineTransform
        Composite transformation (translate → rotate → scale) that aligns `a` to `b`.
    affine_transformation_b : AffineTransform
        Translation-only transformation to centre `b` at the origin.
    """
    if not (isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == 2):
        raise ValueError("Input a must be a (N, 2) numpy array")
    if not (isinstance(b, np.ndarray) and b.ndim == 2 and b.shape[1] == 2):
        raise ValueError("Input b must be a (N, 2) numpy array")

    def fit_ellipse(contour):
        if len(contour) < 5:
            return None
        return cv2.fitEllipse(contour.astype(np.float32))

    ellipse_a = fit_ellipse(a)
    ellipse_b = fit_ellipse(b)
    if ellipse_a is None or ellipse_b is None:
        raise ValueError("No ellipse could be fitted to one or both contours.")

    # Extract translation fit parameter
    centre_a = -np.array(ellipse_a[0])  # Important: Ellipse fit centre does not necessary align with geometric mean!
    centre_b = -np.array(ellipse_b[0])

    # Extract rotation angle as a fit parameter or use precomputed angle
    if rot_ang is None:
        rot_ang = ellipse_b[2] - ellipse_a[2]  # Angle difference
        if rot_ang > 90:
            rot_ang = rot_ang - 180
        elif rot_ang < -90:
            rot_ang = rot_ang + 180
        rot_ang = np.deg2rad(rot_ang)

    ang = rot_ang

    # Extract semi-major and semi-minor axes
    axes_a = np.array(ellipse_a[1])
    axes_b = np.array(ellipse_b[1])

    # Compute scaling factors for scaling in x and y after rotation
    scale_x = axes_b[0] / axes_a[0]  # Scaling factor along the major axis
    scale_y = axes_b[1] / axes_a[1]  # Scaling factor along the minor axis

    # Define Transformation matrix for a and b
    affine_transformation_a = (AffineTransform(translation=(centre_a[0], centre_a[1])) +
                               AffineTransform(rotation=ang) +
                               AffineTransform(scale=(scale_x, scale_y))
                               )
    affine_transformation_b = AffineTransform(translation=(centre_b[0], centre_b[1]))

    return affine_transformation_a, affine_transformation_b


def match_contour_with_bbox(a: np.ndarray, b: np.ndarray, rot_ang: float = 0) -> tuple[AffineTransform, AffineTransform]:
    """
    Compute affine transformation to match contour `a` to contour `b` based on bounding box alignment.

    Parameters
    ----------
    a : np.ndarray of shape (N, 2)
        Source contour.
    b : np.ndarray of shape (M, 2)
        Target contour.
    rot_ang : float or None
        Rotation angle in radians.

    Returns
    -------
    affine_transformation_a : AffineTransform
        Transformation to apply to `a` to align with `b`.
    affine_transformation_b : AffineTransform
        Translation-only transform to centre `b`.
    """
    if not (isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == 2):
        raise ValueError("Input a must be a (N, 2) numpy array")
    if not (isinstance(b, np.ndarray) and b.ndim == 2 and b.shape[1] == 2):
        raise ValueError("Input b must be a (N, 2) numpy array")

    # Translate contours to their geometric centre
    centre_a = -np.mean(a, axis=0)
    a_cent = a + centre_a
    centre_b = -np.mean(b, axis=0)
    b_cent = b + centre_b

    # Rotate contour
    rotation = AffineTransform(rotation=rot_ang)
    a_rot = rotation(a_cent)

    # Get boundary box corner points
    source_points = extract_bbox_corners(a_rot)
    target_points = extract_bbox_corners(b_cent)

    # Scale in x and y using the corner boundary box corner points
    dist_x_target = (target_points[3] - target_points[0])[0]
    dist_x_source = (source_points[3] - source_points[0])[0]
    scale_x = dist_x_target / dist_x_source if dist_x_source != 0 else 1

    dist_y_target = (target_points[1] - target_points[0])[1]
    dist_y_source = (source_points[1] - source_points[0])[1]
    scale_y = dist_y_target / dist_y_source if dist_y_source != 0 else 1

    # Define Transformation matrix for a and b
    affine_transformation_a = (AffineTransform(translation=(centre_a[0], centre_a[1])) +
                               AffineTransform(rotation=rot_ang) +
                               AffineTransform(scale=(scale_x, scale_y))
                               )
    affine_transformation_b = AffineTransform(translation=(centre_b[0], centre_b[1]))

    return affine_transformation_a, affine_transformation_b


def extract_bbox_corners(contour: np.ndarray) -> np.ndarray:
    """
    Extract the four corners of the axis-aligned bounding box enclosing the given contour.

    Parameters
    ----------
    contour : np.ndarray of shape (N, 2)
        A set of 2D points (x, y).

    Returns
    -------
    corners : np.ndarray of shape (4, 2)
        Array of corner points ordered as:
        [bottom-left, top-left, top-right, bottom-right].
    """
    if not isinstance(contour, np.ndarray) or contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be a (N, 2) numpy array")

    x_min, y_min = np.min(contour, axis=0)
    x_max, y_max = np.max(contour, axis=0)

    return np.array([
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_max],
        [x_max, y_min],
    ])


def circularly_shift_contours(contours: List[np.ndarray], template_contour: np.ndarray, init_points: List[np.ndarray])\
        -> List[np.ndarray]:
    """
    Circularly shift contours to align starting point and orientation with a template.

    Parameters
    ----------
    contours : list of np.ndarray
        List of (N, 2) arrays representing contours.
    template_contour : np.ndarray
        (N, 2) array representing the reference contour.
    init_points : list of np.ndarray
        List of points indicating desired starting points for each contour; elements can be None

    Returns
    -------
    modified_contours : list of np.ndarray
        Contours shifted and reoriented to match template.
    """
    if not isinstance(template_contour, np.ndarray) or template_contour.ndim != 2 or template_contour.shape[1] != 2:
        raise ValueError("template_contour must be a (N, 2) numpy array")

    if len(init_points) != len(contours):
        raise ValueError("init_points must be the same length as contours")

    orientation = get_contour_orientation(template_contour)
    modified_contours = []

    for i, contour in enumerate(contours):
        # Topologically orient contour
        if get_contour_orientation(contour) != orientation:
            contour = contour[::-1]  # Reverse the order

        if init_points[i] is not None:
            # Find the closest initial point in init_points
            distances_to_init_points = np.linalg.norm(contour - init_points[i], axis=1)
            shift_index = np.argmin(distances_to_init_points)  # Find the closest point
        else:
            # Calculate the Euclidean distance between the starting point of template_contour and all points in contour
            distances = np.linalg.norm(contour - template_contour[0, :], axis=1)

            # Find the index of the point closest to the starting point of template_contour
            shift_index = np.argmin(distances)

        # Circularly shift the contour to start from the selected point
        shifted_contour = np.roll(contour, -shift_index, axis=0)

        modified_contours.append(shifted_contour)

    return modified_contours


def get_contour_orientation(contour: np.ndarray) -> str:
    """
    Determine the orientation of a 2D closed contour using the Shoelace formula.

    Parameters
    ----------
    contour : np.ndarray of shape (N, 2)
        A closed 2D contour represented as an array of (x, y) points.

    Returns
    -------
    orientation : str
        'clockwise' if the contour is oriented clockwise,
        'counterclockwise' otherwise.
    """
    if not isinstance(contour, np.ndarray) or contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be a (N, 2) numpy array")

    x, y = contour[:, 0], contour[:, 1]
    signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    if signed_area == 0:
        raise ValueError("Signed area enclosed by contour is exactly zero. Check if contour is valid and closed.")

    return 'clockwise' if signed_area < 0 else 'counterclockwise'


def boundary_match_contours(contours: List[np.ndarray], template_index: int = 0, curvature: float = 0.5) -> (
        Tuple)[List[np.ndarray], List[np.ndarray]]:
    """
    Align a list of contours to a reference template using dynamic time warping (DTW) with a curvature-based penalty.

    Parameters
    ----------
    contours : list of (N, 2) numpy arrays
        List of 2D contours to be aligned.
    template_index : int
        Index of the template contour to which others will be aligned.
    curvature : float
        Weight of the curvature penalty in DTW.

    Returns
    -------
    dtw_contours : list of np.ndarray
        Warped versions of each contour aligned to the template.
    dtw_tmp_contours : list of np.ndarray
        Corresponding warped versions of the template to match each input.
    """
    if not contours:
        raise ValueError("contours must not be empty.")
    if not (0 <= template_index < len(contours)):
        raise IndexError("template_index is out of bounds.")
    if not all(isinstance(c, np.ndarray) and c.ndim == 2 and c.shape[1] == 2 for c in contours):
        raise ValueError("Each contour must be a (N, 2) numpy array.")
    # Boundary matching using dynamic time warping with tangent penalty
    dtw_contours, dtw_tmp_contours = [], []

    for i, contour in enumerate(contours):
        contour_dtw, contour_template_dtw = segmented_contour_dtw(contour, contours[template_index],
                                                                  dtw_curvature=curvature)
        dtw_contours.append(contour_dtw)
        dtw_tmp_contours.append(contour_template_dtw)

    return dtw_contours, dtw_tmp_contours
