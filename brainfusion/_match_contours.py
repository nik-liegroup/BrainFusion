import numpy as np
import cv2
from scipy.interpolate import interp1d
import scipy.ndimage as ndi
from skimage.transform import AffineTransform
from scipy.stats import linregress
from collections import defaultdict


def interpolate_contour(contour, num_points):
    assert isinstance(contour, np.ndarray), "Each contour must be a numpy array!"
    assert contour.ndim == 2 and contour.shape[1] == 2, "Each contour must be an Nx2 array!"

    dists = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    cumulative_dists = np.concatenate(([0], np.cumsum(dists)))
    interp_x = interp1d(cumulative_dists, contour[:, 1], kind='linear')
    interp_y = interp1d(cumulative_dists, contour[:, 0], kind='linear')
    new_dists = np.linspace(0, cumulative_dists[-1], num_points)
    return np.vstack((interp_y(new_dists), interp_x(new_dists))).T


def align_contours(contour_list, grid_list, rot_axes=None, init_points=None, template_index=0, fit_routine='ellipse'):
    """
    Align a set of contours to a template contour specified by template index in the contours list. If template_index is
    None, calculate average contour and align to it.
    """
    assert 0 <= template_index < len(contour_list), print("Contour template index out of range!")

    # Match contours using ellipse fit
    matched_contours, matched_grids, matched_points = [], [], []
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

    # Circularly shift contours to match template
    template_contour = matched_contours[template_index]
    shifted_contours = circularly_shift_contours(matched_contours,
                                                 template_contour=template_contour,
                                                 init_points=matched_points)

    return shifted_contours, matched_grids


def angle_between_lines(source_axis, target_axis):
    """Calculate the signed angle between two lines (in radians)."""
    # Compute direction vectors
    v1 = source_axis[1] - source_axis[0]  # Vector of first line
    v2 = target_axis[1] - target_axis[0]  # Vector of second line

    # Compute angle using dot product (magnitude)
    dot_product = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = dot_product / norm if norm != 0 else 1
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)  # Always 0 to π

    # Compute the sign using the 2D cross product
    cross_product = np.cross(v1, v2)  # Scalar value in 2D
    signed_angle = np.sign(cross_product) * angle  # Apply sign

    return signed_angle


def match_contour_with_ellipse(a, b, rot_ang):
    ellipse_a = fit_ellipse(a)
    ellipse_b = fit_ellipse(b)
    assert ellipse_a is not None and ellipse_b is not None, 'No ellipse could be fitted to the contours!'

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


def fit_ellipse(contour):
    if len(contour) < 5:
        return None
    return cv2.fitEllipse(contour.astype(np.float32))


def match_contour_with_bbox(a, b, rot_ang=None):
    """Find affine transformation to match two rigid contours."""
    # Translate contours to their geometric centre
    centre_a = -np.mean(a, axis=0)
    a_cent = a + centre_a
    centre_b = -np.mean(b, axis=0)
    b_cent = b + centre_b

    # Get boundary box corner points
    source_points = extract_bbox_corners(a_cent)
    target_points = extract_bbox_corners(b_cent)

    # Scale in x and y using the corner boundary box corner points
    dist_x_target = (target_points[3] - target_points[0])[0]
    dist_x_source = (source_points[3] - source_points[0])[0]
    scale_x = dist_x_target / dist_x_source if dist_x_source != 0 else 1

    dist_y_target = (target_points[1] - target_points[0])[1]
    dist_y_source = (source_points[1] - source_points[0])[1]
    scale_y = dist_y_target / dist_y_source if dist_y_source != 0 else 1

    ang = regression_alignment_angle(a_cent[:, 0], b_cent) if rot_ang is None else rot_ang

    # Define Transformation matrix for a and b
    affine_transformation_a = (AffineTransform(translation=(centre_a[0], centre_a[1])) +
                               AffineTransform(rotation=ang) +
                               AffineTransform(scale=(scale_x, scale_y))
                               )
    affine_transformation_b = AffineTransform(translation=(centre_b[0], centre_b[1]))

    return affine_transformation_a, affine_transformation_b


def extract_bbox_corners(contour):
    """Extract key points from the four corners of the bounding box."""
    x_min, y_min = np.min(contour, axis=0)
    x_max, y_max = np.max(contour, axis=0)
    return np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])


def regression_alignment_angle(source_contour, target_contour):
    """Fit a line to the right side of the contours and compute rotation."""
    right_source = source_contour[source_contour[:, 0] > np.percentile(source_contour[:, 0], 98)]
    right_target = target_contour[target_contour[:, 0] > np.percentile(target_contour[:, 0], 98)]
    slope_source, _, _, _, _ = linregress(right_source[:, 0], right_source[:, 1])
    slope_target, _, _, _, _ = linregress(right_target[:, 0], right_target[:, 1])
    angle = np.arctan(slope_target) - np.arctan(slope_source)

    # Adjust angle to wrap around within the range of ±π/2
    if angle > np.pi / 2:
        angle -= np.pi
    elif angle < -np.pi / 2:
        angle += np.pi

    return angle


def circularly_shift_contours(contours, template_contour, init_points=None):
    modified_contours = []

    # Get topological orientation of template contour
    orientation = get_contour_orientation(template_contour)

    for i, contour in enumerate(contours):
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

        # Topologically orient contour
        if get_contour_orientation(contour) != orientation:
            shifted_contour = shifted_contour[::-1]  # Reverse the order

        # Close contours
        shifted_contour[-1, :] = shifted_contour[0, :]

        modified_contours.append(shifted_contour)

    return modified_contours


def get_contour_orientation(contour):
    # Calculate the signed area using the Shoelace formula
    x, y = contour[:, 0], contour[:, 1]
    signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    assert signed_area != 0, print("Signed area enclosed by contour is exactly zero!")
    return 'clockwise' if signed_area < 0 else 'counterclockwise'


def boundary_match_contours(contours, template_index=0):
    # Boundary matching using dynamic time warping with tangent penalty
    dtw_contours, dtw_tmp_contours = [], []

    for i, contour in enumerate(contours):
        contour_dtw, contour_template_dtw = dtw_with_curvature_penalty(contour, contours[template_index])
        dtw_contours.append(contour_dtw)
        dtw_tmp_contours.append(contour_template_dtw)

    return dtw_contours, dtw_tmp_contours


def dtw_with_curvature_penalty(contour1, contour2, base_lambda_curvature=0.5, alpha=0.05):
    """Match contours using DTW with a curvature scale-space (CSS) constraint and normalized sigma."""

    def compute_contour_length(contour):
        """Compute total contour length as sum of segment distances."""
        return np.sum(np.linalg.norm(np.diff(contour, axis=0), axis=1))

    def gaussian_smooth_contour(contour, sigma):
        """Smooth contour using Gaussian filter."""
        smoothed_x = ndi.gaussian_filter1d(contour[:, 0], sigma, mode='wrap')
        smoothed_y = ndi.gaussian_filter1d(contour[:, 1], sigma, mode='wrap')
        return np.column_stack((smoothed_x, smoothed_y))

    def compute_css_curvature(contour, sigma):
        """Compute curvature at a given scale using smoothed contours."""
        smoothed_contour = gaussian_smooth_contour(contour, sigma)

        dx = np.gradient(smoothed_contour[:, 0])
        dy = np.gradient(smoothed_contour[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2 + 1e-10) ** (3 / 2)
        return curvature

    # Compute normalized sigma based on contour length
    len1, len2 = compute_contour_length(contour1), compute_contour_length(contour2)
    sigma1 = alpha * (len1 / len(contour1))
    sigma2 = alpha * (len2 / len(contour2))

    n, m = len(contour1), len(contour2)

    # Compute curvature at adaptive scales
    curvatures1 = compute_css_curvature(contour1, sigma1)
    curvatures2 = compute_css_curvature(contour2, sigma2)

    # Normalize lambda_curvature dynamically
    spatial_dists = [np.linalg.norm(contour1[i] - contour2[j]) for i in range(n) for j in range(m)]
    curvature_diffs = [np.abs(curvatures1[i] - curvatures2[j]) for i in range(n) for j in range(m)]

    median_spatial = np.median(spatial_dists)
    median_curvature = np.median(curvature_diffs)

    lambda_curvature = base_lambda_curvature * (median_spatial / (median_curvature + 1e-10))

    def cost_function(i, j):
        """Cost function incorporating spatial distance and CSS-based curvature penalty."""
        p1, p2 = contour1[i], contour2[j]
        spatial_dist = np.linalg.norm(p1 - p2)
        curvature_penalty = lambda_curvature * np.abs(curvatures1[i] - curvatures2[j])
        return spatial_dist ** 2 + curvature_penalty

    # Initialize DP matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Compute DTW
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_function(i - 1, j - 1)
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],  # Insertion
                                          dtw_matrix[i, j - 1],  # Deletion
                                          dtw_matrix[i - 1, j - 1])  # Match

    # Backtrack to find the optimal path
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
        if step == 0:
            i -= 1  # Move up
        elif step == 1:
            j -= 1  # Move left
        else:
            i -= 1  # Move diagonally
            j -= 1

    path.reverse()

    idx1, idx2 = np.array(list(zip(*path)))

    # Dictionary to store values corresponding to idx1
    contour2_mapping = defaultdict(list)

    # Store values of contour2 based on alignment
    for i1, i2 in zip(idx1, idx2):
        contour2_mapping[i1].append(contour2[i2])  # Store actual contour2 values, not indices

    # Compute averaged values
    idx1_unique = np.array(list(contour2_mapping.keys()))
    contour2_averaged = np.array(
        [np.mean(values, axis=0) for values in contour2_mapping.values()])  # Averaging the points

    return contour1[idx1_unique], contour2_averaged
