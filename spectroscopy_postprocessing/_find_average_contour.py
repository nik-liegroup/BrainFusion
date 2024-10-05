import numpy as np
import cv2
from scipy.interpolate import interp1d


def extract_and_interpolate_contours(masks, num_points=1000):
    contours = []
    for mask in masks:
        contour = extract_contours(mask)
        contour_interp = interpolate_contour(contour, num_points)
        contours.append(contour_interp)
    return contours


def extract_contours(mask):
    # Ensure mask is in the correct format (uint8)
    mask = (mask > 0).astype(np.uint8)

    # Find the external contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assuming there is only one main contour, return the largest contour
        contour = max(contours, key=cv2.contourArea)
        # Remove the unnecessary dimension
        contour = np.squeeze(contour)

        # Close contour
        contour[-1, :] = contour[0, :]  # Close contour
    else:
        contour = None

    return contour


def interpolate_contour(contour, num_points):
    dists = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    cumulative_dists = np.concatenate(([0], np.cumsum(dists)))
    interp_x = interp1d(cumulative_dists, contour[:, 1], kind='linear')
    interp_y = interp1d(cumulative_dists, contour[:, 0], kind='linear')
    new_dists = np.linspace(0, cumulative_dists[-1], num_points)
    return np.vstack((interp_y(new_dists), interp_x(new_dists))).T


def match_contours(contours, template_index):
    assert 0 <= template_index < len(contours), print("Contour template index out of range!")
    template_contour = contours[template_index]
    matched_contours = []  # Keep template as the first contour
    for i, contour in enumerate(contours):
        if i == template_index:
            matched_contours.append(template_contour)
            continue

        # Shift and match contours
        shifted_contour = shift_contour(contour, template_contour)
        matched_contour, _, _, _ = match_contour_with_ellipse(shifted_contour, template_contour)

        matched_contours.append(matched_contour)
    return matched_contours, template_contour


def shift_contour(contour, reference_contour):
    # Calculate the Euclidean distance between the starting point of reference_contour and all points in contour
    distances = np.linalg.norm(contour - reference_contour[0], axis=1)

    # Find the index of the point closest to the starting point of the reference_contour
    shift_index = np.argmin(distances)

    # Circularly shift the contour to start from that point
    shifted_contour = np.roll(contour, -shift_index, axis=0)

    return shifted_contour


def match_contour_with_ellipse(A, B):
    ellipse_A = fit_ellipse(A)
    ellipse_B = fit_ellipse(B)
    if ellipse_A is None or ellipse_B is None:
        print("No ellipse could be fitted to the contours! Continue with original shapes.")
        return A  # Return original if ellipse can't be fitted
    center_A = np.array(ellipse_A[0])
    center_B = np.array(ellipse_B[0])
    rotation_angle = ellipse_B[2] - ellipse_A[2]  # Angle difference
    if 90 < rotation_angle:
        rotation_angle = rotation_angle - 180
    elif rotation_angle < -90:
        rotation_angle = rotation_angle + 180
    rotation_angle = np.deg2rad(rotation_angle)
    A_rotated = rotate_coordinate_system(A, rotation_angle, center_A)

    return A_rotated + (center_B - center_A), rotation_angle, center_A, center_B


def fit_ellipse(contour):
    if len(contour) < 5:
        return None
    return cv2.fitEllipse(contour.astype(np.float32))


def rotate_coordinate_system(contour, angle, center):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return (contour - center).dot(R.T) + center


def calculate_median_contour(contours):
    contour = np.median(np.array(contours), axis=0)
    contour[-1, :] = contour[0, :]  # Close contour
    return contour


def find_average_contour(masks, template_index=0):
    # Extract and interpolate contours
    contours_list = extract_and_interpolate_contours(masks)

    # Circularly align and match contours using ellipse fitting
    matched_contour_list, template_contour = match_contours(contours_list, template_index)

    # Calculate the median contour
    median_contour = calculate_median_contour(matched_contour_list)

    return median_contour, contours_list, template_contour, matched_contour_list
