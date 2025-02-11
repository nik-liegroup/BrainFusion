import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy.spatial.distance import directed_hausdorff
from frechetdist import frdist
from shapely.geometry import Polygon, Point, LineString


def interpolate_contour(contour, num_points):
    assert isinstance(contour, np.ndarray), "Each contour must be a numpy array!"
    assert contour.ndim == 2 and contour.shape[1] == 2, "Each contour must be an Nx2 array!"

    dists = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    cumulative_dists = np.concatenate(([0], np.cumsum(dists)))
    interp_x = interp1d(cumulative_dists, contour[:, 1], kind='linear')
    interp_y = interp1d(cumulative_dists, contour[:, 0], kind='linear')
    new_dists = np.linspace(0, cumulative_dists[-1], num_points)
    return np.vstack((interp_y(new_dists), interp_x(new_dists))).T


def align_contours(contour_list, grid_list, template_index, fit_routine='ellipse'):
    # Circularly shift contours to match template and transform all contours to their geometric centre
    modified_contours, centres = circularly_shift_contours(contour_list, template_index)

    # Apply translation to grids
    modified_grids = [grid - centre for grid, centre in zip(grid_list, centres)]

    #  Match contours using fitting routine
    matched_contours, matched_grids = match_contours(modified_contours, modified_grids,
                                                     modified_contours[template_index], fit_routine)

    return matched_contours, matched_grids


def find_average_contour(contours_list, average='star_domain', metric='jaccard'):
    # Calculate the average contour
    avg_contour = calculate_average_contour(contours_list, average=average)

    # Calculate error
    errors = calculate_distances(contours_list, avg_contour, metric=metric)

    return avg_contour, errors


def circularly_shift_contours(contours, template_index=0):
    assert 0 <= template_index < len(contours), print("Contour template index out of range!")

    template_contour = contours[template_index]
    template_centre = np.mean(template_contour, axis=0)
    template_contour = template_contour - template_centre
    modified_contours = []
    centres = []

    # Get topological orientation of template contour
    orientation = get_contour_orientation(template_contour)
    for i, contour in enumerate(contours):
        if i == template_index:
            modified_contours.append(template_contour)
            centres.append(template_centre)
            continue

        # Centre contours around geometric mean
        contour_centre = np.mean(contour, axis=0)
        contour = contour - contour_centre

        # Calculate the Euclidean distance between the starting point of reference_contour and all points in contour
        distances = np.linalg.norm(contour - template_contour[0], axis=1)

        # Find the index of the point closest to the starting point of the reference_contour
        shift_index = np.argmin(distances)

        # Circularly shift the contour to start from that point
        shifted_contour = np.roll(contour, -shift_index, axis=0)

        # Topologically orient contour
        if get_contour_orientation(contour) != orientation:
            shifted_contour = shifted_contour[::-1]  # Reverse the order

        modified_contours.append(shifted_contour)
        centres.append(contour_centre)

    return modified_contours, centres


def get_contour_orientation(contour):
    # Calculate the signed area using the Shoelace formula
    x, y = contour[:, 0], contour[:, 1]
    signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    assert signed_area != 0, print("Signed area enclosed by contour is exactly zero!")
    return 'clockwise' if signed_area < 0 else 'counterclockwise'


def match_contours(contours, grids, template_contour, fit_routine='ellipse'):
    matched_contours = []
    matched_grids = []
    for i, contour in enumerate(contours):
        if fit_routine == 'ellipse':
            matched_contour, grid_rotated, _, _, _ = match_contour_with_ellipse(contour, template_contour, grids[i])
        else:
            matched_contour = contour
            grid_rotated = grids[i]
            print("No fit routine applied to match contours!")

        matched_contours.append(matched_contour)
        matched_grids.append(grid_rotated)
    return matched_contours, matched_grids


def match_contour_with_ellipse(A, B, grid):
    ellipse_A = fit_ellipse(A)
    ellipse_B = fit_ellipse(B)
    if ellipse_A is None or ellipse_B is None:
        print("No ellipse could be fitted to the contours! Continue with original shapes.")
        return A, None, None, None  # Return original if ellipse can't be fitted
    center_A = np.array(ellipse_A[0])
    center_B = np.array(ellipse_B[0])
    rotation_angle = ellipse_B[2] - ellipse_A[2]  # Angle difference
    if 90 < rotation_angle:
        rotation_angle = rotation_angle - 180
    elif rotation_angle < -90:
        rotation_angle = rotation_angle + 180
    rotation_angle = np.deg2rad(rotation_angle)
    A_rotated = rotate_coordinate_system(A, rotation_angle, center_A)
    grid_rotated = rotate_coordinate_system(grid, rotation_angle, center_A)

    return A_rotated, grid_rotated, rotation_angle, center_A, center_B


def fit_ellipse(contour):
    if len(contour) < 5:
        return None
    return cv2.fitEllipse(contour.astype(np.float32))


def rotate_coordinate_system(contour, angle, center):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return (contour - center).dot(R.T) + center


def calculate_average_contour(contours, average='star_domain', num_bins=360):
    if average == 'star_domain':
        def is_star_domain(contour, tol=10):
            centre = [0, 0]
            polygon = Polygon(contour)  # Convert contour to a polygon
            center_point = Point(centre)

            # Step 1: Check if the center is inside the polygon
            if not polygon.contains(center_point):
                return False

            # Step 2: Check if all boundary points are visible from the center
            for point in contour:
                boundary_point = Point(point)
                line_of_sight = LineString([centre, point])

                # Compute intersections with the polygon boundary
                intersection = polygon.boundary.intersection(line_of_sight)

                # Convert intersection points to a list of coordinates
                if isinstance(intersection, Point):  # Single intersection
                    intersections = [intersection]
                elif hasattr(intersection, "geoms"):  # Multiple intersections
                    intersections = list(intersection.geoms)
                else:
                    return False  # Unexpected case

                distances = [boundary_point.distance(inter) for inter in intersections]
                if not all(d < tol for d in distances):
                    return False

            return True

        # Check if all contours are star domains with respect to the given centre
        assert all(is_star_domain(contour) for contour in contours), ('Not all contours define star domains with '
                                                                      'respect to their geometric centre!')

        # Convert contours to polar coordinates
        angles = []
        radii = []

        for contour in contours:
            x, y = contour[:, 0], contour[:, 1]
            theta = np.arctan2(contour[:, 1], contour[:, 0])  # Compute angles
            r = np.sqrt(x ** 2 + y ** 2)
            angles.append(theta)
            radii.append(r)

        # Create a uniform grid of angles
        angle_bins = np.linspace(-np.pi, np.pi, num_bins)

        # Interpolate radii for each contour onto the uniform grid
        interpolated_radii = np.array([
            np.interp(angle_bins, np.sort(theta), r[np.argsort(theta)])
            for theta, r in zip(angles, radii)
        ])

        # Compute the mean radius at each angle bin
        avg_radii = np.mean(interpolated_radii, axis=0)

        # Convert mean polar coordinates back to Cartesian
        avg_x = avg_radii * np.cos(angle_bins)
        avg_y = avg_radii * np.sin(angle_bins)

        contour = np.column_stack((avg_x, avg_y))

    elif average == 'median':
        contour = np.median(np.array(contours), axis=0)

    elif average == 'mean':
        contour = np.mean(np.array(contours), axis=0)

    else:
        raise Exception(f'{average} is not defined for contour averaging.')

    # Close contour
    contour[-1, :] = contour[0, :]
    return contour


def calculate_distances(contours, master_contour, metric='jaccard'):
    def jaccard_distance(curve_a, curve_b):
        # Create Polygon objects
        polya = Polygon(curve_a)
        polyb = Polygon(curve_b)

        # Calculate the intersection and union of the polygons
        intersection_area = polya.intersection(polyb).area
        union_area = polya.union(polyb).area

        # Calculate the Jaccard Index based on areas
        jaccard_index = intersection_area / union_area

        # Calculate the Jaccard Distance
        jaccard_distance = 1 - jaccard_index

        return jaccard_distance

    def frechet_distance(curve_a, curve_b):
        try:
            frechet_dist = frdist(np.array(curve_a), np.array(curve_b))
        except RuntimeWarning:
            print('Recursion error while calculating Frechet_distance!')
            frechet_dist = np.nan

        return frechet_dist

    def hausdorff_distance(curve_a, curve_b):
        return max(
            directed_hausdorff(curve_a, curve_b)[0],
            directed_hausdorff(curve_b, curve_a)[0]
        )

    results = []
    for i, contour in enumerate(contours):
        if metric == 'jaccard':
            dist = jaccard_distance(contour, master_contour)

        elif metric == 'frechet':
            # Fréchet distance
            dist = frechet_distance(contour, master_contour)

        elif metric == 'hausdorff':
            # Hausdorff distance
            dist = hausdorff_distance(contour, master_contour)

        else:
            raise Exception(f'{metric} is not defined as an error metric.')

        # Append distances for the current curve
        results.append(dist)

    return results
