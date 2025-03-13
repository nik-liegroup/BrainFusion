import numpy as np
from scipy.spatial.distance import directed_hausdorff
from frechetdist import frdist
from shapely.geometry import Polygon, Point, LineString
from brainfusion._match_contours import get_contour_orientation


def find_average_contour(contours_list, average='star_domain', num_bins=360, metric='jaccard'):
    # Calculate the average contour
    avg_contour = calculate_average_contour(contours_list, average=average, num_bins=num_bins)

    # Calculate error
    errors = calculate_distances(contours_list, avg_contour, metric=metric)

    return avg_contour, errors


def calculate_average_contour(contours, average='star_domain', num_bins=360):
    if average == 'star_domain':
        def is_star_domain(contour, centre, tol=10):
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
                    raise Exception("An error occurred while checking if polygon is star domain.")

                distances = [boundary_point.distance(inter) for inter in intersections]
                if not all(d < tol for d in distances):
                    return False

            return True

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
        angle_bins = np.linspace(mean_angle, mean_angle + 2 * np.pi, num_bins, endpoint=False)

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

        contour = np.column_stack((avg_x, avg_y))

    elif average == 'median':
        contour = np.median(np.array(contours), axis=0)

    elif average == 'mean':
        contour = np.mean(np.array(contours), axis=0)

    else:
        raise Exception(f'{average} is not defined for contour averaging.')

    # Topologically orient contour
    if get_contour_orientation(contours[0]) != get_contour_orientation(contour):
        contour = contour[::-1]  # Reverse the order

    # Close contour
    contour[-1, :] = contour[0, :]
    return contour


def calculate_distances(contours, master_contour, metric='jaccard'):
    results = []
    for i, contour in enumerate(contours):
        if metric == 'jaccard':
            def jaccard_distance(curve_a, curve_b):
                # Create Polygon objects
                poly_a = Polygon(curve_a)
                poly_b = Polygon(curve_b)

                # Calculate the intersection and union of the polygons
                intersection_area = poly_a.intersection(poly_b).area
                union_area = poly_a.union(poly_b).area

                # Calculate the Jaccard Index based on areas
                jaccard_index = intersection_area / union_area

                # Calculate the Jaccard Distance
                jaccard_dist = 1 - jaccard_index
                return jaccard_dist

            dist = jaccard_distance(contour, master_contour)

        elif metric == 'frechet':
            def frechet_distance(curve_a, curve_b):
                try:
                    frechet_dist = frdist(np.array(curve_a), np.array(curve_b))
                except RuntimeWarning:
                    print('Recursion error while calculating Frechet_distance!')
                    frechet_dist = np.nan
                return frechet_dist

            # Fréchet distance
            dist = frechet_distance(contour, master_contour)

        elif metric == 'hausdorff':
            def hausdorff_distance(curve_a, curve_b):
                return max(
                    directed_hausdorff(curve_a, curve_b)[0],
                    directed_hausdorff(curve_b, curve_a)[0]
                )

            # Hausdorff distance
            dist = hausdorff_distance(contour, master_contour)

        else:
            raise Exception(f'{metric} is not defined as an error metric.')

        # Append distances for the current curve
        results.append(dist)

    return results
