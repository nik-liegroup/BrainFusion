import pytest
import numpy as np
import matplotlib.pyplot as plt
from brainfusion._average_contours import (find_average_contour, calculate_average_contour, is_star_domain,
                                           calculate_error_distances, jaccard_distance, frechet_distance,
                                           hausdorff_distance)


class TestFindAverageContour:

    def test_star_domain_average_and_errors_shape(self):
        # Create 3 circles of radius 1 around origin with slight perturbations
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=True)
        base = np.column_stack((np.cos(angles), np.sin(angles)))

        # Define small translations to simulate perturbed contours
        translations = [[0.1, 0.1], [0.9, 0.0], [-0.1, 0.05]]
        contours = [base + np.array(shift) for shift in translations]

        avg, errors = find_average_contour(contours, average='star_domain', error_metric='jaccard')
        assert isinstance(avg, np.ndarray)
        assert avg.shape[1] == 2
        assert len(errors) == 3

    def test_median_average(self):
        c1 = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        c2 = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        avg, errors = find_average_contour([c1, c2], average='median', error_metric='hausdorff')
        assert avg.shape == c1.shape
        assert errors.shape == (2,)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError, match="contours_list must not be empty"):
            find_average_contour([], average='mean')

        with pytest.raises(ValueError, match="Each contour must be a"):
            find_average_contour([np.array([[1, 2, 3]])], average='mean')


class TestCalculateAverageContour:

    def test_mean_average(self):
        # Two identical square contours
        square1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        square2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        avg = calculate_average_contour([square1, square2], average='mean')
        np.testing.assert_allclose(avg, square1, atol=1e-10)
        # Ensure contour is closed
        assert np.allclose(avg[0], avg[-1])

    def test_median_average(self):
        # Two simple triangles with one shifted
        tri1 = np.array([[0, 0], [1, 0], [0.5, 1], [0, 0]])
        tri2 = np.array([[0, 0], [1, 0], [0.5, 2], [0, 0]])
        avg = calculate_average_contour([tri1, tri2], average='median')
        expected = np.array([[0, 0], [1, 0], [0.5, 1.5], [0, 0]])
        np.testing.assert_allclose(avg, expected)
        # Ensure contour is closed
        assert np.allclose(avg[0], avg[-1])

    def test_star_domain_average_circle(self):
        # Circles around origin → result should approximate circle
        theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
        circle1 = np.column_stack([np.cos(theta), np.sin(theta)])
        circle2 = np.column_stack([1.5*np.cos(theta), 1.5*np.sin(theta)])
        avg = calculate_average_contour([circle1, circle2], average='star_domain', star_bins=100)
        avg_radii = np.sqrt(avg[:, 0]**2 + avg[:, 1]**2)
        # Should be ~ mean radius (1 + 1.5)/2 = 1.25
        assert np.allclose(np.mean(avg_radii), 1.25, rtol=1e-2)
        # Ensure contour is closed
        assert np.allclose(avg[0], avg[-1])

    def test_mean_invalid_shape_raises(self):
        # Different number of points → error
        c1 = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        c2 = np.array([[0, 0], [1, 0], [0, 0]])
        with pytest.raises(ValueError, match="same number of points"):
            calculate_average_contour([c1, c2], average='mean')

    def test_median_invalid_shape_raises(self):
        c1 = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        c2 = np.array([[0, 0], [1, 0], [0, 0]])
        with pytest.raises(ValueError, match="same number of points"):
            calculate_average_contour([c1, c2], average='median')

    def test_invalid_average_type(self):
        c = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        with pytest.raises(Exception, match="not defined for contour averaging"):
            calculate_average_contour([c, c], average='invalid')

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="contours must not be empty"):
            calculate_average_contour([], average='mean')

    def test_bad_contour_shape_raises(self):
        c = np.array([1, 2, 3])  # invalid shape
        with pytest.raises(ValueError, match="Each contour must be a \\(N, 2\\) np.ndarray"):
            calculate_average_contour([c], average='mean')


class TestIsStarDomain:

    def test_square_with_center_inside(self):
        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        assert is_star_domain(contour, centre=(0.5, 0.5)) is True

    def test_square_with_center_outside(self):
        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        assert is_star_domain(contour, centre=(2, 2)) is False

    def test_concave_polygon_not_star(self):
        # A concave polygon with a deep inward dent — star-shaped only from far left
        contour = np.array([
            [0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2], [0, 0]
        ])
        #plot_star_domain_check(contour, (1.5, 0.75))
        assert is_star_domain(contour, centre=(1.5, 0.75)) is False

    def test_concave_polygon_star_from_right_centre(self):
        # Same polygon, but from another centre where all points are visible
        contour = np.array([
            [0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2], [0, 0]
        ])
        #plot_star_domain_check(contour, (0.5, 0.5))
        assert is_star_domain(contour, centre=(0.5, 0.5)) is True

    def test_circle_approximation(self):
        theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
        circle = np.column_stack((np.cos(theta), np.sin(theta)))
        assert is_star_domain(circle, centre=(0, 0)) is True

    def test_invalid_shape_input(self):
        with pytest.raises(ValueError, match="Contour must be a numpy array of shape \\(N, 2\\)"):
            is_star_domain(np.array([1, 2, 3]), centre=(0, 0))

    def test_sensitive_to_tolerance(self):
        # Square with a small inward bump near the centre
        contour = np.array([
            [0, 0], [1, 0], [1, 1], [0.4, 1], [0.6, 0.95], [0.3, 1], [0, 1], [0, 0]
        ])
        centre = (0.5, 0.5)
        # plot_star_domain_check(contour, centre)
        assert is_star_domain(contour, centre=centre, tol_factor=1e-6) is False
        assert is_star_domain(contour, centre=centre, tol_factor=1e-1) is True


class TestCalculateErrorDistances:

    def setup_method(self):
        # Basic square contour
        self.square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        # Slightly offset square
        self.offset_square = self.square + 0.1
        # Scaled square
        self.scaled_square = self.square * 1.5
        self.contours = [self.square, self.offset_square, self.scaled_square]

    def test_jaccard(self):
        dists = calculate_error_distances(self.contours, self.square, metric='jaccard')
        assert dists[0] == pytest.approx(0.0)
        assert all(0.0 <= d <= 1.0 for d in dists)

    def test_frechet(self):
        # Use simplified, equal-length curves for Fréchet
        curve1 = np.array([[0, 0], [1, 1]])
        curve2 = np.array([[0, 0], [0.5, 0.5]])
        curve3 = np.array([[1, 1], [2, 2]])
        dists = calculate_error_distances([curve1, curve2, curve3], curve1, metric='frechet')
        assert dists[0] == pytest.approx(0.0)
        assert dists[1] > 0.0
        assert dists[2] > 0.0

    def test_hausdorff(self):
        dists = calculate_error_distances(self.contours, self.square, metric='hausdorff')
        assert dists[0] == pytest.approx(0.0)
        assert all(d >= 0.0 for d in dists)

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            calculate_error_distances(self.contours, self.square, metric='banana')

    def test_empty_list(self):
        result = calculate_error_distances([], self.square, metric='jaccard')
        assert isinstance(result, np.ndarray)
        assert result.size == 0


class TestJaccardDistance:

    def test_identical_polygons(self):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        dist = jaccard_distance(square, square)
        assert dist == pytest.approx(0.0)

    def test_non_overlapping_polygons(self):
        a = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        b = np.array([[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]])
        dist = jaccard_distance(a, b)
        assert dist == pytest.approx(1.0)

    def test_partial_overlap(self):
        a = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [0, 0]])
        b = np.array([[1, 0], [3, 0], [3, 1], [1, 1], [1, 0]])
        dist = jaccard_distance(a, b)
        # Overlap is 1x1 square, union is 3x1 → 1/3 = 0.333... → dist = 0.666...
        assert dist == pytest.approx(0.6666667, rel=1e-6)

    def test_not_closed_raises(self):
        open_curve = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # Missing [0, 0]
        closed_curve = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        with pytest.raises(ValueError, match="curve_a must be closed"):
            jaccard_distance(open_curve, closed_curve)

    def test_invalid_shape_raises(self):
        bad_input = np.array([0, 1, 2])  # Not (N, 2)
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        with pytest.raises(ValueError, match="curve_a must be a NumPy array"):
            jaccard_distance(bad_input, square)

    def test_too_few_points_raises(self):
        tiny = np.array([[0, 0], [1, 0], [0, 0]])  # 2 distinct points
        with pytest.raises(ValueError, match="curve_a must have at least 3 distinct points"):
            jaccard_distance(tiny, tiny)


class TestFrechetDistance:
    def test_identical_curves(self):
        curve = np.array([[0, 0], [1, 1], [2, 2]])
        dist = frechet_distance(curve, curve)
        assert dist == pytest.approx(0.0)

    def test_offset_curves(self):
        a = np.array([[0, 0], [1, 1], [2, 2]])
        b = np.array([[1, 1], [2, 2], [3, 3]])  # shifted
        dist = frechet_distance(a, b)
        assert dist > 0.0
        assert dist == pytest.approx(np.linalg.norm([1, 1]))

    def test_same_shape_different_sampling_raises(self):
        a = np.array([[0, 0], [1, 1]])
        b = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        with pytest.raises(ValueError, match="same number of points"):
            frechet_distance(a, b)

    def test_invalid_input_shape_raises(self):
        bad = np.array([1, 2, 3])
        good = np.array([[0, 0], [1, 1]])
        with pytest.raises(ValueError, match="curve_a must be a NumPy array of shape"):
            frechet_distance(bad, good)


class TestHausdorffDistance:

    def test_identical_curves(self):
        curve = np.array([[0, 0], [1, 1], [2, 2]])
        dist = hausdorff_distance(curve, curve)
        assert dist == pytest.approx(0.0)

    def test_symmetric_curves(self):
        a = np.array([[0, 0], [1, 0]])
        b = np.array([[1, 0], [0, 0]])  # reversed
        dist_ab = hausdorff_distance(a, b)
        dist_ba = hausdorff_distance(b, a)
        assert dist_ab == pytest.approx(dist_ba)

    def test_offset_curves(self):
        a = np.array([[0, 0], [1, 1]])
        b = np.array([[1, 1], [2, 2]])
        dist = hausdorff_distance(a, b)
        assert dist == pytest.approx(np.sqrt(2))  # furthest point: (0,0) to (1,1)

    def test_different_lengths(self):
        a = np.array([[0, 0]])
        b = np.array([[1, 1], [2, 2]])
        dist = hausdorff_distance(a, b)
        assert dist > 0

    def test_invalid_input_shape(self):
        bad = np.array([1, 2, 3])
        good = np.array([[0, 0], [1, 1]])
        with pytest.raises(ValueError, match="curve_a must be a NumPy array of shape"):
            hausdorff_distance(bad, good)

    def test_input_dim_check(self):
        bad = np.array([[0, 0, 1], [1, 1, 1]])
        good = np.array([[0, 0], [1, 1]])
        with pytest.raises(ValueError, match="must be a NumPy array of shape"):
            hausdorff_distance(good, bad)


# Helper function for visual inspection of star domains when debugging manually
def plot_star_domain_check(contour: np.ndarray, centre: tuple[float, float]):
    """
    Plot the polygon defined by `contour` and lines from `centre` to each contour vertex.
    """
    contour = np.asarray(contour)
    centre = np.asarray(centre)

    plt.figure(figsize=(6, 6))

    # Plot contour
    plt.plot(*contour.T, 'k-', label='Contour', linewidth=2)

    # Plot centre point
    plt.plot(centre[0], centre[1], 'ro', label='Centre')

    # Draw lines to each point
    for pt in contour:
        plt.plot([centre[0], pt[0]], [centre[1], pt[1]], 'r--', alpha=0.4)

    plt.axis('equal')
    plt.legend()
    plt.title('Star Domain Visibility Check')
    plt.show()