import pytest
import numpy as np
from brainfusion._match_contours import (interpolate_contour, angle_between_lines, match_contour_with_ellipse,
                                         match_contour_with_bbox, extract_bbox_corners, circularly_shift_contours,
                                         get_contour_orientation, boundary_match_contours, dtw_with_curvature_penalty)


class TestInterpolateContour:

    def test_linear_contour(self):
        contour = np.array([[0, 0], [1, 0]])  # Straight horizontal line
        num_points = 5
        result = interpolate_contour(contour, num_points)
        expected_x = np.linspace(0, 1, num_points)
        expected = np.column_stack((expected_x, np.zeros_like(expected_x)))
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_square_contour(self):
        contour = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0]  # Closed square
        ])
        result = interpolate_contour(contour, 100)
        assert result.shape == (100, 2)
        assert np.allclose(result[0], contour[0], atol=1e-6)
        assert np.allclose(result[-1], contour[0], atol=1e-6)  # Loop closed

    def test_output_length(self):
        contour = np.random.rand(10, 2)
        result = interpolate_contour(contour, 50)
        assert result.shape == (50, 2)

    def test_same_result_for_already_evenly_spaced(self):
        contour = np.array([[0, 0], [1, 0], [2, 0]])
        result = interpolate_contour(contour, 3)
        np.testing.assert_allclose(result, contour, atol=1e-8)

    def test_invalid_input_type(self):
        with pytest.raises(ValueError, match="contour must be a numpy array"):
            interpolate_contour([[0, 0], [1, 1]], 10)

    def test_invalid_input_shape(self):
        with pytest.raises(ValueError, match="contour must be a 2D array of shape"):
            interpolate_contour(np.array([[0], [1]]), 10)


class TestAngleBetweenLines:

    def test_zero_angle(self):
        axis = np.array([[0, 0], [1, 0]])
        angle = angle_between_lines(axis, axis)
        assert np.isclose(angle, 0.0)

    def test_ninety_degrees_ccw(self):
        source = np.array([[0, 0], [1, 0]])
        target = np.array([[0, 0], [0, 1]])
        angle = angle_between_lines(source, target)
        assert np.isclose(angle, np.pi / 2)

    def test_ninety_degrees_cw(self):
        source = np.array([[0, 0], [1, 0]])
        target = np.array([[0, 0], [0, -1]])
        angle = angle_between_lines(source, target)
        assert np.isclose(angle, -np.pi / 2)

    def test_opposite_direction(self):
        source = np.array([[0, 0], [1, 0]])
        target = np.array([[0, 0], [-1, 0]])
        angle = angle_between_lines(source, target)
        assert np.isclose(angle, 0.0)

    def test_input_validation_type(self):
        with pytest.raises(ValueError, match="source_axis must be a numpy array"):
            angle_between_lines([[0, 0], [1, 0]], np.array([[0, 0], [1, 1]]))

    def test_input_validation_shape(self):
        with pytest.raises(ValueError, match="target_axis must be a numpy array of shape"):
            angle_between_lines(np.array([[0, 0], [1, 0]]), np.array([[0, 0]]))

    def test_zero_length_vector(self):
        source = np.array([[0, 0], [0, 0]])
        target = np.array([[0, 0], [1, 1]])
        with pytest.raises(ValueError, match="zero length"):
            angle_between_lines(source, target)


class TestMatchContourWithEllipse:

    @staticmethod
    def generate_ellipse_contour(center, axes, angle_deg, n=100):
        """Helper to create a 2D ellipse contour."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = axes[0] * np.cos(t)
        y = axes[1] * np.sin(t)
        rot = np.deg2rad(angle_deg)
        R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        ellipse = np.dot(np.stack((x, y), axis=1), R.T) + center
        return ellipse

    def test_identity_alignment(self):
        contour = self.generate_ellipse_contour(center=(0, 0), axes=(2, 1), angle_deg=30)
        T_a, T_b = match_contour_with_ellipse(contour, contour, rot_ang=None)

        # Identity transform: a should map to b with T_a + T_b^-1 ≈ Identity
        combined = T_b.inverse + T_a
        test_pts = contour
        mapped_pts = combined(test_pts)
        np.testing.assert_allclose(mapped_pts, test_pts, atol=1e-6)

    def test_scaled_and_rotated_alignment(self):
        a = self.generate_ellipse_contour(center=(0, 0), axes=(1, 2), angle_deg=0)
        b = self.generate_ellipse_contour(center=(5, -3), axes=(2, 1), angle_deg=45)
        T_a, T_b = match_contour_with_ellipse(a, b, rot_ang=None)

        # Transform a using T_a and b using T_b → both should be aligned to origin
        a_transformed = T_a(a)
        b_transformed = T_b(b)

        # Need to circularly align contours for comparison
        best_shift = np.argmin([
            np.sum(np.linalg.norm(a_transformed - np.roll(b_transformed, shift=s, axis=0), axis=1) ** 2)
            for s in range(len(a_transformed))
        ])
        b_aligned = np.roll(b_transformed, shift=best_shift, axis=0)

        np.testing.assert_allclose(a_transformed, b_aligned, atol=1e-4)

    def test_invalid_input_a(self):
        arr = np.zeros((10, 2))
        with pytest.raises(ValueError, match="Input a must be a"):
            match_contour_with_ellipse("not an array", arr, None)

        with pytest.raises(ValueError, match="Input b must be a"):
            match_contour_with_ellipse(arr, "not an array", None)

    def test_short_inputs(self):
        with pytest.raises(ValueError, match="No ellipse could be fitted to one or both contours."):
            match_contour_with_ellipse(np.array([[0, 0], [1, 1], [1, 1]]), np.array([[0, 0], [1, 1], [1, 1]]),
                                       None)

    def test_angle_override(self):
        a = self.generate_ellipse_contour(center=(0, 0), axes=(1, 2), angle_deg=0)
        b = self.generate_ellipse_contour(center=(0, 0), axes=(1, 2), angle_deg=90)
        # Override with zero rotation
        T_a, T_b = match_contour_with_ellipse(a, b, rot_ang=0)
        transformed_a = T_a(a)
        transformed_b = T_b(b)
        assert not np.allclose(transformed_a, transformed_b, atol=1e-4)


class TestMatchContourWithBBox:

    @staticmethod
    def generate_ellipse_contour(center, axes, angle_deg, n=100):
        """Helper to create a 2D ellipse contour."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = axes[0] * np.cos(t)
        y = axes[1] * np.sin(t)
        rot = np.deg2rad(angle_deg)
        R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        ellipse = np.dot(np.stack((x, y), axis=1), R.T) + center
        return ellipse

    def test_identity_transform(self):
        rect = self.generate_ellipse_contour(center=(0, 0), axes=(1, 2), angle_deg=0)
        T_a, T_b = match_contour_with_bbox(rect, rect, 0)

        combined = T_b.inverse + T_a
        transformed = combined(rect)
        np.testing.assert_allclose(transformed, rect, atol=1e-6)

    def test_translation_scaling_rotation(self):
        a = self.generate_ellipse_contour(center=(1, 0), axes=(1, 2), angle_deg=0)
        b = self.generate_ellipse_contour(center=(0, 0), axes=(2, 1), angle_deg=45)

        T_a, T_b = match_contour_with_bbox(a, b, -np.pi/4)
        a_trans = T_a(a)
        b_trans = T_b(b)

        # Need to circularly align contours for comparison
        best_shift = np.argmin([
            np.sum(np.linalg.norm(a_trans - np.roll(b_trans, shift=s, axis=0), axis=1) ** 2)
            for s in range(len(a_trans))
        ])
        b_aligned = np.roll(b_trans, shift=best_shift, axis=0)

        # Shift b_trans to match a_trans (circular shift not needed here as points correspond)
        # Just compare centroid-aligned forms
        np.testing.assert_allclose(a_trans, b_aligned, atol=1e-5)

    def test_invalid_inputs(self):
        b = np.zeros((10, 2))
        with pytest.raises(ValueError, match="Input a must be a"):
            match_contour_with_bbox("invalid", b)

        a = np.zeros((10, 2))
        with pytest.raises(ValueError, match="Input b must be a"):
            match_contour_with_bbox(a, [[1, 2], [3, 4]])


class TestExtractBboxCorners:

    def test_square_contour(self):
        contour = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ])
        corners = extract_bbox_corners(contour)
        expected = np.array([
            [0, 0], [0, 1], [1, 1], [1, 0]
        ])
        np.testing.assert_array_equal(corners, expected)

    def test_rectangle_shifted(self):
        contour = np.array([
            [5, 5], [8, 5], [8, 7], [5, 7]
        ])
        corners = extract_bbox_corners(contour)
        expected = np.array([
            [5, 5], [5, 7], [8, 7], [8, 5]
        ])
        np.testing.assert_array_equal(corners, expected)

    def test_single_point(self):
        contour = np.array([[2, 3]])
        corners = extract_bbox_corners(contour)
        expected = np.array([[2, 3], [2, 3], [2, 3], [2, 3]])
        np.testing.assert_array_equal(corners, expected)

    def test_invalid_input_type(self):
        with pytest.raises(ValueError, match="contour must be a"):
            extract_bbox_corners([[0, 0], [1, 1]])  # Not a NumPy array

    def test_invalid_input_shape(self):
        with pytest.raises(ValueError):
            extract_bbox_corners(np.array([0, 0]))  # Not 2D


class TestCircularlyShiftContours:

    @staticmethod
    def square_contour(start_index=0, clockwise=True):
        contour = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ])
        if not clockwise:
            contour = contour[::-1]
        return np.roll(contour, -start_index, axis=0)

    def test_alignment_to_template(self):
        contour1 = self.square_contour(start_index=2)
        contour2 = self.square_contour(start_index=3)
        template = self.square_contour(start_index=0)
        init_points = [None, None]
        result = circularly_shift_contours([contour1, contour2], template, init_points)

        for r in result:
            np.testing.assert_allclose(r, template)

    def test_orientation_match(self):
        clockwise = self.square_contour(clockwise=True)
        counter_clockwise = self.square_contour(clockwise=False)
        template = clockwise
        result = circularly_shift_contours([counter_clockwise], template, init_points=[None])
        for r in result:
            np.testing.assert_allclose(r, template)

    def test_shift_to_custom_init_point(self):
        contour = self.square_contour(start_index=0)
        template = self.square_contour(start_index=0)
        init_point = np.array([1, 0])  # Should align to second point
        result = circularly_shift_contours([contour], template, init_points=[init_point])
        np.testing.assert_allclose(result[0][0], init_point)

    def test_mismatched_init_points_length(self):
        contour = self.square_contour()
        template = self.square_contour()
        with pytest.raises(ValueError, match="init_points must be the same length as contours"):
            circularly_shift_contours([contour], template, init_points=[np.array([0, 0]), np.array([1, 1])])

    def test_invalid_template_shape(self):
        contour = self.square_contour()
        bad_template = np.array([0, 1])
        with pytest.raises(ValueError, match="template_contour must be a"):
            circularly_shift_contours([contour], bad_template, init_points=[None])

    def test_invalid_contour_shape(self):
        bad_contour = np.array([0, 1])
        template = self.square_contour()
        with pytest.raises(ValueError, match="contour must be a"):
            circularly_shift_contours([bad_contour], template, init_points=[None])


class TestGetContourOrientation:

    def test_clockwise_orientation(self):
        # Simple square, ordered clockwise
        contour = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0]
        ])[::-1]  # Reverse to make it clockwise
        assert get_contour_orientation(contour) == 'clockwise'

    def test_counterclockwise_orientation(self):
        # Simple square, ordered counterclockwise
        contour = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0]
        ])
        assert get_contour_orientation(contour) == 'counterclockwise'

    def test_invalid_input_shape(self):
        with pytest.raises(ValueError, match="must be a \\(N, 2\\) numpy array"):
            get_contour_orientation(np.array([0, 1, 2]))  # Invalid shape

    def test_zero_area_raises(self):
        # All points on a line → area = 0
        contour = np.array([
            [0, 0],
            [1, 0],
            [2, 0],
            [0, 0]
        ])
        with pytest.raises(ValueError, match="Signed area enclosed by contour is exactly zero"):
            get_contour_orientation(contour)


class TestBoundaryMatchContours:

    @staticmethod
    def generate_circle_contour(radius=1.0, num_points=100, center=(0, 0)):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        return np.stack([x, y], axis=1)

    def test_basic_alignment_returns_same_length(self):
        contour1 = self.generate_circle_contour()
        contour2 = self.generate_circle_contour(radius=1.2, center=(0.5, -0.3))
        dtw_contours, dtw_templates = boundary_match_contours([contour1, contour2], template_index=0)

        assert len(dtw_contours) == 2
        assert len(dtw_templates) == 2
        for warped in dtw_contours + dtw_templates:
            assert isinstance(warped, np.ndarray)
            assert warped.shape[1] == 2  # Must be (N, 2)

    def test_invalid_input_shape(self):
        bad_contour = np.array([[0], [1]])  # Invalid shape: (2, 1) instead of (N, 2)
        with pytest.raises(ValueError, match="Each contour must be a"):
            boundary_match_contours([bad_contour], template_index=0)

    def test_invalid_template_index(self):
        c = self.generate_circle_contour()
        with pytest.raises(IndexError, match="template_index is out of bounds"):
            boundary_match_contours([c], template_index=2)

    def test_empty_contour_list(self):
        with pytest.raises(ValueError, match="contours must not be empty"):
            boundary_match_contours([], template_index=0)


class TestDTWWithCurvaturePenalty:

    @staticmethod
    def generate_circle(radius=1.0, n_points=100):
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        return np.stack((radius * np.cos(t), radius * np.sin(t)), axis=1)

    @staticmethod
    def generate_ellipse(a=1.0, b=0.5, n_points=100):
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        return np.stack((a * np.cos(t), b * np.sin(t)), axis=1)

    def test_identical_shapes(self):
        contour = self.generate_circle(n_points=50)
        warped1, warped2 = dtw_with_curvature_penalty(contour, contour, np.zeros(len(contour)), np.zeros(len(contour)))
        np.testing.assert_allclose(warped1, warped2, atol=1e-6)

    def test_ellipse_to_circle(self):
        circle = self.generate_circle(n_points=60)
        ellipse = self.generate_ellipse(n_points=60)
        warped1, warped2 = dtw_with_curvature_penalty(ellipse, circle,  np.zeros(len(ellipse)),  np.zeros(len(circle)))
        assert warped1.shape == warped2.shape
        assert warped1.shape[1] == 2

    def test_invalid_input_shape(self):
        bad_input = np.array([[0, 0], [1]], dtype=object)  # Ragged array
        circle = self.generate_circle()
        with pytest.raises(ValueError, match="must be a \(N, 2\) numpy array"):
            dtw_with_curvature_penalty(bad_input, circle, np.zeros(len(circle)), np.zeros(len(circle)))
        with pytest.raises(ValueError, match="must be a \(N, 2\) numpy array"):
            dtw_with_curvature_penalty(circle, bad_input, np.zeros(len(circle)), np.zeros(len(circle)))

    def test_output_dimensions_match(self):
        contour1 = self.generate_circle(n_points=120)
        contour2 = self.generate_ellipse(n_points=80)
        warped1, warped2 = dtw_with_curvature_penalty(contour1, contour2, np.zeros(len(contour1)), np.zeros(len(contour2)))
        assert warped1.shape == warped2.shape
        assert warped1.shape[1] == 2

    def test_extreme_curvature_weight(self):
        c1 = self.generate_circle()
        c2 = self.generate_ellipse()
        warped1, warped2 = dtw_with_curvature_penalty(c1, c2, np.zeros(len(c1)), np.zeros(len(c2)), dtw_curvature=10.0)
        assert warped1.shape == warped2.shape
