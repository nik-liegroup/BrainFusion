import pytest
import numpy as np
from brainfusion.fusion.dtw import dtw_with_curvature_penalty, detect_linear_segments_from_curvature, \
    segmented_contour_dtw, align_straight_segments


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
        warped1, warped2 = dtw_with_curvature_penalty(ellipse, circle, np.zeros(len(ellipse)), np.zeros(len(circle)))
        assert warped1.shape == warped2.shape
        assert warped1.shape[1] == 2

    def test_invalid_input_shape(self):
        bad_input = np.array([[0, 0], [1]], dtype=object)  # Ragged array
        circle = self.generate_circle()
        with pytest.raises(ValueError, match=r"must be a \(N, 2\) numpy array"):
            dtw_with_curvature_penalty(bad_input, circle, np.zeros(len(circle)), np.zeros(len(circle)))
        with pytest.raises(ValueError, match=r"must be a \(N, 2\) numpy array"):
            dtw_with_curvature_penalty(circle, bad_input, np.zeros(len(circle)), np.zeros(len(circle)))

    def test_output_dimensions_match(self):
        contour1 = self.generate_circle(n_points=120)
        contour2 = self.generate_ellipse(n_points=80)
        warped1, warped2 = dtw_with_curvature_penalty(contour1, contour2, np.zeros(len(contour1)),
                                                       np.zeros(len(contour2)))
        assert warped1.shape == warped2.shape
        assert warped1.shape[1] == 2

    def test_extreme_curvature_weight(self):
        c1 = self.generate_circle()
        c2 = self.generate_ellipse()
        warped1, warped2 = dtw_with_curvature_penalty(c1, c2, np.zeros(len(c1)), np.zeros(len(c2)),
                                                       dtw_curvature=10.0)
        assert warped1.shape == warped2.shape


class TestDetectLinearSegmentsFromCurvature:

    @staticmethod
    def square_contour(points_per_side=10):
        side = np.linspace(0, 1, points_per_side, endpoint=False)
        bottom = np.column_stack((side, np.zeros_like(side)))
        right = np.column_stack((np.ones_like(side), side))
        top = np.column_stack((1 - side, np.ones_like(side)))
        left = np.column_stack((np.zeros_like(side), 1 - side))
        return np.vstack((bottom, right, top, left))

    def test_straight_sided_polygon_is_detected(self):
        contour = self.square_contour(points_per_side=10)
        curvature = np.zeros(len(contour))
        curvature[[0, 10, 20, 30]] = 1.0  # mark the 4 corners as non-flat, splitting the loop into 4 sides
        mask, seg_info, seg_ids = detect_linear_segments_from_curvature(contour, curvature)
        assert mask.any()
        assert len(seg_info) == 4
        assert all(s["straight"] for s in seg_info)

    def test_circle_has_no_straight_segments(self):
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        contour = np.column_stack((np.cos(theta), np.sin(theta)))
        curvature = np.full(len(contour), 1.0)  # uniformly curved, well above flat_thresh
        mask, seg_info, _ = detect_linear_segments_from_curvature(contour, curvature)
        assert not mask.any()
        assert seg_info == []

    def test_min_frac_filters_out_short_flat_runs(self):
        contour = self.square_contour(points_per_side=10)
        curvature = np.zeros(len(contour))
        _, seg_info_lenient, _ = detect_linear_segments_from_curvature(contour, curvature, min_frac=0.01)
        _, seg_info_strict, _ = detect_linear_segments_from_curvature(contour, curvature, min_frac=0.9)
        assert len(seg_info_lenient) >= len(seg_info_strict)


class TestSegmentedContourDTW:

    @staticmethod
    def generate_circle(radius=1.0, n_points=100, center=(0, 0)):
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        return np.stack((center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)), axis=1)

    def test_identical_circles_align_to_themselves(self):
        contour = self.generate_circle(n_points=80)
        out1, out2 = segmented_contour_dtw(contour, contour)
        np.testing.assert_allclose(out1, out2, atol=1e-6)

    def test_output_is_2d_point_sequences(self):
        c1 = self.generate_circle(radius=1.0, n_points=80)
        c2 = self.generate_circle(radius=1.3, n_points=80, center=(0.2, -0.1))
        out1, out2 = segmented_contour_dtw(c1, c2)
        assert out1.ndim == 2 and out1.shape[1] == 2
        assert out2.ndim == 2 and out2.shape[1] == 2
        assert out1.shape[0] == out2.shape[0]

    def test_falls_back_to_plain_dtw_when_no_straight_segments(self):
        # Circles never trip the straightness threshold, so this must go through the num_segs < 2 fallback
        # to `dtw_with_curvature_penalty` rather than raising.
        c1 = self.generate_circle(n_points=50)
        c2 = self.generate_circle(radius=1.1, n_points=50)
        out1, out2 = segmented_contour_dtw(c1, c2)
        assert out1.shape[1] == 2 and out2.shape[1] == 2


class TestAlignStraightSegments:

    def test_matches_endpoints(self):
        seg1 = np.array([[0, 0], [1, 0], [2, 0]])
        seg2 = np.array([[0, 0], [2, 0]])
        out1, out2 = align_straight_segments(seg1, seg2)
        np.testing.assert_allclose(out1[0], seg1[0])
        np.testing.assert_allclose(out1[-1], seg1[-1])
        np.testing.assert_allclose(out2[0], seg2[0])
        np.testing.assert_allclose(out2[-1], seg2[-1])

    def test_output_length_matches_longer_segment(self):
        seg1 = np.array([[0, 0], [1, 0]])
        seg2 = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        out1, out2 = align_straight_segments(seg1, seg2)
        assert out1.shape[0] == out2.shape[0] == max(len(seg1), len(seg2))

    def test_degenerate_single_point_segment(self):
        seg1 = np.array([[1, 1], [1, 1], [1, 1]])
        seg2 = np.array([[0, 0], [1, 0], [2, 0]])
        out1, out2 = align_straight_segments(seg1, seg2)
        np.testing.assert_allclose(out1, np.ones((3, 2)))
