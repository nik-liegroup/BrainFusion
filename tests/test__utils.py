import pytest
import numpy as np
from brainfusion._utils import mask_contour, regular_grid_on_bbox, bin_2D_image, bin_outline


class TestMaskContour:

    def test_mask_contour_simple_square(self):
        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        points = np.array([
            [0.5, 0.5],  # inside
            [1.5, 0.5],  # outside
            [0, 0],  # on vertex
            [1, 1],  # on vertex
            [0.5, 1],  # on edge
        ])
        mask = mask_contour(contour, points)
        expected = np.array([True, False, True, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_mask_contour_invalid_input(self):
        contour = np.array([[0, 0, 0], [1, 0, 0]])  # wrong shape
        points = np.array([[0, 0]])
        with pytest.raises(ValueError):
            mask_contour(contour, points)


class TestRegularGridOnBbox:

    def test_simple_square_bbox(self):
        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        grid = regular_grid_on_bbox(contour, axis_points=5)

        # Grid should have shape (25, 2)
        assert grid.shape == (25, 2)

        # Grid should cover the bounding box corners
        x_values = grid[:, 0]
        y_values = grid[:, 1]
        assert np.isclose(x_values.min(), 0)
        assert np.isclose(x_values.max(), 1)
        assert np.isclose(y_values.min(), 0)
        assert np.isclose(y_values.max(), 1)

    def test_invalid_input_shape(self):
        contour = np.array([[0, 0, 0], [1, 0, 0]])  # invalid shape
        with pytest.raises(ValueError):
            regular_grid_on_bbox(contour)


class TestBin2DImage:

    def test_mean_binning_crop(self):
        img = np.arange(16).reshape(4, 4)
        binned = bin_2D_image(img, bin_size=2, method='mean', crop=True)
        expected = np.array([[2.5, 4.5],
                             [10.5, 12.5]])
        np.testing.assert_array_almost_equal(binned, expected)

    def test_sum_binning_crop(self):
        img = np.ones((4, 4))
        binned = bin_2D_image(img, bin_size=2, method='sum', crop=True)
        expected = np.full((2, 2), 4)
        np.testing.assert_array_equal(binned, expected)

    def test_max_binning_crop(self):
        img = np.arange(16).reshape(4, 4)
        binned = bin_2D_image(img, bin_size=2, method='max', crop=True)
        expected = np.array([[5, 7],
                             [13, 15]])
        np.testing.assert_array_equal(binned, expected)

    def test_padding_behavior(self):
        img = np.arange(15).reshape(3, 5)
        binned = bin_2D_image(img, bin_size=2, method='sum', crop=False)
        # Image will be padded to (4,6) → binned to (2,3)
        expected_shape = (2, 3)
        assert binned.shape == expected_shape

    def test_invalid_method(self):
        img = np.ones((4, 4))
        with pytest.raises(ValueError):
            bin_2D_image(img, bin_size=2, method='magic', crop=True)

    def test_invalid_bin_size(self):
        img = np.ones((4, 4))
        with pytest.raises(ValueError):
            bin_2D_image(img, bin_size=0, method='mean', crop=True)

    def test_invalid_input_shape(self):
        img = np.ones((4, 4, 3))  # 3D array, invalid
        with pytest.raises(ValueError):
            bin_2D_image(img, bin_size=2, method='mean', crop=True)


class TestBinOutline:

    def test_basic_scaling(self):
        outline = np.array([[0, 0], [10, 10]])
        bin_size = 2
        transformed = bin_outline(outline, bin_size)
        expected = (outline - 0.5) / bin_size + 0.5
        np.testing.assert_allclose(transformed, expected)

    def test_cropping_behavior(self):
        outline = np.array([[0, 0], [50, 50], [99, 99]])
        bin_size = 10
        original_shape = (100, 80)  # Height=100, Width=80
        # After cropping, new height = 100 - (100 % 10) = 100, new width = 80 - (80 % 10) = 80
        # The point (99,99) should be cropped out
        transformed = bin_outline(outline, bin_size, crop=True, original_shape=original_shape)
        # Only the first two points should remain
        expected_outline = outline[:2]
        expected = (expected_outline - 0.5) / bin_size + 0.5
        np.testing.assert_allclose(transformed, expected)
        assert transformed.shape[0] == 2

    def test_invalid_outline_shape(self):
        with pytest.raises(ValueError, match=r"outline must be an \(n, 2\) array of coordinates"):
            bin_outline(np.array([1, 2, 3]), bin_size=5)

    def test_negative_bin_size(self):
        outline = np.array([[0, 0], [10, 10]])
        with pytest.raises(ValueError, match="bin_size must be a positive integer"):
            bin_outline(outline, bin_size=-3)

    def test_crop_without_original_shape(self):
        outline = np.array([[0, 0], [10, 10]])
        with pytest.raises(ValueError, match="original_shape must be provided when crop=True"):
            bin_outline(outline, bin_size=5, crop=True, original_shape=None)
