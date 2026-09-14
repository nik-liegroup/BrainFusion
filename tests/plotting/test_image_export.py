import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import tifffile
import matplotlib.pyplot as plt

from brainfusion.plotting.image_export import render_contour_on_image, save_average_map_tif, plot_map_on_image

SQUARE = np.array([[-1, -1], [4, -1], [4, 4], [-1, 4], [-1, -1]], dtype=float)  # encloses the grids below


def regular_grid(n_rows, n_cols, x_max, y_max):
    xx, yy = np.meshgrid(np.linspace(0, x_max, n_cols), np.linspace(0, y_max, n_rows))
    return np.stack([xx, yy], axis=-1)


class TestRenderContourOnImage:

    def test_masks_points_outside_contour_to_nan(self):
        grid = regular_grid(4, 4, 3, 3)
        img = np.ones((4, 4))
        outside_contour = np.array([[-10, -10], [-9, -10], [-9, -9], [-10, -9], [-10, -10]], dtype=float)

        masked, fig, _ = render_contour_on_image(img, grid, outside_contour, mask=True)
        assert np.isnan(masked).all()
        plt.close(fig)

    def test_computes_pixel_size_from_grid_extent(self):
        grid = regular_grid(4, 4, 3, 3)
        img = np.ones((4, 4))

        _, fig, (pixel_size_x, pixel_size_y) = render_contour_on_image(img, grid, SQUARE, mask=False)
        assert pixel_size_x == pytest.approx(3 / 4)
        assert pixel_size_y == pytest.approx(3 / 4)
        plt.close(fig)


class TestSaveAverageMapTif:

    def test_writes_file_with_resolution_matching_pixel_size(self, tmp_path):
        n_rows, n_cols, extent = 4, 4, 3.0
        grid = regular_grid(n_rows, n_cols, extent, extent)
        value_matrix = np.arange(n_rows * n_cols).reshape(n_rows, n_cols).astype(float)
        output_path = tmp_path / "map.tif"

        save_average_map_tif(value_matrix, grid, SQUARE, str(output_path))
        assert output_path.exists()

        with tifffile.TiffFile(str(output_path)) as tif:
            res_x, res_y = tif.pages[0].resolution
            saved = tif.asarray()

        pixel_size = extent / n_cols
        assert res_x == pytest.approx(1e-6 / pixel_size, rel=1e-3)
        assert saved.shape == (n_rows, n_cols)
        assert not np.isnan(saved).any()  # SQUARE encloses the whole grid


class TestPlotMapOnImage:

    def test_returns_figure(self):
        img = np.zeros((10, 10))
        grid = np.array([[1.0, 1.0], [5.0, 5.0]])
        contour = np.array([[0, 0], [9, 0], [9, 9], [0, 9], [0, 0]], dtype=float)

        fig = plot_map_on_image(img, data=[1.0, 2.0], grid=grid, contour=contour, scale=1)
        assert fig is not None
        plt.close(fig)
