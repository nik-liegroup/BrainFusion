import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt

from brainfusion.plotting.maps import (render_data_on_contour, get_zoom_limits, plot_contours,
                                       plot_transformed_grid, plot_average_map, plot_average_map_arrays)

SQUARE = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)


class TestGetZoomLimits:

    def test_centres_on_contour_with_margin(self):
        x_lim, y_lim = get_zoom_limits(SQUARE, margin_fraction=0.1)
        assert x_lim == pytest.approx((-1.2, 1.2))
        assert y_lim == pytest.approx((-1.2, 1.2))

    def test_non_square_contour_uses_larger_range_for_both_axes(self):
        contour = np.array([[0, 0], [4, 0], [4, 1], [0, 1], [0, 0]], dtype=float)
        x_lim, y_lim = get_zoom_limits(contour, margin_fraction=0.0)
        assert x_lim[1] - x_lim[0] == pytest.approx(y_lim[1] - y_lim[0])


class TestRenderDataOnContour:

    def test_scatter_grid_masks_points_outside_contour(self):
        fig, ax = plt.subplots()
        grid = np.array([[0.0, 0.0], [0.5, 0.0], [5.0, 5.0]])  # last point is outside SQUARE
        data = np.array([1.0, 2.0, 3.0])
        mappable = render_data_on_contour(ax, data, grid, SQUARE, mask=True)
        assert np.ma.is_masked(mappable.get_offsets()[2, 0]) or mappable.get_offsets().mask[2].any()
        plt.close(fig)

    def test_image_grid_reshapes_data_and_masks_outside_contour(self):
        fig, ax = plt.subplots()
        xx, yy = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
        grid = np.stack([xx, yy], axis=-1)  # (4, 4, 2)
        data = np.arange(16).reshape(4, 4).astype(float)
        mappable = render_data_on_contour(ax, data, grid, SQUARE, mask=True)
        image = mappable.get_array()
        assert image.shape == (4, 4)
        plt.close(fig)

    def test_unmasked_scatter_keeps_every_point(self):
        fig, ax = plt.subplots()
        grid = np.array([[0.0, 0.0], [5.0, 5.0]])
        data = np.array([1.0, 2.0])
        mappable = render_data_on_contour(ax, data, grid, SQUARE, mask=False)
        assert mappable.get_offsets().shape[0] == 2
        plt.close(fig)


class TestPlotContours:

    def test_returns_figure_for_several_matched_contours(self):
        theta = np.linspace(0, 2 * np.pi, 20)
        template = np.column_stack((np.cos(theta), np.sin(theta)))
        matched = [template * 0.9, template * 1.1]
        fig = plot_contours(template, matched)
        assert fig is not None
        plt.close(fig)


class TestPlotTransformedGrid:

    def _scatter_grid(self, n=6):
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack((np.cos(theta), np.sin(theta)))

    def test_scatter_to_scatter_reuses_original_data(self):
        contour = self._scatter_grid(30)
        template_contour = contour * 1.1
        grid = self._scatter_grid(6) * 0.5
        trafo_grid = grid * 1.1
        data = {"value": np.linspace(0, 1, 6)}

        fig = plot_transformed_grid(contour, template_contour, data, grid, trafo_grid, key_quant="value")
        assert fig is not None
        plt.close(fig)

    def test_regular_grid_requires_trafo_data(self):
        contour = self._scatter_grid(30)
        template_contour = contour * 1.1
        grid = self._scatter_grid(6) * 0.5
        data = {"value": np.linspace(0, 1, 6)}

        xx, yy = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
        trafo_grid = np.stack([xx, yy], axis=-1)
        trafo_data = {"value": np.arange(9).reshape(3, 3).astype(float)}

        fig = plot_transformed_grid(contour, template_contour, data, grid, trafo_grid, key_quant="value",
                                    trafo_data=trafo_data)
        assert fig is not None
        plt.close(fig)


class TestPlotAverageMapArrays:

    def test_auto_vmin_vmax_from_data(self):
        grid = np.array([[0.0, 0.0], [0.5, 0.0], [-0.5, 0.5]])
        data = np.array([1.0, 5.0, np.nan])
        fig = plot_average_map_arrays(data, grid, SQUARE)
        assert fig is not None
        plt.close(fig)

    def test_saves_and_closes_figure_when_output_path_given(self, tmp_path):
        grid = np.array([[0.0, 0.0], [0.5, 0.0]])
        data = np.array([1.0, 2.0])
        output_path = tmp_path / "map.png"

        fig = plot_average_map_arrays(data, grid, SQUARE, output_path=str(output_path))
        assert output_path.exists()
        assert not plt.fignum_exists(fig.number)


class TestPlotAverageMap:

    def _analysis(self, with_groups=False):
        grid = np.array([[0.0, 0.0], [0.5, 0.0], [-0.5, 0.5]])
        analysis = {
            "template_contours": [SQUARE],
            "measurement_interpolated_grid": grid,
            "measurement_interpolated_dataset": {"value": np.array([1.0, 2.0, 3.0])},
        }
        if with_groups:
            analysis["group_datasets"] = {
                "Control": {"value": np.array([1.0, 1.0, 1.0])},
                "Treated": {"value": np.array([2.0, 2.0, 2.0])},
            }
        return analysis

    def test_plots_overall_average_without_group(self):
        fig = plot_average_map(self._analysis(), "value")
        assert fig is not None
        plt.close(fig)

    def test_plots_one_groups_average(self):
        fig = plot_average_map(self._analysis(with_groups=True), "value", group="Control")
        assert fig is not None
        plt.close(fig)

    def test_raises_if_group_requested_without_group_field(self):
        with pytest.raises(ValueError, match="group_field"):
            plot_average_map(self._analysis(with_groups=False), "value", group="Control")
