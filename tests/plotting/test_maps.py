import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from brainfusion.plotting.map_style import CANVAS_MM
from brainfusion.plotting.maps import (render_data_on_contour, get_zoom_limits, plot_contours,
                                       plot_transformed_grid, plot_average_map, plot_average_map_arrays, scalebar_length,
                                       plot_aligned_outlines, fit_markers_to_spacing)

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

    def test_regular_grid_is_drawn_as_touching_squares(self):
        fig, ax = plt.subplots()
        xx, yy = np.meshgrid(np.arange(-0.4, 0.5, 0.2), np.arange(-0.4, 0.5, 0.2))
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        mappable = render_data_on_contour(ax, np.arange(len(grid), dtype=float), grid, SQUARE, mask=True)
        rectangles = mappable.get_paths()
        assert len(rectangles) == len(grid)
        widths = {round(np.ptp(path.vertices[:, 0]), 6) for path in rectangles}
        assert widths == {0.2}  # one grid pitch: neighbours touch
        plt.close(fig)

    def test_grid_with_different_pitch_in_x_and_y_gets_touching_cells(self):
        fig, ax = plt.subplots()
        xx, yy = np.meshgrid(np.arange(-0.4, 0.5, 0.2), np.arange(-0.45, 0.5, 0.15))
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        mappable = render_data_on_contour(ax, np.arange(len(grid), dtype=float), grid, SQUARE, mask=True)
        sizes = {(round(np.ptp(p.vertices[:, 0]), 6), round(np.ptp(p.vertices[:, 1]), 6)) for p in mappable.get_paths()}
        assert sizes == {(0.2, 0.15)}
        plt.close(fig)

    def test_irregular_points_stay_a_scatter(self):
        fig, ax = plt.subplots()
        rng = np.random.default_rng(0)
        grid = rng.uniform(-0.9, 0.9, size=(30, 2))
        mappable = render_data_on_contour(ax, np.arange(30, dtype=float), grid, SQUARE, mask=True)
        assert mappable.get_offsets().shape[0] == 30
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


class TestAutoMarkerSize:

    def test_markers_are_as_wide_as_the_point_spacing(self):
        rng = np.random.default_rng(0)
        grid = np.column_stack([x.ravel() for x in np.meshgrid(np.arange(10) * 1.0, np.arange(10) * 1.0)])
        grid = grid + rng.normal(0, 0.05, grid.shape)   # slightly warped: not a regular grid any more
        fig, ax = plt.subplots(figsize=(4, 4))
        render_data_on_contour(ax, np.arange(100.0), grid, SQUARE * 20, marker_size='auto')
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 10)
        ax.set_aspect('equal')
        fit_markers_to_spacing(ax)
        side_points = np.sqrt(ax.collections[0].get_sizes()[0])
        axes_width_points = ax.get_window_extent().width * 72 / fig.dpi
        assert side_points == pytest.approx(axes_width_points / 11, rel=0.1)   # one unit of 11 across the axes
        plt.close(fig)

    def test_unknown_marker_size_is_an_error(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError):
            render_data_on_contour(ax, np.arange(5.0), np.random.default_rng(1).random((5, 2)), SQUARE, marker_size='big')
        plt.close(fig)


class TestPlotAlignedOutlines:

    def test_saves_the_overlay_of_the_aligned_outlines(self, tmp_path):
        theta = np.linspace(0, 2 * np.pi, 20)
        template = np.column_stack((np.cos(theta), np.sin(theta)))
        analysis = {"template_contours": [template], "measurement_contours": [template * 0.9, template * 1.1]}
        plot_aligned_outlines(analysis, output_path=str(tmp_path / "outlines.png"))
        assert (tmp_path / "outlines.png").stat().st_size > 0


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

    def test_scalebar_in_the_units_of_the_first_sample(self):
        analysis = self._analysis()
        analysis["measurement_metadata"] = [{"length_unit": "m"}]
        assert scalebar_length(analysis, 100) == pytest.approx(1e-4)
        analysis["measurement_metadata"] = [{"length_unit": "um"}]
        assert scalebar_length(analysis, 100) == pytest.approx(100)
        fig = plot_average_map(analysis, "value", scalebar_um=0.5)
        bars = [a for a in fig.artists if isinstance(a, Rectangle)]   # the scale bar is a figure-level rectangle in figure fractions
        assert len(bars) == 1
        assert bars[0].get_width() * CANVAS_MM[0] == pytest.approx(0.5 * fig._mm_per_unit)
        assert not fig.axes[0].texts   # no label
        plt.close(fig)

    def test_scalebar_needs_a_length_unit(self):
        with pytest.raises(ValueError, match="length unit"):
            scalebar_length(self._analysis(), 100)

    def test_raises_if_group_requested_without_group_field(self):
        with pytest.raises(ValueError, match="group_field"):
            plot_average_map(self._analysis(with_groups=False), "value", group="Control")


class TestRegularGridPitchesUnits:

    @pytest.mark.parametrize("unit", [1.0, 1e-6])
    def test_grid_is_regular_in_metres_and_micrometres(self, unit):
        from brainfusion.plotting.maps import _regular_grid_pitches
        x, y = np.meshgrid(np.arange(20) * 20.89 * unit, np.arange(25) * 20.85 * unit)
        pitches = _regular_grid_pitches(np.column_stack([x.ravel(), y.ravel()]))
        assert pitches == pytest.approx((20.89 * unit, 20.85 * unit))
