import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt

from brainfusion.plotting.correlation import (format_p_value, plot_norm_corr, plot_correlation_with_radii,
                                              plot_correlation_masks)


class TestFormatPValue:

    def test_very_small_p_value_is_capped(self):
        assert format_p_value(0.00005) == "p<0.001"

    def test_gap_between_old_thresholds_is_covered(self):
        # Regression test: 0.0001 <= p < 0.001 used to fall through every branch and return
        # "Invalid p-value" on the plotted label.
        assert format_p_value(0.0005) == "p<0.001"

    def test_boundary_at_one_thousandth_uses_three_decimals(self):
        assert format_p_value(0.001) == "p=0.001"

    def test_mid_range_uses_three_decimals(self):
        assert format_p_value(0.05) == "p=0.050"

    def test_boundary_at_twenty_percent_uses_two_decimals(self):
        assert format_p_value(0.20) == "p=0.20"

    def test_large_p_value_uses_two_decimals(self):
        assert format_p_value(0.83) == "p=0.83"


class TestPlotNormCorr:

    def test_returns_figure_with_stats_annotation(self):
        map1 = np.linspace(0, 1, 10)
        map2 = 2 * map1
        fig = plot_norm_corr(map1, map2, pearson=1.0, p_value=0.001)
        assert fig is not None
        plt.close(fig)

    def test_works_without_stats(self):
        fig = plot_norm_corr([1, 2, 3], [1, 2, 3])
        assert fig is not None
        plt.close(fig)

    def test_raises_on_shape_mismatch(self):
        with pytest.raises(ValueError):
            plot_norm_corr([1, 2, 3], [1, 2])

    def test_saves_and_closes_figure_when_output_path_given(self, tmp_path):
        output_path = tmp_path / "corr.png"
        fig = plot_norm_corr([1, 2, 3], [1, 2, 3], output_path=str(output_path))
        assert output_path.exists()
        assert not plt.fignum_exists(fig.number)


class TestPlotCorrelationWithRadii:

    def test_returns_figure(self):
        contour = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
        reference_grid = np.array([[0.0, 0.0], [0.5, 0.0]])
        other_grid = np.array([[0.1, 0.1], [0.6, 0.1], [-0.5, -0.5]])
        radii = np.array([0.3, 0.3])

        fig = plot_correlation_with_radii(reference_grid, other_grid, contour, radii, data_a=[1.0, 2.0],
                                          data_b=[3.0, 4.0, 5.0])
        assert fig is not None
        plt.close(fig)

    def test_saves_to_results_folder(self, tmp_path):
        contour = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
        reference_grid = np.array([[0.0, 0.0]])
        other_grid = np.array([[0.1, 0.1]])
        radii = np.array([0.3])

        plot_correlation_with_radii(reference_grid, other_grid, contour, radii, data_a=[1.0], data_b=[2.0],
                                    results_folder=str(tmp_path), results_name="test_radii")
        assert (tmp_path / "test_radii.png").exists()


class TestPlotCorrelationMasks:

    def test_numeric_data_both_sides(self, tmp_path):
        grid = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
        contour = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
        data_a = np.array([1.0, 2.0, 3.0])
        data_b = np.array([4.0, 5.0, 6.0])

        plot_correlation_masks(grid, data_a, data_b, contour, results_folder=str(tmp_path))
        assert (tmp_path / "correlation_masks.png").exists()

    def test_boolean_data_both_sides(self):
        grid = np.array([[0.0, 0.0], [0.5, 0.0]])
        contour = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
        data_a = np.array([True, False])
        data_b = np.array([False, True])

        fig = plot_correlation_masks(grid, data_a, data_b, contour)
        assert fig is not None
        plt.close(fig)
