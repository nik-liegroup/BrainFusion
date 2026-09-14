import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from brainfusion.sample import Sample
from brainfusion.fusion.core import brain_fusion
from brainfusion.plotting.results import plot_sample_warps, plot_verification_grids


def circle_sample(cx, cy, r, value, n_contour=40, n_grid=8, filename="s"):
    theta = np.linspace(0, 2 * np.pi, n_contour, endpoint=False)
    contour = np.column_stack((cx + r * np.cos(theta), cy + r * np.sin(theta)))
    grid_theta = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    grid = np.column_stack((cx + (r / 2) * np.cos(grid_theta), cy + (r / 2) * np.sin(grid_theta)))
    dataset = {"value": np.full(n_grid, value)}
    return Sample(contour=contour, grid=grid, dataset=dataset, filename=filename)


@pytest.fixture
def two_sample_analysis():
    samples = [circle_sample(0, 0, 1, value=1.0, filename="a"), circle_sample(0.1, 0, 1.05, value=2.0, filename="b")]
    return brain_fusion(samples, contour_template="average", contour_interp_n=40, clustering="Mean")


class TestPlotVerificationGrids:

    def test_writes_one_png_per_measurement_sample(self, tmp_path, two_sample_analysis):
        plot_verification_grids(two_sample_analysis, str(tmp_path))
        for filename in two_sample_analysis["measurement_filenames"]:
            assert (tmp_path / f"Verification_{filename}.png").exists()


class TestPlotSampleWarps:

    def test_writes_one_png_per_measurement_sample(self, tmp_path, two_sample_analysis):
        results_folder = tmp_path / "warps"
        plot_sample_warps(two_sample_analysis, str(results_folder), key_quant="value")
        for filename in two_sample_analysis["measurement_filenames"]:
            assert (results_folder / f"Transformed_{filename}.png").exists()

    def test_image_dataset_requires_regular_interpolation_grid(self, tmp_path):
        samples = [circle_sample(0, 0, 1, value=1.0, filename="a"), circle_sample(0.1, 0, 1.05, value=2.0,
                                                                                   filename="b")]
        gmm_analysis = brain_fusion(samples, contour_template="average", contour_interp_n=40, clustering="GMM")

        with pytest.raises(ValueError, match="image_dataset"):
            plot_sample_warps(gmm_analysis, str(tmp_path), key_quant="value", image_dataset=True)
