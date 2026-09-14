import os

import pytest
import numpy as np

from brainfusion.sample import Sample, replace
from brainfusion.fusion.core import fuse_boundaries, fuse_grids, fuse_measurement_datasets, brain_fusion, run_fusion


def circle_sample(cx, cy, r, n_contour=40, n_grid=8, filename="s", landmarks=None, value=1.0):
    theta = np.linspace(0, 2 * np.pi, n_contour, endpoint=False)
    contour = np.column_stack((cx + r * np.cos(theta), cy + r * np.sin(theta)))
    grid_theta = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    grid = np.column_stack((cx + (r / 2) * np.cos(grid_theta), cy + (r / 2) * np.sin(grid_theta)))
    dataset = {"value": np.full(n_grid, value)}
    return Sample(contour=contour, grid=grid, dataset=dataset, filename=filename, landmarks=landmarks)


def landmarks_for(cx, cy, r):
    """Three points on the circle of centre (cx, cy) and radius r, for landmark-based alignment tests."""
    return np.array([[cx + r, cy], [cx, cy + r], [cx - r, cy]])


class TestFuseBoundaries:

    def test_average_template_keeps_every_sample_as_measurement(self):
        # "average" synthesizes a *new* template rather than reusing one of the inputs, so unlike
        # "first_element" none of the original samples are excluded from measurement_samples.
        samples = [circle_sample(0, 0, 1, filename="a"), circle_sample(0.2, 0, 1.1, filename="b"),
                  circle_sample(-0.1, 0.2, 0.9, filename="c")]
        template_sample, measurement_samples, template_contours = fuse_boundaries("average", samples)
        assert len(measurement_samples) == len(samples)
        assert len(template_contours) == len(measurement_samples)
        assert template_sample.filename == "template_average"

    def test_first_element_template_uses_that_sample(self):
        samples = [circle_sample(0, 0, 1, filename="a"), circle_sample(0.2, 0, 1.1, filename="b")]
        template_sample, measurement_samples, _ = fuse_boundaries("first_element", samples)
        assert template_sample.filename == "a"
        assert len(measurement_samples) == 1
        assert measurement_samples[0].filename == "b"

    def test_invalid_template_choice_raises(self):
        samples = [circle_sample(0, 0, 1), circle_sample(0.2, 0, 1)]
        with pytest.raises(ValueError, match="is not implemented"):
            fuse_boundaries("magic", samples)

    def test_landmarks_with_average_template_does_not_crash(self):
        # Regression test: landmark-based alignment used to leave contours off-origin, which crashed
        # find_average_contour's star-domain averaging (the default `outline_averaging`).
        samples = [circle_sample(5000, 3000, 1, landmarks=landmarks_for(5000, 3000, 1), filename="a"),
                  circle_sample(5000.2, 3000, 1.1, landmarks=landmarks_for(5000.2, 3000, 1.1), filename="b")]
        template_sample, measurement_samples, _ = fuse_boundaries("average", samples, outline_averaging="star_domain")
        assert template_sample.contour.shape[1] == 2


class TestFuseGrids:

    def setup_method(self):
        samples = [circle_sample(0, 0, 1, filename="a"), circle_sample(0.2, 0, 1.1, filename="b")]
        self.template_sample, self.measurement_samples, self.template_contours = fuse_boundaries(
            "first_element", samples)

    def test_with_ext_grid_resamples_each_sample_onto_it(self):
        ext_grid = np.array([[0.0, 0.0], [0.1, 0.1], [-0.1, -0.1]])
        trafo_data_maps, trafo_grids, _, _, _ = fuse_grids(self.measurement_samples, self.template_contours,
                                                           ext_grid=ext_grid)
        assert len(trafo_data_maps) == len(self.measurement_samples)
        for trafo_data in trafo_data_maps:
            assert set(trafo_data.keys()) == {"value"}
            assert trafo_data["value"].shape == (len(ext_grid),)

    def test_without_ext_grid_skips_resampling(self):
        trafo_data_maps, trafo_grids, _, _, _ = fuse_grids(self.measurement_samples, self.template_contours,
                                                           ext_grid=None)
        assert trafo_data_maps == [None] * len(self.measurement_samples)
        assert all(grid is not None for grid in trafo_grids)


class TestFuseMeasurementDatasets:

    def setup_method(self):
        samples = [circle_sample(0, 0, 1, filename="a"), circle_sample(0.2, 0, 1.1, filename="b", value=3.0)]
        self.template_sample, self.measurement_samples, self.template_contours = fuse_boundaries(
            "first_element", samples)
        self.ext_grid = np.array([[0.0, 0.0], [0.1, 0.1]])
        self.trafo_data_maps, self.trafo_grids, _, _, _ = fuse_grids(
            self.measurement_samples, self.template_contours, ext_grid=self.ext_grid)

    @pytest.mark.parametrize("clustering,expected", [("Mean", 3.0), ("Median", 3.0), ("Sum", 3.0)])
    def test_elementwise_reductions_on_single_measurement_sample(self, clustering, expected):
        # Only one measurement sample (template excluded), so mean/median/sum must all just return its value
        avg_data, avg_grid = fuse_measurement_datasets(self.measurement_samples, self.trafo_data_maps,
                                                        self.trafo_grids, self.ext_grid, clustering=clustering)
        np.testing.assert_allclose(avg_data["value"], expected)
        assert avg_grid is self.ext_grid

    def test_gmm_ignores_ext_grid_and_builds_its_own(self):
        # Regression test: "GMM" used to look up a '_trafo'-suffixed key that never existed, raising KeyError.
        avg_data, avg_grid = fuse_measurement_datasets(self.measurement_samples, self.trafo_data_maps,
                                                        self.trafo_grids, self.ext_grid, clustering="GMM")
        assert avg_grid is not self.ext_grid
        assert avg_grid.shape[0] == avg_data["value"].shape[0]

    def test_invalid_clustering_raises(self):
        with pytest.raises(ValueError, match="is not implemented"):
            fuse_measurement_datasets(self.measurement_samples, self.trafo_data_maps, self.trafo_grids,
                                      self.ext_grid, clustering="magic")


class TestBrainFusion:

    def test_basic_output_structure(self):
        # Default contour_template="average" synthesizes a new template, so all 3 inputs remain measurements.
        samples = [circle_sample(0, 0, 1, filename="a"), circle_sample(0.2, 0, 1.1, filename="b"),
                  circle_sample(-0.1, 0.2, 0.9, filename="c")]
        result = brain_fusion(samples, contour_interp_n=40)
        assert result["measurement_interpolated_dataset"]["value"].shape == \
              (result["measurement_interpolated_grid"].shape[0],)
        assert len(result["measurement_filenames"]) == 3
        assert result["measurement_interpolated_grid_shape"] is not None

    def test_gmm_clustering_skips_regular_grid(self):
        samples = [circle_sample(0, 0, 1, filename="a"), circle_sample(0.2, 0, 1.1, filename="b")]
        result = brain_fusion(samples, clustering="GMM", contour_interp_n=40)
        assert result["measurement_interpolated_grid_shape"] is None
        assert result["measurement_trafo_datasets"] == [None, None]

    def test_landmarks_with_average_template_end_to_end(self):
        # Regression test for the star-domain/off-origin crash, exercised through the full public entry point.
        samples = [circle_sample(5000, 3000, 1, landmarks=landmarks_for(5000, 3000, 1), filename="a"),
                  circle_sample(5000.2, 3000, 1.1, landmarks=landmarks_for(5000.2, 3000, 1.1), filename="b")]
        result = brain_fusion(samples, contour_template="average", outline_averaging="star_domain",
                              contour_interp_n=40)
        assert "measurement_interpolated_dataset" in result

    def test_group_field_caches_group_datasets(self):
        samples = [
            replace(circle_sample(0, 0, 1, value=1.0, filename="a1"), metadata={"condition": "A"}),
            replace(circle_sample(0.05, 0, 1.02, value=1.0, filename="a2"), metadata={"condition": "A"}),
            replace(circle_sample(-0.03, 0.02, 0.98, value=2.0, filename="b1"), metadata={"condition": "B"}),
        ]
        result = brain_fusion(samples, contour_template="average", contour_interp_n=40, clustering="Mean",
                              group_field="condition")
        assert result["group_field"] == "condition"
        assert set(result["group_datasets"].keys()) == {"A", "B"}
        np.testing.assert_allclose(result["group_datasets"]["A"]["value"], 1.0)
        np.testing.assert_allclose(result["group_datasets"]["B"]["value"], 2.0)

    def test_no_group_field_omits_group_keys(self):
        samples = [circle_sample(0, 0, 1, filename="a"), circle_sample(0.2, 0, 1.1, filename="b")]
        result = brain_fusion(samples, contour_interp_n=40)
        assert "group_field" not in result
        assert "group_datasets" not in result


class TestRunFusion:

    def test_computes_and_caches(self, tmp_path):
        samples = [circle_sample(0, 0, 1, filename="a"), circle_sample(0.2, 0, 1.1, filename="b")]
        results_path = str(tmp_path / "analysis.h5")
        result = run_fusion(samples, {"contour_interp_n": 40}, results_path=results_path)
        assert os.path.exists(results_path)
        np.testing.assert_allclose(result["measurement_interpolated_dataset"]["value"], 1.0)

    def test_reuses_cached_result_instead_of_recomputing(self, tmp_path):
        results_path = str(tmp_path / "analysis.h5")
        samples_v1 = [circle_sample(0, 0, 1, filename="a", value=1.0), circle_sample(0.2, 0, 1.1, filename="b",
                                                                                     value=1.0)]
        run_fusion(samples_v1, {"contour_interp_n": 40}, results_path=results_path)

        # Different data, but overwrite=False - the cached (first) result must be returned unchanged
        samples_v2 = [circle_sample(0, 0, 1, filename="a", value=99.0), circle_sample(0.2, 0, 1.1, filename="b",
                                                                                      value=99.0)]
        cached = run_fusion(samples_v2, {"contour_interp_n": 40}, results_path=results_path)
        np.testing.assert_allclose(cached["measurement_interpolated_dataset"]["value"], 1.0)

    def test_overwrite_forces_recompute(self, tmp_path):
        results_path = str(tmp_path / "analysis.h5")
        samples_v1 = [circle_sample(0, 0, 1, filename="a", value=1.0), circle_sample(0.2, 0, 1.1, filename="b",
                                                                                     value=1.0)]
        run_fusion(samples_v1, {"contour_interp_n": 40}, results_path=results_path)

        samples_v2 = [circle_sample(0, 0, 1, filename="a", value=99.0), circle_sample(0.2, 0, 1.1, filename="b",
                                                                                      value=99.0)]
        recomputed = run_fusion(samples_v2, {"contour_interp_n": 40}, results_path=results_path, overwrite=True)
        np.testing.assert_allclose(recomputed["measurement_interpolated_dataset"]["value"], 99.0)
