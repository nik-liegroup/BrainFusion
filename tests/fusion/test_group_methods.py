import numpy as np
import pytest

from brainfusion.sample import Sample
from brainfusion.fusion.core import brain_fusion, average_statistic
from brainfusion.fusion.grouping import extract_groups, group_average_entry
from brainfusion.fusion.interpolation import resample_points
from brainfusion.correlation import correlate_group_maps


def disc_sample(cx, cy, value, filename, group, n_per_side=7, radius=1.0, rng=None):
    """A square grid of points inside a circular outline, value = `value` + x (a gradient along x)."""
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    contour = np.column_stack((cx + radius * np.cos(theta), cy + radius * np.sin(theta)))
    axis = np.linspace(-0.7, 0.7, n_per_side)
    grid = np.array([[cx + x, cy + y] for x in axis for y in axis if x * x + y * y < 0.6])
    noise = 0 if rng is None else rng.normal(0, 0.01, len(grid))
    return Sample(contour=contour, grid=grid, dataset={"value": value + (grid[:, 0] - cx) + noise}, filename=filename,
                  metadata={"group": group})


@pytest.fixture
def samples():
    rng = np.random.default_rng(0)
    out = []
    for i in range(4):
        out.append(disc_sample(0.02 * i, 0, 1.0, f"a{i}", "A", n_per_side=7, rng=rng))
        out.append(disc_sample(0, 0.02 * i, 5.0, f"b{i}", "B", n_per_side=9, rng=rng))
    return out


class TestGroupsForEveryMethod:

    def test_every_method_of_the_list_gets_its_own_group_maps(self, samples):
        analysis = brain_fusion(samples, contour_interp_n=40, clustering=["Median", "GMM"], group_field="group")
        assert set(analysis["group_averaged"]) == {"Median", "GMM"}
        assert extract_groups(analysis) == ["A", "B"]
        for method in ("Median", "GMM"):
            for group in ("A", "B"):
                entry = group_average_entry(analysis, group, method)
                assert set(entry) >= {"grid", "dataset", "n", "spread", "standard_error"}

    def test_gmm_groups_share_the_components_of_one_clustering_of_all_samples(self, samples):
        analysis = brain_fusion(samples, contour_interp_n=40, clustering="GMM", group_field="group")
        pooled = analysis["averaged"]["GMM"]
        for group in ("A", "B"):
            entry = group_average_entry(analysis, group)
            np.testing.assert_array_equal(entry["grid"], pooled["grid"])
            assert np.nanmax(entry["n"]["value"]) <= sum(1 for m in analysis["measurement_metadata"] if m["group"] == group)
        assert "group_datasets" not in analysis      # no regular shared grid, but one grid for every group

    def test_shared_grid_methods_keep_the_group_datasets_form(self, samples):
        analysis = brain_fusion(samples, contour_interp_n=40, clustering=["GMM", "Median"], group_field="group")
        assert set(analysis["group_datasets"]) == {"A", "B"}
        np.testing.assert_array_equal(analysis["group_datasets"]["A"]["value"],
                                      analysis["group_averaged"]["Median"]["A"]["dataset"]["value"])

    def test_average_statistic_of_a_group(self, samples):
        analysis = brain_fusion(samples, contour_interp_n=40, clustering="Median", group_field="group")
        _, n = average_statistic(analysis, "value", statistic="n", group="A")
        assert np.nanmax(n) == 4          # four animals in group A

    def test_unknown_group_is_an_error(self, samples):
        analysis = brain_fusion(samples, contour_interp_n=40, clustering="Median", group_field="group")
        with pytest.raises(ValueError, match="No group"):
            group_average_entry(analysis, "C")


class TestResamplePoints:

    def setup_method(self):
        self.source = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        self.values = np.array([0.0, 10.0, 20.0])

    def test_gaussian_is_a_weighted_mean_between_the_points(self):
        out = resample_points(self.source, self.values, np.array([[0.5, 0.0]]), method="gaussian", sigma=0.3)
        assert out[0] == pytest.approx(5.0)       # halfway between 0 and 10: equal weights (the third point is beyond 3 sigma)

    def test_gaussian_is_nan_far_from_everything(self):
        out = resample_points(self.source, self.values, np.array([[10.0, 0.0]]), method="gaussian", sigma=0.5)
        assert np.isnan(out[0])

    def test_gaussian_ignores_nan_source_values(self):
        values = np.array([0.0, np.nan, 20.0])
        out = resample_points(self.source, values, np.array([[1.0, 0.0]]), method="gaussian", sigma=0.8)
        assert out[0] == pytest.approx(10.0)

    def test_gaussian_default_sigma_is_half_the_spacing(self):
        auto = resample_points(self.source, self.values, np.array([[0.5, 0.0]]))
        explicit = resample_points(self.source, self.values, np.array([[0.5, 0.0]]), sigma=0.5)
        assert auto[0] == pytest.approx(explicit[0])

    def test_radius_and_nearest(self):
        targets = np.array([[0.1, 0.0], [5.0, 0.0]])
        radius = resample_points(self.source, self.values, targets, method="radius", radius=0.5)
        assert radius[0] == pytest.approx(0.0) and np.isnan(radius[1])
        nearest = resample_points(self.source, self.values, targets, method="nearest", radius=1.0)
        assert nearest[0] == pytest.approx(0.0) and np.isnan(nearest[1])

    def test_unknown_method_is_an_error(self):
        with pytest.raises(ValueError):
            resample_points(self.source, self.values, self.source, method="magic")


class TestCorrelateGroupMaps:

    def test_shared_grid_groups_are_correlated_directly(self, samples):
        analysis = brain_fusion(samples, contour_interp_n=40, clustering="Median", group_field="group")
        result = correlate_group_maps(analysis, "value", "A", "B")
        assert result["pearson_correlation"] == pytest.approx(1.0, abs=0.02)   # both are a gradient along x
        assert result["resample"].startswith("none")

    def test_gmm_groups_of_one_fusion_are_correlated_directly(self, samples):
        analysis = brain_fusion(samples, contour_interp_n=40, clustering="GMM", group_field="group")
        result = correlate_group_maps(analysis, "value", "A", "B")
        assert result["resample"] == "none: shared grid"
        assert result["pearson_correlation"] > 0.9

    @staticmethod
    def separate_fusions(samples):
        """Two fusions of the groups on their own (different GMM grids), put into one analysis for correlate_group_maps."""
        parts = {g: brain_fusion([s for s in samples if s.metadata["group"] == g], contour_interp_n=40, clustering="GMM")
                 for g in ("A", "B")}
        return {"template_contours": parts["A"]["template_contours"], "averaged_methods": ["GMM"],
                "group_averaged": {"GMM": {g: parts[g]["averaged"]["GMM"] for g in parts}}}

    def test_maps_of_separate_fusions_are_resampled_with_a_kernel(self, samples):
        result = correlate_group_maps(self.separate_fusions(samples), "value", "A", "B", resample="gaussian")
        assert result["pearson_correlation"] > 0.9
        assert "gaussian" in result["resample"]
        assert len(result["grid"]) == result["n_points"] == len(result["A"]) == len(result["B"])

    def test_the_coarser_map_is_the_target(self, samples):
        analysis = self.separate_fusions(samples)
        grid_a = group_average_entry(analysis, "A")["grid"]
        grid_b = group_average_entry(analysis, "B")["grid"]
        result = correlate_group_maps(analysis, "value", "A", "B")
        coarser = "A" if len(grid_a) < len(grid_b) else "B"      # fewer components = larger spacing
        assert coarser in result["resample"]
