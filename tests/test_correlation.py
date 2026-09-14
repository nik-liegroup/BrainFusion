import numpy as np
import pytest

from brainfusion.sample import Sample
from brainfusion.fusion.core import brain_fusion
from brainfusion.fusion.grouping import group_average_on_shared_grid
from brainfusion.correlation import (correlate_on_shared_grid, pairwise_correlate_groups,
                                     pairwise_correlate_by_density, correlate_groups, correlate_groups_by_density,
                                     average_within_radius, compute_max_radius, correlate_around_reference_grid,
                                     analyse_correlation_percentile, conditional_probability_table)


def circle_sample(cx, cy, r, value, n_contour=40, n_grid=8, filename="s", metadata=None):
    theta = np.linspace(0, 2 * np.pi, n_contour, endpoint=False)
    contour = np.column_stack((cx + r * np.cos(theta), cy + r * np.sin(theta)))
    grid_theta = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    grid = np.column_stack((cx + (r / 2) * np.cos(grid_theta), cy + (r / 2) * np.sin(grid_theta)))
    dataset = {"value": np.full(n_grid, value)}
    return Sample(contour=contour, grid=grid, dataset=dataset, filename=filename, metadata=metadata or {})


class TestCorrelateOnSharedGrid:

    def test_perfect_correlation_for_linearly_related_data(self):
        grid = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        contour = np.array([[-1, -1], [4, -1], [4, 1], [-1, 1], [-1, -1]])
        data_a = np.array([1.0, 2.0, 3.0, 4.0])
        data_b = 2 * data_a

        result = correlate_on_shared_grid(data_a, data_b, grid, contour)
        assert result["pearson_correlation"] == pytest.approx(1.0)
        assert result["n_points"] == 4

    def test_masks_points_outside_contour_and_nans(self):
        grid = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 100.0]])  # last point is outside the contour
        contour = np.array([[-1, -1], [4, -1], [4, 1], [-1, 1], [-1, -1]])
        data_a = np.array([1.0, 2.0, np.nan])
        data_b = np.array([2.0, 4.0, 999.0])

        result = correlate_on_shared_grid(data_a, data_b, grid, contour)
        assert result["n_points"] == 2
        np.testing.assert_array_equal(result["valid_mask"], [True, True, False])

    def test_returns_full_unmasked_datasets_under_their_own_names(self):
        grid = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 100.0]])
        contour = np.array([[-1, -1], [4, -1], [4, 1], [-1, 1], [-1, -1]])
        data_a = np.array([1.0, 2.0, np.nan])
        data_b = np.array([2.0, 4.0, 999.0])

        result = correlate_on_shared_grid(data_a, data_b, grid, contour, name_a="AFM", name_b="Brillouin")
        np.testing.assert_array_equal(result["AFM"], data_a)  # unmasked - still length 3, NaN included
        np.testing.assert_array_equal(result["Brillouin"], data_b)

    def test_raises_when_too_few_valid_points(self):
        grid = np.array([[0.0, 0.0], [1.0, 0.0]])
        contour = np.array([[-1, -1], [4, -1], [4, 1], [-1, 1], [-1, -1]])
        data = np.array([np.nan, 1.0])
        with pytest.raises(ValueError, match="Not enough valid"):
            correlate_on_shared_grid(data, np.array([1.0, 2.0]), grid, contour)


class TestPairwiseCorrelateGroups:

    def test_covers_every_pair_for_three_groups(self):
        samples = [circle_sample(0, 0, 1, value=1.0, filename="a", metadata={"group": "A"}),
                  circle_sample(0.05, 0, 1, value=2.0, filename="b", metadata={"group": "B"}),
                  circle_sample(-0.05, 0, 1, value=3.0, filename="c", metadata={"group": "C"})]
        analysis = brain_fusion(samples, contour_template="average", contour_interp_n=40, clustering="Mean")
        grid, contour, group_maps = group_average_on_shared_grid(analysis, "group")
        pairs = pairwise_correlate_groups(group_maps, grid, contour, "value")
        assert set(pairs.keys()) == {("A", "B"), ("A", "C"), ("B", "C")}


class TestCorrelateGroups:

    def test_wires_group_datasets_and_shared_grid_correlation_together(self):
        grid = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        contour = np.array([[-1, -1], [4, -1], [4, 1], [-1, 1], [-1, -1]])
        analysis = {
            "group_datasets": {
                "A": {"value": np.array([1.0, 2.0, 3.0, 4.0])},
                "B": {"value": np.array([2.0, 4.0, 6.0, 8.0])},
            },
            "measurement_interpolated_grid": grid,
            "template_contours": [contour],
        }

        results = correlate_groups(analysis, "value")
        assert set(results.keys()) == {("A", "B")}
        assert results[("A", "B")]["pearson_correlation"] == pytest.approx(1.0)
        np.testing.assert_array_equal(results[("A", "B")]["grid"], grid)
        np.testing.assert_array_equal(results[("A", "B")]["contour"], contour)

    def test_raises_when_not_fused_with_group_field(self):
        analysis = {"measurement_interpolated_grid": None, "template_contours": [None]}
        with pytest.raises(ValueError, match="group_field"):
            correlate_groups(analysis, "value")


class TestPairwiseCorrelateByDensity:

    @staticmethod
    def _cloud(centers, values, n_per_center, scale, rng):
        grid = np.concatenate([c + rng.normal(scale=0.1, size=(n_per_center, 2)) for c in centers])
        data = np.concatenate([np.full(n_per_center, scale * v) for v in values])
        return grid, data

    def test_auto_picks_sparser_side_as_reference_by_point_count(self):
        centers = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        values = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(0)
        contour = np.array([[-5, -5], [15, -5], [15, 5], [-5, 5], [-5, -5]])

        datasets = {
            "sparse": (centers, values),  # 3 points - fewest, so it should be auto-picked as reference
            "dense": self._cloud(centers, values, n_per_center=20, scale=2.0, rng=rng),
        }

        results = pairwise_correlate_by_density(datasets, contour, radius=1.0)
        assert set(results.keys()) == {("sparse", "dense")}
        assert results[("sparse", "dense")]["pearson_correlation"] == pytest.approx(1.0, abs=1e-6)

    def test_priority_overrides_point_count(self):
        centers = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        values = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(0)
        contour = np.array([[-5, -5], [15, -5], [15, 5], [-5, 5], [-5, -5]])

        datasets = {
            "many_points": self._cloud(centers, values, n_per_center=20, scale=2.0, rng=rng),
            "fewer_points": self._cloud(centers, values, n_per_center=5, scale=3.0, rng=rng),
        }

        # By point count, "fewer_points" (15) would normally be picked as the reference over "many_points"
        # (60) - force the opposite ordering via priority instead.
        results = pairwise_correlate_by_density(datasets, contour, priority=["many_points", "fewer_points"],
                                                 radius=1.0)
        assert set(results.keys()) == {("many_points", "fewer_points")}
        result = results[("many_points", "fewer_points")]
        assert len(result["reference_grid"]) == 60  # every "many_points" point, since it was forced as reference
        assert result["pearson_correlation"] == pytest.approx(1.0, abs=1e-6)

    def test_covers_every_pair_for_three_datasets(self):
        centers = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        values = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(0)
        contour = np.array([[-5, -5], [15, -5], [15, 5], [-5, 5], [-5, -5]])

        datasets = {
            "sparse": (centers, values),
            "medium": self._cloud(centers, values, n_per_center=10, scale=2.0, rng=rng),
            "dense": self._cloud(centers, values, n_per_center=30, scale=3.0, rng=rng),
        }

        results = pairwise_correlate_by_density(datasets, contour, radius=1.0)
        assert set(results.keys()) == {("sparse", "medium"), ("sparse", "dense"), ("medium", "dense")}


class TestCorrelateGroupsByDensity:

    def test_wires_native_extraction_and_density_correlation_together(self):
        centers = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        values = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(0)
        dense_grid = np.concatenate([c + rng.normal(scale=0.1, size=(20, 2)) for c in centers])
        dense_data = np.concatenate([np.full(20, 2 * v) for v in values])
        contour = np.array([[-5, -5], [15, -5], [15, 5], [-5, 5], [-5, -5]])

        analysis = {
            "measurement_metadata": [{"modality": "sparse"}, {"modality": "dense"}],
            "measurement_trafo_grids": [centers, dense_grid],
            "measurement_datasets": [{"value": values}, {"value": dense_data}],
            "template_contours": [contour],
        }

        results = correlate_groups_by_density(analysis, "modality", "value", radius=1.0)
        assert set(results.keys()) == {("sparse", "dense")}
        result = results[("sparse", "dense")]
        assert result["pearson_correlation"] == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_array_equal(result["contour"], contour)
        np.testing.assert_array_equal(result["dense_grid"], dense_grid)
        np.testing.assert_array_equal(result["dense_data"], dense_data)


class TestAverageWithinRadius:

    def test_averages_other_points_within_radius(self):
        reference_grid = np.array([[0.0, 0.0], [10.0, 0.0]])
        other_grid = np.array([[0.1, 0.0], [0.2, 0.0], [10.1, 0.0]])
        other_data = np.array([2.0, 4.0, 100.0])
        avg, radius = average_within_radius(reference_grid, other_grid, other_data, radius=1.0)
        assert avg[0] == pytest.approx(3.0)  # average of the two points near (0, 0)
        assert avg[1] == pytest.approx(100.0)
        assert radius.shape == (2,)

    def test_each_other_point_used_at_most_once(self):
        # Both reference points can reach the single other point - it must be assigned to only one of
        # them, not both (which would silently double-count it).
        reference_grid = np.array([[0.0, 0.0], [0.5, 0.0]])
        other_grid = np.array([[0.25, 0.0]])
        other_data = np.array([5.0])
        avg, _ = average_within_radius(reference_grid, other_grid, other_data, radius=1.0)
        assert np.sum(~np.isnan(avg)) == 1


class TestComputeMaxRadius:

    def test_half_nearest_neighbour_distance(self):
        grid = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
        radii = compute_max_radius(grid)
        np.testing.assert_allclose(radii, [1.0, 1.0, 4.0])


class TestCorrelateAroundReferenceGrid:

    def test_correlates_matched_local_averages(self):
        # other_data = 2 * reference_data, densely sampled in a small cloud around each reference point -
        # the radius-averaged 'other' value at each reference point should recover that relationship.
        reference_grid = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        reference_data = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(0)
        other_grid = np.concatenate([center + rng.normal(scale=0.1, size=(10, 2)) for center in reference_grid])
        other_data = np.concatenate([np.full(10, 2 * value) for value in reference_data])
        contour = np.array([[-5, -5], [15, -5], [15, 5], [-5, 5], [-5, -5]])

        result = correlate_around_reference_grid(reference_data, reference_grid, other_data, other_grid,
                                                  contour, radius=1.0)
        assert result["pearson_correlation"] == pytest.approx(1.0, abs=1e-6)
        assert result["n_points"] == 3
        # name_a/name_b (default 'reference'/'other') hold the FULL (inside-contour) data, unmasked by
        # 'valid_mask' - here every reference point has a match, so masking is a no-op, but the keys
        # themselves are the full arrays, matching 'reference_grid'/'radii' 1:1.
        assert result["reference"].tolist() == pytest.approx([1.0, 2.0, 3.0])
        assert result["other"].tolist() == pytest.approx([2.0, 4.0, 6.0])
        assert len(result["reference"]) == len(result["reference_grid"]) == len(result["radii"])
        np.testing.assert_array_equal(result["valid_mask"], [True, True, True])

    def test_raises_when_nothing_within_radius(self):
        reference_grid = np.array([[0.0, 0.0], [5.0, 0.0]])
        reference_data = np.array([1.0, 2.0])
        other_grid = np.array([[100.0, 100.0]])
        other_data = np.array([1.0])
        contour = np.array([[-5, -5], [15, -5], [15, 5], [-5, 5], [-5, -5]])
        with pytest.raises(ValueError, match="Not enough valid"):
            correlate_around_reference_grid(reference_data, reference_grid, other_data, other_grid, contour,
                                            radius=1.0)


class TestAnalyseCorrelationPercentile:

    def test_returns_pearson_and_contingency_table(self):
        data_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        data_b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        results, stats_results, a_present, b_present = analyse_correlation_percentile(data_a, data_b)
        assert results["pearson_correlation"] == pytest.approx(1.0)
        assert 0.0 <= stats_results["fisher_p_value"] <= 1.0
        np.testing.assert_array_equal(a_present, data_a >= np.median(data_a))


class TestConditionalProbabilityTable:

    def test_contingency_table_matches_manual_counts(self):
        a_present = np.array([True, True, False, False])
        b_present = np.array([True, False, True, False])
        _, stats_results = conditional_probability_table(a_present, b_present)
        # layout: [[n(a absent, b absent), n(a absent, b present)], [n(a present, b absent), n(a present, b present)]]
        np.testing.assert_array_equal(stats_results["contingency_table"], [[1, 1], [1, 1]])
