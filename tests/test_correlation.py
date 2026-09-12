import numpy as np
import pytest

from brainfusion.sample import Sample
from brainfusion.fusion.core import brain_fusion
from brainfusion.correlation import (list_groups, group_average_on_shared_grid, correlate_on_shared_grid,
                                     pairwise_correlate_groups, average_within_radius, compute_max_radius,
                                     correlate_around_reference_grid, analyse_correlation_percentile,
                                     conditional_probability_table)


def circle_sample(cx, cy, r, value, n_contour=40, n_grid=8, filename="s", metadata=None):
    theta = np.linspace(0, 2 * np.pi, n_contour, endpoint=False)
    contour = np.column_stack((cx + r * np.cos(theta), cy + r * np.sin(theta)))
    grid_theta = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    grid = np.column_stack((cx + (r / 2) * np.cos(grid_theta), cy + (r / 2) * np.sin(grid_theta)))
    dataset = {"value": np.full(n_grid, value)}
    return Sample(contour=contour, grid=grid, dataset=dataset, filename=filename, metadata=metadata or {})


@pytest.fixture
def two_group_analysis():
    """Two groups ('A' constant value 1.0, 'B' constant value 2.0), 2 near-identical samples each."""
    samples = [
        circle_sample(0, 0, 1, value=1.0, filename="a1", metadata={"group": "A"}),
        circle_sample(0.05, 0, 1.02, value=1.0, filename="a2", metadata={"group": "A"}),
        circle_sample(-0.03, 0.02, 0.98, value=2.0, filename="b1", metadata={"group": "B"}),
        circle_sample(0.02, -0.03, 1.01, value=2.0, filename="b2", metadata={"group": "B"}),
    ]
    return brain_fusion(samples, contour_template="average", contour_interp_n=40, clustering="Mean")


class TestListGroups:

    def test_lists_unique_groups_sorted(self, two_group_analysis):
        assert list_groups(two_group_analysis, "group") == ["A", "B"]


class TestGroupAverageOnSharedGrid:

    def test_splits_into_correct_per_group_constant_values(self, two_group_analysis):
        grid, contour, group_maps = group_average_on_shared_grid(two_group_analysis, "group")
        assert set(group_maps.keys()) == {"A", "B"}
        np.testing.assert_allclose(group_maps["A"]["value"], 1.0)
        np.testing.assert_allclose(group_maps["B"]["value"], 2.0)
        assert grid.shape[1] == 2
        assert contour.shape[1] == 2

    def test_raises_for_gmm_clustering(self):
        samples = [circle_sample(0, 0, 1, value=1.0, filename="a", metadata={"group": "A"}),
                  circle_sample(0.1, 0, 1, value=2.0, filename="b", metadata={"group": "B"})]
        analysis = brain_fusion(samples, clustering="GMM", contour_interp_n=40)
        with pytest.raises(ValueError, match="clustering"):
            group_average_on_shared_grid(analysis, "group")

    def test_raises_for_unknown_group(self, two_group_analysis):
        with pytest.raises(ValueError, match="No samples found"):
            group_average_on_shared_grid(two_group_analysis, "group", groups=["C"])


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
        assert result["reference_valid"].tolist() == pytest.approx([1.0, 2.0, 3.0])
        assert result["other_valid"].tolist() == pytest.approx([2.0, 4.0, 6.0])

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
