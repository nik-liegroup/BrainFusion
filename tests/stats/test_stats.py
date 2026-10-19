import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from brainfusion.stats import (animal_maps, average_map, pairwise_map_correlations, split_half_reliability, map_consistency,
                               bootstrap_map_correlation, bootstrap_group_correlation, correlation_vs_animals, bootstrap_map_scalar,
                               spatial_surrogates, spatial_null_test, summarise_correlations, disattenuate,
                               save_correlation_results, load_correlation_results)
from brainfusion.stats.surrogates import _Variogram
from scipy.spatial.distance import cdist

N = 16
YY, XX = np.mgrid[0:N, 0:N]
GRID = np.column_stack([XX.ravel(), YY.ravel()]).astype(float)
CONTOUR = np.array([[-1, -1], [N, -1], [N, N], [-1, N], [-1, -1]], dtype=float)
INSIDE = np.ones(N * N, dtype=bool)


def smooth_field(rng, sigma=2.0):
    f = gaussian_filter(rng.standard_normal((N, N)), sigma, mode="wrap").ravel()
    return (f - f.mean()) / f.std()


def animals(rng, pattern, n_animals, noise):
    return np.array([pattern + noise * rng.standard_normal(pattern.size) for _ in range(n_animals)])


class TestConsistency:

    def test_shared_pattern_gives_high_reliability_and_noise_gives_low(self):
        rng = np.random.default_rng(0)
        pattern = smooth_field(rng)
        clean = map_consistency(animals(rng, pattern, 10, 0.3), INSIDE, n_splits=50)
        noisy = map_consistency(animals(rng, pattern, 10, 3.0), INSIDE, n_splits=50)
        assert clean["split_half"]["mean"]["r"] > 0.95
        assert noisy["split_half"]["mean"]["r"] < clean["split_half"]["mean"]["r"]
        assert clean["pairwise"]["pearson_mean"] > noisy["pairwise"]["pearson_mean"]
        assert clean["leave_one_out"]["median"]["r_mean"] > 0.9
        assert clean["pointwise"]["signal_ratio_mean_map"] > noisy["pointwise"]["signal_ratio_mean_map"]

    def test_median_map_is_robust_to_an_odd_animal(self):
        rng = np.random.default_rng(1)
        pattern = smooth_field(rng)
        maps = animals(rng, pattern, 9, 0.3)
        maps[0] = 50 * rng.standard_normal(pattern.size)   # one animal with nothing to do with the pattern
        result = map_consistency(maps, INSIDE, n_splits=60)
        assert result["split_half"]["median"]["r"] > result["split_half"]["mean"]["r"]

    def test_split_half_needs_four_animals(self):
        out = split_half_reliability(np.zeros((3, 10)), np.ones(10, dtype=bool))
        assert np.isnan(out["r"])

    def test_pairwise_correlations_of_identical_maps_are_one(self):
        maps = np.tile(np.arange(60.0), (3, 1))
        pearson, spearman = pairwise_map_correlations(maps, np.ones(60, dtype=bool))
        np.testing.assert_allclose(pearson, 1.0)
        np.testing.assert_allclose(spearman, 1.0)

    def test_average_map_unknown_average_is_an_error(self):
        with pytest.raises(ValueError):
            average_map(np.zeros((2, 3)), "mode")

    def test_animal_maps_without_shared_grid_is_an_error(self):
        with pytest.raises(ValueError, match="shared grid"):
            animal_maps({"measurement_trafo_datasets": [None]}, "value")


class TestBootstrap:

    def test_interval_is_around_the_estimate_and_shrinks_with_more_animals(self):
        rng = np.random.default_rng(2)
        pattern = smooth_field(rng)
        few = bootstrap_map_correlation(animals(rng, pattern, 4, 2.0), animals(rng, pattern, 4, 2.0), INSIDE, n_boot=300)
        many = bootstrap_map_correlation(animals(rng, pattern, 24, 2.0), animals(rng, pattern, 24, 2.0), INSIDE, n_boot=300)
        for result in (few, many):
            low, high = result["pearson_ci"]
            assert low <= result["pearson"] <= high      # centred on the estimate, whatever the bias of the draws
            assert result["pearson_ci_percentile"][1] <= result["pearson"] + 0.05   # the draws themselves sit below it
        assert (many["pearson_ci"][1] - many["pearson_ci"][0]) < (few["pearson_ci"][1] - few["pearson_ci"][0])
        assert many["pearson"] > few["pearson"]          # averaging more animals removes noise

    def test_unrelated_groups_have_an_interval_around_zero(self):
        rng = np.random.default_rng(3)
        a = animals(rng, smooth_field(rng), 10, 1.0)
        b = animals(rng, smooth_field(rng), 10, 1.0)
        low, high = bootstrap_map_correlation(a, b, INSIDE, n_boot=300)["pearson_ci"]
        assert low < 0.35 and high > -0.35

    def test_paired_resampling_needs_equal_group_sizes(self):
        with pytest.raises(ValueError):
            bootstrap_map_correlation(np.zeros((3, 10)), np.zeros((4, 10)), np.ones(10, dtype=bool), paired=True)

    def test_correlation_vs_animals_grows_with_the_number_of_animals(self):
        rng = np.random.default_rng(4)
        pattern = smooth_field(rng)
        a, b = animals(rng, pattern, 12, 3.0), animals(rng, pattern, 12, 3.0)
        curve = correlation_vs_animals(a, b, INSIDE, sizes=[1, 4, 12], n_draws=60)
        assert curve[1]["mean"] < curve[4]["mean"] < curve[12]["mean"]
        assert curve[12]["sd"] == 0.0     # all animals: one draw

    def test_group_wrapper_uses_the_animals_of_each_group(self):
        rng = np.random.default_rng(5)
        pattern = smooth_field(rng)
        maps = np.vstack([animals(rng, pattern, 6, 1.0), animals(rng, pattern, 6, 1.0)])
        analysis = {"measurement_trafo_datasets": [{"value": m} for m in maps], "measurement_interpolated_grid": GRID,
                    "measurement_metadata": [{"g": "A"}] * 6 + [{"g": "B"}] * 6, "template_contours": [CONTOUR],
                    "group_field": "g"}
        result = bootstrap_group_correlation(analysis, "value", "A", "B", n_boot=50)
        assert result["n_a"] == result["n_b"] == 6 and result["pearson"] > 0.8


class TestBootstrapScalar:

    def test_median_of_the_average_map_with_an_interval_that_shrinks_with_more_animals(self):
        rng = np.random.default_rng(9)
        pattern = 10 + smooth_field(rng)
        few = bootstrap_map_scalar(animals(rng, pattern, 4, 2.0), INSIDE, n_boot=300)
        many = bootstrap_map_scalar(animals(rng, pattern, 40, 2.0), INSIDE, n_boot=300)
        for result in (few, many):
            assert result["ci"][0] < result["value"] < result["ci"][1]
            assert result["value"] == pytest.approx(10.0, abs=1.5)
        assert many["ci"][1] - many["ci"][0] < few["ci"][1] - few["ci"][0]
        assert many["n_animals"] == 40

    def test_unknown_statistic_is_an_error(self):
        with pytest.raises(ValueError):
            bootstrap_map_scalar(np.zeros((3, 4)), np.ones(4, dtype=bool), statistic="mode")


class TestSpatialNull:

    def test_surrogates_keep_the_values_and_roughly_the_autocorrelation(self):
        rng = np.random.default_rng(6)
        field = smooth_field(rng)
        surrogates, info = spatial_surrogates(field, GRID, n_surrogates=20, seed=0)
        assert surrogates.shape == (20, field.size)
        np.testing.assert_allclose(np.sort(surrogates[0]), np.sort(field))      # same values, other places
        variogram = _Variogram(cdist(GRID, GRID), 25, 50)
        target, fitted = variogram(field), np.mean([variogram(s) for s in surrogates], axis=0)
        assert np.corrcoef(target, fitted)[0, 1] > 0.9                              # same rise of the variogram with distance
        assert info["bandwidth"] > 0

    def test_independent_smooth_maps_are_not_called_significant_as_often_as_a_plain_test_would(self):
        from scipy.stats import pearsonr
        rng = np.random.default_rng(7)
        plain, spatial = [], []
        for k in range(12):
            a, b = smooth_field(rng, 2.5), smooth_field(rng, 2.5)
            plain.append(pearsonr(a, b).pvalue < 0.05)
            spatial.append(spatial_null_test(a, b, GRID, CONTOUR, n_surrogates=100, seed=k)["p_pearson"] < 0.05)
        assert np.mean(plain) > 0.4                 # smooth maps correlate "significantly" by chance
        assert np.mean(spatial) < np.mean(plain) - 0.2

    def test_a_real_relation_is_significant(self):
        rng = np.random.default_rng(8)
        a = smooth_field(rng)
        b = a + 0.3 * smooth_field(rng)
        result = spatial_null_test(a, b, GRID, CONTOUR, n_surrogates=200, seed=0)
        assert result["pearson"] > 0.9 and result["p_pearson"] < 0.05
        assert result["n_points"] == N * N and len(result["null_pearson_a"]) == 200

    def test_too_few_points_is_an_error(self):
        with pytest.raises(ValueError, match="Too few"):
            spatial_surrogates(np.arange(10.0), np.column_stack([np.arange(10.0), np.zeros(10)]))


class TestSummary:

    def test_fisher_summary_of_animal_correlations(self):
        out = summarise_correlations([0.1, 0.2, 0.3, 0.0, 0.25, 0.15])
        assert out["n"] == 6 and out["n_positive"] == 5
        assert out["ci"][0] < out["mean_r"] < out["ci"][1]
        assert out["p"] < 0.05
        assert out["median_r"] == pytest.approx(0.175)

    def test_summary_needs_three_animals(self):
        with pytest.raises(ValueError):
            summarise_correlations([0.1, 0.2])

    def test_disattenuation(self):
        assert disattenuate(0.4, 0.8, 0.5) == pytest.approx(0.4 / np.sqrt(0.4))
        with pytest.raises(ValueError):
            disattenuate(0.4, 0.0, 0.5)


class TestResults:

    def test_roundtrip_of_a_nested_result(self, tmp_path):
        results = {"bootstrap": {"pearson": 0.65, "pearson_ci": np.array([0.5, 0.75]), "average": "median"},
                   "consistency": {"AFM": {"split_half": {"mean": {"r": 0.7, "values": np.array([0.6, 0.8])}}}},
                   "pairs": [(0.1, 0.2), (0.3, 0.4)], "n": np.int64(26)}
        path = tmp_path / "correlation.h5"
        save_correlation_results(path, results, {"n_boot": 1000, "seed": 0})
        loaded, params = load_correlation_results(path)
        assert loaded["bootstrap"]["pearson"] == pytest.approx(0.65)
        np.testing.assert_allclose(loaded["bootstrap"]["pearson_ci"], [0.5, 0.75])
        assert loaded["bootstrap"]["average"] == "median"
        assert loaded["consistency"]["AFM"]["split_half"]["mean"]["r"] == pytest.approx(0.7)
        assert loaded["n"] == 26
        assert params["n_boot"] == 1000


class TestConditions:

    def test_scalar_difference_finds_a_shift_and_not_noise(self):
        from brainfusion.stats import permutation_scalar_difference
        rng = np.random.default_rng(1)
        shifted = permutation_scalar_difference(rng.normal(10, 1, 15), rng.normal(8, 1, 15), n_perm=2000, n_boot=500)
        same = permutation_scalar_difference(rng.normal(10, 1, 15), rng.normal(10, 1, 15), n_perm=2000, n_boot=500)
        assert shifted["p_value"] < 0.01 and shifted["ci"][0] > 0 and shifted["cliffs_delta"] > 0.5
        assert same["p_value"] > 0.05 and same["ci"][0] < 0 < same["ci"][1]

    def test_map_difference_locates_a_patch_and_global_p_is_uniform_under_null(self):
        from brainfusion.stats import permutation_map_difference
        rng = np.random.default_rng(2)
        pattern = smooth_field(rng)
        a, b = animals(rng, pattern, 10, 0.5), animals(rng, pattern, 10, 0.5)
        patch = (XX.ravel() < 5) & (YY.ravel() < 5)
        a[:, patch] += 2.0
        result = permutation_map_difference(a, b, INSIDE, n_perm=500)
        assert result["global_p"] < 0.01
        assert (result["p_corrected"][patch] < 0.05).mean() > 0.8
        assert (result["p_corrected"][~patch] < 0.05).mean() < 0.1
        null = permutation_map_difference(animals(rng, pattern, 10, 0.5), animals(rng, pattern, 10, 0.5), INSIDE, n_perm=500)
        assert null["global_p"] > 0.01 and np.nanmin(null["p_corrected"]) > 0.01
