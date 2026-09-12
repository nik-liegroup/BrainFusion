import pytest
import numpy as np
from brainfusion.fusion.interpolation import nearest_neighbour_interp, fit_coordinates_gmm


class TestNearestNeighbourInterp:

    def setup_method(self):
        self.org_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        self.values = np.array([10, 20, 30, 40])
        self.target_points = np.array([[0, 0], [0.9, 0], [0.1, 0.9], [1, 1], [2, 2]])

    def test_interpolation_non_unique(self):
        result = nearest_neighbour_interp(self.org_points, self.values, self.target_points, unique=False)
        expected = np.array([10, 20, 30, 40, 40])
        np.testing.assert_array_equal(result, expected)

    def test_interpolation_unique(self):
        result = nearest_neighbour_interp(self.org_points, self.values, self.target_points, unique=True)
        assert result.shape == (len(self.target_points),)
        # Only one source may win per target — values must be in original set or NaN
        assert np.all(np.isnan(result) | np.isin(result, self.values))

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            nearest_neighbour_interp(self.org_points, self.values[:-1], self.target_points)

    def test_empty_sources(self):
        org_points = np.empty((0, 2))
        values = np.array([])
        target_points = np.array([[0, 0], [1, 1]])
        with pytest.raises(ValueError, match="Length of org_points is 0"):
            nearest_neighbour_interp(org_points, values, target_points, unique=False)

    def test_duplicate_target_points_unique(self):
        target_points = np.array([[0, 0], [0, 0]])  # duplicate coordinates
        result = nearest_neighbour_interp(self.org_points, self.values, target_points, unique=True)
        assert result.shape == (2,)
        # Only one of the two will be filled due to same index, the other stays NaN
        assert np.count_nonzero(~np.isnan(result)) == 1
        assert np.isnan(result[0]) or np.isnan(result[1])


class TestFitCoordinatesGmm:

    def setup_method(self):
        # Two tight, well-separated point clusters so the fitted Gaussian components are unambiguous
        rng = np.random.default_rng(0)
        self.cluster_a = rng.normal(loc=(0, 0), scale=0.01, size=(10, 2))
        self.cluster_b = rng.normal(loc=(10, 10), scale=0.01, size=(10, 2))
        self.grids = [self.cluster_a, self.cluster_b]
        self.data_list = [{"value": np.full(10, 1.0)}, {"value": np.full(10, 5.0)}]

    def test_mean_num_components_matches_average_point_count(self):
        centres, avg_data = fit_coordinates_gmm(self.grids, self.data_list, num_components='mean')
        assert centres.shape == (10, 2)  # mean of 10 and 10 points per sample
        assert avg_data["value"].shape == (10,)

    def test_recovers_two_well_separated_clusters(self):
        centres, avg_data = fit_coordinates_gmm([self.cluster_a, self.cluster_b],
                                                 [{"value": np.full(10, 1.0)}, {"value": np.full(10, 5.0)}])
        # More components (10) than real clusters (2) means some components legitimately end up with no
        # points assigned (NaN) - but every non-empty one must cleanly recover one cluster's constant value,
        # never a blend of the two, and both original values must still be represented somewhere.
        values = avg_data["value"]
        non_nan = values[~np.isnan(values)]
        assert set(non_nan.tolist()) == {1.0, 5.0}
        # Every fitted centre must land near one of the two original cluster centres
        for centre in centres:
            assert np.isclose(centre, [0, 0], atol=0.5).all() or np.isclose(centre, [10, 10], atol=0.5).all()

    def test_min_num_components_uses_smallest_sample(self):
        grids = [self.cluster_a, self.cluster_b[:3]]
        data_list = [{"value": np.full(10, 1.0)}, {"value": np.full(3, 5.0)}]
        centres, _ = fit_coordinates_gmm(grids, data_list, num_components='min')
        assert centres.shape[0] == 3

    def test_recovers_clusters_at_tiny_absolute_coordinate_scale(self):
        # Regression test: coordinates in metre-scale units (e.g. ~1e-4) used to make every fitted
        # component collapse onto nearly the same point, because GaussianMixture's default `reg_covar`
        # (1e-6, an absolute value) swamped the real (much smaller) variance of the data.
        rng = np.random.default_rng(0)
        tiny_a = rng.normal(loc=(0, 0), scale=1e-6, size=(10, 2))
        tiny_b = rng.normal(loc=(2e-4, 2e-4), scale=1e-6, size=(10, 2))
        centres, avg_data = fit_coordinates_gmm([tiny_a, tiny_b],
                                                 [{"value": np.full(10, 1.0)}, {"value": np.full(10, 5.0)}])
        assert set(avg_data["value"].tolist()) == {1.0, 5.0}
        assert np.ptp(centres, axis=0).min() > 1e-6  # centres actually spread out, didn't collapse

    def test_invalid_num_components_raises(self):
        with pytest.raises(AssertionError):
            fit_coordinates_gmm(self.grids, self.data_list, num_components='mode')
