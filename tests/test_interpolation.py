import pytest
import numpy as np
from brainfusion._interpolation import nearest_neighbour_interp


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
