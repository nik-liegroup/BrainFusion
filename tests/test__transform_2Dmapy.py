import pytest
import numpy as np
from brainfusion._transform_2Dmap import transform_grid2contour, create_rbf_interpolators, extend_grid


class TestTransformGrid2Contour:

    def test_identity_transform(self):
        grid = np.array([[0.5, 0.5], [0.25, 0.75]])
        out = transform_grid2contour(grid, grid, grid, grid, smooth=0)
        trafo_grid, trafo_test, trafo_contour = out

        # Should match inputs due to identity transform
        np.testing.assert_allclose(trafo_grid, grid)
        np.testing.assert_allclose(trafo_test, grid)

    def test_invalid_input_type(self):
        with pytest.raises(TypeError):
            transform_grid2contour("not an array", np.zeros((4, 2)), np.zeros((2, 2)), np.zeros((2, 2)))

    def test_invalid_input_shape(self):
        with pytest.raises(ValueError):
            transform_grid2contour(np.zeros((4,)), np.zeros((4, 2)), np.zeros((2, 2)), np.zeros((2, 2)))

    def test_invalid_smooth_value(self):
        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        grid = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError):
            transform_grid2contour(contour, contour, grid, grid, smooth="invalid")

    def test_output_shapes(self):
        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        grid = np.array([[0.5, 0.5], [0.25, 0.75]])
        out = transform_grid2contour(contour, contour, grid, grid)
        assert out[0].shape == grid.shape
        assert out[1].shape == grid.shape
        assert out[2].shape == contour.shape


class TestCreateRbfInterpolators:

    def setup_method(self):
        self.original_contour = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        self.deformed_contour = np.array([
            [0, 0],
            [2, 0],
            [2, 2],
            [0, 2]
        ])  # scaled by 2

    def test_returns_callable(self):
        rbf_x, rbf_y = create_rbf_interpolators(self.original_contour, self.deformed_contour)
        assert callable(rbf_x)
        assert callable(rbf_y)

    def test_identity_mapping(self):
        rbf_x, rbf_y = create_rbf_interpolators(self.original_contour, self.original_contour, smooth=0)
        result = np.array([[rbf_x(x, y), rbf_y(x, y)] for x, y in self.original_contour])
        np.testing.assert_allclose(result, self.original_contour, atol=1e-6)

    def test_scaled_mapping(self):
        rbf_x, rbf_y = create_rbf_interpolators(self.original_contour, self.deformed_contour, smooth=0)
        result = np.array([[rbf_x(x, y), rbf_y(x, y)] for x, y in self.original_contour])
        np.testing.assert_allclose(result, self.deformed_contour, atol=1e-6)

    def test_invalid_shape(self):
        bad_contour = np.array([[0, 0, 0], [1, 1, 1]])
        with pytest.raises(ValueError):
            create_rbf_interpolators(bad_contour, self.deformed_contour)

    def test_mismatched_shapes(self):
        with pytest.raises(ValueError):
            create_rbf_interpolators(self.original_contour, self.deformed_contour[:3])

    def test_invalid_function_type(self):
        with pytest.raises(TypeError):
            create_rbf_interpolators(self.original_contour, self.deformed_contour, function=123)

    def test_invalid_function_name(self):
        with pytest.raises(ValueError):
            create_rbf_interpolators(self.original_contour, self.deformed_contour, function="badfunc")

    @pytest.mark.parametrize("smooth", ["invalid", [0.1], None])
    def test_invalid_smooth_type(self, smooth):
        with pytest.raises(TypeError):
            create_rbf_interpolators(self.original_contour, self.deformed_contour, smooth=smooth)

    def test_auto_smooth(self):
        # Should not raise
        rbf_x, rbf_y = create_rbf_interpolators(self.original_contour, self.deformed_contour, smooth='auto')
        assert callable(rbf_x)
        assert callable(rbf_y)


class TestExtendGrid:

    def setup_method(self):
        # Create 3 small 2D grids
        self.grid1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        self.grid2 = np.array([[2, 2], [3, 2], [2, 3], [3, 3]])
        self.grid3 = np.array([[4, 0], [5, 0], [4, 1], [5, 1]])
        self.grids = [self.grid1, self.grid2, self.grid3]

    def test_output_shapes(self):
        extended, shape = extend_grid(self.grids, x_extend=0.1, y_extend=0.1)
        assert isinstance(extended, np.ndarray)
        assert extended.ndim == 2
        assert extended.shape[1] == 2
        assert isinstance(shape, list)
        assert len(shape) == 2

    def test_grid_contains_original_bounds(self):
        extended, _ = extend_grid(self.grids, x_extend=0.0, y_extend=0.0)
        x_all = np.hstack([g[:, 0] for g in self.grids])
        y_all = np.hstack([g[:, 1] for g in self.grids])
        assert extended[:, 0].min() <= x_all.min()
        assert extended[:, 0].max() >= x_all.max()
        assert extended[:, 1].min() <= y_all.min()
        assert extended[:, 1].max() >= y_all.max()

    def test_extension_increases_bounds(self):
        extended_no, _ = extend_grid(self.grids, x_extend=0.0, y_extend=0.0)
        extended_yes, _ = extend_grid(self.grids, x_extend=0.5, y_extend=0.5)
        assert extended_yes[:, 0].min() < extended_no[:, 0].min()
        assert extended_yes[:, 0].max() > extended_no[:, 0].max()
        assert extended_yes[:, 1].min() < extended_no[:, 1].min()
        assert extended_yes[:, 1].max() > extended_no[:, 1].max()

    def test_regular_spacing(self):
        extended, shape = extend_grid(self.grids, x_extend=0.1, y_extend=0.1)
        x_vals = extended[:, 0].reshape(shape)
        y_vals = extended[:, 1].reshape(shape)

        dx = np.diff(x_vals, axis=1)
        dy = np.diff(y_vals, axis=0)

        assert np.allclose(dx, dx[0, 0], atol=1e-8)
        assert np.allclose(dy, dy[0, 0], atol=1e-8)

    def test_invalid_input_type(self):
        with pytest.raises(TypeError):
            extend_grid("not a list", x_extend=0.1, y_extend=0.1)

    def test_invalid_array_shapes(self):
        bad_grids = [np.array([[1, 2, 3]]), np.array([1, 2])]
        with pytest.raises(ValueError):
            extend_grid(bad_grids, x_extend=0.1, y_extend=0.1)
