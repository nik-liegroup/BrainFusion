import numpy as np
from scipy.interpolate import Rbf


def transform_grid2contour(original_contour, deformed_contour, original_grid):
    # Create RBF interpolators
    rbf_x, rbf_y = create_rbf_interpolators(original_contour=original_contour,
                                            deformed_contour=deformed_contour)

    # Transform original contour to the coordinate system of the deformed contour
    trafo_cont_x, trafo_cont_y = evaluate_transformation(rbf_x, rbf_y, original_contour)
    trafo_contour = np.vstack([trafo_cont_x, trafo_cont_y]).T

    # Transform original grid to the coordinate system defined by the average contour
    trafo_grid_x, trafo_grid_y = evaluate_transformation(rbf_x, rbf_y, original_grid)
    trafo_grid = np.vstack([trafo_grid_x, trafo_grid_y]).T

    return trafo_grid, trafo_contour


# ToDo: Implement new interpolation function scipy.interpolate.RBFInterpolator
def create_rbf_interpolators(original_contour, deformed_contour):
    x_orig, y_orig = original_contour[:, 0], original_contour[:, 1]
    x_deform, y_deform = deformed_contour[:, 0], deformed_contour[:, 1]

    rbf_x = Rbf(x_orig, y_orig, x_deform, function='linear', smooth=0.2)
    rbf_y = Rbf(x_orig, y_orig, y_deform, function='linear', smooth=0.2)

    return rbf_x, rbf_y


def evaluate_transformation(rbf_x, rbf_y, grid_points):
    grid_x, grid_y = grid_points[:, 0], grid_points[:, 1]
    trafo_grid_x = rbf_x(grid_x, grid_y)
    trafo_grid_y = rbf_y(grid_x, grid_y)

    return trafo_grid_x, trafo_grid_y


def extend_grid(regular_grid, x_extend, y_extend):
    # Get minimum and maximum coordinate values in x and y
    x_min, y_min = np.min(regular_grid, axis=0)
    x_max, y_max = np.max(regular_grid, axis=0)

    # Round coordinates and extract x and y values
    x_values = regular_grid[:, 0].round(decimals=0)
    y_values = regular_grid[:, 1].round(decimals=0)

    # Find unique sorted x and y coordinates
    x_sorted = np.sort(np.unique(x_values))
    y_sorted = np.sort(np.unique(y_values))

    x_diff = np.diff(x_sorted)
    y_diff = np.diff(y_sorted)

    # Estimate regular spacing using the mean and median (robust to missing points)
    x_median_spacing, x_mean_spacing = np.median(x_diff), np.mean(x_diff)
    y_median_spacing, y_mean_spacing = np.median(y_diff), np.mean(x_diff)

    # Check if standard deviation of spacings fluctuates more than 15% around mean spacing
    #assert (np.std(x_diff) / x_mean_spacing) * 100 < 0.15
    #assert (np.std(y_diff) / y_mean_spacing) * 100 < 0.15

    # Generate new extended coordinates using the estimated spacing
    x_new = np.arange(x_min - x_extend, x_max + x_extend + x_median_spacing, x_median_spacing)
    y_new = np.arange(y_min - y_extend, y_max + y_extend + y_median_spacing, y_median_spacing)

    # Create the extended meshgrid
    x, y = np.meshgrid(x_new, y_new)
    extended_grid = np.stack([x, y], axis=-1)
    extended_grid = np.vstack([extended_grid[:, :, 0].ravel(), extended_grid[:, :, 1].ravel()]).T

    return extended_grid
