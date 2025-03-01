import numpy as np
from scipy.interpolate import Rbf
from tqdm import tqdm


def transform_grid2contour(original_contour, deformed_contour, original_grid, progress=''):
    # Create RBF interpolators
    rbf_x, rbf_y = create_rbf_interpolators(original_contour=original_contour,
                                            deformed_contour=deformed_contour)

    # Apply transformation to original contour to check whether deformation works as expected
    trafo_contour = np.array([
        [rbf_x(x, y), rbf_y(x, y)] for x, y in original_contour
    ])

    # Deform original grid to coordinate system of deformed contour
    trafo_grid = np.array([
        [rbf_x(x, y), rbf_y(x, y)] for x, y in tqdm(original_grid, desc="Transforming grid" + progress)
    ])

    return trafo_grid, trafo_contour


# ToDo: Implement new interpolation function scipy.interpolate.RBFInterpolator
def create_rbf_interpolators(original_contour, deformed_contour, function='thin_plate', smooth=0.2):
    x_orig, y_orig = original_contour[:, 0], original_contour[:, 1]
    x_deform, y_deform = deformed_contour[:, 0], deformed_contour[:, 1]

    rbf_x = Rbf(x_orig, y_orig, x_deform, function=function, smooth=smooth)
    rbf_y = Rbf(x_orig, y_orig, y_deform, function=function, smooth=smooth)

    return rbf_x, rbf_y


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
