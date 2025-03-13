import numpy as np
import matplotlib.path as mpath


def mask_contour(contour, grid):
    # Create the path from the contour
    path = mpath.Path(contour)

    # Find which points are inside the contour
    mask = path.contains_points(grid)

    return mask


def regular_grid_on_contour(contour, axis_points=50):
    # Get min and max x, y coordinates
    min_x, min_y = np.min(contour, axis=0)
    max_x, max_y = np.max(contour, axis=0)

    # Generate linearly spaced grid points
    x_values = np.linspace(min_x, max_x, axis_points)
    y_values = np.linspace(min_y, max_y, axis_points)

    # Create meshgrid
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    return grid


def project_brillouin_dataset(bm_data, bm_metadata, br_intensity_threshold=15):
    bm_data_proj = {}
    if 'brillouin_peak_intensity' in bm_data and 'brillouin_shift_f' in bm_data:
        # Filter out invalid peaks
        mask_peak = bm_data['brillouin_peak_intensity'] > br_intensity_threshold
        # Filter out water shifts
        mask_shift = (4.4 < bm_data['brillouin_shift_f']) & (bm_data['brillouin_shift_f'] < 10.0)

        # Check data distribution visually
        """
        import matplotlib.pyplot as plt
        cumulative_percentage = np.linspace(0, 100, len(sorted_data))
        plt.plot(sorted_data, cumulative_percentage, color='blue', linewidth=2)
        plt.title("Cumulative distribution")
        plt.xlabel("Brillouin shift (GHz)")
        plt.ylabel("Cumulative Percentage (%)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()
        """

        mask = mask_peak & mask_shift & (0 < bm_data['brillouin_peak_fwhm_f']) & (bm_data['brillouin_peak_fwhm_f'] < 5)
    else:
        mask = True
    for key, value in bm_data.items():
        new_value = value.copy()  # Copy the original data to avoid modifying it

        # For distribution analysis
        sorted_data = np.sort(new_value.flatten())
        bm_data_proj[key + '_distribution'] = sorted_data  # Store the sorted distribution

        if key == 'brillouin_peak_intensity':
            continue

        new_value = np.where(mask, new_value, np.nan)

        proj_value = np.nanmedian(new_value, axis=-1).ravel()

        bm_data_proj[key + '_proj'] = proj_value  # Store the projection

    bm_grid_proj = bm_metadata['brillouin_grid'][:, :, 0, :2]  # Use x,y grid of first z-slice
    bm_grid_proj = np.column_stack([bm_grid_proj[:, :, 0].ravel(), bm_grid_proj[:, :, 1].ravel()])

    return bm_data_proj, bm_grid_proj
