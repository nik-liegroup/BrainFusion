import numpy as np
import os
import matplotlib.path as mpath
import h5py
from scipy.spatial.distance import pdist


def get_roi_from_txt(roi_path, delimiter='\t'):
    """
    Load boundary data from text file into Nx2 array.
    """
    if os.path.exists(roi_path):
        # Read the content of the .txt file
        with open(roi_path, 'r') as f:
            # Store the content in a list
            content = f.readlines()
            # Remove newline characters and split coordinates
            coordinates = [tuple(line.strip().split(delimiter)) for line in content]

            # Convert to Nx2 NumPy array
            return np.array(coordinates, dtype=float)
    else:
        print(f'{roi_path} is not a valid .txt file, continuing without!')
        return None


def mask_contour(contour, grid):
    # Create the path from the contour
    path = mpath.Path(contour)

    # Find which points are inside the contour
    mask = path.contains_points(grid)

    return mask


def check_parameters(params_defined, params_loaded):
    # Check if params variables differ
    differences = {}
    for key, value in params_defined.items():
        if key == 'load_experiment_func':
            continue
        elif key in params_loaded:
            if params_loaded[key] != value:
                differences[key] = {
                    'defined': value,
                    'loaded': params_loaded[key]
                }
        else:
            differences[key] = {
                'defined': value,
                'loaded': None
            }

    # Report differences
    if differences:
        print("Differences found between loaded parameters and defined parameters:")
        for key, diff in differences.items():
            print(f"{key}: Old = {diff['loaded']}, Loaded = {diff['defined']}")


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


def rotate3Dgrid(grid, angle, center_x, center_y):
    # Convert angle to radians
    theta = np.radians(angle)

    # Create rotation matrix based on the specified axis
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    # Shift grid to rotation center and rotate
    center = np.array([center_x, center_y, 0])
    translated_grid = grid - center
    rotated_grid = np.dot(translated_grid, rotation_matrix.T)

    # Shift back to original center
    rotated_grid += center

    return rotated_grid


def write_dict_in_h5(h5file, group_path, dic):
    for key, item in dic.items():
        if isinstance(item, dict):
            write_dict_in_h5(h5file, f"{group_path}/{key}", item)
        else:
            h5file.create_dataset(f"{group_path}/{key}", data=item)


def export_analysis(path, analysis, params):
    with h5py.File(path, 'w') as h5file:
        write_dict_in_h5(h5file, '/', analysis)

        # Save parameters as attributes
        for key, value in params.items():
            if key == 'load_experiment_func':
                continue
            elif key in ['vmax', 'vmin']:
                if value is None:
                    value = np.nan

            h5file.attrs[key] = value

    print(f"Results and parameters saved in {path}.")


def read_dict_from_h5(h5file, group_path='/'):
    result = {}
    group = h5file[group_path]

    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = read_dict_from_h5(h5file, f"{group_path}/{key}")
        else:
            result[key] = item[()]  # Reads dataset values

    return result


def import_analysis(path):
    with h5py.File(path, 'r') as h5file:
        # Load the analysis (stored in groups/datasets)
        analysis = read_dict_from_h5(h5file, '/')

        # Load parameters (stored as attributes)
        params = {}
        for key, value in h5file.attrs.items():
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value.item()  # Convert single-element arrays to scalars
            params[key] = value

    return analysis, params
