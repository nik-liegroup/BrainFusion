import numpy as np
import os
import matplotlib.path as mpath
import h5py
import pandas as pd


def read_parquet_file(path, image=False):
    """
    Reads in parquet files either as images in matrix format or as a Nx2 list of coordinates with the respective N values.
    """
    df = pd.read_parquet(path, engine='pyarrow')
    if image:
        df = df.sort_values(by=["y", "x"])
        img = df.pivot(index="y", columns="x", values="value_orig")
        return img
    else:
        grid = df[['x', 'y']].to_numpy()
        data = df[['value_background_corrected']].to_numpy().ravel()
        return grid, data


def append_parquet_file(path, brainfusion_analysis):
    """
    Write brainfusion analysis file to parquet file.
    """
    parquet_file_names = [filename + '_Merged_RAW_ch02_image_roi_linearised.parquet'
                          for filename in brainfusion_analysis['myelin_filenames']]
    for idx, file_name in enumerate(parquet_file_names):
        parquet_file_path = os.path.join(path, file_name)
        df = pd.read_parquet(parquet_file_path, engine='pyarrow')
        trafo_grid = brainfusion_analysis['myelin_trafo_grids'][idx]

        # ToDo: Remove for writing proper data
        trafo_grid = np.zeros((len(df), 2))

        df['x_translated'], df['y_translated'] = trafo_grid[:, 0], trafo_grid[:, 1]
        df.to_parquet(parquet_file_path.removesuffix(".parquet") + '_Trafo' + '.parquet', index=False, engine='pyarrow')


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


def write_dict_in_h5(h5file, group_path, dic):
    for key, item in dic.items():
        if isinstance(item, dict):
            write_dict_in_h5(h5file, f"{group_path}/{key}", item)
        elif isinstance(item, list) and all(isinstance(i, np.ndarray) for i in item):
            # Create a group for the list
            list_group = h5file.create_group(f"{group_path}/{key}")
            for i, arr in enumerate(item):
                list_group.create_dataset(str(i), data=arr)  # Store each array separately
        else:
            h5file.create_dataset(f"{group_path}/{key}", data=item)


def read_dict_from_h5(h5file, group_path='/'):
    result = {}
    group = h5file[group_path]

    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = read_dict_from_h5(h5file, f"{group_path}/{key}")
        else:
            data = item[()]  # Read dataset normally

            # Convert 'myelin_filenames' to list
            if key == 'myelin_filenames' and isinstance(data, np.ndarray):
                result[key] = [s.decode('utf-8') if isinstance(s, bytes) else s for s in data.tolist()]
            else:
                result[key] = data  # Keep as is for other cases

    # If all keys are numerical, convert dict to list
    if all(k.isdigit() for k in result.keys()):
        sorted_keys = sorted(result.keys(), key=int)  # Sort keys numerically
        result = [result[k] for k in sorted_keys]  # Convert to list

    return result


def import_analysis(path):
    print(f'Importing analysis file from: {os.path.basename(path)}.')
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
