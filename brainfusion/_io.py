import numpy as np
import os
import h5py
import pandas as pd


def check_parameters(params_defined, params_loaded):
    """
    Compare two dictionaries of parameters and report differences, allowing element-wise comparison for lists.
    """
    differences = {}

    for key, value in params_defined.items():
        if key == 'load_experiment_func':
            continue
        elif key in params_loaded:
            loaded_value = params_loaded[key]

            # Handle lists separately to compare their content rather than object identity
            if isinstance(value, np.ndarray) and isinstance(loaded_value, list):
                if set(value) != set(loaded_value) or len(value) != len(loaded_value):
                    differences[key] = {
                        'defined': value,
                        'loaded': loaded_value
                    }
            elif loaded_value != value:
                differences[key] = {
                    'defined': value,
                    'loaded': loaded_value
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


def export_analysis(path, analysis, params):
    with h5py.File(path, 'w') as h5file:
        write_dict_in_h5(h5file, '/', analysis)

        # Save parameters as attributes
        for key, value in params.items():
            if key == 'load_experiment_func':
                continue

            h5file.attrs[key] = value

    print(f"Results and parameters saved in {path}.")


def write_dict_in_h5(h5file, group_path, dic):
    """Recursively writes a dictionary (or nested dicts/lists) into an HDF5 file."""
    for key, item in dic.items():
        if isinstance(item, dict):
            # Recursively store dictionaries as groups
            write_dict_in_h5(h5file, f"{group_path}/{key}", item)

        elif isinstance(item, list):
            if all(isinstance(i, np.ndarray) for i in item):
                # Store list of arrays as a group of datasets
                list_group = h5file.create_group(f"{group_path}/{key}")
                for i, arr in enumerate(item):
                    list_group.create_dataset(str(i), data=arr)

            elif all(isinstance(i, dict) for i in item):
                # Store list of dictionaries as a group with subgroups
                list_group = h5file.create_group(f"{group_path}/{key}")
                for i, sub_dict in enumerate(item):
                    write_dict_in_h5(list_group, str(i), sub_dict)

            elif all(isinstance(i, str) for i in item):
                # Store list of strings using variable-length string type
                dt = h5py.string_dtype(encoding="utf-8")
                h5file.create_dataset(f"{group_path}/{key}", data=np.array(item, dtype=dt))

            elif all(isinstance(i, (int, float, bool, np.integer, np.floating, np.bool_)) for i in item):
                # Store list of numbers or booleans as a dataset
                h5file.create_dataset(f"{group_path}/{key}", data=np.array(item, dtype=np.float64 if any(isinstance(i, float) for i in item) else np.int64))

            else:
                raise ValueError(f"Unsupported list type in key '{key}': {type(item[0])}")

        elif isinstance(item, str):
            # Store single string as variable-length dataset
            dt = h5py.string_dtype(encoding="utf-8")
            h5file.create_dataset(f"{group_path}/{key}", data=np.array(item, dtype=dt))

        elif isinstance(item, bool):
            # Convert bool to np.bool_ (or store it as an int)
            h5file.create_dataset(f"{group_path}/{key}", data=np.array(item, dtype=np.bool_))

        elif isinstance(item, (np.integer, np.floating)):
            # Convert NumPy scalars to proper dataset values
            h5file.create_dataset(f"{group_path}/{key}", data=item.item())

        elif isinstance(item, np.ndarray):
            # Store NumPy arrays directly
            h5file.create_dataset(f"{group_path}/{key}", data=item)

        elif isinstance(item, (int, float)):
            # Store native Python numbers
            h5file.create_dataset(f"{group_path}/{key}", data=item)

        else:
            raise TypeError(f"Unsupported data type in key '{key}': {type(item)}")


def import_analysis(path):
    print(f'Importing analysis file from: {os.path.basename(path)}.')
    with h5py.File(path, 'r') as h5file:
        # Load the analysis (stored in groups/datasets)
        analysis = read_dict_from_h5(h5file, '/')

        # Load parameters (stored as attributes)
        params = {}
        for key, value in h5file.attrs.items():
            if isinstance(value, np.generic):
                value = value.item()   # Extract the scalar while keeping its NumPy type
            elif key == 'afm_variables' and isinstance(value, np.ndarray):
                value = value.tolist()
            params[key] = value

    return analysis, params


def read_dict_from_h5(h5file, group_path='/'):
    result = {}
    group = h5file[group_path]

    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = read_dict_from_h5(h5file, f"{group_path}/{key}")
        else:
            data = item[()]  # Read dataset normally

            # Convert 'measurement_filenames' to list
            if key == 'measurement_filenames' and isinstance(data, np.ndarray):
                result[key] = [s.decode('utf-8') if isinstance(s, bytes) else s for s in data.tolist()]
            else:
                result[key] = data  # Keep as is for other cases

    # If all keys are numerical, convert dict to list
    if all(k.isdigit() for k in result.keys()):
        sorted_keys = sorted(result.keys(), key=int)  # Sort keys numerically
        result = [result[k] for k in sorted_keys]  # Convert to list

    return result


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


def read_parquet_file(path, image=False):
    """
    Reads in parquet files either as images in matrix format or as a Nx2 list of coordinates with the respective N values.
    """
    df = pd.read_parquet(path, engine='pyarrow')
    if image:
        df = df.sort_values(by=["y", "x"])
        img = df.pivot(index="y", columns="x", values="value_orig").to_numpy()
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
                          for filename in brainfusion_analysis['measurement_filenames']]
    for idx, file_name in enumerate(parquet_file_names):
        parquet_file_path = os.path.join(path, file_name)
        df = pd.read_parquet(parquet_file_path, engine='pyarrow')
        trafo_grid = brainfusion_analysis['measurement_trafo_grids'][idx]

        # ToDo: REMOVE
        trafo_grid = np.zeros((len(df['x_translated']), 2))
        #

        df['x_translated'], df['y_translated'] = trafo_grid[:, 0], trafo_grid[:, 1]
        df.to_parquet(parquet_file_path.removesuffix(".parquet") + '_Trafo' + '.parquet', index=False, engine='pyarrow')