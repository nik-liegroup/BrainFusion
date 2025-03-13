import os
import numpy as np
from PIL import Image
import re
import pandas as pd
import h5py
import threading
from brainfusion._io import get_roi_from_txt
from brainfusion._utils import project_brillouin_dataset


def load_brillouin_experiment(folder_path):
    """
    Function to load a Brillouin experiment analysed with the BMicro Python package.
    """
    # Load Brillouin metadata file
    bm_h5_path = os.path.join(folder_path, 'RawData', 'Brillouin.h5')
    bm_metadata, bm_rep_numbers = get_brillouin_metadata(bm_h5_path, 'Brillouin')

    # Load Brillouin data files
    bm_path = os.path.join(folder_path, 'Export')
    bm_data = get_brillouin_data(bm_path)

    # Choose replicate number and apply to metadata and data dicts
    bm_chosen_rep = choose_rep_number(bm_rep_numbers)
    bm_metadata_rep = bm_metadata[bm_chosen_rep]
    bm_data_rep = bm_data[bm_chosen_rep]

    # Load bright-field metadata file
    bf_h5_path = os.path.join(folder_path, 'RawData', 'Brillouin.h5')
    bf_metadata, bf_rep_numbers = get_brillouin_metadata(bf_h5_path, 'Fluorescence')

    # Load bright-field data files
    bf_path = os.path.join(folder_path, 'Plots')
    bf_data = get_brillouin_images(bf_path)

    # Choose replicate number and apply to metadata and data dicts
    bf_chosen_rep = 0
    bf_data_rep = bf_data[bf_chosen_rep]

    # Load brain tissue boundary contour
    contour = get_roi_from_txt(os.path.join(folder_path, 'Plots', 'brain_outline.txt'))

    # Rotate bright-field image
    bf_data_rep = np.fliplr(np.rot90(bf_data_rep, 1))

    # Flip image up-down for imshow function with origin=lower
    bf_data_rep = np.flipud(bf_data_rep)

    # Rotate data grid
    grid = bm_metadata_rep['brillouin_grid']
    rot_grid = grid.copy()
    rot_grid[:, :, :, 0], rot_grid[:, :, :, 1] = grid[:, :, :, 1], grid[:, :, :, 0]
    bm_metadata_rep['brillouin_grid'] = rot_grid

    # Brillouin scale to pix/µm
    assert np.isclose(np.abs(bm_metadata_rep['pixPerMicrometerX'][0, 1]),
                      np.abs(bm_metadata_rep['pixPerMicrometerY'][0, 0]),
                      rtol=1.e-3)
    scale = np.abs(bm_metadata_rep['pixPerMicrometerX'][0, 1])

    # Create 2D Brillouin map from 3D dataset and ravel datasets
    bm_data_rep, bm_metadata_rep = project_brillouin_dataset(bm_data_rep, bm_metadata_rep)

    # Scale Brillouin contour to µm
    contour = contour / scale

    return bm_data_rep, bm_metadata_rep, bf_data_rep, contour, scale


def get_brillouin_metadata(h5_path, data_var):
    """
    Load metadata from Brillouin .h5 file (BMicro).
    """
    assert os.path.exists(h5_path), f'{h5_path} does not exist.'
    assert data_var in ['Brillouin', 'Fluorescence'], (f'{data_var} is not available in h5 file. '
                                                       f'Please choose Brillouin or Fluorescence.')

    # Get number of replicates
    reps = get_h5rep(h5_path, data_var)
    assert isinstance(reps, list) and all(isinstance(rep, int) for rep in reps), (f'Reps {reps} '
                                                                                  f'must be a list of integers.')

    metadata_dict = {}
    with h5py.File(h5_path, 'r') as h5_file:
        for rep in reps:
            # Pixels per micrometer is correct!
            pixPerMicrometerX = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/micrometerToPixX']
            pixPerMicrometerX_x, pixPerMicrometerX_y = pixPerMicrometerX.attrs['x'], pixPerMicrometerX.attrs['y']

            pixPerMicrometerY = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/micrometerToPixY']
            pixPerMicrometerY_x, pixPerMicrometerY_y = pixPerMicrometerY.attrs['x'], pixPerMicrometerY.attrs['y']

            origin = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/origin']
            origin_x, origin_y = origin.attrs['x'], origin.attrs['y']

            stage = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/positionStage']
            stage_x, stage_y = stage.attrs['x'], stage.attrs['y']

            scanner = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/positionScanner']
            scanner_x, scanner_y = scanner.attrs['x'], scanner.attrs['y']

            if data_var == 'Brillouin':
                # Get grid coordinates from Brillouin measurement
                brillouin_grid_x = h5_file[f'Brillouin/{rep}/payload/positions-x'][:]
                brillouin_grid_y = h5_file[f'Brillouin/{rep}/payload/positions-y'][:]
                brillouin_grid_z = h5_file[f'Brillouin/{rep}/payload/positions-z'][:]

                # Shift Brillouin measurement grid to stage centre, bring coordinates in correct order and scale
                # ToDo: Fix the y-translation error!
                shift_y = 1023/2
                brillouin_grid_x = np.transpose(brillouin_grid_x - stage_x, (1, 2, 0))
                brillouin_grid_y = np.transpose(brillouin_grid_y - stage_y + shift_y, (1, 2, 0))
                brillouin_grid_z = np.transpose(brillouin_grid_z - np.min(brillouin_grid_z), (1, 2, 0))  # in µm

                brillouin_grid = np.stack((brillouin_grid_x, brillouin_grid_y, brillouin_grid_z), axis=-1)
            else:
                brillouin_grid = None

            # Create a dictionary for the current replicate
            replicate_metadata = {
                'pixPerMicrometerX': np.stack((pixPerMicrometerX_x, pixPerMicrometerX_y), axis=-1),
                'pixPerMicrometerY': np.stack((pixPerMicrometerY_x, pixPerMicrometerX_y), axis=-1),
                'origin': np.stack((origin_x, origin_y), axis=-1),
                'scanner': np.stack((scanner_x, scanner_y), axis=-1),
                'stage': np.stack((stage_x, stage_y), axis=-1),
                'brillouin_grid': brillouin_grid
            }

            metadata_dict[rep] = replicate_metadata

    return metadata_dict, reps


def get_brillouin_data(bm_path):
    """
    Load data from Brillouin csv file (BMicro).
    """
    assert os.path.exists(bm_path), f'{bm_path} does not exist!'
    pattern = re.compile(r'Brillouin_BMrep(\d+)_(\w+)_slice-(\d+)\.csv')

    bm_dict = {}
    for filename in os.listdir(bm_path):
        # Match the filename pattern
        match = pattern.match(filename)
        if match:
            rep = int(match.group(1))  # Extract the replicate number
            variable = match.group(2)  # Extract the variable name
            slice_num = int(match.group(3))  # Extract the slice number

            # Read the CSV file
            file_path = os.path.join(bm_path, filename)
            data_slice = pd.read_csv(file_path, skiprows=2, header=None).to_numpy()
            data_slice = np.expand_dims(data_slice, axis=-1)  # Shape becomes (rows, cols, 1)

            # Initialize the variable dictionary if it doesn't exist
            if rep not in bm_dict:
                bm_dict[rep] = {}

            # Initialize the variable array if it doesn't exist
            if variable not in bm_dict[rep]:
                bm_dict[rep][variable] = []

            # Append the expanded data slice to the appropriate variable for the replicate
            bm_dict[rep][variable].append(data_slice)

    # Convert lists to arrays along the last dimension for each variable
    for rep in bm_dict:
        for variable in bm_dict[rep]:
            bm_dict[rep][variable] = np.concatenate(bm_dict[rep][variable], axis=-1)

    return bm_dict


def get_brillouin_images(bf_path):
    """
    Load Brillouin background bright-field images.
    """
    pattern = re.compile(r'Brillouin_FLrep(\d+)_channelBrightfield_aligned.png')
    bf_dict = {}
    for filename in os.listdir(bf_path):
        # Match the filename pattern
        match = pattern.match(filename)
        if match:
            rep = int(match.group(1))  # Extract the replicate number

            # Import the png file
            bf_path_join = os.path.join(bf_path, filename)
            bf_img = Image.open(bf_path_join).convert('L')
            bf_img = np.array(bf_img)

            # Initialize the replicate dictionary if it doesn't exist
            if rep not in bf_dict:
                bf_dict[rep] = {}

            # Assign the data slice to the appropriate slice number
            bf_dict[rep] = bf_img

    return bf_dict


def choose_rep_number(rep_numbers):
    """
    Given multiple replicates in a single Brillouin experiment, extract the one to analyse.
    """
    # Select Brillouin replicate if more than one available
    if len(rep_numbers) == 1:
        chosen_rep = int(rep_numbers[0])
    else:
        try:
            user_input = get_user_input(f'Please enter a replicate number from: {rep_numbers} (timeout in 10 seconds): ',
                                        timeout=10)
            if user_input and user_input.isdigit() and int(user_input) in rep_numbers:
                chosen_rep = int(user_input)
                print(f'Valid input received. Script will continue!')
            else:
                raise ValueError("Invalid input or no input provided.")
        except Exception:
            # If user fails to provide valid input, choose the max
            chosen_rep = np.max(rep_numbers)
            print(f'No valid input received. Defaulting to max replicate: {chosen_rep}')

    return chosen_rep


def get_user_input(prompt, timeout=10) -> int:
    user_input = [None]  # To store user input

    def ask_input():
        user_input[0] = input(prompt)

    # Create and start a thread to ask for input
    input_thread = threading.Thread(target=ask_input)
    input_thread.start()

    # Wait for the input thread to finish or timeout
    input_thread.join(timeout)

    # Return the user input if it was provided, otherwise return None
    return user_input[0]


def get_h5rep(h5_path, data_var):
    assert os.path.exists(h5_path), print(f'{h5_path} does not exist.')
    assert data_var in ['Brillouin', 'Fluorescence'], (f'{data_var} is not available in h5 file. '
                                                       f'Please choose Brillouin or Fluorescence.')

    with h5py.File(h5_path, 'r') as h5_file:
        brillouin_group = h5_file[data_var]
        rep_numbers = [int(key) for key in brillouin_group.keys() if key.isdigit()]

    return rep_numbers
