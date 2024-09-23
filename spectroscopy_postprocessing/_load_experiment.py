import os
import numpy as np
from PIL import Image
import re
import pandas as pd
from _utilis import get_h5metadata, get_user_input


def choose_rep_number(rep_numbers):
    # Select Brillouin replicate if more than one available
    chosen_rep = 0
    if len(rep_numbers) == 1:
        chosen_rep = int(rep_numbers[0])
    else:
        try:
            user_input = get_user_input(f'Please enter a replicate number from: {rep_numbers} '
                                        f'(timeout in 10 seconds): ', timeout=10)
            if user_input and user_input.isdigit() and int(user_input) in rep_numbers:
                chosen_rep = int(user_input)
                print(f'Valid input received. Script will continue!')
            else:
                raise ValueError("Invalid input or no input provided.")
        except:
            # If user fails to provide valid input, choose the max
            chosen_rep = np.max(rep_numbers)
            print(f'No valid input received. Defaulting to max replicate: {chosen_rep}')

    return chosen_rep


def get_brillouin_data(bm_path):
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


def get_brightfield_data(bf_path):
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


def get_mask(folder_path):
    mask_path = os.path.join(folder_path, 'Plots', 'brain_outline.png')
    if os.path.exists(mask_path):
        mask = Image.open(mask_path).convert('L')  # Load mask in grayscale
        mask = np.array(mask) / 255.0  # Normalize mask (1 inside ROI, 0 outside)
        return mask
    else:
        print(f'No mask was found for {folder_path}. Continuing without.')
        return None


def load_brillouin_experiment(folder_path):
    # Load Brillouin metadata file
    bm_h5_path = os.path.join(folder_path, 'RawData', 'Brillouin.h5')
    bm_metadata, bm_rep_numbers = get_h5metadata(bm_h5_path, 'Brillouin')

    # Load Brillouin data files
    bm_path = os.path.join(folder_path, 'Export')
    bm_data = get_brillouin_data(bm_path)

    # Choose replicate number and apply to metadata and data dicts
    bm_chosen_rep = choose_rep_number(bm_rep_numbers)
    bm_metadata_rep = bm_metadata[bm_chosen_rep]
    bm_data_rep = bm_data[bm_chosen_rep]

    # Load bright-field metadata file
    bf_h5_path = os.path.join(folder_path, 'RawData', 'Brillouin.h5')
    bf_metadata, bf_rep_numbers = get_h5metadata(bf_h5_path, 'Fluorescence')

    # Load bright-field data files
    bf_path = os.path.join(folder_path, 'Plots')
    bf_data = get_brightfield_data(bf_path)

    # Choose replicate number and apply to metadata and data dicts
    bf_chosen_rep = 0
    bf_metadata_rep = bf_metadata[bf_chosen_rep]
    bf_data_rep = bf_data[bf_chosen_rep]

    # Load mask
    mask = get_mask(folder_path)

    return bm_data_rep, bm_metadata_rep, bf_data_rep, bf_metadata_rep, mask


def load_afm_experiment(folder_path):
    # Load the CSV data file and background image
    data_path = os.path.join(folder_path, 'region analysis', 'data.csv')
    data = pd.read_csv(data_path)

    img_path = os.path.join(folder_path, 'Pics', 'calibration', 'overview.tif')
    img = Image.open(img_path).convert('L')
    img = np.array(img)

    # Load mask
    mask_path = os.path.join(folder_path, 'Pics', 'calibration')
    orientation = []
    if os.path.exists(os.path.join(mask_path, 'brain_outline_OriLeft.png')):
        orientation.append('left')
        mask_path = os.path.join(mask_path, 'brain_outline_OriLeft.png')

    elif os.path.exists(os.path.join(mask_path, 'brain_outline_OriRight.png')):
        orientation.append('right')
        mask_path = os.path.join(mask_path, 'brain_outline_OriRight.png')
    else:
        print(f'No matching mask was found for {folder_path}!')
        exit()

    mask = Image.open(mask_path).convert('L')  # Load mask in grayscale
    mask = np.array(mask) / 255.0  # Normalize mask (1 inside ROI, 0 outside)

    return data, img, mask