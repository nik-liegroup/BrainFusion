import os
import numpy as np
from PIL import Image
import pandas as pd
import re
from .._utils import read_parquet_file, get_roi_from_txt


def load_afm_brain(folder_path):
    """
    Function to load a simple AFM experiment analysed with the batchforce Matlab library.
    """
    # Load the AFM results csv file and extract data and grid coordinates
    data_path = os.path.join(folder_path, 'region analysis', 'data.csv')
    data = pd.read_csv(data_path)
    afm_data = {'modulus': data['modulus'], 'beta_pyforce': data['beta_pyforce'],
                'k0_pyforce': data['k0_pyforce'], 'k_pyforce': data['k_pyforce']}
    afm_data = {key: np.array(value) for key, value in afm_data.items()}

    # Load AFM grid
    afm_grid = np.stack((np.array(data['x_image']), np.array(data['y_image'])), axis=-1)

    # Load background image
    img_path = os.path.join(folder_path, 'Pics', 'calibration', 'overview.tif')
    img = Image.open(img_path).convert('L')
    img = np.array(img)

    # Load contour
    contour_path = os.path.join(folder_path, 'Pics', 'calibration')

    if os.path.exists(os.path.join(contour_path, 'brain_outline_OriLeft.txt')):
        # Flip BF image and AFM grid
        img = np.flipud(img)
        afm_grid[:, 1] = img.shape[0] - afm_grid[:, 1]

        # Import contour and flip
        roi_path = os.path.join(contour_path, 'brain_outline_OriLeft.txt')
        contour = get_roi_from_txt(roi_path)
        contour[:, 1] = img.shape[0] - contour[:, 1]

    elif os.path.exists(os.path.join(contour_path, 'brain_outline_OriRight.txt')):
        # Import contour
        roi_path = os.path.join(contour_path, 'brain_outline_OriRight.txt')
        contour = get_roi_from_txt(roi_path)
    else:
        print(f'No matching contour was found for {folder_path}!')
        exit()

    # Calculate scaling factor in pix/µm
    scale = data['pix_per_m'][0] * 1e-6

    # Scale AFM grid and contour to µm
    afm_grid = afm_grid / scale
    contour = contour / scale

    return afm_data, afm_grid, img, contour, scale


def load_sc_afm_myelin(folder_path, boundary_filename, key_point_filename=None, rot_axis_filename=None,
                       sampling_size=None):
    """
    Function to load myelin images of multiple spinal cord sections and the corresponding AFM experiment analysed with
    the Matlab library 'batchforce'.
    """
    # Get the experiment number from the folder name
    folder_name = os.path.basename(os.path.normpath(folder_path))
    match = re.search(r'#(\d+)', folder_name)
    exp_num = int(match.group(1)) if match else None

    # Load all myelin parquet filenames
    myelin_i_filenames = [f for f in os.listdir(folder_path) if
                          f'ani{exp_num}' in f and f.endswith("image_roi_linearised.parquet")]

    # Import myelin images and pixel grid coordinates
    myelin_grids, myelin_datasets, myelin_filenames = [], [], []
    for filename in myelin_i_filenames:
        myelin_filenames.append(re.match(r"^(.*?)(?=_Merged_RAW)", filename).group(1))
        image_path = os.path.join(folder_path, filename)
        myelin_grid, myelin_data = read_parquet_file(image_path, False)

        # Randomly sample datasets for faster calculation
        if type(sampling_size) is int:
            print('Attention: Data sampling is activated to improve calculation time. Deactivate for proper analysis!')
            sample_idx = np.random.choice(len(myelin_data), size=sampling_size, replace=False)
            myelin_grid = np.stack((myelin_grid[:, 0][sample_idx], myelin_grid[:, 1][sample_idx]), axis=1)
            myelin_data = myelin_data[sample_idx]

        myelin_grids.append(myelin_grid)
        myelin_datasets.append(myelin_data)

    # Load all contour filenames corresponding to myelin images
    myelin_c_filenames = [f for f in os.listdir(folder_path) if f'ani{exp_num}' in f and
                          f.endswith(boundary_filename + ".txt")]

    # Import contours corresponding to myelin images
    myelin_contours = []
    for index, filename in enumerate(myelin_c_filenames):
        file_path = os.path.join(folder_path, filename)
        myelin_contour = get_roi_from_txt(file_path, delimiter=',')
        myelin_contours.append(myelin_contour)

    # Load the AFM bright-field image used to define the measurement grid
    afm_i_filename = os.path.join(folder_path, f'overview_#{exp_num}_image_roi_linearised.parquet')
    afm_image = read_parquet_file(afm_i_filename, True)

    # Load the AFM results file and extract grid coordinates with data values
    data_path = os.path.join(folder_path, 'data_FAKE_FOR_CODE.csv')  # ToDo: Return to proper naming for correlation
    if os.path.exists(data_path):  # ToDo: Replace with an assert statement once the correlation part is implemented
        data = pd.read_csv(data_path)

        # Extract measurement values
        afm_dataset = {'modulus': data['modulus']}  # Save as dictionary to include additional measurements (e.g. fluidity)
        afm_dataset = {key: np.array(value) for key, value in afm_dataset.items()}

        # Extract AFM grid
        afm_grid = np.stack((np.array(data['x_image']), np.array(data['y_image'])), axis=-1)
    else:
        afm_dataset, afm_grid = None, None
        print('No AFM data file found, continuing without!')

    # Load the contour associated to the AFM measurement
    afm_c_filename = os.path.join(folder_path, f'overview_#{exp_num}_{boundary_filename}.txt')
    afm_contour = get_roi_from_txt(os.path.join(folder_path, afm_c_filename), delimiter=',')

    # To make the boundary matching algorithm more robust, additional information like a landmark point similar on all
    # contours and an axis used to align contours can be included

    # Load all myelin and AFM associated key-points in a list
    myelin_keypoints, afm_keypoint = [], []
    keypoint_idx = 0  # If more than one keypoint is defined, use the first  # ToDo: Allow for multiple key-points
    if type(key_point_filename) is str:
        myelin_p_filenames = [p for p in os.listdir(folder_path) if f'ani{exp_num}' in p and
                              p.endswith(key_point_filename + ".txt")]
        for index, filename in enumerate(myelin_p_filenames):
            file_path = os.path.join(folder_path, filename)
            myelin_keypoint = get_roi_from_txt(file_path, delimiter=',')[keypoint_idx]
            myelin_keypoints.append(myelin_keypoint)

        afm_p_filename = os.path.join(folder_path, f'overview_#{exp_num}_{key_point_filename}.txt')
        afm_keypoint = get_roi_from_txt(os.path.join(folder_path, afm_p_filename), delimiter=',')[keypoint_idx]

    # Load all myelin rotation axes in a list
    myelin_axes, afm_axis = [], []
    if type(rot_axis_filename) is str:
        myelin_r_filenames = [p for p in os.listdir(folder_path) if f'ani{exp_num}' in p and
                              p.endswith(rot_axis_filename + ".txt")]
        for index, filename in enumerate(myelin_r_filenames):
            file_path = os.path.join(folder_path, filename)
            myelin_axis = get_roi_from_txt(file_path, delimiter=',')
            myelin_axes.append(myelin_axis)

        afm_r_filename = os.path.join(folder_path, f'overview_#{exp_num}_{rot_axis_filename}.txt')
        afm_axis = get_roi_from_txt(os.path.join(folder_path, afm_r_filename), delimiter=',')

    # Add AFM data as the first list item
    grids = [afm_grid] + myelin_grids
    datasets = [afm_dataset] + myelin_datasets
    contours = [afm_contour] + myelin_contours
    keypoints = [afm_keypoint] + myelin_keypoints
    axes = [afm_axis] + myelin_axes

    return grids, datasets, contours, keypoints, axes, myelin_filenames, afm_image
