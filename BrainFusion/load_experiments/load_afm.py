import os
import numpy as np
from PIL import Image
import pandas as pd
import re
from .._utils import get_roi_from_txt


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


def load_afm_spinalcord(folder_path):
    """
    Function to load an AFM experiment analysed with the batchforce Matlab library to correlate stiffness values with
    imaging data.
    """

    # Get the experiment number from the folder name
    folder_name = os.path.basename(os.path.normpath(folder_path))
    match = re.search(r'#(\d+)', folder_name)
    exp_num = int(match.group(1)) if match else None

    # Load all myelin data in a list
    myelin_i_filenames = [f for f in os.listdir(folder_path) if
                          'ani' in f and f.endswith("_image_roi_linearised.parquet")]
    myelin_grids, myelin_datasets = [], []
    for filename in myelin_i_filenames:
        image_path = os.path.join(folder_path, filename)
        meyelin_grid, myelin_data = read_parquet_file(image_path, False)
        if 'right' in filename:
            pass
        myelin_grids.append(meyelin_grid)
        myelin_datasets.append(myelin_data)

    # Load all myelin contours in a list
    myelin_c_filenames = [f for f in os.listdir(folder_path) if 'ani' in f and f.endswith("_whitematter_outline.txt")]
    myelin_contours = []
    for index, filename in enumerate(myelin_c_filenames):
        file_path = os.path.join(folder_path, filename)
        myelin_contour = get_roi_from_txt(file_path, delimiter=',')
        if 'right' in filename:
            pass
        myelin_contours.append(myelin_contour)

    # Load the template AFM image
    afm_i_filename = os.path.join(folder_path, f'overview_#{exp_num}_image_roi_linearised.parquet')
    afm_image = read_parquet_file(afm_i_filename, True)
    if 'right' in afm_i_filename:
        pass

    # Load the template AFM contour
    afm_c_filename = os.path.join(folder_path, f'overview_#{exp_num}_whitematter_outline.txt')
    afm_contour = get_roi_from_txt(os.path.join(folder_path, afm_c_filename), delimiter=',')
    if 'right' in afm_c_filename:
        pass

    # Load the AFM results csv file and extract data and grid coordinates
    data_path = os.path.join(folder_path, 'data.csv')
    data = pd.read_csv(data_path)
    afm_data = {'modulus': data['modulus']}
    afm_data = {key: np.array(value) for key, value in afm_data.items()}

    # Load AFM grid
    afm_grid = np.stack((np.array(data['x_image']), np.array(data['y_image'])), axis=-1)
    if 'right' in afm_c_filename:
        pass

    # Calculate scaling factor in pix/µm
    afm_scale = data['pix_per_m'][0] * 1e-6
    myelin_scale = 4  #ToDo: FIX

    # Scale AFM / Myelin contours and grids to µm
    afm_grid = afm_grid / afm_scale
    afm_contour = afm_contour / afm_scale
    myelin_grids = [grid / myelin_scale for grid in myelin_grids]
    myelin_contours = [contour / myelin_scale for contour in myelin_contours]

    return myelin_grids, myelin_datasets, myelin_contours, afm_image, afm_contour, afm_data, afm_grid, afm_scale, myelin_scale


def read_parquet_file(image_path, image=False):
    df = pd.read_parquet(image_path, engine='pyarrow')
    if image:
        img = df.pivot(index="y", columns="x", values="value_background_corrected").to_numpy()
        return img
    else:
        grid = df[['x', 'y']].to_numpy()
        data = df[['value_background_corrected']].to_numpy().ravel()
        return grid, data
