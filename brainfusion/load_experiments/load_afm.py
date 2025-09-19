import os
import numpy as np
from PIL import Image
import pandas as pd
import re
from skimage.transform import AffineTransform, estimate_transform
from brainfusion._io import read_parquet_file, get_roi_from_txt


def load_batchforce_all(base_path, afm_variables, batchforce_filename, key_point_filename, rot_axis_filename,
                        grid_conv_filename, boundary_filename, **kwargs):
    """
    Load multiple batchforce experiments in a directory.
    """
    filenames, grids, scale_matrices, datasets, contours, points, axes, bg_images_list = [], [], [], [], [], [], [], []
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path) and ('#' in folder_name):
            # Load data from experiment folder
            grid, scale_matrix, dataset, contour, point, axis, background_image = load_batchforce_single(
                folder_path,
                afm_variables=afm_variables,
                batchforce_filename=batchforce_filename,
                key_point_filename=key_point_filename,
                rot_axis_filename=rot_axis_filename,
                grid_conv_filename=grid_conv_filename,
                boundary_filename=boundary_filename)

            # Save imported data
            grids.append(grid)
            scale_matrices.append(scale_matrix)
            datasets.append(dataset)
            contours.append(contour)
            points.append(point)
            axes.append(axis)
            filenames.append(folder_name)
            bg_images_list.append(background_image)

    results = {"grids": grids,
               "reg_grid_dims": "None",
               "scales": scale_matrices,
               "datasets": datasets,
               "contours": contours,
               "points": points,
               "axes": axes,
               "filenames": filenames,
               "bg_images": bg_images_list}

    return results


def load_batchforce_single(folder_path, afm_variables, batchforce_filename='data.csv', key_point_filename="None",
                           rot_axis_filename="None", grid_conv_filename='GridInversionMatrix.csv',
                           boundary_filename='brain_outline', stage_image_angle=-90):
    """
    Load an AFM experiment analysed with the batchforce Matlab library and the outline coordinates.
    """
    # Load the AFM analysis file
    data_path = os.path.join(folder_path, 'region analysis', batchforce_filename)
    assert os.path.exists(data_path), f'The given path does not point to a AFM analysis file: {data_path}'

    # Get filetype extension
    extension = os.path.splitext(batchforce_filename)[1]
    if extension == '.mat':
        raise ValueError(f"Importing {extension} files is not implemented yet, use writetable(data, 'data.csv') in Matlab")
    elif extension == '.csv':
        data = pd.read_csv(data_path)
        afm_data = {i: np.array(data[i]) for i in afm_variables}
    else:
        raise ValueError(f"{extension} files containing AFM analysis data are not supported!")

    # Extract image coordinates
    afm_grid = np.stack((np.array(data['x_image']), np.array(data['y_image'])), axis=-1)

    # Load transformation matrix to scale to stage coordinates (to µm)
    grid_vars_path = os.path.join(folder_path, grid_conv_filename)
    assert os.path.exists(grid_vars_path), f'The given path does not point to a grid conversion variables file: {grid_vars_path}'

    # Get filetype extension
    extension = os.path.splitext(batchforce_filename)[1]
    if extension == '.mat':
        raise ValueError(f"Importing {extension} files is not implemented yet, use writematrix([M [r; s]; 0 0 1],"
                         f"'GridInversionMatrix.csv') in Matlab to save full 3x3 conversion matrix")
    elif extension == '.csv':
        df = pd.read_csv(grid_vars_path, header=None, sep=',')
        afm_scale_matrix = df.to_numpy()  # Transforms from stage coordinates to image coordinates
    else:
        print(f"{extension} files containing AFM analysis data are not supported! Estimating transformation matrix.")
        afm_grid_stage = np.stack((np.array(data['x']), np.array(data['y'])), axis=-1)
        afm_scale_matrix = estimate_transform('affine', afm_grid_stage, afm_grid).params

    # Rotate stage coordinates to preserves the grid orientation in relation to image
    theta = np.radians(stage_image_angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    afm_scale_matrix = afm_scale_matrix @ rotation_matrix

    # Load background image
    img_path = os.path.join(folder_path, 'Pics', 'calibration', 'overview.tif')
    img = Image.open(img_path).convert('L')
    img = np.array(img)

    # Load contour
    contour_path = os.path.join(folder_path, 'Pics', 'calibration')

    if os.path.exists(os.path.join(contour_path, f'{boundary_filename}_OriLeft.txt')):
        # Flip BF image and AFM grid
        img = np.flipud(img)
        afm_grid[:, 1] = img.shape[0] - afm_grid[:, 1]

        # Import contour and flip
        roi_path = os.path.join(contour_path, f'{boundary_filename}_OriLeft.txt')
        contour = get_roi_from_txt(roi_path)
        contour[:, 1] = img.shape[0] - contour[:, 1]

    elif os.path.exists(os.path.join(contour_path, f'{boundary_filename}_OriRight.txt')):
        # Import contour
        roi_path = os.path.join(contour_path, f'{boundary_filename}_OriRight.txt')
        contour = get_roi_from_txt(roi_path)
    else:
        raise ValueError(f"No matching contour was found for {folder_path}\n!"
                         f"Make sure filename is of type: '<boundary_filename>_OriRight.txt' or"
                         f" '<boundary_filename>_OriLeft.txt'")

    # ToDo: Implement
    point = None
    axis = None

    # Transform coordinates from image to micro meter
    inv_matrix = np.linalg.inv(afm_scale_matrix)
    aff = AffineTransform(matrix=inv_matrix)
    afm_grid = aff(afm_grid)
    contour = aff(contour)

    return afm_grid, afm_scale_matrix, afm_data, contour, point, axis, img


def load_sc_afm_myelin(folder_path, boundary_filename, key_point_filename=None, rot_axis_filename=None,
                       sampling_size=None, **kwargs):
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
        myelin_datasets.append({"myelin_intensity": myelin_data})

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
    # ToDo: Make choice of points more robust for multiple key-points
    myelin_keypoints, afm_keypoint = [], []
    if key_point_filename != "None":
        myelin_p_filenames = [p for p in os.listdir(folder_path) if f'ani{exp_num}' in p and
                              p.endswith(key_point_filename + ".txt")]
        for index, filename in enumerate(myelin_p_filenames):
            file_path = os.path.join(folder_path, filename)
            myelin_keypoint_list = get_roi_from_txt(file_path, delimiter=',')
            myelin_keypoint = min(myelin_keypoint_list, key=lambda p: p[1])
            myelin_keypoints.append(myelin_keypoint)

        afm_p_filename = os.path.join(folder_path, f'overview_#{exp_num}_{key_point_filename}.txt')
        afm_keypoint_list = get_roi_from_txt(os.path.join(folder_path, afm_p_filename), delimiter=',')
        afm_keypoint = min(afm_keypoint_list, key=lambda p: p[1])

    # Load all myelin rotation axes in a list
    myelin_axes, afm_axis = [], []
    if rot_axis_filename != "None":
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

    scale_matrices = "None"

    results = {"grids": grids,
               "scales": scale_matrices,
               "datasets": datasets,
               "contours": contours,
               "points": keypoints,
               "axes": axes,
               "filenames": myelin_filenames,
               "bg_images": afm_image}

    return results


def load_salini_afm(base_path, boundary_filename, key_point_filename=None, rot_axis_filename=None,
                    sampling_size=None, **kwargs):
    """
    Function to load AFM experiments analysed with the Matlab library 'batchforce'.
    """
    foldernames, grids, scale_matrices, datasets, contours, points, axes, bg_images_list = [], [], [], [], [], [], [], []

    # Iterate over experiments
    experiment_folders = [f for f in os.listdir(base_path) if f'#' in f]
    for folder_name in experiment_folders:

        # Get the experiment number from the folder name
        match = re.search(r'#(\d+)', folder_name)
        exp_num = int(match.group(1)) if match else None

        # Load parquet data file
        parquet_name = re.sub(r"_(left|right)$", r"_afm_measurements_fortranslation_\1", folder_name)
        parquet_path = os.path.join(base_path, folder_name, f"{parquet_name}.parquet")
        if os.path.exists(parquet_path):
            grid, data = read_parquet_file(parquet_path, False, x_var='x_image', y_var='y_image', data_var="modulus")
            dataset = {"modulus": data}
        else:
            grid, dataset = None, None

        # Load contour file
        contour_name = re.sub(r"_(left|right)$", fr"_{boundary_filename}_\1", folder_name)
        contour_path = os.path.join(base_path, folder_name, f"{contour_name}.txt")
        contour = get_roi_from_txt(contour_path, delimiter=',')

        # Save imported data
        foldernames.append(folder_name)
        grids.append(grid)
        datasets.append(dataset)
        contours.append(contour)

        # To make the boundary matching algorithm more robust, additional information like a landmark point similar on all
        # contours and an axis used to align contours can be included
        # Load keypoint
        if key_point_filename != "None":
            keypoint_name = re.sub(r"_(left|right)$", fr"_{key_point_filename}_\1", folder_name)
            keypoint_path = os.path.join(base_path, folder_name, f"{keypoint_name}.txt")
            keypoint = get_roi_from_txt(keypoint_path, delimiter=',')
            keypoint = min(keypoint, key=lambda p: p[1])
            points.append(keypoint)
        else:
            points.append(None)

        # Load axis
        if rot_axis_filename != "None":
            axis_name = re.sub(r"_(left|right)$", fr"_{rot_axis_filename}_\1", folder_name)
            axis_path = os.path.join(base_path, folder_name, f"{axis_name}.txt")
            axis = get_roi_from_txt(axis_path, delimiter=',')
            axes.append(axis)
        else:
            axes.append(None)

    # Load target
    folder_name = "Saliani_2019_mC6_left"
    contour_name = re.sub(r"_(left|right)$", fr"_{boundary_filename}_\1", folder_name)
    contour_path = os.path.join(base_path, folder_name, f"{contour_name}.txt")
    target_contour = get_roi_from_txt(contour_path, delimiter=',')

    keypoint_name = re.sub(r"_(left|right)$", fr"_{key_point_filename}_\1", folder_name)
    keypoint_path = os.path.join(base_path, folder_name, f"{keypoint_name}.txt")
    target_keypoint = get_roi_from_txt(keypoint_path, delimiter=',')

    axis_name = re.sub(r"_(left|right)$", fr"_{rot_axis_filename}_\1", folder_name)
    axis_path = os.path.join(base_path, folder_name, f"{axis_name}.txt")
    target_axis = get_roi_from_txt(axis_path, delimiter=',')

    # Use Salini atlas data as the first list item
    grids = [None] + grids
    datasets = [None] + datasets
    contours = [target_contour] + contours
    points = [target_keypoint] + points
    axes = [target_axis] + axes
    scale_matrices = None
    bg_images_list = None
    reg_grid_dims = None

    results = {"grids": grids,
               "reg_grid_dims": reg_grid_dims,
               "scales": scale_matrices,
               "datasets": datasets,
               "contours": contours,
               "points": points,
               "axes": axes,
               "filenames": foldernames,
               "bg_images": bg_images_list}

    return results