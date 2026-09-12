import os

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import AffineTransform, estimate_transform

from brainfusion.io import get_roi_from_txt, attach_metadata, parse_name
from brainfusion.load_experiments.base import iter_experiment_folders
from brainfusion.sample import Sample


def load_batchforce_all(base_path, afm_variables, batchforce_filename, grid_conv_filename, boundary_filename,
                        landmarks_filename=None, name_pattern=None, name_converters=None,
                        **kwargs) -> list[Sample]:
    """
    Load every batchforce AFM experiment folder (name containing '#') found directly below `base_path`.

    See `load_batchforce_single` for what `landmarks_filename` does. If `name_pattern` is given, it is
    matched against each folder's name (see `brainfusion.io.parse_name`) and the extracted fields are
    stored in the sample's `.metadata`, e.g. animal number, condition, stage.
    """
    samples = []
    for folder_name, folder_path in iter_experiment_folders(base_path):
        sample = load_batchforce_single(folder_path, afm_variables=afm_variables,
                                        batchforce_filename=batchforce_filename,
                                        grid_conv_filename=grid_conv_filename, boundary_filename=boundary_filename,
                                        landmarks_filename=landmarks_filename)
        if name_pattern is not None:
            sample = attach_metadata(sample, parse_name(folder_name, name_pattern, name_converters))
        samples.append(sample)
    return samples


def load_batchforce_single(folder_path, afm_variables, batchforce_filename='data.csv',
                           grid_conv_filename='GridInversionMatrix.csv', boundary_filename='brain_outline',
                           landmarks_filename=None, **kwargs) -> Sample:
    """
    Load a single AFM experiment analysed with the Matlab 'batchforce' library, together with its outline.

    `afm_variables` is normally a list of column names to read directly out of `batchforce_filename` (one
    column per quantity). Some batchforce versions instead export a long/tidy table with one row per
    (point, quantity) - a 'result_parameter' column naming the quantity (e.g. 'Reduced apparent elastic
    modulus') and a 'result' column holding its value - which is auto-detected and pivoted into one row per
    point. For that format, pass `afm_variables` as a dict mapping the desired `Sample.dataset` key to the
    quantity's `result_parameter` label instead, e.g. `{'modulus': 'Reduced apparent elastic modulus'}`.

    If `landmarks_filename` is given, looks for '<landmarks_filename>.txt' next to the outline and stores its
    points as `Sample.landmarks`.
    """
    folder_name = os.path.basename(os.path.normpath(folder_path))
    variable_map = afm_variables if isinstance(afm_variables, dict) else {name: name for name in afm_variables}

    # Load the AFM analysis file
    data_path = os.path.join(folder_path, 'region analysis', batchforce_filename)
    assert os.path.exists(data_path), f'The given path does not point to an AFM analysis file: {data_path}'

    data_extension = os.path.splitext(batchforce_filename)[1]
    if data_extension == '.mat':
        raise ValueError(f"Importing {data_extension} files is not implemented yet, use "
                         f"writetable(data, 'data.csv') in Matlab")
    elif data_extension == '.csv':
        data = pd.read_csv(data_path)
        if 'result_parameter' in data.columns and 'result' in data.columns:
            wide = data.pivot_table(index=['x_image', 'y_image'], columns='result_parameter', values='result',
                                    aggfunc='first').reset_index()
            if 'x' in data.columns and 'y' in data.columns:
                stage_xy = data.drop_duplicates(subset=['x_image', 'y_image'])[['x_image', 'y_image', 'x', 'y']]
                wide = wide.merge(stage_xy, on=['x_image', 'y_image'], how='left')
            data = wide
        afm_data = {key: np.array(data[label]) for key, label in variable_map.items()}
    else:
        raise ValueError(f"{data_extension} files containing AFM analysis data are not supported!")

    # Extract image coordinates
    afm_grid = np.stack((np.array(data['x_image']), np.array(data['y_image'])), axis=-1)

    # Load transformation matrix to scale to stage coordinates (to um)
    grid_vars_path = os.path.join(folder_path, grid_conv_filename)
    assert os.path.exists(grid_vars_path), (f'The given path does not point to a grid conversion variables file: '
                                            f'{grid_vars_path}')

    grid_extension = os.path.splitext(grid_conv_filename)[1]
    if grid_extension == '.csv':
        afm_scale_matrix = pd.read_csv(grid_vars_path, header=None, sep=',').to_numpy()
    else:
        print(f"Importing {grid_extension} files is not implemented yet, use writematrix([M [r; s]; 0 0 1],"
                         f"'GridInversionMatrix.csv') in Matlab to save the full 3x3 conversion matrix."
             f"Estimating transformation matrix instead now.")
        afm_grid_stage = np.stack((np.array(data['x']), np.array(data['y'])), axis=-1)
        afm_scale_matrix = estimate_transform('affine', afm_grid_stage, afm_grid).params

    # Rotate stage coordinates to preserve the grid orientation in relation to the image
    stage_image_angle = -90
    theta = np.radians(stage_image_angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    afm_scale_matrix = afm_scale_matrix @ rotation_matrix

    # Load background image
    img_path = os.path.join(folder_path, 'Pics', 'calibration', 'overview.tif')
    bg_image = np.array(Image.open(img_path).convert('L'))

    # Load contour, flipping it (and the grid/image) if it was defined on the left orientation
    contour_dir = os.path.join(folder_path, 'Pics', 'calibration')
    left_path = os.path.join(contour_dir, f'{boundary_filename}_oriLeft.txt')
    right_path = os.path.join(contour_dir, f'{boundary_filename}_oriRight.txt')

    # Landmarks are just user-annotated points in a fixed order, so unlike the contour they don't need
    # separate left/right files - the orientation detected from the boundary file is enough to know whether
    # to flip them too.
    landmarks = None
    if landmarks_filename is not None:
        landmarks_path = os.path.join(contour_dir, f'{landmarks_filename}.txt')

    if os.path.exists(left_path):
        bg_image = np.flipud(bg_image)
        afm_grid[:, 1] = bg_image.shape[0] - afm_grid[:, 1]
        contour = get_roi_from_txt(left_path)
        contour[:, 1] = bg_image.shape[0] - contour[:, 1]
        if landmarks_filename is not None:
            landmarks = get_roi_from_txt(landmarks_path)
            landmarks[:, 1] = bg_image.shape[0] - landmarks[:, 1]
    elif os.path.exists(right_path):
        contour = get_roi_from_txt(right_path)
        if landmarks_filename is not None:
            landmarks = get_roi_from_txt(landmarks_path)
    else:
        raise ValueError(f"No matching contour was found for {folder_path}!\n"
                         f"Make sure filename is of type: '<boundary_filename>_OriRight.txt' or "
                         f"'<boundary_filename>_OriLeft.txt'")

    # Transform coordinates from image pixels to micrometers
    aff = AffineTransform(matrix=np.linalg.inv(afm_scale_matrix))
    afm_grid = aff(afm_grid)
    contour = aff(contour)
    if landmarks is not None:
        landmarks = aff(landmarks)

    return Sample(contour=contour, grid=afm_grid, dataset=afm_data, scale=afm_scale_matrix, landmarks=landmarks,
                 filename=folder_name, bg_image=bg_image)
