import os

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import AffineTransform

from brainfusion.io import (attach_metadata, parse_name, read_annotation_file, mirror_pixels, annotation_mirror_flags,
                            ANNOTATION_FILENAME)
from brainfusion.load_experiments.base import iter_experiment_folders
from brainfusion.sample import Sample

# The roi-annotator's file sits next to the image it was drawn on
AFM_ANNOTATION_DIR = os.path.join('Pics', 'calibration')


def load_batchforce_all(base_path, afm_variables, batchforce_filename='data.csv',
                        grid_conv_filename='GridInversionMatrix.csv', use_landmarks=True, apply_mirror=True,
                        name_pattern=None, name_converters=None) -> list[Sample]:
    """
    Load every batchforce AFM experiment folder (name containing '#') found directly below `base_path`.

    See `load_batchforce_single` for the other parameters. If `name_pattern` is given, it is matched against
    each folder's name (see `brainfusion.io.parse_name`) and the extracted fields are stored in the sample's
    `.metadata`, e.g. animal number, condition, stage.
    """
    samples = []
    for folder_name, folder_path in iter_experiment_folders(base_path):
        sample = load_batchforce_single(folder_path, afm_variables=afm_variables,
                                        batchforce_filename=batchforce_filename,
                                        grid_conv_filename=grid_conv_filename, use_landmarks=use_landmarks,
                                        apply_mirror=apply_mirror)
        if name_pattern is not None:
            sample = attach_metadata(sample, parse_name(folder_name, name_pattern, name_converters))
        samples.append(sample)
    return samples


def load_batchforce_single(folder_path, afm_variables, batchforce_filename='data.csv',
                           grid_conv_filename='GridInversionMatrix.csv', use_landmarks=True, apply_mirror=True,
                           annotation_file=None) -> Sample:
    """
    Load a single AFM experiment analysed with the Matlab 'batchforce' library, together with the outline (and
    landmarks) annotated with the roi-annotator.

    `afm_variables` is normally a list of column names to read directly out of `batchforce_filename` (one
    column per quantity). Some batchforce versions instead export a long/tidy table with one row per
    (point, quantity) - a 'result_parameter' column naming the quantity (e.g. 'Reduced apparent elastic
    modulus') and a 'result' column holding its value - which is auto-detected and pivoted into one row per
    point. For that format, pass `afm_variables` as a dict mapping the desired `Sample.dataset` key to the
    quantity's `result_parameter` label instead, e.g. `{'modulus': 'Reduced apparent elastic modulus'}`. A per-point
    column of that table which is not a quantity (e.g. 'indent_depth') can be requested in the same way.

    The annotation is 'Pics/calibration/outline_brainfusion.txt' of the experiment, or `annotation_file` if
    given (e.g. the file of another run on the same sample with an identical 'overview.tif'); see
    `brainfusion.io.read_annotation_file`. It must have been drawn on 'overview.tif', the image the data's
    'x_image'/'y_image' refer to. Its (0-based) pixels are converted to the data's Matlab (1-based) pixels.
    With `apply_mirror`, the mirroring noted in its header (so that all brains look the same way) is applied
    to grid, outline, landmarks and background image. `use_landmarks=False` ignores the landmarks.
    """
    folder_name = os.path.basename(os.path.normpath(folder_path))
    variable_map = afm_variables if isinstance(afm_variables, dict) else {name: name for name in afm_variables}

    # --- AFM data (needs 'data.csv' from AFM_BatchMapping.m) ---
    data_path = os.path.join(folder_path, 'region analysis', batchforce_filename)
    assert os.path.exists(data_path), f'The given path does not point to an AFM analysis file: {data_path}'
    assert data_path.endswith('.csv'), "AFM analysis data must be a .csv file (writetable(data, 'data.csv') in Matlab)"

    data = pd.read_csv(data_path)
    if 'result_parameter' in data.columns and 'result' in data.columns:
        wide = data.pivot_table(index=['x_image', 'y_image'], columns='result_parameter', values='result',
                                aggfunc='first').reset_index()
        # plain per-point columns of the table (e.g. 'indent_depth') can be requested like a quantity
        per_point = [label for label in variable_map.values() if label not in wide.columns and label in data.columns]
        extra = ['x', 'y'] if 'x' in data.columns and 'y' in data.columns else []
        if extra or per_point:
            first = data.drop_duplicates(subset=['x_image', 'y_image'])[['x_image', 'y_image'] + extra + per_point]
            wide = wide.merge(first, on=['x_image', 'y_image'], how='left')
        data = wide
    afm_data = {key: np.array(data[label]) for key, label in variable_map.items()}

    # --- Grid in image pixels, and the matrix to micrometers ---
    # The data's pixels are Matlab's (first pixel centre = 1); the annotation is 0-based like numpy. Everything
    # is brought to 0-based here and back to Matlab's convention only for the matrix.
    afm_grid = np.stack((np.array(data['x_image']), np.array(data['y_image'])), axis=-1) - 1

    grid_vars_path = os.path.join(folder_path, grid_conv_filename)
    assert os.path.exists(grid_vars_path), (f'The given path does not point to a grid conversion variables file: '
                                            f'{grid_vars_path}')
    assert grid_vars_path.endswith('.csv'), ("The grid conversion must be a .csv file: writematrix([M [r; s]; 0 0 1], "
                                             "'GridInversionMatrix.csv') in Matlab")
    afm_scale_matrix = pd.read_csv(grid_vars_path, header=None, sep=',').to_numpy()

    # Rotate stage coordinates to preserve the grid orientation in relation to the image
    theta = np.radians(-90)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    afm_scale_matrix = afm_scale_matrix @ rotation_matrix

    # --- Background image ---
    calibration_dir = os.path.join(folder_path, AFM_ANNOTATION_DIR)
    bg_image = np.array(Image.open(os.path.join(calibration_dir, 'overview.tif')).convert('L'))

    # --- Annotation ---
    annotation_file = annotation_file or os.path.join(calibration_dir, ANNOTATION_FILENAME)
    annotation = read_annotation_file(annotation_file)
    assert annotation is not None, f"No roi-annotator file with an outline at {annotation_file}"
    assert annotation['outline'] is not None, f"No [outline] in {annotation_file}"
    annotated_image = annotation['header'].get('image', 'overview.tif')
    assert annotated_image == 'overview.tif', (
        f"{annotation_file}: the outline was drawn on '{annotated_image}', but the grid (x_image/y_image) is in "
        f"pixels of 'overview.tif' - annotate on overview.tif")
    contour = annotation['outline']
    landmarks = annotation['landmarks'] if use_landmarks else None

    if apply_mirror:
        horizontal, vertical = annotation_mirror_flags(annotation)
        shape = bg_image.shape[:2]
        afm_grid = mirror_pixels(afm_grid, shape, horizontal, vertical)
        contour = mirror_pixels(contour, shape, horizontal, vertical)
        if landmarks is not None:
            landmarks = mirror_pixels(landmarks, shape, horizontal, vertical)
        if horizontal:
            bg_image = np.fliplr(bg_image)
        if vertical:
            bg_image = np.flipud(bg_image)

    # --- Image pixels to micrometers (matrix in Matlab pixels: +1) ---
    to_um = AffineTransform(matrix=np.linalg.inv(afm_scale_matrix))
    afm_grid, contour = to_um(afm_grid + 1), to_um(contour + 1)
    if landmarks is not None:
        landmarks = to_um(landmarks + 1)

    return Sample(contour=contour, grid=afm_grid, dataset=afm_data, scale=afm_scale_matrix, landmarks=landmarks,
                  filename=folder_name, bg_image=bg_image)
