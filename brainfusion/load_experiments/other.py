"""
Loaders that don't (yet) fit the general "one function per method" shape the rest of this package is built
around - usually because they're tied to one specific experiment's non-standard file layout, or bundle two
concerns (loading + a specific templating choice) that the general system now handles via composition in the
calling script instead (see `brainfusion.load_experiments.base.load_template_sample`,
`brainfusion.load_experiments.load_parquet.load_parquet_samples`).

Not re-exported from `brainfusion` or `brainfusion.load_experiments` - import directly from this module.
"""

import os
import re

import numpy as np
import pandas as pd

from brainfusion.io import get_roi_from_txt, read_parquet_file, attach_metadata, parse_name
from brainfusion.load_experiments.base import iter_experiment_folders, load_template_sample
from brainfusion.sample import Sample


def load_sc_afm_single(folder_path, boundary_filename, landmarks_filename=None, name_pattern=None,
                       name_converters=None, **kwargs) -> Sample:
    """
    Load one spinal cord AFM measurement (the 'batchforce' part of the SC/myelin experiments).

    This is a stopgap: it reads a plain 'data_FAKE_FOR_CODE.csv' rather than the standard batchforce layout
    that `load_batchforce_single` expects (see ToDo below) - fix that and this function can likely be
    replaced by `load_batchforce_single`. It exists separately from the myelin image loading (now
    `brainfusion.load_experiments.load_parquet.load_parquet_samples`) so the two can be composed as needed -
    e.g. one AFM sample as the alignment template for that same animal's myelin sections.

    If `name_pattern` is given, it is matched against the folder's name and the extracted fields are stored
    in the sample's `.metadata` (see `brainfusion.metadata.parse_name`).
    """
    folder_name = os.path.basename(os.path.normpath(folder_path))
    match = re.search(r'#(\d+)', folder_name)
    exp_num = int(match.group(1)) if match else None

    # Load the AFM bright-field image used to define the measurement grid
    bg_image = read_parquet_file(os.path.join(folder_path, f'overview_#{exp_num}_image_roi_linearised.parquet'),
                                 image=True)

    # Load the AFM results file and extract grid coordinates with data values
    data_path = os.path.join(folder_path, 'data_FAKE_FOR_CODE.csv')  # ToDo: Return to proper naming for correlation
    if os.path.exists(data_path):  # ToDo: Replace with an assert statement once the correlation part is implemented
        data = pd.read_csv(data_path)
        dataset = {'modulus': np.array(data['modulus'])}
        grid = np.stack((np.array(data['x_image']), np.array(data['y_image'])), axis=-1)
    else:
        dataset, grid = None, None
        print('No AFM data file found, continuing without!')

    # Load the contour and optional landmark points associated with the AFM measurement
    contour = get_roi_from_txt(os.path.join(folder_path, f'overview_#{exp_num}_{boundary_filename}.txt'),
                               delimiter=',')

    landmarks = None
    if landmarks_filename is not None:
        landmarks_path = os.path.join(folder_path, f'overview_#{exp_num}_{landmarks_filename}.txt')
        landmarks = get_roi_from_txt(landmarks_path, delimiter=',')

    sample = Sample(contour=contour, grid=grid, dataset=dataset, landmarks=landmarks, bg_image=bg_image,
                    filename=folder_name)
    if name_pattern is not None:
        sample = attach_metadata(sample, parse_name(folder_name, name_pattern, name_converters))
    return sample


def load_salini_afm(base_path, boundary_filename, landmarks_filename=None, name_pattern=None, name_converters=None,
                    **kwargs) -> list[Sample]:
    """
    Load AFM experiments analysed with the Matlab library 'batchforce', aligned to the Saliani 2019 atlas.

    If `landmarks_filename` is given, its points (matched by position, including for the atlas target) are
    stored as each sample's `Sample.landmarks` and used directly in the affine pre-alignment step. If
    `name_pattern` is given, it is matched against each folder's name and the extracted fields are stored
    in that sample's `.metadata` (see `brainfusion.metadata.parse_name`). The atlas target sample is not
    matched against `name_pattern` since it isn't one of the experiment folders.
    """
    samples = []
    for folder_name, folder_path in iter_experiment_folders(base_path):
        parquet_name = re.sub(r"_(left|right)$", r"_afm_measurements_fortranslation_\1", folder_name)
        parquet_path = os.path.join(folder_path, f"{parquet_name}.parquet")
        if os.path.exists(parquet_path):
            grid, data = read_parquet_file(parquet_path, image=False, x_var='x_image', y_var='y_image',
                                           data_var="modulus")
            dataset = {"modulus": data}
        else:
            grid, dataset = None, None

        contour_name = re.sub(r"_(left|right)$", fr"_{boundary_filename}_\1", folder_name)
        contour = get_roi_from_txt(os.path.join(folder_path, f"{contour_name}.txt"), delimiter=',')

        # To make the boundary matching algorithm more robust, corresponding landmark points (matched by
        # position across samples) can be provided and are used directly in the affine pre-alignment step
        landmarks = None
        if landmarks_filename is not None:
            landmarks_name = re.sub(r"_(left|right)$", fr"_{landmarks_filename}_\1", folder_name)
            landmarks = get_roi_from_txt(os.path.join(folder_path, f"{landmarks_name}.txt"), delimiter=',')

        sample = Sample(contour=contour, grid=grid, dataset=dataset, landmarks=landmarks, filename=folder_name)
        if name_pattern is not None:
            sample = attach_metadata(sample, parse_name(folder_name, name_pattern, name_converters))
        samples.append(sample)

    # Load the atlas target contour as the alignment template (first element)
    target_folder = "Saliani_2019_mC6_left"
    contour_name = re.sub(r"_(left|right)$", fr"_{boundary_filename}_\1", target_folder)
    contour_path = os.path.join(base_path, target_folder, f"{contour_name}.txt")

    landmarks_path = None
    if landmarks_filename is not None:
        landmarks_name = re.sub(r"_(left|right)$", fr"_{landmarks_filename}_\1", target_folder)
        landmarks_path = os.path.join(base_path, target_folder, f"{landmarks_name}.txt")

    target_sample = load_template_sample(contour_path, landmarks_path=landmarks_path, filename=target_folder)

    return [target_sample] + samples
