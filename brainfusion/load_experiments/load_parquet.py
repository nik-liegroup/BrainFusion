"""Generic loader for "one folder full of parquet measurements, each with its own outline" experiments."""

import os
import re
from typing import Optional

import numpy as np

from brainfusion.io import get_roi_from_txt, read_parquet_file, attach_metadata, parse_name
from brainfusion.load_experiments.base import list_matching_files
from brainfusion.sample import Sample


def load_parquet_samples(folder_path: str, data_pattern: str, contour_pattern: str,
                         landmarks_pattern: Optional[str] = None, x_var: str = 'x_image', y_var: str = 'y_image',
                         data_var: str = 'value_background_corrected', dataset_key: Optional[str] = None,
                         sampling_size: Optional[int] = None, name_pattern: Optional[str] = None,
                         name_converters: Optional[dict] = None) -> list:
    """
    Load every parquet file in `folder_path` matching the regex `data_pattern`, together with its contour
    (file matching `contour_pattern`) and, if given, its landmark points (file matching `landmarks_pattern`).

    This function doesn't know about any specific lab's file naming - it just needs the regexes that describe
    it - so it works for any "one folder full of parquet measurements, each with its own outline" experiment
    (e.g. myelin sections, or a parquet-based AFM measurement).

    Files for each role are matched in sorted order and paired by position, so `data_pattern`,
    `contour_pattern` and `landmarks_pattern` must each match exactly one file per sample, in that same order.
    If `data_pattern` defines a named group called 'name', that group's value is used as each sample's
    filename/identity (handy for stripping a messy suffix); otherwise the full filename (without extension) is
    used.

    Parameters
    ----------
    x_var, y_var, data_var : str
        Column names to read from each parquet file (see `brainfusion.io.read_parquet_file`).
    dataset_key : str, optional
        Key under which the loaded data is stored in `Sample.dataset` (defaults to `data_var`).
    sampling_size : int, optional
        If given, randomly subsample each dataset to this many points (for a quick test run).
    name_pattern, name_converters
        If `name_pattern` is given, it is matched against each sample's filename and the extracted fields are
        stored in that sample's `.metadata` (see `brainfusion.metadata.parse_name`).
    """
    dataset_key = dataset_key or data_var

    data_filenames = list_matching_files(folder_path, lambda f: re.search(data_pattern, f) is not None)
    contour_filenames = list_matching_files(folder_path, lambda f: re.search(contour_pattern, f) is not None)
    assert len(data_filenames) == len(contour_filenames), (
        f"Found {len(data_filenames)} data file(s) matching '{data_pattern}' but {len(contour_filenames)} "
        f"contour file(s) matching '{contour_pattern}' in {folder_path}.")

    landmarks_filenames = [None] * len(data_filenames)
    if landmarks_pattern is not None:
        landmarks_filenames = list_matching_files(folder_path, lambda f: re.search(landmarks_pattern, f) is not None)
        assert len(landmarks_filenames) == len(data_filenames), (
            f"Found {len(data_filenames)} data file(s) but {len(landmarks_filenames)} landmark file(s) matching "
            f"'{landmarks_pattern}' in {folder_path}.")

    samples = []
    for data_filename, contour_filename, landmarks_filename in zip(data_filenames, contour_filenames,
                                                                    landmarks_filenames):
        grid, data = read_parquet_file(os.path.join(folder_path, data_filename), image=False, x_var=x_var,
                                       y_var=y_var, data_var=data_var)

        if isinstance(sampling_size, int):
            print('Attention: Data sampling is activated to improve calculation time. Deactivate for the real analysis!')
            sample_idx = np.random.choice(len(data), size=sampling_size, replace=False)
            grid, data = grid[sample_idx], data[sample_idx]

        contour = get_roi_from_txt(os.path.join(folder_path, contour_filename), delimiter=',')
        landmarks = (get_roi_from_txt(os.path.join(folder_path, landmarks_filename), delimiter=',')
                    if landmarks_filename is not None else None)

        name_match = re.search(data_pattern, data_filename)
        filename = name_match.groupdict().get('name') or os.path.splitext(data_filename)[0]

        sample = Sample(contour=contour, grid=grid, dataset={dataset_key: data}, landmarks=landmarks,
                        filename=filename)
        if name_pattern is not None:
            sample = attach_metadata(sample, parse_name(filename, name_pattern, name_converters))
        samples.append(sample)

    return samples
