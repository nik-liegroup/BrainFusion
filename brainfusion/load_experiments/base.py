"""Shared helpers reused across the individual experiment loaders."""

import os
import re
from typing import Iterator, Optional, Tuple

import numpy as np

from brainfusion.io import get_roi_from_txt, read_parquet_file
from brainfusion.metadata import attach_metadata, parse_name
from brainfusion.sample import Sample


def iter_experiment_folders(base_path: str, marker: str = '#') -> Iterator[Tuple[str, str]]:
    """
    Yield (folder_name, folder_path) for every subdirectory of `base_path` whose name contains `marker`.

    Folders are visited in sorted order so that repeated runs process samples in the same order -
    this matters because the first sample in the list is used as the alignment template.
    """
    for folder_name in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path) and marker in folder_name:
            yield folder_name, folder_path


def list_matching_files(folder_path: str, predicate) -> list:
    """Return sorted filenames directly inside `folder_path` for which `predicate(filename)` is True."""
    return sorted(f for f in os.listdir(folder_path) if predicate(f))


def load_parquet_images(folder_path: str, data_pattern: str, contour_pattern: str,
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


def load_template_sample(contour_path: str, delimiter: str = ',', dataset: Optional[dict] = None,
                         landmarks_path: Optional[str] = None, filename: Optional[str] = None) -> Sample:
    """
    Load a fixed template `Sample` from a contour file, for aligning measurements onto a pre-defined shape
    instead of an average or first-element sample.

    `dataset`, if given, lets the template carry its own spatial data (e.g. a labelled anatomical-region map)
    that measurement grids get compared against once aligned. `landmarks_path`, if given, is loaded as the
    template's `Sample.landmarks`.
    """
    contour = get_roi_from_txt(contour_path, delimiter=delimiter)
    landmarks = get_roi_from_txt(landmarks_path, delimiter=delimiter) if landmarks_path is not None else None
    return Sample(contour=contour, dataset=dataset, landmarks=landmarks,
                 filename=filename or os.path.splitext(os.path.basename(contour_path))[0])
