"""Shared helpers reused across the individual experiment loaders."""

import os
from typing import Iterator, Optional, Tuple

from brainfusion.io import get_roi_from_txt
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
