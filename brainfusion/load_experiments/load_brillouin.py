import os

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import AffineTransform

from brainfusion.io import get_roi_from_txt, attach_metadata, parse_name
from brainfusion.load_experiments.base import iter_experiment_folders, list_matching_files
from brainfusion.sample import Sample


def load_brillouin_all(base_path, brillouin_variables, data_filename=None, transform_filename=None,
                       boundary_filename='brain_outline', landmarks_filename=None, name_pattern=None,
                       name_converters=None, **kwargs) -> list[Sample]:
    """
    Load every Brillouin experiment folder (name containing '#') found directly below `base_path`.

    See `load_brillouin_experiment` for the other parameters. If `name_pattern` is given, it is matched
    against each folder's name (see `brainfusion.io.parse_name`) and the extracted fields are stored in
    the sample's `.metadata`.
    """
    samples = []
    for folder_name, folder_path in iter_experiment_folders(base_path):
        sample = load_brillouin_experiment(folder_path, brillouin_variables, data_filename=data_filename,
                                           transform_filename=transform_filename,
                                           boundary_filename=boundary_filename,
                                           landmarks_filename=landmarks_filename)
        if name_pattern is not None:
            sample = attach_metadata(sample, parse_name(folder_name, name_pattern, name_converters))
        samples.append(sample)
    return samples


def load_brillouin_experiment(folder_path, brillouin_variables, data_filename=None, transform_filename=None,
                              boundary_filename='brain_outline', landmarks_filename=None, name_pattern=None,
                              name_converters=None, **kwargs) -> Sample:
    """
    Load a Brillouin experiment exported by bmlab's combined-CSV exporter, together with the outline drawn
    on one of its overview/fluorescence images.

    Expects, directly below `folder_path`:
    - 'Export/<...>_BMrep<n>_data.csv': one row per measured grid point, with 'x'/'y' stage position (um)
      columns plus one column per key requested in `brillouin_variables` ('#'-prefixed metadata lines are
      skipped automatically). If `data_filename` is None, the sole '*_data.csv' file in 'Export' is used;
      pass an explicit filename if a folder can contain more than one (e.g. several repetitions). The 'z'
      column is always included in `Sample.dataset` too - it isn't a stacking dimension (grid alignment stays
      purely 2D, on 'x'/'y'), just each point's own z offset, so it rides through like any other channel.
    - 'Plots/TransformMatrices/<image_stem>_transform.csv': the 3x3 matrix mapping a stage position (x, y,
      in um - the same coordinates as the data CSV's 'x'/'y' columns) onto that image's own pixel coordinates
      ([col, row, 1] = M @ [x_um, y_um, 1]) - the same layout as this package's AFM 'GridInversionMatrix.csv'.
      If `transform_filename` is None, the sole '*_transform.csv' file in that folder is used. The matching
      image (same stem, in 'Plots/') is loaded as the sample's background image.
    - 'Plots/<boundary_filename>.txt': the outline, drawn in that same image's pixel coordinates.
    - if `landmarks_filename` is given, 'Plots/<landmarks_filename>.txt': matching landmark points, also
      drawn in that image's pixel coordinates.

    If `name_pattern` is given, it is matched against the folder's name and the extracted fields are stored
    in the sample's `.metadata` (see `brainfusion.io.parse_name`).
    """
    folder_name = os.path.basename(os.path.normpath(folder_path))

    # --- Brillouin data: grid (stage position, um) plus the requested variables ---
    export_dir = os.path.join(folder_path, 'Export')
    if data_filename is None:
        candidates = list_matching_files(export_dir, lambda f: f.endswith('_data.csv'))
        assert len(candidates) == 1, (
            f"Expected exactly one '*_data.csv' file in {export_dir}, found {len(candidates)}: {candidates}. "
            f"Pass `data_filename` explicitly to pick one.")
        data_filename = candidates[0]

    data = pd.read_csv(os.path.join(export_dir, data_filename), comment='#')
    grid = np.stack((np.array(data['x']), np.array(data['y'])), axis=-1)
    dataset = {variable: np.array(data[variable]) for variable in brillouin_variables}
    # 'z' isn't a stacking dimension here - each grid point just has its own z offset (e.g. from surface
    # unevenness), not a shared discrete plane - so it rides along as an ordinary data channel like any other
    # requested variable, warped/interpolated/averaged the same way, rather than needing any special handling.
    dataset.setdefault('z', np.array(data['z']))

    # --- Transform matrix (stage um -> image pixels) and the background image it belongs to ---
    transform_dir = os.path.join(folder_path, 'Plots', 'TransformMatrices')
    if transform_filename is None:
        candidates = list_matching_files(transform_dir, lambda f: f.endswith('_transform.csv'))
        assert len(candidates) == 1, (
            f"Expected exactly one '*_transform.csv' file in {transform_dir}, found {len(candidates)}: "
            f"{candidates}. Pass `transform_filename` explicitly to pick one.")
        transform_filename = candidates[0]

    transform_matrix = pd.read_csv(os.path.join(transform_dir, transform_filename), header=None).to_numpy()

    image_dir = os.path.join(folder_path, 'Plots')
    image_stem = transform_filename.removesuffix('_transform.csv')
    image_candidates = list_matching_files(image_dir, lambda f: os.path.splitext(f)[0] == image_stem)
    assert len(image_candidates) == 1, (
        f"Expected exactly one image named '{image_stem}.*' in {image_dir}, found {len(image_candidates)}: "
        f"{image_candidates}.")
    bg_image = np.array(Image.open(os.path.join(image_dir, image_candidates[0])))

    # --- Contour and optional landmarks, drawn on that same image, transformed into the data's stage (um)
    # coordinates so they line up with `grid` above ---
    aff = AffineTransform(matrix=np.linalg.inv(transform_matrix))
    contour = aff(get_roi_from_txt(os.path.join(folder_path, 'Plots', f'{boundary_filename}.txt')))

    landmarks = None
    if landmarks_filename is not None:
        landmarks_path = os.path.join(folder_path, 'Plots', f'{landmarks_filename}.txt')
        landmarks = aff(get_roi_from_txt(landmarks_path))

    sample = Sample(contour=contour, grid=grid, dataset=dataset, scale=transform_matrix, landmarks=landmarks,
                    bg_image=bg_image, filename=folder_name)
    if name_pattern is not None:
        sample = attach_metadata(sample, parse_name(folder_name, name_pattern, name_converters))
    return sample
