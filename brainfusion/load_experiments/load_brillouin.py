import os

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import AffineTransform

from brainfusion.fusion.outline_repeats import read_outline
from brainfusion.io import attach_metadata, parse_name, mirror_pixels, annotation_mirror_flags
from brainfusion.load_experiments.base import iter_experiment_folders, list_matching_files
from brainfusion.sample import Sample


DATA_SUFFIX = '_data.csv'  # the exported data file of a repetition


def load_brillouin_all(base_path, brillouin_variables, data_filename=None, use_landmarks=True, apply_mirror=True,
                       quality_column=None, name_pattern=None, name_converters=None,
                       data_suffix=DATA_SUFFIX, image_frame=False, min_depth_um=None) -> list[Sample]:
    """
    Load every Brillouin experiment folder (name containing '#') found directly below `base_path`.

    See `load_brillouin_experiment` for the other parameters. If `name_pattern` is given, it is matched
    against each folder's name (see `brainfusion.io.parse_name`) and the extracted fields are stored in
    the sample's `.metadata`.
    """
    samples = []
    for folder_name, folder_path in iter_experiment_folders(base_path):
        sample = load_brillouin_experiment(folder_path, brillouin_variables, data_filename=data_filename,
                                           use_landmarks=use_landmarks, apply_mirror=apply_mirror,
                                           quality_column=quality_column, data_suffix=data_suffix,
                                           image_frame=image_frame, min_depth_um=min_depth_um)
        if name_pattern is not None:
            sample = attach_metadata(sample, parse_name(folder_name, name_pattern, name_converters))
        samples.append(sample)
    return samples


def load_brillouin_experiment(folder_path, brillouin_variables, data_filename=None, use_landmarks=True,
                              apply_mirror=True, quality_column=None, outline='outline_brainfusion.txt',
                              outline_kwargs=None, data_suffix=DATA_SUFFIX, image_frame=False,
                              min_depth_um=None) -> Sample:
    """
    Load a Brillouin experiment exported by bmlab's combined-CSV exporter, together with the outline (and
    landmarks) from an annotation file.

    Expects, directly below `folder_path`:
    - 'Export/<...>_BMrep<n>_data.csv' (a different ending can be given with `data_suffix`): one row per measured grid point, with 'x'/'y'
      stage position (um) columns plus one column per key requested in `brillouin_variables` ('#'-prefixed
      metadata lines are skipped automatically). If `data_filename` is None, the sole file in 'Export' ending in
      `data_suffix` is used (no other file is substituted); with several (one per repetition) pass the one to
      use. The 'z' column is always
      included in `Sample.dataset` too - it isn't a stacking dimension (grid alignment stays purely 2D, on
      'x'/'y'), just each point's own z offset, so it rides through like any other channel.
      The CSV's 'x'/'y' must be absolute stage positions (what bmlab exports now; older exports centred them on
      the scan grid and must be exported again).
    - 'Plots/<outline>': the outline and landmarks, drawn in the pixels of the image named in its
      header (see `brainfusion.fusion.read_outline`): a name ending in '.txt' is that single file, any other name a
      folder of repeated outlines (e.g. 'outline_repeats'), averaged with the outline settings of `outline_kwargs`
      (the fusion kwargs). The image is found relative to 'Plots', e.g. 'Overview_BMrep0/overview_1_....tif'.
    - the 3x3 matrix of that image, 'TransformMatrices/<image stem>_transform.csv' next to it: it maps a stage
      position (x, y, in um - the coordinates of the data CSV's 'x'/'y' columns) onto the image's pixels
      ([col, row, 1] = M @ [x_um, y_um, 1]), the same layout as this package's AFM 'GridInversionMatrix.csv'.

    `use_landmarks=False` ignores the landmarks. `quality_column`: name of a 0/1 column in the data CSV (bmlab
    writes 'quality_pass'); rows where it is 0 are dropped. With `apply_mirror`, the mirroring noted in the
    annotation header (so that all brains look the same way) is applied to the grid, outline, landmarks and
    background image.

    `min_depth_um`: keep only the z planes at least this deep (um) below the surface the acquisition found in each
    grid column (surface follow), e.g. 0 to leave out the planes the scan starts above it. Plane k of a column lies at
    grid_z_min_um + k * grid_z_step_um ('#'-lines of the CSV header, written by bmlab from the raw file). An export
    without them (raw files recorded before the z range was stored) keeps all its planes, with a printed note; one
    without any plane that deep raises ValueError.

    The sample lives in the stage frame (um, metadata 'length_unit': 'um') of the setup, and the stage frame of a Brillouin setup is usually a
    reflection of the image (the matrix has a negative determinant), so a map drawn in it looks mirrored against the
    image. `image_frame=True` turns grid, outline and landmarks into the frame of the image instead (still um), with
    the y axis pointing up as in a plot: the linear part of the matrix, normalised to unit scale, then y negated. A map
    drawn from it looks like the image. This is the frame the AFM samples of `load_batchforce_single` are in, so use it
    when the sample is fused with AFM data: the outline alignment only rotates, it does not mirror.
    """
    folder_name = os.path.basename(os.path.normpath(folder_path))
    export_dir = os.path.join(folder_path, 'Export')
    annotation = read_outline(os.path.join(folder_path, 'Plots'), outline, outline_kwargs)
    annotated_image = annotation['header'].get('image')
    assert annotated_image, f"{annotation['path']} does not name the image the outline was drawn on"

    # --- Brillouin data: grid (stage position, um) plus the requested variables ---
    if data_filename is None:
        candidates = list_matching_files(export_dir, lambda f: f.endswith(data_suffix))
        assert len(candidates) == 1, (
            f"Expected exactly one '*{data_suffix}' file in {export_dir}, found {len(candidates)}: {candidates}. "
            f"Pass `data_filename` explicitly to pick one.")
        data_filename = candidates[0]

    data = pd.read_csv(os.path.join(export_dir, data_filename), comment='#')
    if min_depth_um is not None:
        data = _planes_from_depth(data, os.path.join(export_dir, data_filename), min_depth_um)
    if quality_column is not None:
        data = data[data[quality_column].fillna(0) > 0].reset_index(drop=True)
    grid = np.stack((np.array(data['x']), np.array(data['y'])), axis=-1)
    dataset = {variable: np.array(data[variable]) for variable in brillouin_variables}
    # 'z' isn't a stacking dimension here - each grid point just has its own z offset (e.g. from surface
    # unevenness), not a shared discrete plane - so it rides along as an ordinary data channel like any other
    # requested variable, warped/interpolated/averaged the same way, rather than needing any special handling.
    dataset.setdefault('z', np.array(data['z']))

    # --- The annotated image and its matrix (stage um -> image pixels) ---
    image_path = os.path.join(annotation['image_dir'], *annotated_image.split('/'))
    transform_path = os.path.join(os.path.dirname(image_path), 'TransformMatrices',
                                  os.path.splitext(os.path.basename(image_path))[0] + '_transform.csv')
    assert os.path.isfile(image_path), f"The annotated image {image_path} does not exist"
    if not os.path.isfile(transform_path):
        raise FileNotFoundError(f"{folder_name}: the annotated image ({annotated_image}) has no transform matrix "
                                f"({transform_path}), so the outline cannot be related to the measurement grid")
    transform_matrix = pd.read_csv(transform_path, header=None).to_numpy()
    bg_image = np.array(Image.open(image_path))

    # --- Annotation (image pixels), mirrored so all brains look the same way ---
    contour = annotation['outline']
    landmarks = annotation['landmarks'] if use_landmarks else None
    to_pixels = AffineTransform(matrix=transform_matrix)
    to_um = AffineTransform(matrix=np.linalg.inv(transform_matrix))
    if apply_mirror:
        horizontal, vertical = annotation_mirror_flags(annotation)
        shape = bg_image.shape[:2]
        # the grid is in stage um: into image pixels, mirror there, and back with the same matrix
        grid = to_um(mirror_pixels(to_pixels(grid), shape, horizontal, vertical))
        contour = mirror_pixels(contour, shape, horizontal, vertical)
        if landmarks is not None:
            landmarks = mirror_pixels(landmarks, shape, horizontal, vertical)
        if horizontal:
            bg_image = bg_image[:, ::-1]
        if vertical:
            bg_image = bg_image[::-1]

    # --- Outline and landmarks into the data's stage (um) coordinates so they line up with `grid` ---
    contour = to_um(contour)
    landmarks = to_um(landmarks) if landmarks is not None else None

    # A grid in another coordinate frame (e.g. an older export centred on the scan grid) lies thousands of um away
    reach = 3 * np.ptp(contour, axis=0).max()
    if not np.any(np.all((grid >= contour.min(axis=0) - reach) & (grid <= contour.max(axis=0) + reach), axis=1)):
        raise ValueError(f"{folder_name}: the measurement grid of {data_filename} lies far away from the outline - "
                         f"its x/y are probably not absolute stage positions (older bmlab export): export again")

    if image_frame:
        linear = transform_matrix[:2, :2] / np.sqrt(abs(np.linalg.det(transform_matrix[:2, :2])))
        linear = np.diag([1.0, -1.0]) @ linear   # pixels count downwards, plots upwards
        contour, grid = contour @ linear.T, grid @ linear.T
        landmarks = landmarks @ linear.T if landmarks is not None else None

    return Sample(contour=contour, grid=grid, dataset=dataset, scale=transform_matrix, landmarks=landmarks,
                  bg_image=bg_image, filename=folder_name, metadata={'length_unit': 'um'})


def _read_header(path):
    """The '#key,value' lines at the top of a bmlab export as a dict of strings."""
    header = {}
    with open(path, encoding='utf-8') as file:
        for line in file:
            if not line.startswith('#'):
                break
            key, _, value = line[1:].rstrip('\n').partition(',')
            header[key] = value
    return header


def _planes_from_depth(data, path, min_depth_um):
    """The rows of `data` whose plane lies at least `min_depth_um` below the found surface (see load_brillouin_experiment).
    The planes of a grid column share x_orig/y_orig (if the file has these columns) or x/y; its smallest z (z counts into the sample) is plane 0."""
    header = _read_header(path)
    try:
        z_min, z_step = float(header['grid_z_min_um']), float(header['grid_z_step_um'])
    except (KeyError, ValueError):
        print(f"Note: {os.path.basename(path)} does not record its z range (grid_z_min_um) - all planes kept")
        return data
    columns = ['x_orig', 'y_orig'] if {'x_orig', 'y_orig'} <= set(data.columns) else ['x', 'y']
    column = data[columns].round(1).apply(tuple, axis=1)
    plane = np.round((data['z'] - data['z'].groupby(column).transform('min')) / z_step).astype(int)
    depth = z_min + plane * z_step
    if not (depth >= min_depth_um - 1e-6).any():
        raise ValueError(f"{os.path.basename(path)}: no plane at least {min_depth_um} um below the surface "
                         f"(z range from {z_min} um in steps of {z_step} um)")
    return data[depth >= min_depth_um - 1e-6].reset_index(drop=True)
