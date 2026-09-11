import os

import numpy as np
from tifffile import TiffFile
from skimage.transform import AffineTransform

from brainfusion.io import get_roi_from_txt, attach_metadata, parse_name
from brainfusion.load_experiments.base import list_matching_files
from brainfusion.sample import Sample
from brainfusion.utils import bin_2D_image, bin_outline


def load_microscopy_all(folder_path, boundary_filename='BrainBoundary', landmarks_filename=None, bin_size="None",
                        bit_depth=16, normalize_percentile="None", clip=False, invert=True, name_pattern=None,
                        name_converters=None, **kwargs) -> list[Sample]:
    """
    Load every .tif image in `folder_path`. See `load_microscopy_single` for the other parameters.

    If `name_pattern` is given, it is matched against each image's own filename (without extension) and the
    extracted fields are stored in that sample's `.metadata` (see `brainfusion.metadata.parse_name`).
    """
    tif_filenames = list_matching_files(folder_path, lambda f: f.lower().endswith('.tif'))

    samples = []
    for filename in tif_filenames:
        sample = load_microscopy_single(folder_path, filename, boundary_filename=boundary_filename,
                                        landmarks_filename=landmarks_filename, bin_size=bin_size,
                                        bit_depth=bit_depth, normalize_percentile=normalize_percentile, clip=clip,
                                        invert=invert)
        if name_pattern is not None:
            sample = attach_metadata(sample, parse_name(sample.filename, name_pattern, name_converters))
        samples.append(sample)
    return samples


def load_microscopy_single(folder_path, filename, boundary_filename='BrainBoundary', landmarks_filename=None,
                           bin_size="None", bit_depth=16, normalize_percentile="None", clip=False, invert=True,
                           **kwargs) -> Sample:
    """
    Load one .tif image (all of its channels) together with its tissue outline.

    Expects '<image_stem><boundary_filename>.txt' next to the image for the outline, drawn in the image's own
    (un-flipped, un-binned) pixel coordinates. If `landmarks_filename` is given, looks for
    '<image_stem><landmarks_filename>.txt' in the same convention and stores its points as `Sample.landmarks`.
    """
    stem = os.path.splitext(filename)[0]

    contour = get_roi_from_txt(os.path.join(folder_path, stem + boundary_filename + '.txt'), delimiter='\t', skip=1)

    landmarks = None
    if landmarks_filename is not None:
        landmarks_path = os.path.join(folder_path, stem + landmarks_filename + '.txt')
        landmarks = get_roi_from_txt(landmarks_path, delimiter='\t')

    with TiffFile(os.path.join(folder_path, filename)) as tif:
        image = tif.asarray()
        page = tif.pages[0]
        x_res = page.coords['width'][1]
        y_res = page.coords['height'][1]

    # Treat a single-channel image as one "channel" so the rest of this function doesn't need to branch on
    # dimensionality - channel numbering ('Channel_1', 'Channel_2', ...) comes out the same either way.
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    elif image.ndim != 3:
        raise ValueError(f"Invalid image dimension: {image.ndim}")

    if invert:
        height = image.shape[1]
        contour[:, 1] = height - 1 - contour[:, 1]  # Invert contour y-axis
        if landmarks is not None:
            landmarks[:, 1] = height - 1 - landmarks[:, 1]

    binned = isinstance(bin_size, int)
    channels = {}
    for i in range(image.shape[0]):
        channel_image = image[i][::-1, :]  # Flip y-axis
        if binned:
            channel_image = bin_2D_image(channel_image, bin_size=bin_size, crop=True)
        channels[f'Channel_{i + 1}'] = channel_image.ravel()
    height, width = channel_image.shape

    if binned:
        print('Attention: Data binning is activated to improve calculation time!')
        contour = bin_outline(contour, bin_size=bin_size, crop=True, original_shape=image.shape[-2:])
        if landmarks is not None:
            landmarks = bin_outline(landmarks, bin_size=bin_size, crop=True, original_shape=image.shape[-2:])
        x_res *= bin_size
        y_res *= bin_size

    # Generate pixel coordinates grid
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    pixel_grid = np.column_stack((xx.ravel(), yy.ravel()))

    # Re-normalize image and clip high intensity values
    if isinstance(normalize_percentile, int) or isinstance(normalize_percentile, float):
        if bit_depth not in [8, 12, 16]:
            raise ValueError("bit_depth must be 8, 12, or 16")
        print(f'Attention: Channels are dynamically re-normalized to the {normalize_percentile}th percentile!')

        if clip is True:
            print(f'Attention: Intensity values above the {normalize_percentile}th percentile are clipped to max value!')

        max_val = 2 ** bit_depth - 1
        dtype = np.uint16 if bit_depth > 8 else np.uint8

        for key, c in channels.items():
            # Compute scaling value from percentile
            scale = np.percentile(c, normalize_percentile)
            if scale <= 0:
                scaled = np.zeros_like(c, dtype=dtype)
            else:
                # First scale to [0..1] range
                scaled = c / scale

                if clip is False:
                    # Values >1 → 1
                    scaled = np.minimum(scaled, 1.0)
                else:
                    # Values >1 → 0
                    scaled[scaled > 1.0] = 0.0

                # Now scale to full bit depth
                scaled = scaled * max_val

            # Final integer conversion
            channels[key] = scaled.astype(dtype)

    # Transform coordinates from image coordinates to micro meter
    scale_matrix = np.array([
        [x_res, 0, 0],
        [0, y_res, 0],
        [0, 0, 1]])

    aff = AffineTransform(matrix=scale_matrix)
    pixel_grid = aff(pixel_grid)
    contour = aff(contour)
    if landmarks is not None:
        landmarks = aff(landmarks)

    return Sample(contour=contour, grid=pixel_grid, dataset=channels, scale=np.linalg.inv(scale_matrix),
                 landmarks=landmarks, grid_shape=np.array([height, width]), filename=stem)
