import os
from tifffile import TiffFile
import numpy as np
from skimage.transform import AffineTransform

from brainfusion.io import get_roi_from_txt
from brainfusion.load_experiments.base import list_matching_files
from brainfusion.metadata import attach_metadata, parse_name
from brainfusion.sample import Sample
from brainfusion.utils import bin_2D_image, bin_outline


def load_microscopy_experiment(folder_path, boundary_filename='BrainBoundary', landmarks_filename=None,
                               bin_size="None", bit_depth=16, normalize_percentile="None", clip=False, invert=True,
                               name_pattern=None, name_converters=None, **kwargs) -> list[Sample]:
    """
    Load .tif files including all channels and their tissue outlines.

    If `landmarks_filename` is given, looks for '<image_stem><landmarks_filename>.txt' next to each image
    (same convention as `boundary_filename`) and stores its points as that sample's `Sample.landmarks`. If
    `name_pattern` is given, it is matched against each image's own filename (without extension) and the
    extracted fields are stored in that sample's `.metadata` (see `brainfusion.metadata.parse_name`).
    """
    # Load all image files with corresponding brain outlines
    tif_i_filenames = list_matching_files(folder_path, lambda f: f.lower().endswith('.tif'))

    # Import tif images
    tif_grids, tif_grids_dims, scale_matrices, tif_datasets, tif_contours, tif_landmarks, filenames = \
        [], [], [], [], [], [], []
    for filename in tif_i_filenames:
        file_path = os.path.join(folder_path, filename)
        stem = filename.removesuffix(".tif")

        # Import contours corresponding to tif images
        contour_path = os.path.join(folder_path, stem + boundary_filename + '.txt')
        tif_contour = get_roi_from_txt(contour_path, delimiter='\t', skip=1)

        # Import optional landmark points, matched by position against the template's own landmarks
        landmarks = None
        if landmarks_filename is not None:
            landmarks_path = os.path.join(folder_path, stem + landmarks_filename + '.txt')
            landmarks = get_roi_from_txt(landmarks_path, delimiter='\t')

        # Load resolution metadata
        with TiffFile(file_path) as tif:
            image = tif.asarray()
            page = tif.pages[0]
            x_res = page.coords['width'][1]
            y_res = page.coords['height'][1]

        # Handle single-channel and multi-channel images
        channels = {}
        binned = False

        if image.ndim == 2:  # Single-channel
            height = image.shape[0]
            if invert:
                tif_contour[:, 1] = height - 1 - tif_contour[:, 1]  # Invert contour y-axis
                if landmarks is not None:
                    landmarks[:, 1] = height - 1 - landmarks[:, 1]

            tmp_img = image[::-1, :] # Flip y-axis
            if isinstance(bin_size, int):
                tmp_img = bin_2D_image(tmp_img, bin_size=bin_size, crop=True)
                tif_contour = bin_outline(tif_contour, bin_size=bin_size, crop=True,
                                          original_shape=image.shape[-2:])
                if landmarks is not None:
                    landmarks = bin_outline(landmarks, bin_size=bin_size, crop=True,
                                            original_shape=image.shape[-2:])
                binned = True
            channels['Channel_1'] = tmp_img.ravel()
            height, width = tmp_img.shape[-2:]

        elif image.ndim == 3:  # Multi-channel
            height = image[0].shape[0]
            if invert:
                tif_contour[:, 1] = height - 1 - tif_contour[:, 1]  # Invert contour y-axis
                if landmarks is not None:
                    landmarks[:, 1] = height - 1 - landmarks[:, 1]
            for i in range(image.shape[0]):
                tmp_img = image[i][::-1, :] # Flip y-axis
                if isinstance(bin_size, int):
                    tmp_img = bin_2D_image(tmp_img, bin_size=bin_size, crop=True)
                    binned = True
                channels[f'Channel_{i + 1}'] = tmp_img.ravel()

            if isinstance(bin_size, int):
                tif_contour = bin_outline(tif_contour, bin_size=bin_size, crop=True,
                                          original_shape=image.shape[-2:])
                if landmarks is not None:
                    landmarks = bin_outline(landmarks, bin_size=bin_size, crop=True,
                                            original_shape=image.shape[-2:])
            height, width = tmp_img.shape[-2:]
        else:
            raise ValueError(f"Invalid image dimension: {image.ndim}")

        if binned:
            print('Attention: Data binning is activated to improve calculation time!')
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
                print(
                    f'Attention: Intensity values above the {normalize_percentile}th percentile are clipped to max value!')

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
        tif_contour = aff(tif_contour)
        if landmarks is not None:
            landmarks = aff(landmarks)

        # Store data as dictionary and contour
        tif_contours.append(tif_contour)
        tif_grids.append(pixel_grid)
        tif_landmarks.append(landmarks)
        scale_matrices.append(np.linalg.inv(scale_matrix))
        tif_grids_dims.append(np.array([height, width]))
        tif_datasets.append(channels)
        filenames.append(os.path.splitext(filename)[0])

    samples = [
        Sample(contour=contour, grid=grid, dataset=dataset, scale=scale, landmarks=landmarks,
              grid_shape=grid_shape, filename=filename)
        for contour, grid, dataset, scale, landmarks, grid_shape, filename in
        zip(tif_contours, tif_grids, tif_datasets, scale_matrices, tif_landmarks, tif_grids_dims, filenames)
    ]
    if name_pattern is not None:
        samples = [attach_metadata(s, parse_name(s.filename, name_pattern, name_converters)) for s in samples]
    return samples
