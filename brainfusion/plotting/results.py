import os

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform

from brainfusion.plotting.maps import plot_transformed_grid
from brainfusion.plotting.image_export import plot_map_on_image


def plot_sample_warps(analysis, results_folder, key_quant, image_dataset=False, cbar_label='', cmap='afmhot',
                      marker_size=20, mask=True, vmin=None, vmax=None, plot_background=False,
                      invert_y=False, **kwargs):
    """
    Render each sample's DTW/warp diagnostic (its data before and after warping onto the template) for one
    `brain_fusion` result - the per-sample counterpart to `plot_verification_grids`.

    Averaged-map and matched-contour plots are not part of this anymore (grouping means there can be
    several averaged maps, not just one) - use `plot_average_map`/`plot_contours` directly instead.

    `image_dataset=True` treats each sample's data as an image (reshaping flat arrays back to (H, W) using
    the stored grid shapes) rather than a scatter of points - only correct for samples actually loaded as
    images (e.g. HCR/microscopy). It additionally requires the analysis to carry a regular interpolation
    grid, so it's incompatible with `clustering="GMM"` (which doesn't build one).
    """
    print(f'Plotting: {os.path.basename(os.path.dirname(results_folder))}.')
    os.makedirs(results_folder, exist_ok=True)

    _plot_per_sample(analysis, results_folder, key_quant, image_dataset=image_dataset, cbar_label=cbar_label,
                     cmap=cmap, marker_size=marker_size, mask=mask, vmin=vmin, vmax=vmax,
                     plot_background=plot_background, invert_y=invert_y)


def plot_verification_grids(analysis, results_folder, marker_size=20, mask=True, invert_y=False):
    """
    For each measurement sample, plot placeholder data on the pre-/post-warp verification grids (a regular
    grid over that sample's own bounding box, independent of its real measurement grid) to sanity-check the
    RBF warp itself - e.g. to check for folding or excessive distortion - without involving any real data.
    """
    os.makedirs(results_folder, exist_ok=True)
    key = 'verification'

    for index, filename in enumerate(analysis['measurement_filenames']):
        matched_contour = analysis['measurement_contours'][index]
        template_contour = analysis['template_contours'][index]
        matched_grid = analysis['verification_grids'][index]
        trafo_grid = analysis['verification_trafo_grids'][index]
        raw_data = {key: np.random.choice(np.linspace(1, 10, 10), size=matched_grid.shape[0])}

        fig = plot_transformed_grid(matched_contour, template_contour, raw_data, matched_grid, trafo_grid,
                                    key_quant=key, cmap='viridis', marker_size=marker_size, mask=mask,
                                    invert_y=invert_y)
        fig.savefig(os.path.join(results_folder, f'Verification_{filename}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)


def _plot_per_sample(analysis, results_folder, key_quant, image_dataset, cbar_label, cmap, marker_size, mask, vmin,
                    vmax, plot_background, invert_y):
    interpolated_grid_shape = analysis.get('measurement_interpolated_grid_shape')
    interpolated_grid_image = None
    if image_dataset:
        if interpolated_grid_shape is None:
            raise ValueError("image_dataset=True needs a regular shared interpolation grid, but this "
                             "analysis's clustering (e.g. 'GMM') doesn't produce one.")
        H, W = (int(v) for v in interpolated_grid_shape)
        interpolated_grid_image = analysis['measurement_interpolated_grid'].reshape(H, W, 2)

    for index, filename in enumerate(analysis['measurement_filenames']):
        matched_contour = analysis['measurement_contours'][index]
        template_contour = analysis['template_contours'][index]
        matched_grid = analysis['measurement_grids'][index]
        trafo_grid = analysis['measurement_trafo_grids'][index]
        raw_data = dict(analysis['measurement_datasets'][index])
        trafo_data = None

        if image_dataset:
            h, w = (int(v) for v in analysis['measurement_grids_shape'][index])
            matched_grid = matched_grid.reshape(h, w, 2)
            raw_data = {key: value.reshape(h, w) for key, value in raw_data.items()}
            trafo_grid = interpolated_grid_image
            resampled = analysis['measurement_trafo_datasets'][index]
            trafo_data = {key: value.reshape(H, W) for key, value in resampled.items()}

        fig = plot_transformed_grid(matched_contour, template_contour, raw_data, matched_grid, trafo_grid,
                                    key_quant=key_quant, trafo_data=trafo_data, cbar_label=cbar_label,
                                    cmap=cmap, marker_size=marker_size, vmin=vmin, vmax=vmax, mask=mask,
                                    invert_y=invert_y)
        fig.savefig(os.path.join(results_folder, f'Transformed_{filename}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        if plot_background:
            _plot_sample_on_background(analysis, index, matched_grid, matched_contour, raw_data, key_quant,
                                       results_folder, filename, cmap, marker_size, vmin, vmax)


def _plot_sample_on_background(analysis, index, matched_grid, matched_contour, raw_data, key_quant, results_folder,
                               filename, cmap, marker_size, vmin, vmax):
    # `affine_matrices`/`scale_matrices` carry the template at index 0, so measurement sample `index`
    # lives at `index + 1`.
    affine_matrix = analysis['affine_matrices'][index + 1]
    scale_matrix = analysis['scale_matrices'][index + 1]
    affine_trafo = AffineTransform(matrix=scale_matrix @ affine_matrix)

    fig = plot_map_on_image(analysis['background_image'][index], raw_data[key_quant],
                            affine_trafo(matched_grid), affine_trafo(matched_contour), scale=1, cmap=cmap,
                            marker_size=marker_size / 50, vmin=vmin, vmax=vmax, mask=True, alpha=0.9)
    fig.savefig(os.path.join(results_folder, f'Original_{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
