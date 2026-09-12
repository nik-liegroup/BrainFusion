import numpy as np
import matplotlib.pyplot as plt
from tifffile import imwrite

from brainfusion.utils import mask_contour


def render_contour_on_image(img, grid, contour, cmap='grey', mask=False, invert_y=False):
    """
    Mask `img` (an (N, M) array already sampled on the regular `grid`) to the region inside `contour`.

    Returns the masked image, a preview figure, and the (x, y) physical pixel size implied by `grid`'s
    extent - used by `save_average_map_tif` to write a properly scaled .tif.
    """
    img = img.astype(float)
    N, M = img.shape
    coords = grid.reshape(-1, 2)

    if mask:
        inside = mask_contour(contour, coords).reshape(N, M)
        img = np.where(inside, img, np.nan)

    x_min, x_max = np.min(grid[:, :, 0]), np.max(grid[:, :, 0])
    y_min, y_max = np.min(grid[:, :, 1]), np.max(grid[:, :, 1])
    extent = [x_min, x_max, y_min, y_max]

    pixel_size_x = (x_max - x_min) / M
    pixel_size_y = (y_max - y_min) / N

    fig, ax = plt.subplots()
    ax.imshow(img, extent=extent, origin='lower' if invert_y else 'upper', cmap=cmap)
    ax.plot(contour[:, 0], contour[:, 1], 'b--', linewidth=2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return img, fig, (pixel_size_x, pixel_size_y)


def save_average_map_tif(value_matrix, grid, contour, output_path, invert_y=False):
    """Save a fused map (already on a regular (H, W) grid) as a calibrated, contour-masked .tif."""
    matrix, fig, (pixel_size_x, pixel_size_y) = render_contour_on_image(value_matrix, grid, contour, cmap='grey',
                                                                        mask=True, invert_y=invert_y)
    plt.close(fig)

    if not invert_y:
        matrix = np.flipud(matrix)  # match origin='lower' in imshow

    imwrite(output_path, matrix.astype('float32'), imagej=True,
           resolution=(1e-6 / pixel_size_x, 1e-6 / pixel_size_y), metadata={'unit': 'um', 'axes': 'YX'})


def plot_map_on_image(img, data, grid, contour, scale=1, label='', cmap='viridis', marker_size=15, vmin=None,
                      vmax=None, mask=False, alpha=0.9):
    """Overlay scattered `data` at `grid` positions on top of a raw background image `img`."""
    fig, ax = plt.subplots()

    height_in_mu = img.shape[0] / scale
    width_in_mu = img.shape[1] / scale
    ax.imshow(img, cmap='gray', aspect='equal', origin='lower', extent=[0, width_in_mu, 0, height_in_mu],
             alpha=alpha)

    inside = mask_contour(contour, grid) if mask else np.full(grid.shape[0], True)
    heatmap = ax.scatter(np.ma.masked_where(~inside, grid[:, 0]), np.ma.masked_where(~inside, grid[:, 1]), c=data,
                         cmap=cmap, s=marker_size, marker='s', edgecolors='none', alpha=1, vmin=vmin, vmax=vmax)

    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label, size=20)

    return fig
