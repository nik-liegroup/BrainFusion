import numpy as np
import matplotlib.pyplot as plt

from brainfusion.utils import mask_contour


def render_data_on_contour(ax, data, grid, contour, cmap='afmhot', vmin=None, vmax=None, marker_size=15, mask=True,
                           invert_y=False, marker='s'):
    """
    Draw `data` at `grid` positions on `ax`, clipped to `contour`.

    `grid` of shape (H, W, 2) is drawn as an image (`data` reshaped to (H, W)); a flat (N, 2) point cloud is
    drawn as a scatter instead - the two cases used to be duplicated wherever a map needed drawing. Returns
    the drawn mappable, e.g. for building a colorbar from it.
    """
    if grid.ndim == 3:
        image = np.asarray(data, dtype=float)
        if mask:
            inside = mask_contour(contour, grid.reshape(-1, 2)).reshape(grid.shape[:2])
            image = np.where(inside, image, np.nan)
        extent = [grid[..., 0].min(), grid[..., 0].max(), grid[..., 1].min(), grid[..., 1].max()]
        return ax.imshow(image, extent=extent, origin='lower' if invert_y else 'upper', cmap=cmap, vmin=vmin,
                         vmax=vmax)

    inside = mask_contour(contour, grid) if mask else np.full(grid.shape[0], True)
    return ax.scatter(np.ma.masked_where(~inside, grid[:, 0]), np.ma.masked_where(~inside, grid[:, 1]), c=data,
                      cmap=cmap, s=marker_size, marker=marker, vmin=vmin, vmax=vmax)


def get_zoom_limits(contour, margin_fraction=0.1):
    """Compute x/y limits that fit `contour` with equal aspect ratio and a margin around it."""
    x_min, x_max = contour[:, 0].min(), contour[:, 0].max()
    y_min, y_max = contour[:, 1].min(), contour[:, 1].max()

    x_range, y_range = x_max - x_min, y_max - y_min
    max_range = max(x_range, y_range)
    margin = margin_fraction * max_range

    x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
    x_lim = (x_mid - max_range / 2 - margin, x_mid + max_range / 2 + margin)
    y_lim = (y_mid - max_range / 2 - margin, y_mid + max_range / 2 + margin)
    return x_lim, y_lim


def plot_contours(template_contour, matched_contours, invert_y=False):
    """Overlay every sample's matched contour together with the template contour."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for i, contour in enumerate(matched_contours):
        label = 'DTW Matched Contours' if i == 0 else None
        point_label = 'Initial Coordinate' if i == 0 else None
        ax.plot(contour[:, 0], contour[:, 1], color='grey', linestyle='-', linewidth=1.5, alpha=0.5, label=label)
        ax.scatter(contour[0, 0], contour[0, 1], color='grey', s=25, label=point_label)

    ax.plot(template_contour[:, 0], template_contour[:, 1], color='blue', linestyle='--', linewidth=3,
            label='Template Contour')
    ax.scatter(template_contour[0, 0], template_contour[0, 1], color='blue', s=30, zorder=6,
              label='Initial Tmp. Coordinate')

    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    if invert_y:
        ax.invert_yaxis()

    return fig


def plot_transformed_grid(contour, template_contour, data, grid, trafo_grid, key_quant, trafo_data=None,
                          cbar_label='', cmap='afmhot', marker_size=30, vmin=None, vmax=None, mask=False,
                          invert_y=False):
    """
    Show a sample's DTW correspondence to the template, plus its data map before and after warping.

    Warping only moves point positions, so the transformed panel normally reuses `data` (the sample's own
    values) at `trafo_grid`'s warped positions - `trafo_data` is only needed when `trafo_grid` is a regular
    (H, W, 2) grid (i.e. the shared, resampled interpolation grid, not this sample's own warped points).
    """
    trafo_data = data if trafo_grid.ndim != 3 else trafo_data
    values = data[key_quant]
    vmin = np.nanmin(values) if vmin is None else vmin
    vmax = np.nanmax(values) if vmax is None else vmax

    fig, axes = plt.subplots(1, 3, figsize=(25, 7))

    axes[0].plot(contour[:, 0], contour[:, 1], c='grey', linestyle='-')
    axes[0].scatter(contour[:, 0], contour[:, 1], c='k', s=5)
    axes[0].plot(template_contour[:, 0], template_contour[:, 1], color='blue', linestyle='-')
    axes[0].scatter(template_contour[:, 0], template_contour[:, 1], c='b', s=5)
    axes[0].quiver(contour[:, 0], contour[:, 1], template_contour[:, 0] - contour[:, 0],
                   template_contour[:, 1] - contour[:, 1], angles='xy', scale_units='xy', scale=1, color='r',
                   alpha=0.8, zorder=3)
    axes[0].set_title('DTW with Curvature Penalty', fontsize=20)

    axes[1].plot(contour[:, 0], contour[:, 1], color='grey', linestyle='-', linewidth=3, label='Original Contour',
                alpha=0.75)
    render_data_on_contour(axes[1], data[key_quant], grid, contour, cmap=cmap, vmin=vmin, vmax=vmax,
                           marker_size=marker_size, mask=mask, invert_y=invert_y)
    axes[1].set_title('Original Data Map', fontsize=20)

    axes[2].plot(template_contour[:, 0], template_contour[:, 1], color='blue', linestyle='-', linewidth=4,
                label='Template Contour', alpha=0.75)
    trafo_map = render_data_on_contour(axes[2], trafo_data[key_quant], trafo_grid, template_contour, cmap=cmap,
                                       vmin=vmin, vmax=vmax, marker_size=marker_size, mask=mask, invert_y=invert_y)
    axes[2].set_title('Transformed Data Map', fontsize=20)

    cbar = fig.colorbar(trafo_map, ax=axes[2])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, size=30)

    x_lim, y_lim = get_zoom_limits(template_contour)
    for ax in axes:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        if invert_y and grid.ndim != 3:  # image panels already flip via `origin='lower'`
            ax.invert_yaxis()

    plt.tight_layout()
    return fig


def plot_average_map_arrays(data_avg, grid_avg, template_contour, cbar_label='', cmap='viridis', marker_size=15,
                            vmin=None, vmax=None, mask=True, invert_y=False, output_path=None):
    """
    Plot one fused/averaged data map over the template contour, from plain arrays - e.g. a derived map you
    computed yourself (a difference map, a percentile mask, ...) that has no corresponding `analysis` dict
    to pull it back out of. For plotting straight from a `run_fusion`/`brain_fusion` result instead, use
    `plot_average_map`, which wraps this.

    If `output_path` is given, saves the figure there (dpi=300, tight bbox) and closes it - the figure is
    still returned, but by then it's already closed, so treat that as "saved, not for further use".
    """
    data_avg = np.asarray(data_avg, dtype=float)
    vmin = np.nanmin(data_avg) if vmin is None else vmin
    vmax = np.nanmax(data_avg) if vmax is None else vmax

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(template_contour[:, 0], template_contour[:, 1], 'b--', linewidth=5, label='Template Contour')

    heatmap = render_data_on_contour(ax, data_avg, grid_avg, template_contour, cmap=cmap, vmin=vmin, vmax=vmax,
                                     marker_size=marker_size, mask=mask, invert_y=invert_y, marker='s')

    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(cbar_label, size=20)

    x_lim, y_lim = get_zoom_limits(template_contour)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if invert_y and grid_avg.ndim != 3:
        ax.invert_yaxis()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_average_map(analysis, key_quant, group=None, cbar_label='', cmap='viridis', marker_size=15, vmin=None,
                     vmax=None, mask=True, invert_y=False, output_path=None):
    """
    Plot one averaged map straight from a `run_fusion`/`brain_fusion` result - a thin wrapper around
    `plot_average_map_arrays` that pulls the right grid/contour/data out of `analysis` first, so you don't
    need to keep those arrays around separately (works equally well on an `analysis` re-loaded from its .h5
    file). For plotting a derived array you computed yourself (no `analysis` to pull it from), use
    `plot_average_map_arrays` directly instead.

    Without `group`, plots the overall average across every sample. With `group` given (e.g.
    `group='Control'`), plots just that one group's average, read straight from
    `analysis['group_datasets']` - `analysis` must have been fused with `run_fusion(...,
    group_field=...)`. Use `extract_groups(analysis)` to see what groups exist.
    """
    template_contour = analysis['template_contours'][0]
    grid = analysis['measurement_interpolated_grid']
    if group is None:
        data = analysis['measurement_interpolated_dataset'][key_quant]
    else:
        if 'group_datasets' not in analysis:
            raise ValueError("A 'group' was given but this analysis wasn't fused with a group_field - see "
                             "run_fusion(..., group_field=...).")
        data = analysis['group_datasets'][group][key_quant]

    return plot_average_map_arrays(data, grid, template_contour, cbar_label=cbar_label, cmap=cmap,
                                   marker_size=marker_size, vmin=vmin, vmax=vmax, mask=mask, invert_y=invert_y,
                                   output_path=output_path)
