import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from brainfusion.fusion.core import average_statistic
from brainfusion.fusion.grouping import group_average_entry
from brainfusion.fusion.interpolation import typical_spacing
from brainfusion.utils import mask_contour


def _regular_grid_pitches(grid, tolerance=0.1, min_fraction=0.95):
    """(pitch in x, pitch in y) of an axis-aligned regular grid given as an (N, 2) point list - the two may differ -
    else None (a warped or scattered cloud). Regular: almost all points sit on integer multiples of the pitch of their
    axis."""
    if len(grid) < 4:
        return None
    pitches, on_lattice = [], np.ones(len(grid), dtype=bool)
    for axis in (0, 1):
        coordinate = grid[:, axis]
        extent = np.ptp(coordinate)
        if extent == 0:
            return None
        # rounded relative to the extent, so grids in metres and in micrometres are treated alike
        spacings = np.diff(np.unique(np.round(coordinate / extent, 9))) * extent
        spacings = spacings[spacings > 1e-3 * extent]
        if len(spacings) == 0:
            return None
        pitch = np.median(spacings)
        offsets = (coordinate - coordinate.min()) / pitch
        on_lattice &= np.abs(offsets - np.round(offsets)) < tolerance
        pitches.append(pitch)
    return tuple(pitches) if np.mean(on_lattice) >= min_fraction else None


def fit_markers_to_spacing(ax):
    """
    Give every scatter that was drawn with `marker_size='auto'` (see `render_data_on_contour`) markers one point spacing wide
    on this axes, so neighbours touch. Needs the final size of the axes: call it after the layout (`tight_layout`) and
    after the axis limits and aspect are set. Independent of the figure size and the resolution.
    """
    ax.apply_aspect()
    width_points = ax.get_window_extent().width * 72 / ax.figure.dpi
    x_min, x_max = ax.get_xlim()
    points_per_unit = width_points / abs(x_max - x_min)
    for collection in ax.collections:
        spacing = getattr(collection, '_marker_spacing', None)
        if spacing:
            collection.set_sizes([(spacing * points_per_unit) ** 2])


def render_data_on_contour(ax, data, grid, contour, cmap='afmhot', vmin=None, vmax=None, marker_size=15, mask=True,
                           invert_y=False, marker='s'):
    """
    Draw `data` at `grid` positions on `ax`, clipped to `contour`.

    `grid` of shape (H, W, 2) is drawn as an image (`data` reshaped to (H, W)); a flat (N, 2) point cloud is
    drawn as a scatter instead - or, if it is an axis-aligned regular grid and `marker` is 's', as cells of one
    grid pitch (in x and in y) each that touch without gaps (`marker_size` is then not used) - the two cases used to be duplicated wherever a map needed drawing. Returns
    the drawn mappable, e.g. for building a colorbar from it.

    `marker_size='auto'` (scatter only): the markers are as wide as the typical spacing of the points, so a warped or
    scattered cloud is drawn about as densely as a regular grid; the size is set by `fit_markers_to_spacing` once the layout
    of the axes is known (the plotting functions of this module call it).
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
    pitches = _regular_grid_pitches(grid) if marker == 's' else None
    if pitches is not None:
        # a regular grid: cells of exactly one grid pitch, so neighbours touch (edge in the face colour: no seams)
        pitch_x, pitch_y = pitches
        values = np.ma.masked_invalid(np.asarray(data, dtype=float)[inside])
        squares = PatchCollection([Rectangle((x - pitch_x / 2, y - pitch_y / 2), pitch_x, pitch_y)
                                   for x, y in grid[inside]],
                                  cmap=cmap, linewidths=0.4, antialiased=True)
        squares.set_array(values)
        squares.set_clim(vmin, vmax)
        squares.set_edgecolor('face')
        ax.add_collection(squares)
        ax.autoscale_view()
        return squares
    auto = isinstance(marker_size, str)
    if auto and marker_size != 'auto':
        raise ValueError(f"marker_size must be a number or 'auto', not {marker_size!r}")
    scatter = ax.scatter(np.ma.masked_where(~inside, grid[:, 0]), np.ma.masked_where(~inside, grid[:, 1]), c=data,
                         cmap=cmap, s=15 if auto else marker_size, marker=marker, vmin=vmin, vmax=vmax)
    if auto:
        scatter._marker_spacing = typical_spacing(grid[inside])
    return scatter


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


def plot_aligned_outlines(analysis, invert_y=False, output_path=None):
    """
    Overlay the outlines of all samples of a fusion after the affine alignment (grey, one per sample: centred, rotated and
    scaled onto the ellipse of the template; the outline's shape is untouched) with the average outline (blue) that is
    computed from them - how well the outlines agree before the point-wise (DTW) matching, and which samples stand out.
    Works on an `analysis` from `run_fusion`/`brain_fusion` or one re-loaded from its .h5 file. If `output_path` is given,
    saves the figure there (dpi=300, tight bbox) and closes it.
    """
    template = np.asarray(analysis['template_contours'][0])
    contours = [np.asarray(c) for c in analysis['measurement_contours']]
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, contour in enumerate(contours):
        ax.plot(contour[:, 0], contour[:, 1], color='grey', lw=1, alpha=0.5,
                label=f'aligned outline of each sample ({len(contours)})' if i == 0 else None)
    ax.plot(template[:, 0], template[:, 1], color='blue', ls='--', lw=3, label='average outline (template)')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    if invert_y:
        ax.invert_yaxis()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_transformed_grid(contour, template_contour, data, grid, trafo_grid, key_quant, trafo_data=None,
                          cbar_label='', cmap='afmhot', marker_size='auto', vmin=None, vmax=None, mask=False,
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
    for ax in axes:
        fit_markers_to_spacing(ax)
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
    fit_markers_to_spacing(ax)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_average_map(analysis, key_quant, group=None, cbar_label='', cmap='viridis', marker_size=15, vmin=None,
                     vmax=None, mask=True, invert_y=False, output_path=None, method=None):
    """
    Plot one averaged map straight from a `run_fusion`/`brain_fusion` result - a thin wrapper around
    `plot_average_map_arrays` that pulls the right grid/contour/data out of `analysis` first, so you don't
    need to keep those arrays around separately (works equally well on an `analysis` re-loaded from its .h5
    file). For plotting a derived array you computed yourself (no `analysis` to pull it from), use
    `plot_average_map_arrays` directly instead.

    Without `group`, plots the overall average across every sample. With `group` given (e.g.
    `group='Control'`), plots just that one group's average - `analysis` must have been fused with
    `run_fusion(..., group_field=...)`. Use `extract_groups(analysis)` to see what groups exist.

    `method` ('Mean', 'Median', 'GMM', ...) plots that clustering of a fusion made with several (`clustering=[...]`)
    instead of the first one; GMM maps live on the GMM components (also for a group).
    """
    template_contour = analysis['template_contours'][0]
    grid = analysis['measurement_interpolated_grid']
    if group is not None:
        entry = group_average_entry(analysis, group, method)
        grid, data = entry['grid'], entry['dataset'][key_quant]
    elif method is not None:
        grid = analysis['averaged'][method]['grid']
        data = analysis['averaged'][method]['dataset'][key_quant]
    else:
        data = analysis['measurement_interpolated_dataset'][key_quant]

    return plot_average_map_arrays(data, grid, template_contour, cbar_label=cbar_label, cmap=cmap,
                                   marker_size=marker_size, vmin=vmin, vmax=vmax, mask=mask, invert_y=invert_y,
                                   output_path=output_path)


_STATISTIC_LABELS = {'n': 'samples per point', 'spread': 'spread between samples', 'relative_spread':
                     'spread between samples (% of the average)', 'standard_error': 'standard error',
                     'relative_standard_error': 'standard error (% of the average)'}


def plot_average_statistic(analysis, key_quant, statistic='standard_error', method=None, cbar_label=None, cmap='viridis',
                           marker_size=15, vmin=0, vmax=None, mask=True, invert_y=False, output_path=None):
    """
    Map of how well an averaged map is known: the number of samples, the spread or the standard error at every point
    (see `brainfusion.fusion.core.average_statistic` for the definitions), for one clustering `method` of a fusion.
    """
    grid, values = average_statistic(analysis, key_quant, method, statistic)
    contour = analysis['template_contours'][0]
    if vmax is None and mask:   # the 99th percentile inside the outline: a few outliers outside it would stretch the scale
        inside = values[mask_contour(contour, grid) & np.isfinite(values)]
        vmax = np.percentile(inside, 99) if len(inside) else None
    return plot_average_map_arrays(values, grid, contour,
                                   cbar_label=cbar_label or _STATISTIC_LABELS[statistic], cmap=cmap,
                                   marker_size=marker_size, vmin=vmin, vmax=vmax, mask=mask, invert_y=invert_y,
                                   output_path=output_path)
