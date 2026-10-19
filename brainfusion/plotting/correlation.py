import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from scipy import stats

from .maps import render_data_on_contour


def format_p_value(p):
    if p < 0.001:
        return "p<0.001"
    elif p < 0.20:
        return f"p={p:.3f}"
    else:
        return f"p={p:.2f}"


def plot_norm_corr(map1, map2, pearson=None, p_value=None, label1="X-axis", label2="Y-axis", output_path=None,
                   spearman=None, n_points=None, fit_line=True, note=None, figsize=(3.6, 3.4)):
    """
    Scatter of two maps' values at the same points, in the units of the data (put them in the labels, e.g.
    "AFM modulus (Pa)"), with a least-squares line and its 95 % confidence band, and the correlation numbers given.
    Both axes are scaled to their own data. Sized for a single-column figure (`figsize`, inches; fonts 7-8 pt).

    `pearson`, `spearman`, `p_value` are printed if given (see `correlate_on_shared_grid`); `p_value` belongs to the
    Pearson r. The numbers sit in the lower right corner, which is empty for a positive relation. `n_points` defaults to the number of values. `note`: one more line of text under the numbers, e.g. why
    there is no p-value. If `output_path` is given, saves the figure there (dpi=300, tight bbox) and closes it.
    """
    map1, map2 = np.asarray(map1, dtype=float), np.asarray(map2, dtype=float)
    if map1.shape != map2.shape:
        raise ValueError("map1 and map2 must have the same shape.")
    n_points = map1.size if n_points is None else n_points

    with plt.rc_context({'font.size': 7, 'axes.labelsize': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(map1, map2, s=7, color='#1f4e79', alpha=0.55, linewidths=0, rasterized=True, zorder=2)

        if fit_line and map1.size > 2 and np.ptp(map1) > 0:
            slope, intercept = np.polyfit(map1, map2, 1)
            x = np.linspace(map1.min(), map1.max(), 100)
            residual_sd = np.sqrt(np.sum((map2 - (slope * map1 + intercept)) ** 2) / (map1.size - 2))
            half_width = (stats.t.ppf(0.975, map1.size - 2) * residual_sd
                          * np.sqrt(1 / map1.size + (x - map1.mean()) ** 2 / np.sum((map1 - map1.mean()) ** 2)))
            ax.fill_between(x, slope * x + intercept - half_width, slope * x + intercept + half_width, color='#c0392b',
                            alpha=0.18, linewidths=0, zorder=3)
            ax.plot(x, slope * x + intercept, color='#c0392b', lw=1.2, zorder=4)

        lines = []
        if pearson is not None:
            lines.append(f'Pearson r = {pearson:.2f}' + (f' ({format_p_value(p_value)})' if p_value is not None else ''))
        if spearman is not None:
            lines.append(f'Spearman ρ = {spearman:.2f}')
        lines.append(f'n = {n_points}')
        if note:
            lines.append(note)
        ax.text(0.97, 0.03, '\n'.join(lines), transform=ax.transAxes, va='bottom', ha='right', fontsize=7, zorder=5)

        x_margin, y_margin = 0.05 * np.ptp(map1), 0.05 * np.ptp(map2)
        ax.set_xlim(map1.min() - x_margin, map1.max() + x_margin)
        ax.set_ylim(map2.min() - y_margin, map2.max() + y_margin)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(pad=0.6)

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    return fig


def plot_correlation_with_radii(reference_grid, other_grid, contour, radii, data_a, data_b, label_a="Reference",
                                label_b="Other", results_folder=None, results_name="CorrelationRadii", title=""):
    """
    Plot the reference and other grids, each colored by its own value, together with circles of the given
    radii drawn around each reference point - the circles let you sanity-check a
    `correlate_around_reference_grid` call's radius choice visually (no excessive overlap, no big gaps)
    before trusting the correlation, while the coloring shows each side's spatial pattern directly (the
    density-route counterpart of `plot_correlation_masks`).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    sc_b = ax.scatter(other_grid[:, 0], other_grid[:, 1], c=data_b, s=10, cmap='grey', alpha=0.5)
    fig.colorbar(sc_b, ax=ax, orientation='vertical', label=label_b, shrink=0.6)
    sc_a = ax.scatter(reference_grid[:, 0], reference_grid[:, 1], c=data_a, s=15, cmap='hot')
    fig.colorbar(sc_a, ax=ax, orientation='vertical', label=label_a, shrink=0.6)
    ax.plot(contour[:, 0], contour[:, 1], 'k-', linewidth=2, label="Contour")

    for i, point in enumerate(reference_grid):
        ax.add_patch(patches.Circle(point, radii[i], color='blue', alpha=0.2))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=18)
    ax.set_aspect('equal')
    ax.legend()

    if isinstance(results_folder, str):
        os.makedirs(results_folder, exist_ok=True)
        fig.savefig(os.path.join(results_folder, f'{results_name}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_correlation_masks(grid, data_a, data_b, contour, label_a="A", label_b="B", results_folder=None,
                           results_name="correlation_masks", cmap_a="hot", cmap_b="hot", cbar_label_a="", cbar_label_b=""):
    """Plot two datasets living on the same grid side by side (e.g. two groups from `pairwise_correlate_groups`).
    Supports boolean or numeric data. A regular grid is drawn as cells that touch (see `render_data_on_contour`)."""
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    bool_cmap = mcolors.ListedColormap(['black', 'red'])
    for ax, data, label, cmap, cbar_label in ((axs[0], data_a, label_a, cmap_a, cbar_label_a),
                                              (axs[1], data_b, label_b, cmap_b, cbar_label_b)):
        is_bool = data.dtype == bool or np.array_equal(data, data.astype(bool))
        if is_bool:
            render_data_on_contour(ax, data.astype(float), grid, contour, cmap=bool_cmap, vmin=0, vmax=1, mask=False)
        else:
            cells = render_data_on_contour(ax, np.asarray(data, dtype=float), grid, contour, cmap=cmap, mask=False)
            fig.colorbar(cells, ax=ax, orientation='vertical', label=cbar_label)
        ax.plot(contour[:, 0], contour[:, 1], 'k-', linewidth=2, label="Contour")
        ax.set_title(label)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    if results_folder:
        os.makedirs(results_folder, exist_ok=True)
        fig.savefig(os.path.join(results_folder, f"{results_name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig
