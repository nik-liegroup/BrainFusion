import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.stats import f


def format_p_value(p):
    if p < 0.0001:
        return "p<0.001"
    elif 0.001 <= p < 0.20:
        return f"p={p:.3f}"
    elif p >= 0.20:
        return f"p={p:.2f}"
    else:
        return "Invalid p-value"


def plot_norm_corr(map1, map2, pearson=None, p_value=None, label1="X-axis", label2="Y-axis", output_path=None,
                   square_root_fit=False, map1_fit_limits=None, map2_fit_limits=None):
    map1, map2 = np.asarray(map1), np.asarray(map2)
    if map1.shape != map2.shape:
        raise ValueError("map1 and map2 must have the same shape.")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(map1, map2, color='blue', s=10, label='Normalized data')

    mask1 = mask2 = np.full(map1.shape, True, dtype=bool)
    if map1_fit_limits:
        mask1 = (map1 > map1_fit_limits[0]) & (map1 < map1_fit_limits[1])
    if map2_fit_limits:
        mask2 = (map2 > map2_fit_limits[0]) & (map2 < map2_fit_limits[1])
    map1, map2 = map1[mask1 & mask2], map2[mask1 & mask2]

    if square_root_fit:
        def sqrt_model(x, a, b):
            return a * np.sqrt(x) + b

        try:
            params, _ = curve_fit(sqrt_model, map1, map2)
            x_fit = np.linspace(min(map1), max(map1), 100)
            y_fit = sqrt_model(x_fit, *params)

            y_pred = sqrt_model(map1, *params)
            rss_alt = np.sum((map2 - y_pred) ** 2)
            tss = np.sum((map2 - np.mean(map2)) ** 2)
            r_squared = 1 - (rss_alt / tss)
            df_model = len(params) - 1
            df_residual = len(map2) - len(params)
            f_stat = ((tss - rss_alt) / df_model) / (rss_alt / df_residual)
            p_value = 1 - f.cdf(f_stat, df_model, df_residual)

            ax.plot(x_fit, y_fit, color='red', label=r'Fit: $y = a\sqrt{x + b} + c$')
            pval = format_p_value(p_value)
            ax.text(0.60 - len(pval) / 200, 0.88, f'$\\mathbf{{R^2}}$: {np.round(r_squared, 3)}\np-value: {pval}',
                   fontsize=12, fontweight='bold', color='red', ha='left', transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.75, edgecolor='red', boxstyle='round,pad=0.5'))
        except RuntimeError:
            ax.text(0.5, 0.9, 'Fit failed', fontsize=12, color='red', ha='center', transform=ax.transAxes)
    elif pearson is not None and p_value is not None:
        pval = format_p_value(p_value)
        ax.text(0.60 - len(pval) / 200, 0.88, f'Pearson: {np.round(pearson, 3)}\np-value: {pval}', fontsize=12,
               fontweight='bold', color='red', ha='left', transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.75, edgecolor='red', boxstyle='round,pad=0.5'))

    ax.set_xlabel(label1, fontsize=18)
    ax.set_ylabel(label2, fontsize=18)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.tick_params(axis='both', which='major', length=4, width=1.5)
    plt.axis('equal')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    if square_root_fit:
        ax.legend(loc='upper left')
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_correlation_with_radii(reference_grid, other_grid, contour, radii, label_a="Reference", label_b="Other",
                                results_folder=None, results_name="CorrelationRadii", title=""):
    """Plot the reference and other grids together with circles of the given radii drawn around each
    reference point - use it to sanity-check a `correlate_around_reference_grid` call's radius choice
    visually (no excessive overlap, no big gaps) before trusting the correlation."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(other_grid[:, 0], other_grid[:, 1], s=10, color='grey', label=label_b, alpha=0.5)
    ax.scatter(reference_grid[:, 0], reference_grid[:, 1], s=15, color='blue', label=label_a)
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
                           results_name="correlation_masks"):
    """Plot two datasets living on the same grid side by side (e.g. two groups from `pairwise_correlate_groups`).
    Supports boolean or numeric data."""
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    is_a_bool = data_a.dtype == bool or np.array_equal(data_a, data_a.astype(bool))
    is_b_bool = data_b.dtype == bool or np.array_equal(data_b, data_b.astype(bool))
    bool_cmap = mcolors.ListedColormap(['black', 'red'])
    bool_norm = mcolors.BoundaryNorm([-.5, 0.5, 1.5], bool_cmap.N)

    if is_a_bool:
        axs[0].scatter(grid[:, 0], grid[:, 1], c=data_a.astype(int), cmap=bool_cmap, norm=bool_norm, s=100,
                      marker='s')
    else:
        sc1 = axs[0].scatter(grid[:, 0], grid[:, 1], c=data_a, cmap='grey', s=100, marker='s')
        fig.colorbar(sc1, ax=axs[0], orientation='vertical', label='Value')

    axs[0].plot(contour[:, 0], contour[:, 1], 'k-', linewidth=2, label="Contour")
    axs[0].set_title(label_a)
    axs[0].set_aspect('equal')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    if is_b_bool:
        axs[1].scatter(grid[:, 0], grid[:, 1], c=data_b.astype(int), cmap=bool_cmap, norm=bool_norm, s=100,
                      marker='s')
    else:
        sc2 = axs[1].scatter(grid[:, 0], grid[:, 1], c=data_b, cmap='hot', s=100, marker='s')
        fig.colorbar(sc2, ax=axs[1], orientation='vertical', label='Value')

    axs[1].plot(contour[:, 0], contour[:, 1], 'k-', linewidth=2, label="Contour")
    axs[1].set_title(label_b)
    axs[1].set_aspect('equal')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    if results_folder:
        os.makedirs(results_folder, exist_ok=True)
        fig.savefig(os.path.join(results_folder, f"{results_name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig
