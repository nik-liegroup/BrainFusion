import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.stats import f

from brainfusion._utils import mask_contour
from brainfusion._transform_2Dmap import transform_grid2contour

plt.rcParams['svg.fonttype'] = 'none'


def plot_brainfusion_results(analysis_file, results_folder, key_quant, label='', cmap='afmhot',
                             marker_size=20, vmin=None, vmax=None, verify_trafo=False, **kwargs):
    print(f'Plotting: {os.path.basename(os.path.dirname(results_folder))}.')
    os.makedirs(results_folder, exist_ok=True)

    # Choose first DTW matched template contour as template contour
    template_contour = analysis_file['template_contours'][0]

    # Plot transformation fields and transformation results
    for index, c in enumerate(analysis_file['measurement_datasets']):
        # Extract arrays
        matched_contour = analysis_file['measurement_contours'][index]
        raw_data = analysis_file['measurement_datasets'][index]
        matched_grid = analysis_file['measurement_grids'][index]
        trafo_grid = analysis_file['measurement_trafo_grids'][index]

        if verify_trafo:
            cmap = 'viridis'  # Changes colour mapping to avoid confusion with real data
            matched_grid = analysis_file['verification_grids'][index]
            trafo_grid = analysis_file['verification_trafo_grids'][index]
            raw_data = np.random.choice(np.linspace(1, 10, 10), size=matched_grid.shape[0])

        # Plot trafo field and the original and transformed data grids
        fig = plot_transformed_grid(matched_contour,
                                    analysis_file['template_contours'][index],
                                    raw_data,
                                    matched_grid,
                                    trafo_grid,
                                    label=label,
                                    cmap=cmap,
                                    marker_size=marker_size,
                                    vmin=vmin,
                                    vmax=vmax,
                                    mask=True)
        output_path = os.path.join(results_folder, f'Transformed_{analysis_file['measurement_filenames'][index]}.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Plot all original contours and the averaged contour
    cont = analysis_file['measurement_contours']
    iterator = list(cont.values()) if type(cont) is dict else cont
    fig = plot_contours(template_contour, iterator)
    output_path = os.path.join(results_folder, 'matched_contours.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot averaged data map
    fig = plot_average_map(analysis_file['measurement_interpolated_dataset'][key_quant],
                           analysis_file['measurement_interpolated_grid'],
                           template_contour,
                           label=label,
                           cmap=cmap,
                           marker_size=marker_size,
                           vmin=vmin,
                           vmax=vmax,
                           mask=True)

    output_path = os.path.join(results_folder, f'Averaged_Myelin_Maps.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_transformed_grid(contour, template_contour, data, grid, trafo_grid, label='', cmap='afmhot', marker_size=30,
                          vmin=None, vmax=None, mask=False):
    # Create subplots with three side-by-side plots
    fig, axes = plt.subplots(1, 3, figsize=(25, 7))

    # DTW transformation field
    axes[0].plot(contour[:, 0], contour[:, 1], c='grey', linestyle='-')
    axes[0].scatter(contour[:, 0], contour[:, 1], c='k', s=5)
    axes[0].plot(template_contour[:, 0], template_contour[:, 1], 'blue', linestyle='-')
    axes[0].scatter(template_contour[:, 0], template_contour[:, 1], c='b', s=5)

    # Plot vectors between contours
    axes[0].quiver(contour[:, 0], contour[:, 1],
                   template_contour[:, 0] - contour[:, 0], template_contour[:, 1] - contour[:, 1],
                   angles='xy', scale_units='xy', scale=1, color='r', alpha=0.8, zorder=3)

    axes[0].set_title('DTW with Curvature Penalty', fontsize=20)
    axes[0].axis('equal')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Original contour and grid
    axes[1].plot(contour[:, 0], contour[:, 1], color='grey', linestyle='-', linewidth=3,
                 label='Original Contour')

    # Set mask
    if mask:
        mask_1 = mask_contour(contour, grid)
        mask_tmp = data >= 0
        mask_1 = mask_1 & mask_tmp
    else:
        mask_1 = np.full(grid.shape[1], True)

    # Plot original data points
    axes[1].scatter(np.ma.masked_where(~mask_1, grid[:, 0]),
                    np.ma.masked_where(~mask_1, grid[:, 1]),
                    c=data,
                    cmap=cmap,
                    s=marker_size,
                    vmin=vmin,
                    vmax=vmax)

    axes[1].set_title('Original Data Map', fontsize=20)
    axes[1].axis('equal')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Transformed contour and grid
    axes[2].plot(template_contour[:, 0], template_contour[:, 1], color='blue', linestyle='--', linewidth=4,
                 label='Template Contour')

    # Set mask
    if mask:
        mask_2 = mask_contour(template_contour, trafo_grid)
        mask_tmp = data >= 0
        mask_2 = mask_2 & mask_tmp
    else:
        mask_2 = np.full(trafo_grid.shape[0], True)

    # Plot transformed data points
    trafo_map = axes[2].scatter(np.ma.masked_where(~mask_2, trafo_grid[:, 0]),
                                np.ma.masked_where(~mask_2, trafo_grid[:, 1]),
                                c=data,
                                cmap=cmap,
                                s=marker_size,
                                vmin=vmin,
                                vmax=vmax)

    axes[2].set_title('Transformed Data Map', fontsize=20)
    axes[2].axis('equal')
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(trafo_map, ax=axes[2])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label, size=30)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig


def plot_contours(template_contour, matched_contours):
    fig = plt.figure(figsize=(8, 8))

    for i, contour in enumerate(matched_contours):
        # Plot the matched contours (only label the first one)
        if i == 0:
            plt.plot(contour[:, 0], contour[:, 1], color='grey', linestyle='-', linewidth=1.5, alpha=0.5,
                     label='DTW Matched Contours')
            plt.scatter(contour[0, 0], contour[0, 1], color='grey', s=25, label='Initial Coordinate')
        else:
            # No label for subsequent contours
            plt.plot(contour[:, 0], contour[:, 1], color='grey', linestyle='-', linewidth=1.5, alpha=0.5)
            plt.scatter(contour[0, 0], contour[0, 1], color='grey', s=25)

    # Plot the averaged contour
    plt.plot(template_contour[:, 0], template_contour[:, 1], color='blue', linestyle='--', linewidth=3,
             label='Template Contour')
    plt.scatter(template_contour[0, 0], template_contour[0, 1], color='blue', s=30, zorder=6,
                label='Initial Tmp. Coordinate')

    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.legend()

    return fig


def plot_average_map(data_avg, grid_avg, template_contour, label='', cmap='viridis', marker_size=15, vmin=None,
                     vmax=None, mask=True):
    # Create plot
    fig = plt.figure(figsize=(8, 8))

    # Plot template contour
    plt.plot(template_contour[:, 0], template_contour[:, 1], 'b--', linewidth=5, label='Template Contour')

    # Set mask
    if mask is True:
        mask = mask_contour(template_contour, grid_avg)
        mask_tmp = data_avg >= 0
        mask = mask & mask_tmp
    else:
        mask = np.full(grid_avg.shape[0], True)

    # Plot averaged data
    heatmap = plt.scatter(np.ma.masked_where(~mask, grid_avg[:, 0]),
                          np.ma.masked_where(~mask, grid_avg[:, 1]),
                          c=data_avg,
                          cmap=cmap,
                          s=marker_size / 20,
                          marker='s',
                          alpha=1,
                          vmin=vmin,
                          vmax=vmax)

    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])

    # Add colorbar
    cbar = fig.colorbar(heatmap)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label, size=20)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig


def plot_map_on_image(img, data, grid, contour, folder_name, scale=1, label='', cmap='viridis', marker_size=15,
                      vmin=None, vmax=None, log=False, mask=False, alpha=0.9):
    # Create figure and axis
    fig, ax = plt.subplots()

    # Scale image to µm
    height_in_mu = img.shape[0] / scale
    width_in_mu = img.shape[1] / scale

    # Plot background image and heatmap
    ax.imshow(img, cmap='gray', aspect='equal', origin='lower', extent=[0, width_in_mu, 0, height_in_mu], alpha=alpha)

    # Use logarithmic normalization
    if vmin or vmin:
        norm = None
    else:
        norm = mcolors.LogNorm() if log else None

    if mask:
        mask = mask_contour(contour, grid)
    else:
        mask = np.full(grid.shape[0], True)

    heatmap = ax.scatter(np.ma.masked_where(~mask, grid[:, 0]),
                         np.ma.masked_where(~mask, grid[:, 1]),
                         c=data,
                         cmap=cmap,
                         s=marker_size,
                         marker='s',
                         edgecolors='none',
                         alpha=1,
                         vmin=vmin,
                         vmax=vmax,
                         norm=norm)

    ax.set_title(f'{folder_name}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label, size=20)

    return fig


def plot_corr_maps(average_contour, afm_map, brillouin_map, grid, mask=True, marker_size=40, vmin=None, vmax=None):
    # Mask average data
    assert type(mask) is bool, print('The provided mask argument is not boolean!')
    if mask is True:
        mask = mask_contour(average_contour, grid)
    else:
        mask = np.full(grid.shape[0], True)

    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the AFM map
    axes[0].plot(average_contour[:, 0], average_contour[:, 1], 'r-', label='Average Contour')
    heatmap_brillouin = axes[0].scatter(np.ma.masked_where(~mask, grid[:, 0]),
                                        np.ma.masked_where(~mask, grid[:, 1]),
                                        c=afm_map,
                                        marker='s',
                                        cmap='hot',
                                        s=marker_size)

    axes[0].legend()
    axes[0].set_title('Transformed AFM data map')
    axes[0].grid()
    axes[0].axis('equal')

    # Plot the AFM map
    axes[1].plot(average_contour[:, 0], average_contour[:, 1], 'r-', label='Average Contour')
    heatmap_afm = axes[1].scatter(np.ma.masked_where(~mask, grid[:, 0]),
                                  np.ma.masked_where(~mask, grid[:, 1]),
                                  c=brillouin_map,
                                  marker='s',
                                  cmap='hot',
                                  s=marker_size,
                                  vmin=vmin,
                                  vmax=vmax)

    axes[1].legend()
    axes[1].set_title('Transformed Brillouin data map')
    axes[1].grid()
    axes[1].axis('equal')

    # Plot colorbars
    cbar = fig.colorbar(heatmap_brillouin, ax=axes[0])
    cbar.set_label('Reduced elastic modulus (Pa)')

    cbar = fig.colorbar(heatmap_afm, ax=axes[1])
    cbar.set_label('Brillouin shift (GHz)')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig


def plot_norm_corr(map1, map2, pearson=None, p_value=None, label1="X-axis", label2="Y-axis",
                   output_path=None, square_root_fit=False, map1_fit_limits=None, map2_fit_limits=None):
    # Ensure map1 and map2 are arrays
    map1, map2 = np.asarray(map1), np.asarray(map2)
    if map1.shape != map2.shape:
        raise ValueError("map1 and map2 must have the same shape.")

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(map1, map2, color='blue', s=10, label='Normalized data')

    # Filter for fitting
    mask1 = mask2 = np.full(map1.shape, True, dtype=bool)
    if map1_fit_limits:
        mask1 = (map1 > map1_fit_limits[0]) & (map1 < map1_fit_limits[1])

    if map2_fit_limits:
        mask2 = (map2 > map2_fit_limits[0]) & (map2 < map2_fit_limits[1])

    # Apply the combined mask
    combined_mask = mask1 & mask2
    map1, map2 = map1[combined_mask], map2[combined_mask]

    if square_root_fit:
        def sqrt_model(x, a, b):
            return a * np.sqrt(x) + b

        try:
            # Fit the square-root model
            params, _ = curve_fit(sqrt_model, map1, map2)
            a, b = params

            # Generate fitted values
            x_fit = np.linspace(min(map1), max(map1), 100)
            y_fit = sqrt_model(x_fit, *params)

            # Evaluate fit quality
            y_pred = sqrt_model(map1, *params)
            rss_alt = np.sum((map2 - y_pred) ** 2)
            y_mean = np.mean(map2)
            tss = np.sum((map2 - y_mean) ** 2)
            r_squared = 1 - (rss_alt / tss)
            rss_null = tss
            df_model = len(params) - 1
            df_residual = len(map2) - len(params)
            f_stat = ((rss_null - rss_alt) / df_model) / (rss_alt / df_residual)
            p_value = 1 - f.cdf(f_stat, df_model, df_residual)

            # Plot the fit
            ax.plot(x_fit, y_fit, color='red', label=r'Fit: $y = a\sqrt{x + b} + c$')

            # Annotate with R^2 and p-value
            pval = format_p_value(p_value)
            ax.text(0.60 - len(pval) / 200, 0.88, f'$\\mathbf{{R^2}}$: {np.round(r_squared, 3)}\np-value: {pval}',
                    fontsize=12, fontweight='bold', color='red', ha='left', transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='red', boxstyle='round,pad=0.5'))
        except RuntimeError:
            ax.text(0.5, 0.9, 'Fit failed', fontsize=12, color='red', ha='center', transform=ax.transAxes)
            r_squared = None

    else:
        # Add Pearson correlation and p-value
        if pearson is not None and p_value is not None:
            pval = format_p_value(p_value)
            ax.text(0.60 - len(pval) / 200, 0.88, f'Pearson: {np.round(pearson, 3)}\np-value: {pval}',
                    fontsize=12, fontweight='bold', color='red', ha='left', transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='red', boxstyle='round,pad=0.5'))

    # Customize plot
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

    plt.show()
    return fig


def plot_cumulative(analysis_file, raw_key, projected=True, results_folder=None, bin=0.01, label='', log=False,
                    loc='upper left', x_ticks=6, round=1):
    results_folder = results_folder or './figures_for_thesis'

    # Prepare bins
    all_values = np.concatenate([
        exp_value['raw_data'][f'{raw_key}_distribution']
        for exp_key, exp_value in analysis_file.items()
        if '#' in exp_key
    ])
    lower, upper = np.nanpercentile(all_values, 1), np.nanpercentile(all_values, 99)
    bins = np.arange(lower, upper + bin, bin)

    raw_results = []
    projected_results = []

    for exp_key, exp_value in analysis_file.items():
        if '#' in exp_key:
            # Raw data
            raw_data = exp_value['raw_data'][f'{raw_key}_distribution']
            dist, _ = np.histogram(raw_data, bins=bins)
            ndist = dist / sum(dist)
            cdist = np.cumsum(ndist) * 100
            raw_results.append(cdist)

            # Projected data
            if projected:
                proj_data = exp_value['raw_data'][f'{raw_key}_proj']
                proj_data_flat = np.sort(proj_data.flatten())
                proj_dist, _ = np.histogram(proj_data_flat, bins=bins)
                proj_ndist = proj_dist / sum(proj_dist)
                proj_cdist = np.cumsum(proj_ndist) * 100
                projected_results.append(proj_cdist)

    # Averaging raw results
    raw_cdist_aver = np.nanmean(raw_results, axis=0)
    raw_cdist_std = np.nanstd(raw_results, axis=0)
    raw_lower_bound = raw_cdist_aver - raw_cdist_std
    raw_upper_bound = raw_cdist_aver + raw_cdist_std

    # Averaging projected results if needed
    if projected:
        proj_cdist_aver = np.nanmean(projected_results, axis=0)
        proj_cdist_std = np.nanstd(projected_results, axis=0)
        proj_lower_bound = proj_cdist_aver - proj_cdist_std
        proj_upper_bound = proj_cdist_aver + proj_cdist_std

    # Plot cumulative distribution
    plt.figure(figsize=(4, 6))
    # Raw data plot
    plt.plot(bins[:-1], raw_cdist_aver, color='black', linewidth=3, label='3D dataset', linestyle='-')
    plt.fill_between(bins[:-1], raw_lower_bound, raw_upper_bound, color='#A9A9A9', alpha=0.4, edgecolor='none')

    # Projected data plot
    if projected:
        plt.plot(bins[:-1], proj_cdist_aver, color='blue', linewidth=3, label='Median z-axis projection',
                 linestyle='-')
        plt.fill_between(bins[:-1], proj_lower_bound, proj_upper_bound, color='#1E90FF', alpha=0.4, edgecolor='none')

    # Customize plot labels and title
    plt.xlabel(label, fontsize=21)
    plt.ylabel('Cumulative Percentage (%)', fontsize=21)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tick_params(axis='both', which='major', length=4, width=1.5)
    plt.ylim([0, 100])

    plt.xlim([lower, upper])
    plt.xticks(np.round(np.linspace(lower, upper, x_ticks), round), fontsize=14)

    if log:
        plt.xscale('log')

    if loc is not None:
        plt.legend(loc=loc, fontsize=12)

    # Save and show the plot
    save_path = os.path.join(results_folder, f'CumulativePercentage_{raw_key}.svg')
    os.makedirs(results_folder, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_correlation_with_radii(afm_grid, myelin_grid, afm_contour, radii, title=""):
    """
    Plots AFM and myelin grids with circles of given radius around AFM points.
    """
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot Myelin Points
    ax.scatter(myelin_grid[:, 0], myelin_grid[:, 1], s=10, color='grey', label="Myelin Points", alpha=0.5)

    # Plot AFM Points
    ax.scatter(afm_grid[:, 0], afm_grid[:, 1], s=15, color='blue', label="AFM Points")

    # Plot AFM Contour (Black Line)
    ax.plot(afm_contour[:, 0], afm_contour[:, 1], 'k-', linewidth=2, label="AFM Contour")

    # Draw Circles Around AFM Points
    for i, afm_point in enumerate(afm_grid):
        circle = patches.Circle(afm_point, radii[i], color='blue', alpha=0.2)
        ax.add_patch(circle)

    # Labels and Legend
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=18)
    # ax.legend(loc="lower left", fontsize=15)
    ax.set_aspect('equal')

    plt.show()
    return fig


def format_p_value(p):
    if p < 0.0001:
        return "p<0.001"
    elif 0.001 <= p < 0.20:
        return f"p={p:.3f}"
    elif p >= 0.20:
        return f"p={p:.2f}"
    else:
        return "Invalid p-value"
