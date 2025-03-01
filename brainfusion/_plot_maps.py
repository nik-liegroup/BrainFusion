import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from scipy.stats import f

from brainfusion._utils import mask_contour
from brainfusion._transform_2Dmap import transform_grid2contour
plt.rcParams['svg.fonttype'] = 'none'


def plot_experiments(analysis_file, results_folder, raw_data_key, average='interp', label='', cmap='viridis',
                     marker_size=20, vmin=None, vmax=None, **kwargs):
    avg_contour = analysis_file['average_contour']
    if average == 'interp':
        avg_data = analysis_file['interpolated_data']
        avg_grid = analysis_file['interpolated_grid']
    elif average == 'gmm':
        avg_data = analysis_file['gmm_data']
        avg_grid = analysis_file['gmm_grid']
    else:
        raise ValueError(f"Invalid value for 'average': {average}. Expected 'interp' or 'gmm'.")

    # Plot original maps on background images and plot original/transformed maps next to each other
    matched_contours = []
    for exp_key, exp_value in analysis_file.items():
        if '#' in exp_key:
            fig = plot_map_on_image(exp_value['brightfield_image'],
                                    exp_value['raw_data'][f'{raw_data_key}'],
                                    exp_value['raw_grid'],
                                    exp_value['original_contour'],
                                    exp_key,
                                    scale=exp_value['pix_per_um'],
                                    label=label,
                                    cmap=cmap,
                                    marker_size=marker_size,
                                    vmin=vmin,
                                    vmax=vmax,
                                    log=False,
                                    mask=True)
            output_path = os.path.join(results_folder, f'{exp_key}.png')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Plot original and transformed data grids
            fig = plot_trafo_map(exp_value['matched_contour'],
                                 avg_contour,
                                 exp_value['raw_data'][f'{raw_data_key}'],
                                 exp_value['matched_grid'],
                                 exp_value['trafo_grid'],
                                 label=label,
                                 cmap=cmap,
                                 marker_size=120,
                                 vmin=vmin,
                                 vmax=vmax,
                                 mask=True)
            output_path = os.path.join(results_folder, f'Transformed_{exp_key}.png')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Save matched contour
            matched_contours.append(exp_value['matched_contour'])

    # Plot all original contours and the averaged contour
    fig = plot_contours(avg_contour, matched_contours)
    output_path = os.path.join(results_folder, 'contours.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Overlay all transformed heatmaps and the averaged heatmap
    data_list = [value['raw_data'] for key, value in analysis_file.items() if '#' in key]
    grid_trafo_list = [value['trafo_grid'] for key, value in analysis_file.items() if '#' in key]

    fig = plot_average_map(data_list,
                           grid_trafo_list,
                           avg_data,
                           avg_grid,
                           avg_contour,
                           f'{raw_data_key}',
                           label=label,
                           cmap=cmap,
                           marker_size=120,
                           vmin=vmin,
                           vmax=vmax)
    output_path = os.path.join(results_folder, f'Averaged_{raw_data_key.capitalize()}_Maps.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_sc_experiments(analysis_file, results_folder, label='', cmap='grey', marker_size=20,
                        vmin=None, vmax=None, verify_trafo=False, **kwargs):
    print(f'Plotting spinal cord data: {os.path.basename(os.path.dirname(results_folder))}.')
    os.makedirs(results_folder, exist_ok=True)

    avg_data = analysis_file['myelin_interpolated_dataset']
    avg_grid = analysis_file['myelin_interpolated_grid']
    afm_contours = analysis_file['afm_contours']

    # Plot original maps on background images and plot original/transformed maps next to each other
    for index, c in enumerate(analysis_file['myelin_datasets']):
        # Extract arrays
        matched_contour = analysis_file['myelin_contours'][index]
        raw_data = analysis_file['myelin_datasets'][index]
        matched_grid = analysis_file['myelin_grids'][index]
        trafo_grid = analysis_file['myelin_trafo_grids'][index]

        if verify_trafo:
            cmap = 'viridis'
            num_points_per_axis = 50

            # Get min and max x, y coordinates
            min_x, min_y = np.min(matched_grid, axis=0) - [100, 100]
            max_x, max_y = np.max(matched_grid, axis=0) + [100, 100]

            # Generate linearly spaced grid points
            x_values = np.linspace(min_x, max_x, num_points_per_axis)
            y_values = np.linspace(min_y, max_y, num_points_per_axis)

            # Create meshgrid
            x_grid, y_grid = np.meshgrid(x_values, y_values)
            matched_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

            # Randomly sample values from the discrete list for each grid point
            raw_data = np.random.choice(np.linspace(1, 10, 10), size=matched_grid.shape[0])

            trafo_grid, _ = transform_grid2contour(matched_contour, afm_contours[index], matched_grid)

        # Plot original and transformed data grids
        fig = plot_trafo_map(matched_contour,
                             afm_contours[index],
                             raw_data,
                             matched_grid,
                             trafo_grid,
                             label=label,
                             cmap=cmap,
                             marker_size=marker_size,
                             vmin=vmin,
                             vmax=vmax,
                             mask=True)
        output_path = os.path.join(results_folder, f'Transformed_{analysis_file['myelin_filenames'][index]}.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Plot all original contours and the averaged contour
    cont = analysis_file['myelin_contours']
    iterator = list(cont.values()) if type(cont) is dict else cont
    fig = plot_contours(afm_contours[0], iterator)
    output_path = os.path.join(results_folder, 'matched_contours.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    fig = plot_average_map2(avg_data,
                            avg_grid,
                            afm_contours[0],
                            label=label,
                            cmap=cmap,
                            marker_size=marker_size,
                            vmin=vmin,
                            vmax=vmax)

    output_path = os.path.join(results_folder, f'Averaged_Myelin_Maps.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_contours(average_contour, matched_contours):
    fig = plt.figure(figsize=(8, 8))

    for i, contour in enumerate(matched_contours):
        # Plot the matched contours (only label the first one)
        if i == 0:
            plt.plot(contour[:, 0], contour[:, 1], color='grey', linestyle='-', linewidth=1.5, alpha=0.5,
                     label='Contours (Matched)')
            plt.scatter(contour[0, 0], contour[0, 1], color='orange', s=25, label='First Coordinate')
        else:
            # No label for subsequent contours
            plt.plot(contour[:, 0], contour[:, 1], color='grey', linestyle='-', linewidth=1.5, alpha=0.5)
            plt.scatter(contour[0, 0], contour[0, 1], color='orange', s=25)

    # Plot the averaged contour
    plt.plot(average_contour[:, 0], average_contour[:, 1], color='blue', linestyle='--', linewidth=4,
             label='Averaged Contour')
    plt.scatter(average_contour[0, 0], average_contour[0, 1], color='blue', s=25, zorder=6,
                label='First Avg. Coordinate')

    plt.legend()
    plt.axis('equal')

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


def plot_trafo_map(contour, avg_contour, data, grid, trafo_grid, label='', cmap='viridis', marker_size=30,
                   vmin=None, vmax=None, mask=False):
    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 3, figsize=(25, 7))

    axes[0].plot(contour[:, 0], contour[:, 1], c='grey', linestyle='-')
    axes[0].scatter(contour[:, 0], contour[:, 1], c='k', s=5)
    axes[0].plot(avg_contour[:, 0], avg_contour[:, 1], 'blue', linestyle='-')
    axes[0].scatter(avg_contour[:, 0], avg_contour[:, 1], c='b', s=5)

    axes[0].quiver(contour[:, 0], contour[:, 1],
               avg_contour[:, 0] - contour[:, 0], avg_contour[:, 1] - contour[:, 1],
               angles='xy', scale_units='xy', scale=1, color='r', alpha=0.8, zorder=3)

    axes[0].set_title('DTW with Curvature Penalty', fontsize=20)
    axes[0].axis('equal')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].plot(contour[:, 0], contour[:, 1], color='grey', linestyle='-', linewidth=3,
                 label='Original Contour')

    # Original grid
    if mask:
        mask_1 = mask_contour(contour, grid)
        mask_tmp = data >= 0
        mask_1 = mask_1 & mask_tmp
    else:
        mask_1 = np.full(grid.shape[1], True)

    axes[1].scatter(np.ma.masked_where(~mask_1, grid[:, 0]),
                    np.ma.masked_where(~mask_1, grid[:, 1]),
                    c=data,
                    cmap=cmap,
                    s=marker_size,
                    vmin=vmin,
                    vmax=vmax)

    #axes[1].legend()
    axes[1].set_title('Original Data Map', fontsize=20)
    #axes[1].grid()
    axes[1].axis('equal')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Plot the transformed contours
    #axes[2].plot(contour[:, 0], contour[:, 1], color='grey', linestyle='-', linewidth=3,
    #             label='Original Contour')
    axes[2].plot(avg_contour[:, 0], avg_contour[:, 1], color='blue', linestyle='--', linewidth=4,
                 label='Average Contour')

    # Transformed grid
    if mask:
        mask_2 = mask_contour(avg_contour, trafo_grid)
        mask_tmp = data >= 0
        mask_2 = mask_2 & mask_tmp
    else:
        mask_2 = np.full(trafo_grid.shape[0], True)

    heatmap = axes[2].scatter(np.ma.masked_where(~mask_2, trafo_grid[:, 0]),
                              np.ma.masked_where(~mask_2, trafo_grid[:, 1]),
                              c=data,
                              cmap=cmap,
                              s=marker_size,
                              vmin=vmin,
                              vmax=vmax)

    #axes[2].legend(loc=1)
    axes[2].set_title('Transformed Data Map', fontsize=20)
    #axes[2].grid()
    axes[2].axis('equal')
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # Plot colorbar
    cbar = fig.colorbar(heatmap, ax=axes[2])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label, size=30)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig


def calculate_arc_length(contour):
    # Calculate the arc length along the contour
    distances = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    arc_length = np.insert(np.cumsum(distances), 0, 0)  # Cumulative sum of distances
    return arc_length


def plot_average_map(data_maps, grid_trafos, data_avg, grid_avg, average_contour, data_variable, label='',
                     cmap='viridis', marker_size=15, vmin=None, vmax=None, mask=True):
    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Overlay all heatmaps on top of each other
    # Average contour
    axes[0].plot(average_contour[:, 0], average_contour[:, 1], 'b--', linewidth=5, label='Average Contour')

    # Heatmaps
    for i, _ in enumerate(data_maps):
        if mask is True:
            mask_1 = mask_contour(average_contour, grid_trafos[i])
        else:
            mask_1 = np.full(grid_trafos[i].shape[0], True)

        axes[0].scatter(np.ma.masked_where(~mask_1, grid_trafos[i][:, 0]),
                        np.ma.masked_where(~mask_1, grid_trafos[i][:, 1]),
                        c=data_maps[i][data_variable],
                        cmap=cmap,
                        s=marker_size,
                        alpha=0.5,
                        vmin=vmin,
                        vmax=vmax)

    axes[0].legend()
    axes[0].set_title('Pooled Data Maps', fontsize=20)
    axes[0].grid()
    axes[0].axis('equal')

    # Average contour
    axes[1].plot(average_contour[:, 0], average_contour[:, 1], 'b--', linewidth=5, label='Median Contour')

    # Mask average data
    if mask is True:
        mask_2 = mask_contour(average_contour, grid_avg)
    else:
        mask_2 = np.full(grid_avg.shape[0], True)

    # Plot average data
    heatmap = axes[1].scatter(np.ma.masked_where(~mask_2, grid_avg[:, 0]),
                              np.ma.masked_where(~mask_2, grid_avg[:, 1]),
                              c=data_avg[f'{data_variable}'],
                              cmap=cmap,
                              s=marker_size,
                              marker='s',
                              alpha=1,
                              vmin=vmin,
                              vmax=vmax)

    axes[1].legend()
    axes[1].set_title('Average Data Maps')
    axes[1].grid()
    axes[1].axis('equal')

    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=axes[1])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label, size=30)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig


def plot_average_map2(data_avg, grid_avg, average_contour, label='', cmap='viridis', marker_size=15,
                      vmin=None, vmax=None, mask=True):
    # Create plot
    fig = plt.figure(figsize=(8, 8))

    # Average contour
    plt.plot(average_contour[:, 0], average_contour[:, 1], 'b--', linewidth=5, label='AFM Contour')

    # Mask average data
    if mask is True:
        mask_2 = mask_contour(average_contour, grid_avg)
    else:
        mask_2 = np.full(grid_avg.shape[0], True)

    # Plot average data
    heatmap = plt.scatter(np.ma.masked_where(~mask_2, grid_avg[:, 0]),
                          np.ma.masked_where(~mask_2, grid_avg[:, 1]),
                          c=data_avg,
                          cmap=cmap,
                          s=marker_size,
                          marker='s',
                          alpha=1,
                          vmin=vmin,
                          vmax=vmax)

    plt.legend()
    plt.grid()
    plt.axis('equal')

    # Add colorbar
    cbar = fig.colorbar(heatmap)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label, size=30)

    # Adjust layout to avoid overlap
    plt.tight_layout()

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


def format_p_value(p):
    if p < 0.0001:
        return "p<0.001"
    elif 0.001 <= p < 0.20:
        return f"p={p:.3f}"
    elif p >= 0.20:
        return f"p={p:.2f}"
    else:
        return "Invalid p-value"