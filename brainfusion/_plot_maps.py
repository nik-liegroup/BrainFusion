import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.stats import f
from tifffile import imwrite
from skimage.transform import AffineTransform

from brainfusion._utils import mask_contour

plt.rcParams['svg.fonttype'] = 'none'


def plot_brainfusion_results(analysis_file, results_folder, key_quant, image_dataset=False, cbar_label='', cmap='afmhot',
                             marker_size=20, mask=True, vmin=None, vmax=None, verify_trafo=False, plot_background=False,
                             correlation=False, **kwargs):
    print(f'Plotting: {os.path.basename(os.path.dirname(results_folder))}.')
    os.makedirs(results_folder, exist_ok=True)

    #--- Plot overlayed contours and average contour ---#
    # Choose first DTW matched template contour as template contour
    template_contour = analysis_file['template_contours'][0]

    # Plot all original contours and the averaged contour
    cont = analysis_file['measurement_contours']
    iterator = list(cont.values()) if type(cont) is dict else cont
    fig = plot_contours(template_contour, iterator)
    output_path = os.path.join(results_folder, 'matched_contours.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    #--- Plot transformation fields and transformation results ---#
    # Get extended interpolation grid
    interpolated_grid = analysis_file["measurement_interpolated_grid"]
    interpolated_grid_shape = tuple(analysis_file["measurement_interpolated_grid_shape"])
    interpolated_avg_data = analysis_file['measurement_interpolated_dataset'][key_quant] if not correlation else None

    # Transform into regular grid and matrix if image data
    if image_dataset:
        H, W = tuple(analysis_file["measurement_interpolated_grid_shape"])
        interpolated_grid = interpolated_grid.reshape(H, W, 2)
        interpolated_avg_data = analysis_file['measurement_interpolated_dataset'][key_quant].reshape(H, W)

    for index, c in enumerate(analysis_file['measurement_datasets']):
        # Extract arrays
        matched_contour = analysis_file['measurement_contours'][index]
        raw_data = analysis_file['measurement_datasets'][index]
        matched_grid = analysis_file['measurement_grids'][index]
        trafo_data = analysis_file['measurement_trafo_datasets'][index]
        trafo_grid = analysis_file['measurement_trafo_grids'][index]

        # Transform into regular grid and matrix if image data
        if image_dataset:
            h, w = tuple(analysis_file['measurement_grids_shape'][index])
            matched_grid = matched_grid.reshape(h, w, 2)
            raw_data = {key: value.reshape(h, w) for key, value in raw_data.items()}
            trafo_grid = interpolated_grid
            trafo_data = {key: value.reshape(H, W) for key, value in trafo_data.items()}

        if verify_trafo:
            cmap = 'viridis'  # Changes colour mapping to avoid confusion with real data
            matched_grid = analysis_file['verification_grids'][index]
            trafo_grid = analysis_file['verification_trafo_grids'][index]
            raw_data = {key_quant: np.random.choice(np.linspace(1, 10, 10), size=matched_grid.shape[0])}

        # Plot trafo field and the original and transformed data grids
        fig = plot_transformed_grid(matched_contour,
                                    analysis_file['template_contours'][index],
                                    raw_data,
                                    matched_grid,
                                    trafo_grid,
                                    affine_matrix=analysis_file['affine_matrices'][index],
                                    key_quant=key_quant,
                                    trafo_data=trafo_data,
                                    cbar_label=cbar_label,
                                    cmap=cmap,
                                    marker_size=marker_size,
                                    vmin=vmin,
                                    vmax=vmax,
                                    mask=mask)

        output_path = os.path.join(results_folder, f'Transformed_{analysis_file['measurement_filenames'][index]}.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Plot original grid on brightfield background image
        if plot_background:
            affine_matrix = analysis_file['affine_matrices'][index]
            scale_matrix = analysis_file['scale_matrices'][index]
            comp_matrix = scale_matrix @ affine_matrix
            affine_trafo = AffineTransform(matrix=comp_matrix)
            grid_trafo = affine_trafo(matched_grid)
            contour_trafo = affine_trafo(matched_contour)

            fig = plot_map_on_image(analysis_file['background_image'][index],
                                    raw_data[key_quant],
                                    grid_trafo,
                                    contour_trafo,
                                    scale=1,
                                    label='',
                                    cmap=cmap,
                                    marker_size=marker_size / 50,
                                    vmin=vmin,
                                    vmax=vmax,
                                    mask=True,
                                    alpha=0.9)

            output_path = os.path.join(results_folder,
                                       f'Original_{analysis_file['measurement_filenames'][index]}.png')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

    if not correlation:
        #--- Plot averaged data map ---#
        fig = plot_average_map(interpolated_avg_data,
                               interpolated_grid,
                               template_contour,
                               cbar_label=cbar_label,
                               cmap=cmap,
                               marker_size=marker_size,
                               vmin=vmin,
                               vmax=vmax,
                               mask=mask)
        output_path = os.path.join(results_folder, f'Averaged_Maps.svg')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save averaged data map as tif
        H, W = tuple(analysis_file["measurement_interpolated_grid_shape"])
        regular_grid = analysis_file['measurement_interpolated_grid'].reshape(H, W, 2)
        value_matrix = analysis_file['measurement_interpolated_dataset'][key_quant].reshape(H, W)

        # Save as .tif
        matrix, _, (pixel_size_x, pixel_size_y) = plot_contour_on_image(value_matrix, regular_grid, template_contour,
                                                                        cmap='grey', mask=True)
        matrix_to_save = np.flipud(matrix)  # Flip vertically to match origin='lower' in imshow

        # Save cropped image
        output_path = os.path.join(results_folder, f'Averaged_Maps_Image.tif')
        imwrite(output_path,
                matrix_to_save.astype('float32'),
                imagej=True,
                resolution=(1e-6 / pixel_size_x, 1e-6 / pixel_size_y),
                metadata={
                    'unit': 'um',
                    'axes': 'YX'
                }
                )

        plt.close()


def plot_contours(template_contour, matched_contours):
    fig, ax = plt.subplots(figsize=(8, 8))

    for i, contour in enumerate(matched_contours):
        # Plot the matched contours (only label the first one)
        if i == 0:
            ax.plot(contour[:, 0], contour[:, 1], color='grey', linestyle='-', linewidth=1.5, alpha=0.5,
                     label='DTW Matched Contours')
            ax.scatter(contour[0, 0], contour[0, 1], color='grey', s=25, label='Initial Coordinate')
        else:
            # No label for subsequent contours
            ax.plot(contour[:, 0], contour[:, 1], color='grey', linestyle='-', linewidth=1.5, alpha=0.5)
            ax.scatter(contour[0, 0], contour[0, 1], color='grey', s=25)

    # Plot the averaged contour
    ax.plot(template_contour[:, 0], template_contour[:, 1], color='blue', linestyle='--', linewidth=3,
             label='Template Contour')
    ax.scatter(template_contour[0, 0], template_contour[0, 1], color='blue', s=30, zorder=6,
                label='Initial Tmp. Coordinate')

    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()

    return fig


def plot_transformed_grid(contour, template_contour, data, grid, trafo_grid, affine_matrix, key_quant, trafo_data=None,
                          cbar_label='', cmap='afmhot', marker_size=30, vmin=None, vmax=None, mask=False):
    # Compute vmin and vmax
    values = data[key_quant]
    vmin = np.nanmin(values) if vmin is None else vmin
    vmax = np.nanmax(values) if vmax is None else vmax

    # Create subplots with three side-by-side plots
    fig, axes = plt.subplots(1, 3, figsize=(25, 7))

    # --- DTW transformation field ---#
    # Plot contours
    axes[0].plot(contour[:, 0], contour[:, 1], c='grey', linestyle='-')
    axes[0].scatter(contour[:, 0], contour[:, 1], c='k', s=5)
    axes[0].plot(template_contour[:, 0], template_contour[:, 1], 'blue', linestyle='-')
    axes[0].scatter(template_contour[:, 0], template_contour[:, 1], c='b', s=5)

    # Plot vectors between contours
    axes[0].quiver(contour[:, 0], contour[:, 1],
                   template_contour[:, 0] - contour[:, 0], template_contour[:, 1] - contour[:, 1],
                   angles='xy', scale_units='xy', scale=1, color='r', alpha=0.8, zorder=3)

    axes[0].set_title('DTW with Curvature Penalty', fontsize=20)

    # --- Original and transformed contours ---#
    axes[1].plot(contour[:, 0], contour[:, 1], color='grey', linestyle='-', linewidth=3,
                 label='Original Contour')

    # Transformed contour
    axes[2].plot(template_contour[:, 0], template_contour[:, 1], color='blue', linestyle='--', linewidth=4,
                 label='Template Contour')

    # Plot image data
    if len(grid.shape) == 3:
        # Original Grid
        N, M, _ = grid.shape
        coords = grid.reshape(-1, 2)

        # Test which points are inside contour
        if mask:
            mask1 = mask_contour(contour, coords)
            mask1 = mask1.reshape(N, M)
            for key, value in data.items():
                value = value.astype(float)
                value[~mask1] = np.nan
                data[key] = value

        # Get extent from grid
        x_min = np.min(grid[:, :, 0])
        x_max = np.max(grid[:, :, 0])
        y_min = np.min(grid[:, :, 1])
        y_max = np.max(grid[:, :, 1])
        extent = [x_min, x_max, y_min, y_max]

        axes[1].imshow(data[key_quant], extent=extent, origin='lower', cmap=cmap)

        # Transformed Grid
        N, M, _ = trafo_grid.shape
        coords = trafo_grid.reshape(-1, 2)

        # Test which points are inside contour
        if mask:
            mask2 = mask_contour(template_contour, coords)
            mask2 = mask2.reshape(N, M)
            for key, value in trafo_data.items():
                value = value.astype(float)
                value[~mask2] = np.nan
                trafo_data[key] = value

        # Get extent from grid
        x_min = np.min(trafo_grid[:, :, 0])
        x_max = np.max(trafo_grid[:, :, 0])
        y_min = np.min(trafo_grid[:, :, 1])
        y_max = np.max(trafo_grid[:, :, 1])
        extent = [x_min, x_max, y_min, y_max]

        # Plot
        trafo_map = axes[2].imshow(trafo_data[key_quant], extent=extent, origin='lower', cmap=cmap)

    # Plot scattered data
    else:
        # Set mask
        if mask:
            mask_1 = mask_contour(contour, grid)
        else:
            mask_1 = np.full(grid.shape[0], True)

        # Plot original data points
        axes[1].scatter(np.ma.masked_where(~mask_1, grid[:, 0]),
                        np.ma.masked_where(~mask_1, grid[:, 1]),
                        c=data[key_quant],
                        cmap=cmap,
                        s=marker_size,
                        vmin=vmin,
                        vmax=vmax)

        # Transformed contour and grid
        axes[2].plot(template_contour[:, 0], template_contour[:, 1], color='blue', linestyle='--', linewidth=4,
                     label='Template Contour')

        # Set mask
        if mask:
            mask_2 = mask_contour(template_contour, trafo_grid)
        else:
            mask_2 = np.full(trafo_grid.shape[0], True)

        # Plot transformed data points
        trafo_map = axes[2].scatter(np.ma.masked_where(~mask_2, trafo_grid[:, 0]),
                                    np.ma.masked_where(~mask_2, trafo_grid[:, 1]),
                                    c=data[key_quant],
                                    cmap=cmap,
                                    s=marker_size,
                                    vmin=vmin,
                                    vmax=vmax)

    axes[1].set_title('Original Data Map', fontsize=20)
    axes[2].set_title('Transformed Data Map', fontsize=20)

    # Add colorbar
    cbar = fig.colorbar(trafo_map, ax=axes[2])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, size=30)

    # --- Auto-zoom with 10% margin ---
    x_lim, y_lim = get_zoom_limits(template_contour)
    for ax in axes:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # Ensures same unit length in x and y
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig


def plot_average_map(data_avg, grid_avg, template_contour, cbar_label='', cmap='viridis', marker_size=15, vmin=None,
                     vmax=None, mask=True):
    # Compute vmin and vmax
    data_avg = data_avg.astype(float)
    vmin = np.nanmin(data_avg) if vmin is None else vmin
    vmax = np.nanmax(data_avg) if vmax is None else vmax

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot template contour
    ax.plot(template_contour[:, 0], template_contour[:, 1], 'b--', linewidth=5, label='Template Contour')

    # Plot image data
    if grid_avg.ndim == 3:
        N, M, _ = grid_avg.shape
        coords = grid_avg.reshape(-1, 2)

        if mask:
            mask1 = mask_contour(template_contour, coords)
            mask1 = mask1.reshape(N, M)
            data_avg[~mask1] = np.nan

        # Get extent from grid
        x_min = np.min(grid_avg[:, :, 0])
        x_max = np.max(grid_avg[:, :, 0])
        y_min = np.min(grid_avg[:, :, 1])
        y_max = np.max(grid_avg[:, :, 1])
        extent = [x_min, x_max, y_min, y_max]

        heatmap = ax.imshow(data_avg, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    # Plot scattered data
    else:
        if mask:
            mask_values = mask_contour(template_contour, grid_avg)
        else:
            mask_values = np.full(grid_avg.shape[0], True)

        heatmap = ax.scatter(
            np.ma.masked_where(~mask_values, grid_avg[:, 0]),
            np.ma.masked_where(~mask_values, grid_avg[:, 1]),
            c=data_avg, cmap=cmap, s=marker_size, marker='s', vmin=vmin, vmax=vmax
        )

    # Colorbar settings
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(cbar_label, size=20)

    # --- Auto-zoom with 10% margin ---
    x_lim, y_lim = get_zoom_limits(template_contour)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')  # Ensures same unit length in x and y
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    return fig


def plot_contour_on_image(img, grid, contour, cmap='grey', mask=False):
    """
    Normalize a matrix to [0, max] and convert to uint8 or uint16 and add contour to image.
    """
    img = img.astype(float)
    N, M = img.shape
    coords = grid.reshape(-1, 2)

    # Test which points are inside contour
    if mask:
        mask = mask_contour(contour, coords)
        mask = mask.reshape(N, M)
        img[~mask] = np.nan

    # Get extent from grid
    x_min = np.min(grid[:, :, 0])
    x_max = np.max(grid[:, :, 0])
    y_min = np.min(grid[:, :, 1])
    y_max = np.max(grid[:, :, 1])
    extent = [x_min, x_max, y_min, y_max]

    # Compute pixel-to-µm scale
    x_extent_um = x_max - x_min  # Width in µm
    y_extent_um = y_max - y_min  # Height in µm

    pixel_size_x = x_extent_um / M  # µm per pixel in X
    pixel_size_y = y_extent_um / N  # µm per pixel in Y

    # Plot
    fig, ax = plt.subplots()
    ax.imshow(img, extent=extent, origin='lower', cmap=cmap)
    ax.plot(contour[:, 0], contour[:, 1], 'b--', linewidth=2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return img, fig, (pixel_size_x, pixel_size_y)


def plot_map_on_image(img, data, grid, contour, scale=1, label='', cmap='viridis', marker_size=15,
                      vmin=None, vmax=None, mask=False, alpha=0.9):
    # Create figure and axis
    fig, ax = plt.subplots()

    # Scale image to µm
    height_in_mu = img.shape[0] / scale
    width_in_mu = img.shape[1] / scale

    # Plot background image and heatmap
    ax.imshow(img, cmap='gray', aspect='equal', origin='lower', extent=[0, width_in_mu, 0, height_in_mu], alpha=alpha)

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
                         vmax=vmax)

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


def plot_correlation_with_radii(sparse_grid, dense_grid, contour, radii, results_folder=None, title=""):
    """
    Plots sparse and dense grids with circles of given radius around sparse points.
    """
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot dense Points
    ax.scatter(dense_grid[:, 0], dense_grid[:, 1], s=10, color='grey', label="Dense Points", alpha=0.5)

    # Plot sparse Points
    ax.scatter(sparse_grid[:, 0], sparse_grid[:, 1], s=15, color='blue', label="Sparse Points")

    # Plot contour
    ax.plot(contour[:, 0], contour[:, 1], 'k-', linewidth=2, label="Contour")

    # Draw Circles Around sparse Points
    for i, sparse_point in enumerate(sparse_grid):
        circle = patches.Circle(sparse_point, radii[i], color='blue', alpha=0.2)
        ax.add_patch(circle)

    # Labels and Legend
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=18)
    # ax.legend(loc="lower left", fontsize=15)
    ax.set_aspect('equal')

    if isinstance(results_folder, str):
        save_path = os.path.join(results_folder, f'CorrelationAnalysis.png')
        os.makedirs(results_folder, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def plot_correlation_masks(sparse_grid, sparse_data, dense_data, contour, results_folder=None):
    """
    Plots sparse and dense grids next to each other with color-coded values.
    Supports boolean or numeric data.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Detect if data is boolean
    is_sparse_bool = sparse_data.dtype == bool or np.array_equal(sparse_data, sparse_data.astype(bool))
    is_dense_bool = dense_data.dtype == bool or np.array_equal(dense_data, dense_data.astype(bool))

    # Define discrete colormap for booleans
    bool_cmap = mcolors.ListedColormap(['black', 'red'])
    bool_norm = mcolors.BoundaryNorm([-.5, 0.5, 1.5], bool_cmap.N)

    # Sparse points
    if is_sparse_bool:
        sc1 = axs[0].scatter(sparse_grid[:, 0], sparse_grid[:, 1], c=sparse_data.astype(int),
                             cmap=bool_cmap, norm=bool_norm, s=100, marker='s')
    else:
        sc1 = axs[0].scatter(sparse_grid[:, 0], sparse_grid[:, 1], c=sparse_data, cmap='grey', s=100, marker='s')
        plt.colorbar(sc1, ax=axs[0], orientation='vertical', label='Value')

    axs[0].plot(contour[:, 0], contour[:, 1], 'k-', linewidth=2, label="Contour")
    axs[0].set_title("Sparse Data")
    axs[0].set_aspect('equal')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # Dense points
    if is_dense_bool:
        sc2 = axs[1].scatter(sparse_grid[:, 0], sparse_grid[:, 1], c=dense_data.astype(int),
                             cmap=bool_cmap, norm=bool_norm, s=100, marker='s')
    else:
        sc2 = axs[1].scatter(sparse_grid[:, 0], sparse_grid[:, 1], c=dense_data, cmap='hot', s=100, marker='s')
        plt.colorbar(sc2, ax=axs[1], orientation='vertical', label='Value')

    axs[1].plot(contour[:, 0], contour[:, 1], 'k-', linewidth=2, label="Contour")
    axs[1].set_title("Dense Data")
    axs[1].set_aspect('equal')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    if results_folder:
        import os
        os.makedirs(results_folder, exist_ok=True)
        save_path = os.path.join(results_folder, "correlation_masks.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def get_zoom_limits(contour, margin_fraction=0.1):
    """
    Calculates x and y limits for a plot to ensure equal aspect ratio and a margin around the contour.
    """
    x_min, x_max = contour[:, 0].min(), contour[:, 0].max()
    y_min, y_max = contour[:, 1].min(), contour[:, 1].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    margin = margin_fraction * max_range

    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2

    x_lim = (x_mid - max_range / 2 - margin, x_mid + max_range / 2 + margin)
    y_lim = (y_mid - max_range / 2 - margin, y_mid + max_range / 2 + margin)

    return x_lim, y_lim



def format_p_value(p):
    if p < 0.0001:
        return "p<0.001"
    elif 0.001 <= p < 0.20:
        return f"p={p:.3f}"
    elif p >= 0.20:
        return f"p={p:.2f}"
    else:
        return "Invalid p-value"
