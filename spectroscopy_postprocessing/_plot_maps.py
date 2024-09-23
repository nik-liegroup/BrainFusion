import matplotlib.pyplot as plt
from _utilis import scatter_with_touching_squares


def plot_brillouin_maps(background_image, bm_data, bm_grid, folder_name):
    fig, ax = plt.subplots()

    ax.imshow(background_image, cmap='gray', aspect='equal')
    heatmap = ax.scatter(bm_grid[:, 0],  # Use x-y grid of first z-slice
                         bm_grid[:, 1],
                         c=bm_data,
                         cmap='viridis',
                         s=15,
                         alpha=0.75,
                         vmax=5.45,
                         vmin=5.10
                         )

    ax.set_title(f'{folder_name}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Brillouin shift (GHz)')

    return fig


def plot_afm_maps(background_image, data, grid, folder_name):
    # Create figure and axis
    fig, ax = plt.subplots()

    ax.imshow(background_image, origin='lower', cmap='gray', aspect='equal')

    # Plot heatmap
    heatmap = ax.scatter(grid[:, 0],
                         grid[:, 1],
                         c=data,
                         cmap='hot',
                         s=5,
                         alpha=0.75
                         )
    ax.set_title(f'{folder_name}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a colorbar and label it
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Reduced elastic modulus (Pa)')

    return fig


def plot_cont_func(original_contour, deformed_contour, trafo_contour, bm_data, bm_data_trafo_list,
                   bm_grid_points, grid_points_trafo, extended_grid):
    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the deformed grid using transformed grid values
    # Contours
    axes[0].plot(original_contour[:, 0], original_contour[:, 1], 'b-', label='Original Contour')
    axes[0].plot(deformed_contour[:, 0], deformed_contour[:, 1], 'r-', label='Deformed Contour')
    axes[0].plot(trafo_contour[:, 0], trafo_contour[:, 1], 'g--', label='Transformed Original Contour')

    # Original grid
    axes[0].scatter(bm_grid_points[:, 0],
                    bm_grid_points[:, 1],
                    c=bm_data,
                    cmap='viridis',
                    s=15,
                    label='Original Grid',
                    alpha=1)

    axes[0].invert_yaxis()
    axes[0].legend()
    axes[0].set_title('Original data map')
    axes[0].grid()
    axes[0].axis('equal')

    # Plot the deformed grid using new coordinates and transformed data values (griddata)
    # Contours
    axes[1].plot(original_contour[:, 0], original_contour[:, 1], 'b-', label='Original Contour')
    axes[1].plot(deformed_contour[:, 0], deformed_contour[:, 1], 'r-', label='Deformed Contour')
    axes[1].plot(trafo_contour[:, 0], trafo_contour[:, 1], 'g--', label='Transformed Original Contour')

    # Original and deformed data
    axes[1].scatter(grid_points_trafo[:, 0],
                    grid_points_trafo[:, 1],
                    c=bm_data,
                    cmap='viridis',
                    s=15,
                    label='Transformed Grid (New coordinates)',
                    alpha=1)
    axes[1].scatter(extended_grid[:, 0],
                    extended_grid[:, 1],
                    c=bm_data_trafo_list,
                    cmap='viridis',
                    s=15,
                    label='Transformed Grid (Griddata)',
                    alpha=0)

    axes[1].invert_yaxis()
    axes[1].legend()
    axes[1].set_title('Transformed data map')
    axes[1].grid()
    axes[1].axis('equal')

    #axes[0].set_xlim([0, 1023])
    #axes[0].set_ylim([200, 1023])
    #axes[1].set_xlim([0, 1023])
    #axes[1].set_ylim([200, 1023])

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig


def plot_cont_func_afm(original_contour, deformed_contour, trafo_contour, data, grid, data_trafo, grid_trafo):
    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the deformed grid using transformed grid values
    # Contours
    axes[0].plot(original_contour[:, 0], original_contour[:, 1], 'b-', label='Original Contour')
    axes[0].plot(deformed_contour[:, 0], deformed_contour[:, 1], 'r-', label='Deformed Contour')
    axes[0].plot(trafo_contour[:, 0], trafo_contour[:, 1], 'g--', label='Transformed Original Contour')

    # Original grid
    axes[0].scatter(grid[:, 0],
                    grid[:, 1],
                    c=data,
                    cmap='hot',
                    s=15,
                    label='Original Grid',
                    alpha=1)

    axes[0].invert_yaxis()
    axes[0].legend()
    axes[0].set_title('Original AFM data map')
    axes[0].grid()
    axes[0].axis('equal')

    # Plot the deformed grid using new coordinates and transformed data values (griddata)
    # Contours
    axes[1].plot(original_contour[:, 0], original_contour[:, 1], 'b-', label='Original Contour')
    axes[1].plot(deformed_contour[:, 0], deformed_contour[:, 1], 'r-', label='Deformed Contour')
    axes[1].plot(trafo_contour[:, 0], trafo_contour[:, 1], 'g--', label='Transformed Original Contour')

    # Original and deformed data
    axes[1].scatter(grid_trafo[:, 0],
                    grid_trafo[:, 1],
                    c=data,
                    cmap='viridis',
                    s=15,
                    label='Transformed Grid (New coordinates)',
                    alpha=1)
    axes[1].scatter(grid[:, 0],
                    grid[:, 1],
                    c=data_trafo,
                    cmap='viridis',
                    s=1,
                    label='Transformed Grid (Interpolated on regular grid)',
                    alpha=0)

    axes[1].invert_yaxis()
    axes[1].legend()
    axes[1].set_title('Transformed AFM data map')
    axes[1].grid()
    axes[1].axis('equal')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig


def plot_contours(median_contour, template_contour, matched_contours):
    """Plot the template, matched contours, and the median contour"""
    fig = plt.figure(figsize=(8, 8))

    # Plot the template contour
    plt.plot(template_contour[:, 0], template_contour[:, 1], 'g-', linewidth=2, label='Mask (Template)')
    plt.scatter(template_contour[0, 0], template_contour[0, 1], color='orange', s=12, label='First contour coordinate')

    # Plot the matched contours (only label the first one)
    for i, contour in enumerate(matched_contours[1:]):
        if i == 0:
            plt.plot(contour[:, 0], contour[:, 1], 'b-', label='Masks (Matched)')
            plt.scatter(contour[0, 0], contour[0, 1], color='orange', s=12)
        else:
            plt.plot(contour[:, 0], contour[:, 1], 'b-')  # No label for subsequent contours
            plt.scatter(contour[0, 0], contour[0, 1], color='orange', s=12)

    # Plot the median contour
    plt.plot(median_contour[:, 0], median_contour[:, 1], 'r--', linewidth=2, label='Median Contour')

    plt.gca().invert_yaxis()
    plt.legend()
    plt.axis('equal')
    plt.title('Path Density with Contours')

    return fig


def plot_average_heatmap(data_maps, data_map_avg, grid, average_contour, data_variable):
    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Overlay all heatmaps on top of each other
    # Median contour
    axes[0].plot(average_contour[:, 0], average_contour[:, 1], 'r--', linewidth=2, label='Median Contour')

    # Heatmaps
    for i, data_map in enumerate(data_maps[1:]):
        if i == 0:
            axes[0].scatter(grid[:, 0],
                            grid[:, 1],
                            c=data_map[data_variable],
                            cmap='viridis',
                            s=25,
                            label='Transformed Grid',
                            alpha=0.1)
        else:
            axes[0].scatter(grid[:, 0],
                            grid[:, 1],
                            c=data_map[data_variable],
                            cmap='viridis',
                            s=25,
                            alpha=0.1)

    axes[0].legend()
    axes[0].set_title('Layered data maps of transformed spatial maps')
    axes[0].grid()
    axes[0].axis('equal')

    # Average heatmap
    # Median contour
    axes[1].plot(average_contour[:, 0], average_contour[:, 1], 'r--', linewidth=2, label='Median Contour')

    # Original and deformed data
    axes[1].scatter(grid[:, 0],
                    grid[:, 1],
                    c=data_map_avg,
                    cmap='viridis',
                    s=25,
                    label='Transformed Grid',
                    alpha=1)

    axes[1].legend()
    axes[1].set_title('Average data map of transformed spatial maps')
    axes[1].grid()
    axes[1].axis('equal')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig
