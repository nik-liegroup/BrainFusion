import os
import matplotlib.pyplot as plt
import numpy as np
from _load_experiment import load_afm_experiment
from _find_average_contour import find_average_contour
from _transform_2Dmap import transform_map2contour, transform_grid2contour
from _calculate_average_heatmap import average_heatmap
from _plot_maps import plot_afm_maps, plot_contours, plot_cont_func, plot_average_heatmap

# Start analysis
afm_raw_data = 'modulus'
base_folder = os.path.join('path/to/data')
results_folder = os.path.join('path/to/data')
os.makedirs(results_folder, exist_ok=True)

folder_names, afm_data_list, afm_grid_list = [], [], []
bf_data_list, bf_metadata_list, mask_list = [], [], []

for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
    if os.path.isdir(folder_path) and ('#' in folder_name):
        folder_names.append(folder_name)

        afm_data_full, bf_data, mask = load_afm_experiment(folder_path)
        afm_data = {'modulus': afm_data_full['modulus'], 'beta_pyforce': afm_data_full['beta_pyforce'],
                    'k0_pyforce': afm_data_full['k0_pyforce'], 'k_pyforce': afm_data_full['k_pyforce']}
        afm_data = {key: np.array(value) for key, value in afm_data.items()}

        # Save imported data
        afm_data_list.append(afm_data)
        afm_grid_list.append(np.stack((afm_data_full['x_image'], afm_data_full['y_image']), axis=-1))
        bf_data_list.append(bf_data)
        mask_list.append(mask)

# Calculate average mask and plot result
median_contour, contours_list, template_contour, matched_contour_list = find_average_contour(mask_list)
fig = plot_contours(median_contour, template_contour, matched_contour_list)
output_path = os.path.join(results_folder, 'matched_mask_contours.png')
fig.savefig(output_path, dpi=300, bbox_inches='tight')
# fig.show()
plt.close()

# Transform AFM maps to average map
bm_data_trafo_list, bm_grid_trafo_list, contour_trafo_list = [], [], []
for index, contour in enumerate(contours_list):
    # Transform original grid to coordinate system of deformed contour
    trafo_grid_points, trafo_contour = transform_grid2contour(contour, median_contour, afm_grid_list[index])

    # Transform maps to deformed coordinate system and interpolate on rectangular grid
    bm_data_trafo = {}
    for key, data_map in afm_data_list[index].items():  # Access both key and data
        data_map_trafo, extended_grid = transform_map2contour(trafo_grid_points, afm_grid_list[0], data_map)
        # Reshape data_map_trafo back to the original data dimensions and store it in the new dictionary
        bm_data_trafo[key + '_trafo'] = data_map_trafo

    bm_data_trafo_list.append(bm_data_trafo)
    bm_grid_trafo_list.append(trafo_grid_points)
    contour_trafo_list.append(trafo_contour)

    # Plot regular heatmaps
    fig = plot_afm_maps(bf_data_list[index],
                        afm_data_list[index][f'{afm_raw_data}'],
                        afm_grid_list[index],
                        folder_names[index])
    output_path = os.path.join(results_folder, f'{folder_names[index]}.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    # fig.show()
    plt.close()

    # Plot transformed heatmap and contours
    fig = plot_cont_func(contour,
                         median_contour,
                         trafo_contour,
                         afm_data_list[index][f'{afm_raw_data}'],
                         bm_data_trafo_list[index][f'{afm_raw_data}_trafo'],
                         afm_grid_list[index],
                         bm_grid_trafo_list[index],
                         extended_grid)

    output_path = os.path.join(results_folder, f'MeanContourTransformed_{folder_names[index]}.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    # fig.show()
    plt.close()

# Calculate average heatmap
bm_data_trafo_avg = average_heatmap(bm_data_trafo_list, f'{afm_raw_data}_trafo')

# Plot all transformed heatmaps on top of each other into one plot and the averaged heatmap
fig = plot_average_heatmap(bm_data_trafo_list,
                           bm_data_trafo_avg,
                           extended_grid,
                           median_contour,
                           f'{afm_raw_data}_trafo')
output_path = os.path.join(results_folder, 'AveragedAFMMaps.png')
fig.savefig(output_path, dpi=300, bbox_inches='tight')
# fig.show()
plt.close()
