import os
import matplotlib.pyplot as plt
from _load_experiment import load_brillouin_experiment, load_afm_experiment
from _find_average_contour import find_average_contour
from _transform_2Dmap import transform_map2contour, transform_grid2contour
from _plot_maps import plot_contours, plot_experiments
from _utils import project_brillouin_dataset, export_analysis, import_analysis, check_parameters
from _br_afm_correlation import fit_coordinates_gmm, afm_brillouin_corr


def process_experiment(experiment, base_folder, results_folder, load_experiment_func, **kwargs):
    os.makedirs(results_folder, exist_ok=True)

    folder_names, data_list, grid_list, grid_shape_list = [], [], [], []
    bf_data_list, mask_list = [], []

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path) and ('#' in folder_name):
            folder_names.append(folder_name)

            # Load data from experiment folder
            data, grid, bf_data, mask = load_experiment_func(folder_path)

            if experiment == 'brillouin':
                # Create 2D Brillouin map from 3D dataset and ravel datasets
                data, grid = project_brillouin_dataset(data, grid)

            grid_shape_list.append(grid.shape)

            # Save imported data
            data_list.append(data)
            grid_list.append(grid)
            bf_data_list.append(bf_data)
            mask_list.append(mask)

    # Calculate average mask and plot result
    med_contour, contours_list, template_contour, matched_contour_list = find_average_contour(mask_list)
    fig = plot_contours(med_contour, template_contour, matched_contour_list)
    output_path = os.path.join(results_folder, 'matched_mask_contours.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Transform maps to average map
    data_trafo_list, grid_trafo_list, contour_trafo_list, extended_grid = [], [], [], []
    for index, contour in enumerate(contours_list):
        # Transform original grid to coordinate system of deformed contour
        trafo_grid_points, trafo_contour = transform_grid2contour(contour, med_contour, grid_list[index])

        # Transform maps to deformed coordinate system and interpolate on rectangular grid
        data_trafo = {}
        for key, data_map in data_list[index].items():
            data_map_trafo, extended_grid = transform_map2contour(trafo_grid_points, grid_list[0], data_map)
            data_trafo[key + '_trafo'] = data_map_trafo

        data_trafo_list.append(data_trafo)
        grid_trafo_list.append(trafo_grid_points)
        contour_trafo_list.append(trafo_contour)

    # Calculate average heatmap
    grid_avg, data_median_dict = fit_coordinates_gmm(grid_trafo_list, data_list, same_maps=True, num_components='mean')

    # Creating the analysis dictionary
    structured_data = {}
    for i, folder in enumerate(folder_names):
        structured_data[folder] = {
            'raw_data': data_list[i],  # For Brillouin data, this is already projected
            'trafo_data': data_trafo_list[i],
            'raw_grid': grid_list[i],  # For Brillouin data, this is already projected
            'grid_shape': grid_shape_list[i],
            'trafo_grid': grid_trafo_list[i],
            'contour': contours_list[i],
            'trafo_contour': contour_trafo_list[i],
            'brightfield_image': bf_data_list[i]
        }

    # Adding common data at folder level
    structured_data['median_contour'] = med_contour
    structured_data['template_contour'] = template_contour
    structured_data['average_data'] = data_median_dict
    structured_data['average_grid'] = grid_avg
    structured_data['extended_grid'] = extended_grid

    return structured_data


# Parameters for Brillouin data
brillouin_params = {
    "experiment": 'brillouin',
    "load_experiment_func": load_brillouin_experiment,
    "base_folder": 'path/to/data',
    "results_folder": 'path/to/data',
    "raw_data_key": 'brillouin_shift_f_proj',
    "label": 'Brillouin shift (GHz)',
    "cmap": 'viridis',
    "marker_size": 35
}

# Parameters for AFM data
afm_params = {
    "experiment": 'afm',
    "load_experiment_func": load_afm_experiment,
    "base_folder": 'path/to/data',
    "results_folder": 'path/to/data',
    "raw_data_key": 'modulus',
    "label": 'Reduced elastic modulus (Pa)',
    "cmap": 'hot',
    "marker_size": 5
}

# Process Brillouin data
brillouin_analysis_path = os.path.join(brillouin_params['results_folder'], 'BrillouinAnalysis.h5')
if not os.path.exists(brillouin_analysis_path):
    brillouin_analysis = process_experiment(**brillouin_params)
    export_analysis(brillouin_analysis_path, brillouin_analysis, brillouin_params)
else:
    brillouin_analysis, brillouin_params_loaded = import_analysis(brillouin_analysis_path)
    check_parameters(brillouin_params, brillouin_params_loaded)
    brillouin_params = brillouin_params_loaded

# Process AFM data
afm_analysis_path = os.path.join(afm_params['results_folder'], 'AfmAnalysis.h5')
if not os.path.exists(afm_analysis_path):
    afm_analysis = process_experiment(**afm_params)
    export_analysis(afm_analysis_path, afm_analysis, afm_params)
else:
    afm_analysis, afm_params_loaded = import_analysis(afm_analysis_path)
    check_parameters(afm_params, afm_params_loaded)
    afm_params = afm_params_loaded

# ToDo: CHECK if loading the dataset versus direct calculation yields different results!

# Plot results for AFM and Brillouin experiments
plot_experiments(brillouin_analysis, **brillouin_params)
plot_experiments(afm_analysis, **afm_params)

# Extract relevant data for correlation map calculation
# ToDo: AFM and Brillouin grids in pixels and not µm!
afm_scale = 2
afm_analysis['average_grid'] = afm_analysis['average_grid'] * afm_scale
afm_analysis['median_contour'] = afm_analysis['median_contour'] * afm_scale

# Calculate correlation between AFM and Brillouin datasets
results_folder = './correlation'
os.makedirs(results_folder, exist_ok=True)
correlation, median_contour, afm_data_interp, br_data_interp, grid_average = afm_brillouin_corr(afm_analysis,
                                                                                                afm_params,
                                                                                                brillouin_analysis,
                                                                                                brillouin_params,
                                                                                                results_folder)
