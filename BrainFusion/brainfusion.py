import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from ._find_average_contour import interpolate_contour, align_contours, find_average_contour
from ._transform_2Dmap import extend_grid, transform_grid2contour
from ._gmm_correlation import fit_coordinates_gmm
from ._plot_maps import plot_contours


def process_multiple_experiments(load_experiment_func, base_folder, results_folder, **kwargs):
    os.makedirs(results_folder, exist_ok=True)

    folder_names, data_list, grid_list, grid_shape_list = [], [], [], []
    bf_data_list, contour_list, scale_list = [], [], []

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path) and ('#' in folder_name):
            folder_names.append(folder_name)

            # Load 2D data from experiment folder
            data, grid, bf_data, contour, pix_per_um = load_experiment_func(folder_path)

            grid_shape_list.append(grid.shape)

            # Save imported data
            data_list.append(data)
            grid_list.append(grid)
            bf_data_list.append(bf_data)
            contour_list.append(contour)
            scale_list.append(pix_per_um)

    # Interpolate contours to same length
    contours_interp = []
    for contour in contour_list:
        contours_interp.append(interpolate_contour(contour, num_points=1000))

    # Align contours and apply respective transformations to data maps
    tmp_index = 0
    matched_contours, matched_grids = align_contours(contours_interp, grid_list, template_index=tmp_index)

    # Calculate average contour
    avg_contour, errors = find_average_contour(matched_contours)
    avg_contour = interpolate_contour(avg_contour, num_points=1000)

    # Transform maps to average map
    trafo_data_maps, trafo_grids, trafo_contours, extended_grid = [], [], [], []
    for index, contour in enumerate(matched_contours):
        # Transform original grid to coordinate system of deformed contour
        trafo_grid, trafo_contour = transform_grid2contour(contour, avg_contour, matched_grids[index])

        # Interpolate maps from deformed grids to common regular grid
        trafo_data = {}
        extended_grid = extend_grid(matched_grids[tmp_index], 30, 30)
        for key, data_map in data_list[index].items():
            if '_distribution' in key:
                continue
            trafo_data[key + '_trafo'] = griddata(trafo_grid, data_map.ravel(), extended_grid, method='nearest')

        trafo_data_maps.append(trafo_data)
        trafo_grids.append(trafo_grid)
        trafo_contours.append(trafo_contour)

    # Cluster data points using Gaussian Mixture Model and average using median, avg_dict contains mean of interpolated maps
    gmm_grid, gmm_dict, avg_dict = fit_coordinates_gmm(trafo_grids, data_list, trafo_data_maps, same_maps=True, num_components='mean')

    # Creating the analysis dictionary
    structured_data = {}
    for i, folder in enumerate(folder_names):
        structured_data[folder] = {
            'raw_data': data_list[i],
            'trafo_data': trafo_data_maps[i],
            'raw_grid': grid_list[i],
            'matched_grid': matched_grids[i],
            'trafo_grid': trafo_grids[i],
            'grid_shape': grid_shape_list[i],
            'matched_contour': matched_contours[i],
            'original_contour': contours_interp[i],
            'trafo_contour': trafo_contours[i],
            'brightfield_image': bf_data_list[i],
            'pix_per_um': scale_list[i]
        }

    # Adding common data at folder level
    structured_data['average_contour'] = avg_contour
    structured_data['template_contour'] = matched_contours[tmp_index]
    structured_data['gmm_data'] = gmm_dict
    structured_data['gmm_grid'] = gmm_grid
    structured_data['interpolated_data'] = avg_dict
    structured_data['interpolated_grid'] = extended_grid

    return structured_data


def process_single_experiments(load_experiment_func, base_folder, folder_name, **kwargs):
    # Create results folder
    results_folder = os.path.join(base_folder, folder_name, 'results')
    os.makedirs(results_folder, exist_ok=True)

    # Load 2D data from experiment folder
    myelin_grids, myelin_datasets, myelin_contours, afm_image, afm_contour, afm_data, afm_grid, afm_scale, myelin_scale = load_experiment_func(os.path.join(base_folder, folder_name))

    # Use AFM contour as standard
    tmp_index = 0
    myelin_contours.insert(tmp_index, afm_contour)

    # Insert dummy map for 'align_contours' function
    myelin_grids.insert(tmp_index, myelin_grids[tmp_index])

    # Interpolate contours to same length
    contours_interp = []
    for contour in myelin_contours:
        contours_interp.append(interpolate_contour(contour, num_points=1000))

    # Align contours and apply respective transformations to data maps
    matched_contours, matched_grids = align_contours(contours_interp, myelin_grids, template_index=tmp_index,
                                                     fit_routine=None)

    # Use AFM contour as average contour for myelin data
    avg_contour = matched_contours[0]
    avg_contour = interpolate_contour(avg_contour, num_points=1000)
    matched_contours.pop(0)
    myelin_grids.pop(0)

    fig = plot_contours(avg_contour, matched_contours)
    output_path = os.path.join(results_folder, 'matched_mask_contours.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Transform maps to average map
    trafo_data_maps, trafo_grids, trafo_contours, extended_grid = [], [], [], []
    for index, contour in enumerate(matched_contours):
        # Transform original grid to coordinate system of deformed contour
        trafo_grid, trafo_contour = transform_grid2contour(contour, avg_contour, matched_grids[index])

        # Interpolate maps from deformed grids to common regular grid
        trafo_data = {}
        extended_grid = extend_grid(matched_grids[tmp_index], 30, 30)
        for key, data_map in data_list[index].items():
            if '_distribution' in key:
                continue
            trafo_data[key + '_trafo'] = griddata(trafo_grid, data_map.ravel(), extended_grid, method='nearest')

        trafo_data_maps.append(trafo_data)
        trafo_grids.append(trafo_grid)
        trafo_contours.append(trafo_contour)

    # Cluster data points using Gaussian Mixture Model and average using median, avg_dict contains mean of interpolated maps
    gmm_grid, gmm_dict, avg_dict = fit_coordinates_gmm(trafo_grids, data_list, trafo_data_maps, same_maps=True, num_components='mean')

    # Creating the analysis dictionary
    structured_data = {}
    for i, folder in enumerate(folder_names):
        structured_data[folder] = {
            'raw_data': data_list[i],
            'trafo_data': trafo_data_maps[i],
            'raw_grid': grid_list[i],
            'matched_grid': matched_grids[i],
            'trafo_grid': trafo_grids[i],
            'grid_shape': grid_shape_list[i],
            'matched_contour': matched_contours[i],
            'original_contour': contours_interp[i],
            'trafo_contour': trafo_contours[i],
            'brightfield_image': bf_data_list[i],
            'pix_per_um': scale_list[i]
        }

    # Adding common data at folder level
    structured_data['average_contour'] = avg_contour
    structured_data['template_contour'] = matched_contours[tmp_index]
    structured_data['gmm_data'] = gmm_dict
    structured_data['gmm_grid'] = gmm_grid
    structured_data['interpolated_data'] = avg_dict
    structured_data['interpolated_grid'] = extended_grid

    return structured_data
