import os
import numpy as np
from scipy.interpolate import griddata

from brainfusion._match_contours import (interpolate_contour, align_contours, find_average_contour,
                                         circularly_shift_contours, align_sc_contours)
from brainfusion._transform_2Dmap import extend_grid, transform_grid2contour
from brainfusion._gmm_correlation import fit_coordinates_gmm
from brainfusion.load_experiments.load_afm import load_sc_afm_myelin


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
    shifted_contours, _ = circularly_shift_contours([matched_contours[tmp_index], avg_contour],
                                                    template_index=0,
                                                    centre_contours=True)
    avg_contour = shifted_contours[1]

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
    gmm_grid, gmm_dict, avg_dict = fit_coordinates_gmm(trafo_grids,
                                                       data_list,
                                                       trafo_data_maps,
                                                       same_maps=True,
                                                       num_components='mean')

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


def process_sc_experiment(path, boundary_filename, key_point_filename, rot_axis_filename, contour_interp_n=200,
                          sampling_size=None, **kwargs):
    """
    Use this function to match spinal cord boundary outlines from myelin staining data to the spinal cord outline from a
    single AFM measurement. Myelin images are transformed to match AFM contour and corresponding datasets correlated.
    """
    print(f'Processing folder: {os.path.basename(path)}.')

    # Load Myelin and AFM data from experiment folder; the first list element represent the AFM dataset
    afm_index = 0
    grids, datasets, contours, points, axes, myelin_filenames, afm_image = load_sc_afm_myelin(folder_path=path,
                                                                                              boundary_filename=boundary_filename,
                                                                                              key_point_filename=key_point_filename,
                                                                                              rot_axis_filename=rot_axis_filename,
                                                                                              sampling_size=sampling_size)

    # Interpolate all contours to same length
    contours_interp = []
    for contour in contours:
        contours_interp.append(interpolate_contour(contour, num_points=contour_interp_n))

    # Align contours using affine transformations and match boundaries using DTW
    matched_contours, matched_templates, matched_grids = align_sc_contours(contours_interp, grids, rot_axes=axes,
                                                                           init_points=points, template_index=afm_index)

    # Split AFM and myelin data
    afm_grid = matched_grids[afm_index]
    myelin_grids = matched_grids[afm_index + 1:]

    afm_dataset = datasets[afm_index]
    myelin_datasets = datasets[afm_index + 1:]

    afm_contours = matched_templates[afm_index + 1:]
    myelin_contours = matched_contours[afm_index + 1:]

    # Initialise grid to interpolate transformed data on
    extended_grid = extend_grid(myelin_grids[0], 50, 50)  # ToDo: Change myelin_grids[0] to AFM grid

    # Using boundary matched contours to model plane transformation with thin plate spline interpolation
    trafo_data_maps, trafo_grids, trafo_contours = [], [], []
    for index, contour in enumerate(myelin_contours):
        trafo_grid, trafo_contour = transform_grid2contour(myelin_contours[index],
                                                           afm_contours[index],
                                                           myelin_grids[index],
                                                           progress=f' {index + 1} out of {len(matched_contours) - 1}')

        # Interpolate maps from deformed grids to common regular grid
        trafo_data = griddata(myelin_grids[index], myelin_datasets[index].ravel(), extended_grid, method='nearest')

        trafo_data_maps.append(trafo_data)
        trafo_grids.append(trafo_grid)
        trafo_contours.append(trafo_contour)

    # Calculate the mean of interpolated maps
    avg_data = np.nanmean(np.array(trafo_data_maps), axis=0)

    # Creating the analysis dictionary
    structured_data = {
        'afm_contours': afm_contours,  # This contour is matched with the first myelin contour!
        'myelin_contours': myelin_contours,
        'myelin_trafo_contours': trafo_contours,
        'afm_grid': afm_grid,
        'myelin_grids': myelin_grids,
        'myelin_trafo_grids': trafo_grids,
        'myelin_interpolated_grid': extended_grid,
        'afm_dataset': afm_dataset,
        'myelin_datasets': myelin_datasets,
        'myelin_trafo_datasets': trafo_data_maps,
        'myelin_interpolated_dataset': avg_data,
        'myelin_filenames': myelin_filenames,
        'afm_overview_image': afm_image
    }

    return structured_data
