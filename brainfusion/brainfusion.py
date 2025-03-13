import numpy as np
from scipy.interpolate import griddata

from brainfusion._match_contours import interpolate_contour, align_contours, boundary_match_contours
from brainfusion._average_contours import find_average_contour
from brainfusion._transform_2Dmap import extend_grid, transform_grid2contour
from brainfusion._gmm_correlation import fit_coordinates_gmm
from brainfusion._utils import regular_grid_on_contour


def fuse_measurement_datasets(base_path, template, grids, scales, datasets, contours, points, axes, filenames,
                              bg_images, contour_interp_n=200, clustering=None, **kwargs):
    """
    Use this function to match tissue boundary outlines with measurement grids defined in enclosed area. Boundaries can
    be matched using a given template or by calculating average tissue shape. The original measurement grids are then
    transformed to the template shape and together with the corresponding data averaged by interpolation to a regular
    grid or clustering using a Gaussian Mixture Model.
    """
    print(f'Processing folder: {base_path}.')

    # Interpolate all contours to same length
    contours_interp = []
    for contour in contours:
        contours_interp.append(interpolate_contour(contour, num_points=contour_interp_n))

    # Align contours and grids using affine transformations
    template_index = 0  # Must be 0
    aligned_contours, aligned_grids = align_contours(contours_interp,
                                                     grids,
                                                     rot_axes=axes,
                                                     init_points=points,
                                                     template_index=template_index)

    if template == 'average':
        # Calculate average contour and add as first element in list
        template_contour, errors = find_average_contour(aligned_contours, average='star_domain',
                                                        num_bins=contour_interp_n)
        aligned_contours.insert(template_index, template_contour)
        aligned_grids.insert(template_index, np.empty((1, 2)))
        datasets.insert(template_index, np.empty((1, 2)))
    elif template == 'first_element':
        pass
    else:
        raise ValueError(f'Choice of template: {template} is not implemented!')

    # Match boundaries using DTW
    dtw_contours, dtw_template_contours = boundary_match_contours(aligned_contours, template_index=template_index)

    # Split template data from rest
    template_grid = aligned_grids[0]
    measurement_grids = aligned_grids[1:]

    template_dataset = datasets[0]
    measurement_datasets = datasets[1:]

    template_contours = dtw_template_contours[1:]
    measurement_contours = dtw_contours[1:]

    # Initialise grid to interpolate transformed data on
    extended_grid = extend_grid(measurement_grids[0], 150, 150)

    # Create regular test grids for verifying correct transformation
    verification_grids = [regular_grid_on_contour(c) for c in measurement_contours]

    # Using boundary matched contours to model plane transformation with thin plate spline interpolation
    trafo_data_maps, trafo_grids, trafo_contours, trafo_ver_grids = [], [], [], []
    for index, contour in enumerate(measurement_contours):
        trafo_grid, trafo_ver_grid, trafo_contour = transform_grid2contour(measurement_contours[index],
                                                                           template_contours[index],
                                                                           measurement_grids[index],
                                                                           verification_grids[index],
                                                                           progress=f' {index + 1} out of {len(dtw_contours) - 1}')

        # Interpolate maps from deformed grids to common regular grid
        trafo_data = {}
        for key, data_map in measurement_datasets[index].items():
            trafo_data[key] = griddata(trafo_grid, data_map.ravel(), extended_grid, method='nearest')

        trafo_data_maps.append(trafo_data)
        trafo_grids.append(trafo_grid)
        trafo_ver_grids.append(trafo_ver_grid)
        trafo_contours.append(trafo_contour)

    # Average transformed datasets
    if clustering == 'GMM':
        # Cluster data points using Gaussian Mixture Model and average using median
        gmm_grid, gmm_dict, avg_data = fit_coordinates_gmm(trafo_grids,
                                                           measurement_datasets,
                                                           trafo_data_maps,
                                                           same_maps=True,
                                                           num_components='mean')
    elif clustering == "None":
        # Compute mean array for each key
        keys = trafo_data_maps[0].keys()
        avg_data = {
            key: np.nanmean(np.array([d[key] for d in trafo_data_maps]), axis=0)
            for key in keys
        }
    else:
        raise ValueError(f'Clustering option {clustering} is not implemented!')

    # Creating the analysis dictionary
    structured_data = {
        'template_contours': template_contours,  # Template contours but matched to respective measurement contours!
        'measurement_contours': measurement_contours,
        'measurement_trafo_contours': trafo_contours,
        'template_grid': template_grid,
        'measurement_grids': measurement_grids,
        'verification_grids': verification_grids,
        'measurement_trafo_grids': trafo_grids,
        'verification_trafo_grids': trafo_ver_grids,
        'measurement_interpolated_grid': extended_grid,
        'template_dataset': template_dataset,
        'measurement_datasets': measurement_datasets,
        'measurement_trafo_datasets': trafo_data_maps,
        'measurement_interpolated_dataset': avg_data,
        'measurement_filenames': filenames,
        'background_image': bg_images
    }

    return structured_data
