import numpy as np

from brainfusion._match_contours import interpolate_contour, align_contours, boundary_match_contours
from brainfusion._average_contours import find_average_contour
from brainfusion._transform_2Dmap import extend_grid, transform_grid2contour
from brainfusion._interpolation import nearest_neighbour_interp, fit_coordinates_gmm
from brainfusion._utils import regular_grid_on_bbox


def brain_fusion(grids, datasets, contours, scales, contour_template="average", reg_grid_dims="None", points="None",
                 axes="None", filenames="None", bg_images="None", contour_interp_n=200, clustering='Mean',
                 outline_averaging='star_domain', smooth='auto', curvature=0.5, fit_routine='ellipse', **kwargs):
    """
    Perform contour-based alignment, deformation, and interpolation of spatial datasets for averaging.

    This function aligns user-defined tissue boundaries (contours) and measurement grids using affine and
    non-rigid transformations. Each dataset is warped to match a common template shape, and the data are
    interpolated onto a regular grid or averaged using clustering (e.g., GMM). The result is a spatially
    standardised representation suitable for comparing biological measurements across samples.

    Parameters
    ----------
    grids : list of np.ndarray
        List of 2D coordinate arrays (shape Nx2) representing original measurement grids.
    datasets : list of dict
        Each dictionary contains measurement maps (e.g. fluorescence, force) corresponding to a grid.
    contours : list of np.ndarray
        List of 2D arrays (shape Nx2) representing the outer boundaries of each sample.
    scales : list
        Scaling factors or shape information associated with each measurement grid (used in visualisation/export).
    contour_template : str, default="average"
        Method for choosing the reference contour. Options: "average", "first_element".
    reg_grid_dims : str or list, default="None"
        Shape of original regular grids (can be inferred or passed explicitly).
    points : str or list, default="None"
        Optional landmarks for affine alignment.
    axes : str or list, default="None"
        Optional rotation axes used for initial alignment.
    filenames : list, default="None"
        Filenames corresponding to the original input data (used for labelling/output).
    bg_images : list, default="None"
        Background images associated with each sample (for visual reference).
    contour_interp_n : int, default=200
        Number of points to interpolate each contour to before alignment.
    clustering : str, default='Mean'
        Averaging method across transformed datasets. Options: "Mean", "Median", "Sum", "GMM".
    outline_averaging : str, default='star_domain'
        Method used to average contours when template is "average".
    smooth : float, 'auto' or 'weighted', default='auto'
        RBF interpolation smoothing factor for non-rigid transformation.
    curvature : float, default=0.5
        DTW matching penalty for curvature deviation in boundary alignment.
    fit_routine : str, default='ellipse'
        Method for initial affine contour alignment. Options: "ellipse", "PCA", etc.

    Returns
    -------
    structured_data : dict
        Dictionary containing all intermediate and final results, including:
        - aligned and matched contours
        - original and transformed measurement grids
        - warped measurement datasets
        - averaged dataset on interpolated grid
        - affine transformation matrices
        - background and metadata references

    Notes
    -----
    This function is designed to integrate biological measurements across spatially variable samples.
    It is especially useful for microscopy-based tissue studies with annotated outlines and measurement maps.
    """

    print("Starting brainfusion analysis.")

    # Part I: Affine alignment of contours and subsequent boundary matching
    aligned_grids, datasets, dtw_contours, dtw_template_contours, affine_matrices = (
        fuse_boundaries(contour_template, grids, datasets, contours, points, axes, contour_interp_n=contour_interp_n,
                        outline_averaging=outline_averaging, curvature=curvature, fit_routine=fit_routine)
    )

    # Split template data from rest
    template_grid, measurement_grids = aligned_grids[0], aligned_grids[1:]
    template_dataset, measurement_datasets = datasets[0], datasets[1:]
    template_contours, measurement_contours = dtw_template_contours[1:], dtw_contours[1:]

    # Part II: Coordinate plane warping based on contour matching
    # Initialise grid to interpolate transformed data on
    ext_grid, ext_grid_shape = extend_grid(measurement_grids, 0.05, 0.05)
    trafo_data_maps, trafo_grids, trafo_ver_grids, verification_grids, trafo_contours = (
        fuse_grids(measurement_grids, ext_grid, measurement_datasets, template_contours, measurement_contours,
                   smooth=smooth)
    )

    # Part III: Averaging across multiple datasets in transformed space
    avg_data = fuse_measurement_datasets(measurement_datasets, trafo_data_maps, trafo_grids, clustering=clustering)

    # Create analysis dictionary for export to .h5 file
    structured_data = {
        'affine_matrices': affine_matrices,
        "scale_matrices": scales,
        'template_contours': template_contours,
        'measurement_contours': measurement_contours,
        'measurement_trafo_contours': trafo_contours,
        'template_grid': template_grid,
        'measurement_grids': measurement_grids,
        'measurement_grids_shape': reg_grid_dims,
        'verification_grids': verification_grids,
        'measurement_trafo_grids': trafo_grids,
        'verification_trafo_grids': trafo_ver_grids,
        'measurement_interpolated_grid': ext_grid,
        'measurement_interpolated_grid_shape': ext_grid_shape,
        'template_dataset': template_dataset,
        'measurement_datasets': measurement_datasets,
        'measurement_trafo_datasets': trafo_data_maps,
        'measurement_interpolated_dataset': avg_data,
        'measurement_filenames': filenames,
        'background_image': bg_images
    }

    return structured_data


def brain_fusion_correlation(grids, datasets, contours, scales, contour_template="average", reg_grid_dims="None",
                             points="None", axes="None", filenames="None", bg_images="None", contour_interp_n=200,
                             clustering='Mean', outline_averaging='star_domain', smooth='auto', curvature=0.5,
                             fit_routine='ellipse', **kwargs):
    print("Starting correlation analysis.")

    # Part I: Affine alignment of contours and subsequent boundary matching
    aligned_grids, datasets, dtw_contours, dtw_template_contours, affine_matrices = (
        fuse_boundaries(contour_template, grids, datasets, contours, points, axes, contour_interp_n=contour_interp_n,
                        outline_averaging=outline_averaging, curvature=curvature, fit_routine=fit_routine)
    )

    # Split template data from rest
    template_grid, measurement_grids = aligned_grids[0], aligned_grids[1:]
    template_dataset, measurement_datasets = datasets[0], datasets[1:]
    template_contours, measurement_contours = dtw_template_contours[1:], dtw_contours[1:]

    # Part II: Coordinate plane warping based on contour matching
    # Initialise individual grids to interpolate transformed data on
    ext_grids, ext_grids_shape, trafo_data_maps, trafo_grids = [], [], [], []
    trafo_ver_grids, verification_grids, trafo_contours = [], [], []
    for index, contour in enumerate(measurement_contours):
        ext_grid, ext_grid_shape = extend_grid([measurement_grids[index]], 0.05, 0.05)

        trafo_data_map, trafo_grid, trafo_ver_grid, verification_grid, trafo_contour = (
            fuse_grids([measurement_grids[index]],
                       ext_grid,
                       [measurement_datasets[index]],
                       [template_contours[index]],
                       [measurement_contours[index]],
                       smooth=smooth)
        )

        ext_grids.append(ext_grid)
        ext_grids_shape.append(np.array(ext_grid_shape))
        trafo_data_maps.extend(trafo_data_map)
        trafo_grids.extend(trafo_grid)
        trafo_ver_grids.extend(trafo_ver_grid)
        verification_grids.extend(verification_grid)
        trafo_contours.extend(trafo_contour)

    # Create analysis dictionary for export to .h5 file
    structured_data = {
        'affine_matrices': affine_matrices,
        "scale_matrices": scales,
        'template_contours': template_contours,
        'measurement_contours': measurement_contours,
        'measurement_trafo_contours': trafo_contours,
        'template_grid': template_grid,
        'measurement_grids': measurement_grids,
        'measurement_grids_shape': reg_grid_dims,
        'verification_grids': verification_grids,
        'measurement_trafo_grids': trafo_grids,
        'verification_trafo_grids': trafo_ver_grids,
        'measurement_interpolated_grid': ext_grids,
        'measurement_interpolated_grid_shape': ext_grids_shape,
        'template_dataset': template_dataset,
        'measurement_datasets': measurement_datasets,
        'measurement_trafo_datasets': trafo_data_maps,
        'measurement_filenames': filenames,
        'background_image': bg_images
    }

    return structured_data


def fuse_boundaries(template, grids, datasets, contours, points, axes, contour_interp_n=200,
                    outline_averaging='star_domain', curvature=0.5, fit_routine='ellipse', **kwargs):
    # Close all contours
    closed_contours = []
    for contour in contours:
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])  # Append the first point at the end
        closed_contours.append(contour)

    # Interpolate all contours to same length
    contours_interp = []
    for contour in closed_contours:
        contours_interp.append(interpolate_contour(contour, num_points=contour_interp_n))

    # Align contours and grids using affine transformations
    template_index = 0  # Must be 0
    aligned_contours, aligned_grids, affine_matrices = align_contours(contours_interp,
                                                                      grids,
                                                                      rot_axes=axes,
                                                                      init_points=points,
                                                                      template_index=template_index,
                                                                      fit_routine=fit_routine)

    if template == 'average':
        # Calculate average contour and add as first element in list
        template_contour, errors = find_average_contour(aligned_contours,
                                                        average=outline_averaging,
                                                        star_bins=contour_interp_n,
                                                        error_metric='frechet')
        aligned_contours.insert(template_index, template_contour)
        aligned_grids.insert(template_index, np.empty((1, 2)))
        datasets.insert(template_index, np.empty((1, 2)))
    elif template == 'first_element':
        pass
    else:
        raise ValueError(f'Choice of template: {template} is not implemented!')

    # Match boundaries using DTW
    dtw_contours, dtw_template_contours = boundary_match_contours(aligned_contours,
                                                                  template_index=template_index,
                                                                  curvature=curvature)

    return aligned_grids, datasets, dtw_contours, dtw_template_contours, affine_matrices


def fuse_grids(measurement_grids, ext_grid, measurement_datasets, template_contours, measurement_contours,
               smooth='auto', **kwargs):
    # Create regular test grids for verifying correct transformation
    verification_grids = [regular_grid_on_bbox(c) for c in measurement_contours]

    # Using boundary matched contours to model plane transformation with RBF interpolation
    trafo_data_maps, trafo_grids, trafo_contours, trafo_ver_grids = [], [], [], []
    for index, contour in enumerate(measurement_contours):
        trafo_grid, trafo_ver_grid, trafo_contour = transform_grid2contour(
            measurement_contours[index],
            template_contours[index],
            measurement_grids[index],
            verification_grids[index],
            smooth=smooth,
            progress=f' {index + 1} out of {len(measurement_contours)}')

        # Interpolate maps from deformed grids to common regular grid
        trafo_data = {}

        for key, data_map in measurement_datasets[index].items():
            trafo_data[key] = nearest_neighbour_interp(trafo_grid,
                                                       data_map.ravel(),
                                                       ext_grid,
                                                       unique=False)

        trafo_data_maps.append(trafo_data)
        trafo_grids.append(trafo_grid)
        trafo_ver_grids.append(trafo_ver_grid)
        trafo_contours.append(trafo_contour)

    return trafo_data_maps, trafo_grids, trafo_ver_grids, verification_grids, trafo_contours


def fuse_measurement_datasets(measurement_datasets, trafo_data_maps, trafo_grids, clustering='Mean'):
    # Average transformed datasets
    if clustering == 'GMM':
        # Cluster data points using Gaussian Mixture Model and average using median
        gmm_grid, gmm_dict, avg_data = fit_coordinates_gmm(trafo_grids,
                                                           measurement_datasets,
                                                           trafo_data_maps,
                                                           same_maps=True,
                                                           num_components='mean')
    elif clustering in ["Sum", "Median", "Mean"]:
        # Compute clustered array for each key
        keys = trafo_data_maps[0].keys()
        projection = None
        if clustering == "Sum":
            projection = np.nansum
        elif clustering == "Median":
            projection = np.nanmedian
        elif clustering == "Mean":
            projection = np.nanmean

        avg_data = {
            key: projection(np.array([d[key] for d in trafo_data_maps]), axis=0)
            for key in keys
        }
    else:
        raise ValueError(f'Clustering option {clustering} is not implemented!')

    return avg_data
