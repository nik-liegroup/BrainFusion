import numpy as np
import matplotlib.path as mpath
import os.path
import h5py
import threading
from scipy.spatial.distance import pdist


def get_user_input(prompt, timeout=10):
    user_input = [None]  # To store user input

    def ask_input():
        user_input[0] = input(prompt)

    # Create and start a thread to ask for input
    input_thread = threading.Thread(target=ask_input)
    input_thread.start()

    # Wait for the input thread to finish or timeout
    input_thread.join(timeout)

    # Return the user input if it was provided, otherwise return None
    return user_input[0]


def mask_contour(contour, grid):
    # Create the path from the contour
    path = mpath.Path(contour)

    # Find which points are inside the contour
    points = np.vstack((grid[:, :, 0].ravel(), grid[:, :, 1].ravel())).T
    mask_points = path.contains_points(points)

    # Reshape the result back to the shape of the mask
    mask = mask_points.reshape(grid[:, :, 0].shape)

    return mask


def center_contour(contour):
    # Calculate the centroid (center of mass) of the contour
    centroid = np.mean(contour, axis=0)

    # Translate all points so the centroid is at the origin
    centered_contour = contour - centroid

    return centered_contour, centroid


def get_h5rep(h5_path, data_var):
    assert os.path.exists(h5_path), print(f'{h5_path} does not exist.')
    assert data_var in ['Brillouin', 'Fluorescence'], (f'{data_var} is not available in h5 file. '
                                                       f'Please choose Brillouin or Fluorescence.')

    with h5py.File(h5_path, 'r') as h5_file:
        brillouin_group = h5_file[data_var]
        rep_numbers = [int(key) for key in brillouin_group.keys() if key.isdigit()]

    return rep_numbers


def get_h5metadata(h5_path, data_var):
    assert os.path.exists(h5_path), f'{h5_path} does not exist.'
    assert data_var in ['Brillouin', 'Fluorescence'], (f'{data_var} is not available in h5 file. '
                                                       f'Please choose Brillouin or Fluorescence.')

    # Get number of replicates
    reps = get_h5rep(h5_path, data_var)
    assert isinstance(reps, list) and all(isinstance(rep, int) for rep in reps), (f'Reps {reps} '
                                                                                  f'must be a list of integers.')

    metadata_dict = {}
    with h5py.File(h5_path, 'r') as h5_file:
        for rep in reps:
            pixToMicrometerX = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/micrometerToPixX']
            pixToMicrometerX_x, pixToMicrometerX_y = pixToMicrometerX.attrs['x'], pixToMicrometerX.attrs['y']

            pixToMicrometerY = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/micrometerToPixY']
            pixToMicrometerY_x, pixToMicrometerY_y = pixToMicrometerY.attrs['x'], pixToMicrometerY.attrs['y']

            origin = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/origin']
            origin_x, origin_y = origin.attrs['x'], origin.attrs['y']

            stage = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/positionStage']
            stage_x, stage_y = stage.attrs['x'], stage.attrs['y']

            scanner = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/positionScanner']
            scanner_x, scanner_y = scanner.attrs['x'], scanner.attrs['y']

            # Transform scanner position to coordinate system of image
            scanner_x = scanner_x * pixToMicrometerX_y
            scanner_y = scanner_y * pixToMicrometerY_x

            if data_var == 'Brillouin':
                # Get grid coordinates from Brillouin measurement
                brillouin_grid_x = h5_file[f'Brillouin/{rep}/payload/positions-x'][:]
                brillouin_grid_y = h5_file[f'Brillouin/{rep}/payload/positions-y'][:]
                brillouin_grid_z = h5_file[f'Brillouin/{rep}/payload/positions-z'][:]

                # Transform Brillouin measurement grid to coordinate system of image
                brillouin_grid_x = (brillouin_grid_x - stage_x) * pixToMicrometerX_y
                brillouin_grid_y = (brillouin_grid_y - stage_y) * pixToMicrometerY_x
                brillouin_grid_z = (brillouin_grid_z - np.min(brillouin_grid_z))  # Already in µm

                brillouin_grid = np.stack((brillouin_grid_x, brillouin_grid_y, brillouin_grid_z), axis=-1)
                brillouin_grid = np.transpose(brillouin_grid, (2, 1, 0, 3))
            else:
                brillouin_grid = None

            # Create a dictionary for the current replicate
            replicate_metadata = {
                'pixToMicrometerX': np.stack((pixToMicrometerX_x, pixToMicrometerX_y), axis=-1),
                'pixToMicrometerY': np.stack((pixToMicrometerY_x, pixToMicrometerY_y), axis=-1),
                'origin': np.stack((origin_x, origin_y), axis=-1),
                'scanner': np.stack((scanner_x, scanner_y), axis=-1),
                'stage': np.stack((stage_x, stage_y), axis=-1),
                'brillouin_grid': brillouin_grid
            }

            metadata_dict[rep] = replicate_metadata

    return metadata_dict, reps


# ToDo: Implement more elaborate analysis method
def project_brillouin_dataset(bm_data):
    bm_data_proj = {}
    for key, value in bm_data.items():
        new_value = value.copy()  # Copy the original data to avoid modifying it
        new_value[new_value < 4.4] = np.nan
        proj_value = np.max(new_value, axis=-1).ravel()

        bm_data_proj[key + '_proj'] = proj_value  # Store the projection

    return bm_data_proj


def scatter_with_touching_squares(x, y, plot_size=0):
    # Compute pairwise distances between points
    coords = np.vstack([x, y]).T
    pairwise_distances = pdist(coords)
    mask = ~np.isclose(pairwise_distances, 0)
    pairwise_distances = pairwise_distances[mask]

    # Find the minimum distance between any two points
    min_distance = np.min(pairwise_distances)

    # Calculate the size of the squares to make them touch but not overlap
    desired_size = min_distance / 2  # Size in coordinate units

    # Set the marker size (area) for squares
    marker_size = (desired_size ** 2) * np.pi

    return marker_size
