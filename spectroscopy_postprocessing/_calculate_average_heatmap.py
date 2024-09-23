import numpy as np


def average_heatmap(data_maps, data_variable):
    arrays = [d[data_variable] for d in data_maps]
    stacked_arrays = np.stack(arrays, axis=0)
    mean_array = np.nanmedian(stacked_arrays, axis=0)

    return mean_array
