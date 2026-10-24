import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'

from .results import plot_sample_warps, plot_verification_grids
from .maps import (plot_contours, plot_aligned_outlines, plot_transformed_grid, plot_average_map, plot_average_map_arrays,
                   plot_average_statistic,
                   render_data_on_contour, get_zoom_limits, scalebar_length, draw_scalebar)
from .map_style import map_canvas, decorate_map, save_map
from .image_export import plot_map_on_image, render_contour_on_image, save_average_map_tif
from .correlation import plot_correlation_masks, plot_correlation_with_radii, plot_norm_corr, format_p_value

__all__ = [
    "plot_sample_warps",
    "plot_verification_grids",
    "plot_contours",
    "plot_aligned_outlines",
    "plot_transformed_grid",
    "plot_average_map",
    "plot_average_map_arrays",
    "plot_average_statistic",
    "render_data_on_contour",
    "get_zoom_limits",
    "scalebar_length",
    "draw_scalebar",
    "map_canvas",
    "decorate_map",
    "save_map",
    "plot_map_on_image",
    "render_contour_on_image",
    "save_average_map_tif",
    "plot_correlation_masks",
    "plot_correlation_with_radii",
    "plot_norm_corr",
    "format_p_value",
]
