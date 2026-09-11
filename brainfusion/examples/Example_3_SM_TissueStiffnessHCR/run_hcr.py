"""BrainFusion analysis for the in-situ HCR mRNA staining half of the Xenopus brain tissue example.

Put the .tif images (and their *_BrainBoundary.txt outlines) under ./data/HCR_Data.
"""

import os

from brainfusion import load_microscopy_experiment, run_fusion, plot_brainfusion_results

here = os.path.dirname(__file__)
data_folder = os.path.join(here, "data", "HCR_Data")

LOADER_KWARGS = dict(
    boundary_filename="BrainBoundary",
    landmarks_filename=None,
)

FUSION_KWARGS = dict(
    contour_template="average",
    contour_interp_n=200,
    clustering="Sum",
    outline_averaging="median",
)

samples = load_microscopy_experiment(data_folder, **LOADER_KWARGS)
analysis = run_fusion(samples, FUSION_KWARGS, results_path=os.path.join(data_folder, "results", "analysis.h5"),
                      overwrite=False)

plot_brainfusion_results(analysis, results_folder=os.path.join(data_folder, "results"), key_quant="Channel_3",
                         image_dataset=True, cbar_label="mRNA intensity", cmap="Greens", vmin=None, vmax=1024)
