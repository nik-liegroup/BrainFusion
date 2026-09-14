"""
Case 1: fuse several experiments of ONE modality onto a shared shape and make an averaged map.

Put batchforce experiment folders under ./data (see `load_batchforce_all`), then run this script.
"""

import os

from brainfusion import load_batchforce_all, run_fusion, plot_sample_warps, plot_average_map

here = os.path.dirname(__file__)
data_folder = os.path.join(here, "data")
results_folder = os.path.join(here, "results")
key_quant = "modulus"

LOADER_KWARGS = dict(
    afm_variables={"modulus": "Reduced apparent elastic modulus"},
    batchforce_filename="data.csv",
    grid_conv_filename="GridInversionMatrix.csv",
    boundary_filename="outline",
)

FUSION_KWARGS = dict(contour_template="average", outline_averaging="median", contour_interp_n=200,
                     clustering="Mean")
# To align onto a fixed outline instead of the samples' own average shape: load it with
# `load_template_sample(...)`, put it first in `samples` below, and set contour_template="first_element".

samples = load_batchforce_all(data_folder, **LOADER_KWARGS)
analysis = run_fusion(samples, FUSION_KWARGS, results_path=os.path.join(results_folder, "analysis.h5"))

# Per-sample diagnostics: each animal's data before/after warping onto the shared template
plot_sample_warps(analysis, results_folder, key_quant=key_quant, cmap="hot", vmin=0, vmax=500,
                         cbar_label="Reduced elastic modulus (Pa)")

# The one averaged map across every sample
plot_average_map(analysis, key_quant, cbar_label="Reduced elastic modulus (Pa)", cmap="hot", vmin=0, vmax=500,
                 output_path=os.path.join(results_folder, "Averaged_Map.png"))
