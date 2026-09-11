"""BrainFusion analysis for the AFM half of the Xenopus brain tissue example. Put data under ./data/AFM_Data."""

import os

from brainfusion import load_batchforce_all, run_fusion, plot_brainfusion_results

here = os.path.dirname(__file__)
data_folder = os.path.join(here, "data", "AFM_Data")

LOADER_KWARGS = dict(
    afm_variables=["modulus", "beta_pyforce"],
    batchforce_filename="data.csv",
    grid_conv_filename="GridInversionMatrix.csv",
    boundary_filename="brain_outline",
    landmarks_filename=None,
)

FUSION_KWARGS = dict(
    contour_template="average",
    contour_interp_n=200,
    clustering="Mean",
)

samples = load_batchforce_all(data_folder, **LOADER_KWARGS)
analysis = run_fusion(samples, FUSION_KWARGS, results_path=os.path.join(data_folder, "results", "analysis.h5"),
                      overwrite=False)

plot_brainfusion_results(analysis, results_folder=os.path.join(data_folder, "results"), key_quant="modulus",
                         cbar_label="Reduced elastic modulus (Pa)", cmap="hot", marker_size=20, vmin=0, vmax=500)
