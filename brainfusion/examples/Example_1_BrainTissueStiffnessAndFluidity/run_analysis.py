"""
BrainFusion analysis for AFM brain-tissue stiffness/fluidity maps.

Put the batchforce experiment folders under ./data, then just run this script.
"""

import os

from brainfusion import load_batchforce_all, run_fusion, plot_brainfusion_results

here = os.path.dirname(__file__)
data_folder = os.path.join(here, "data")

# --- Loader parameters ---
LOADER_KWARGS = dict(
    afm_variables=["modulus", "beta_pyforce"],
    batchforce_filename="data.csv",
    grid_conv_filename="GridInversionMatrix.csv",
    boundary_filename="brain_outline",
    landmarks_filename=None,  # Set to a filename suffix to use corresponding landmark points for pre-alignment
    # Folder names look like '#1_XenopusExposedBrain_Control_Stage37_20260729'. The named groups below are
    # stored in each sample's `.metadata` dict - adapt this regex to your own folder naming convention.
    name_pattern=r"#(?P<animal_number>\d+)_.*?_(?P<condition>[A-Za-z]+)_Stage(?P<stage>\d+)_(?P<date>\d+)",
    name_converters={"animal_number": int, "stage": int},
)

# --- Fusion parameters ---
FUSION_KWARGS = dict(
    contour_template="average",
    contour_interp_n=200,
    clustering="Mean",
)

OVERWRITE = False

# --- Run ---
samples = load_batchforce_all(data_folder, **LOADER_KWARGS)
print("Loaded samples:", [(s.filename, s.metadata) for s in samples])

analysis = run_fusion(samples, FUSION_KWARGS, results_path=os.path.join(here, "results", "analysis.h5"),
                      overwrite=OVERWRITE)

plot_brainfusion_results(analysis, results_folder=os.path.join(here, "results"), key_quant="modulus",
                         cbar_label="Reduced elastic modulus (Pa)", cmap="hot", marker_size=20, vmin=0, vmax=500)
