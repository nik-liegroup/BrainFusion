"""
BrainFusion analysis for spinal cord myelin/AFM sections.

Every subfolder named with a unique '#<number>...' below ./data is one section: an AFM measurement plus one
or more myelin-stained images of the same tissue. Each section is fused independently, using its own AFM
sample as the alignment template for its myelin images. Just run this script.

Note: `load_sc_afm_single` reads a non-standard, stopgap file layout specific to this experiment (see its
docstring) rather than the general AFM loader, which is why it's imported from `.other` here.
"""

import os
import re

from brainfusion import load_parquet_samples, correlate_afm_myelin, run_fusion, plot_brainfusion_results, \
    parse_name, attach_metadata
from brainfusion.load_experiments.base import iter_experiment_folders
from brainfusion.load_experiments.other import load_sc_afm_single

here = os.path.dirname(__file__)
data_folder = os.path.join(here, "data")

BOUNDARY_FILENAME = "whitematter_outline"
LANDMARKS_FILENAME = "midline_landmarks"  # Corresponding points used directly in the affine pre-alignment

# Folder names look like '#1_JB_SpinalCord_P7_20260729'. Adapt this regex to your own naming convention.
NAME_PATTERN = r"#(?P<animal_number>\d+)_.*?_P(?P<postnatal_day>\d+)_"
NAME_CONVERTERS = {"animal_number": int, "postnatal_day": int}

FUSION_KWARGS = dict(
    contour_template="first_element",  # Use the AFM sample (first element) as the alignment template
    contour_interp_n=500,
    clustering="Mean",
)

OVERWRITE = False

# --- Run: each section gets its own fusion result, stored next to its data ---
for folder_name, folder_path in iter_experiment_folders(data_folder):
    exp_num_match = re.search(r'#(\d+)', folder_name)
    exp_num = int(exp_num_match.group(1))

    afm_sample = load_sc_afm_single(folder_path, boundary_filename=BOUNDARY_FILENAME,
                                    landmarks_filename=LANDMARKS_FILENAME)

    # This lab's myelin images are named e.g. 'ani1_something_Merged_RAW_ch02_image_roi_linearised.parquet',
    # with matching 'ani1_something_<boundary>.txt' / '..._<landmarks>.txt' outline/landmark files. The
    # 'name' group below strips the '_Merged_RAW...' suffix so each sample gets a clean filename/identity.
    # Note: NAME_PATTERN describes the *folder* name, not these per-image filenames, so metadata is attached
    # separately below rather than passed as name_pattern to this loader.
    myelin_samples = load_parquet_samples(
        folder_path,
        data_pattern=rf'(?P<name>ani{exp_num}_.*?)_Merged_RAW.*image_roi_linearised\.parquet$',
        contour_pattern=rf'ani{exp_num}_.*{re.escape(BOUNDARY_FILENAME)}\.txt$',
        landmarks_pattern=rf'ani{exp_num}_.*{re.escape(LANDMARKS_FILENAME)}\.txt$',
        dataset_key="myelin_intensity",
        sampling_size=None)  # Set to an int to randomly subsample the myelin data for a quick test run

    # Every sample from this folder shares the same animal/condition metadata, parsed once from the folder name
    metadata = parse_name(folder_name, NAME_PATTERN, NAME_CONVERTERS)
    samples = attach_metadata([afm_sample] + myelin_samples, metadata)
    print(f"{folder_name}: loaded samples", [(s.filename, s.metadata) for s in samples])

    analysis = run_fusion(samples, FUSION_KWARGS, results_path=os.path.join(folder_path, "results", "analysis.h5"),
                          overwrite=OVERWRITE)

    plot_brainfusion_results(analysis, results_folder=os.path.join(folder_path, "results"),
                             key_quant="myelin_intensity", cmap="grey", marker_size=35, verify_trafo=True)

    correlate_afm_myelin(analysis, radius="max", verify_corr=True)
