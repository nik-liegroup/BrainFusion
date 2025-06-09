import os
from brainfusion import *
from params import afm_params
from params import hcr_params

# This script was made to explore the BrainFusion Python package using AFM stiffness maps of Xenopus brain tissue and
# in situ hybridization chain reaction mRNA staining images

# Set the base folder containing all AFM experiments (folder indicated with a # followed by a unique number)
base_folder = r'path/to/data'
afm_analysis_path = os.path.join(base_folder, 'BrainFusion_AFM_Analysis.h5')

"""
# Run the analysis in case the resulting h5 has not been created yet or the results should be overwritten
if not os.path.exists(afm_analysis_path) or afm_params["overwrite_analysis"]:
    loaded = load_batchforce_all(base_path=base_folder, **afm_params)
    afm_analysis = fuse_measurement_datasets(base_path=base_folder, **loaded, **afm_params)
    export_analysis(afm_analysis_path, afm_analysis, afm_params)
# Import the pre-computed analysis file for further post-processing steps
else:
    afm_analysis, afm_params_loaded = import_analysis(afm_analysis_path)

    # Check if the currently defined analysis parameters match with the imported analysis parameters
    check_parameters(afm_params, afm_params_loaded)

# Plot the transformation results
plot_brainfusion_results(afm_analysis, **afm_params,
                         results_folder=os.path.join(base_folder, 'results'), key_quant='modulus',
                         label='', cmap='grey', marker_size=35, vmin=None, vmax=None, verify_trafo=True)
"""

# Set the base folder containing an in situ HCR experiment
base_folder = r'path/to/data'
hcr_analysis_path = os.path.join(base_folder, 'BrainFusion_HCR_Analysis.h5')

# Run the analysis in case the resulting h5 has not been created yet or the results should be overwritten
if not os.path.exists(hcr_analysis_path) or hcr_params["overwrite_analysis"]:
    loaded = load_hcr_experiment(folder_path=base_folder, **hcr_params)
    hcr_analysis = fuse_measurement_datasets(base_path=base_folder, **loaded, **hcr_params, outline_averaging='median')
    export_analysis(hcr_analysis_path, hcr_analysis, hcr_params)
# Import the pre-computed analysis file for further post-processing steps
else:
    hcr_analysis, hcr_params_loaded = import_analysis(hcr_analysis_path)

    # Check if the currently defined analysis parameters match with the imported analysis parameters
    #check_parameters(hcr_params, hcr_params_loaded)

# Plot the transformation results
plot_brainfusion_results(hcr_analysis, **hcr_params,
                         results_folder=os.path.join(base_folder, 'results'), key_quant='Channel_3',
                         label='', cmap='Greens', marker_size=25, vmin=None, vmax=1024, verify_trafo=False, mask=True)
