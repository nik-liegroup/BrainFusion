import os
from brainfusion import *
from params import afm_params

# This script was made to explore the BrainFusion Python package using AFM stiffness maps of spinal cord tissue and image
# data quantifying myelin staining

# Set the base folder containing all experimental folders indicated with a # followed by a unique number
base_folder = r'path/to/data'

for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
    if os.path.isdir(folder_path) and ('#' in folder_name):
        # Define where the results of this analysis are stored
        afm_analysis_path = os.path.join(folder_path, f'BrainFusion_SC_Analysis_{folder_name}.h5')

        # Run the analysis in case the resulting h5 has not been created yet or the results should be overwritten
        if not os.path.exists(afm_analysis_path) or afm_params["overwrite_analysis"]:
            afm_analysis = process_sc_experiment(**afm_params, path=os.path.join(base_folder, folder_name))
            export_analysis(afm_analysis_path, afm_analysis, afm_params)
            append_parquet_file(os.path.join(base_folder, folder_name), afm_analysis)
        # Import the pre-computed analysis file for further post-processing steps
        else:
            afm_analysis, afm_params_loaded = import_analysis(afm_analysis_path)

            # Check if the currently defined analysis parameters match with the imported analysis parameters
            check_parameters(afm_params, afm_params_loaded)

        # Plot results
        plot_sc_experiments(afm_analysis, **afm_params, results_folder=os.path.join(base_folder, folder_name, 'results'),
                            label='', cmap='grey', marker_size=35, vmin=None, vmax=None, verify_trafo=True)
