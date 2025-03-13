import os
from brainfusion import *
from params import myelin_params

# This script was made to explore the BrainFusion Python package using AFM stiffness maps of spinal cord tissue and image
# data quantifying myelin staining

# Set the base folder containing all experimental folders indicated with a # followed by a unique number
base_folder = r'path/to/data'

for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
    if os.path.isdir(folder_path) and ('#' in folder_name):
        # Define where the results of this analysis are stored
        myelin_analysis_path = os.path.join(folder_path, f'BrainFusion_SC_Analysis_{folder_name}.h5')

        # Run the analysis in case the resulting h5 has not been created yet or the results should be overwritten
        if not os.path.exists(myelin_analysis_path) or myelin_params["overwrite_analysis"]:
            loaded = load_sc_afm_myelin(folder_path=folder_path, **myelin_params)
            myelin_analysis = fuse_measurement_datasets(base_path=base_folder, **loaded, **myelin_params)
            export_analysis(myelin_analysis_path, myelin_analysis, myelin_params)
            append_parquet_file(os.path.join(base_folder, folder_name), myelin_analysis)
        # Import the pre-computed analysis file for further post-processing steps
        else:
            myelin_analysis, myelin_params_loaded = import_analysis(myelin_analysis_path)

            # Check if the currently defined analysis parameters match with the imported analysis parameters
            check_parameters(myelin_params, myelin_params_loaded)

        # Plot the transformation results
        plot_brainfusion_results(myelin_analysis, **myelin_params,
                                 results_folder=os.path.join(base_folder, folder_name, 'results'),
                                 key_quant='myelin_intensity', label='', cmap='grey', marker_size=35, vmin=None,
                                 vmax=None, verify_trafo=True)

        # Calculate correlation between AFM and myelin data
        corr_result = correlate_afm_myelin(myelin_analysis, radius='max', verify_corr=True)
