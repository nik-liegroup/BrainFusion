from BrainFusion import *
from params import afm_params

# This script is made to explore the BrainFusion Python package using AFM stiffness maps of Xenopus laevis brain tissue

# Define where the results of this analysis are stored
afm_analysis_path = os.path.join(afm_params['results_folder'], 'BrainFusion_AFM_Analysis.h5')

# Run the analysis in case the resulting h5 has not been created yet
if not os.path.exists(afm_analysis_path):
    afm_analysis = process_multiple_experiments(**afm_params)
    export_analysis(afm_analysis_path, afm_analysis, afm_params)
# Import the pre-computed analysis file for further post-processing steps
else:
    afm_analysis, afm_params_loaded = import_analysis(afm_analysis_path)
    check_parameters(afm_params, afm_params_loaded)

# Plot results
plot_experiments(afm_analysis, **afm_params, raw_data_key='modulus', label='Stiffness K', cmap='hot', marker_size=9,
                 vmin=0, vmax=500)
