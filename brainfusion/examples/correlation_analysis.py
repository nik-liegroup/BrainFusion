from spectroscopy_postprocessing import *
from params import brillouin_params, afm_params
from plot_results import plot_average_heatmap

# Load Brillouin data
brillouin_analysis_path = os.path.join(brillouin_params['results_folder'], 'BrillouinAnalysis.h5')
brillouin_analysis, brillouin_params_loaded = import_analysis(brillouin_analysis_path)
check_parameters(brillouin_params, brillouin_params_loaded)

# Load AFM data
afm_analysis_path = os.path.join(afm_params['results_folder'], 'AfmAnalysis.h5')
afm_analysis, afm_params_loaded = import_analysis(afm_analysis_path)
check_parameters(afm_params, afm_params_loaded)

# Calculate transformation between AFM and Brillouin datasets
results_folder = './correlation'
os.makedirs(results_folder, exist_ok=True)
trafo = afm_brillouin_transformation(afm_analysis,
                                     afm_params,
                                     brillouin_analysis,
                                     brillouin_params,
                                     results_folder)
afm_data_interp, br_data_interp, grid_avg, avg_contour = trafo

# Plot AFM and Brillouin trafo results
fig = plot_average_heatmap(afm_data_interp, grid_avg, avg_contour, 'modulus', vmin=None, vmax=None,
                           label='Reduced elastic modulus (Pa)', cmap='hot', marker_size=45, color='r--')
save_path = os.path.join('.', 'figures_for_thesis', 'AFMStiffness_CorrelationHeatmap.svg')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
fig.show()

fig = plot_average_heatmap(afm_data_interp, grid_avg, avg_contour, 'beta_pyforce', vmin=None, vmax=None,
                           label='Fluidity (Pa a)', cmap='cool', marker_size=45, color='b--')
save_path = os.path.join('.', 'figures_for_thesis', 'AFMFluidity_CorrelationHeatmap.svg')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
fig.show()

fig = plot_average_heatmap(br_data_interp, grid_avg, avg_contour, 'brillouin_shift_f_proj', vmin=None, vmax=None,
                           label='Brillouin shift (GHz)', cmap='hot', marker_size=45, color='r--')
save_path = os.path.join('.', 'figures_for_thesis', 'BrillouinShift_CorrelationHeatmap.svg')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
fig.show()

fig = plot_average_heatmap(br_data_interp, grid_avg, avg_contour, 'brillouin_peak_fwhm_f_proj', vmin=None, vmax=None,
                           label='Brillouin FWHM (GHz)', cmap='cool', marker_size=45, color='b--')
save_path = os.path.join('.', 'figures_for_thesis', 'BrillouinFWHM_CorrelationHeatmap.svg')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
fig.show()

# Calculate cross-correlation between stiffness and Brillouin shift maps and plot results
modulus_shift_corr = bin_and_correlate(afm_data_interp['modulus_median'],
                                       br_data_interp['brillouin_shift_f_proj_median'],
                                       grid_avg,
                                       bin_size=1)
plot_norm_corr(**modulus_shift_corr, label1='Norm. Reduced elastic modulus', label2='Norm. Brillouin shift',
               output_path=os.path.join(results_folder, f'ModulusShift_Correlation.svg'))

# Calculate cross-correlation between fluidity and Brillouin FWHM maps
fluidity_fwhm_corr = bin_and_correlate(afm_data_interp['beta_pyforce_median'],
                                       br_data_interp['brillouin_peak_fwhm_f_proj_median'],
                                       grid_avg,
                                       bin_size=1)
plot_norm_corr(**fluidity_fwhm_corr, label1='Norm. Fluidity', label2='Norm. Brillouin FWHM',
               output_path=os.path.join(results_folder, f'FluidityFWHM_Correlation.svg'))

# Calculate auto-correlation between stiffness and fluidity maps
modulus_fluidity_corr = bin_and_correlate(afm_data_interp['modulus_median'],
                                          afm_data_interp['beta_pyforce_median'],
                                          grid_avg,
                                          bin_size=1)
plot_norm_corr(**modulus_fluidity_corr, label1='Norm. Reduced elastic modulus', label2='Norm. Fluidity',
               output_path=os.path.join(results_folder, f'ModulusFluidity_Correlation.svg'))

# Calculate auto-correlation between Brillouin shift and Brillouin FWHM maps
shift_fwhm_corr = bin_and_correlate(br_data_interp['brillouin_shift_f_proj_median'],
                                    br_data_interp['brillouin_peak_fwhm_f_proj_median'],
                                    grid_avg,
                                    bin_size=1,
                                    map1_fit_limits=[0.4, 1.0],
                                    map2_fit_limits=[0, 0.3])
plot_norm_corr(**shift_fwhm_corr, label1='Norm. Brillouin shift', label2='Norm. Brillouin FWHM',
               output_path=os.path.join(results_folder, f'ShiftFWHM_Correlation.svg'))
