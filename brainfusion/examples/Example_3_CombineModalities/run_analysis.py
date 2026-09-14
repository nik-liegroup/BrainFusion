"""
Case 3: two modalities (e.g. AFM and Brillouin) fused SEPARATELY - each with its own loader, own run_fusion
call, own results/analysis.h5 (see Example_1/Example_2).
"""

import os

from brainfusion import load_fused_analysis, pairwise_correlate_by_density, plot_correlation_with_radii, \
    plot_norm_corr

here = os.path.dirname(__file__)
results_folder = os.path.join(here, "results")

h5_path_a = os.path.join(here, "..", "Example_1_AverageMap", "results", "analysis.h5")  # e.g. AFM
h5_path_b = os.path.join(here, "..", "Example_2_ConditionGroups", "results", "analysis.h5")  # e.g. Brillouin

# Each file becomes one Sample carrying its own averaged map, own grid, own (shared) contour.
sample_a = load_fused_analysis(h5_path_a, key_quant="modulus")[0]
sample_b = load_fused_analysis(h5_path_b, key_quant="modulus")[0]  # swap key_quant for Brillouin's own key

datasets = {"AFM": (sample_a.grid, sample_a.dataset["modulus"]),
           "Brillouin": (sample_b.grid, sample_b.dataset["modulus"])}
contour = sample_a.contour  # both already on the same template, so either file's contour works

for (ref_name, other_name), result in pairwise_correlate_by_density(datasets, contour).items():
    print(f"{ref_name} vs {other_name}: r={result['pearson_correlation']:.3f}, "
         f"p={result['pearson_p_value']:.3g}, n={result['n_points']}")

    other_grid = datasets[other_name][0]
    plot_correlation_with_radii(result["reference_grid"], other_grid, contour, result["radii"],
                                label_a=ref_name, label_b=other_name, results_folder=results_folder,
                                results_name=f"CorrelationRadii_{ref_name}_vs_{other_name}")
    plot_norm_corr(result[f"{ref_name}_valid"], result[f"{other_name}_valid"],
                  pearson=result["pearson_correlation"], p_value=result["pearson_p_value"],
                  label1=ref_name, label2=other_name,
                  output_path=os.path.join(results_folder, f"{ref_name}_vs_{other_name}_correlation.png"))
