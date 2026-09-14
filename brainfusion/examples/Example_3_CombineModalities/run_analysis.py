"""
Case 3: two modalities (e.g. AFM and Brillouin) fused SEPARATELY - each with its own loader, own run_fusion
call, own results/analysis.h5 (see Example_1/Example_2), each onto ITS OWN cohort's average shape - so the
two files do NOT share a coordinate space and can't be correlated as-is.

If your two files DO already share a template (e.g. both used contour_template="first_element" pointing at
the same fixed outline), none of the re-fusion below is needed - see Example_4 instead.
"""

import os

from brainfusion import (load_fused_analysis, run_fusion, plot_sample_warps, merge_keys,
                         correlate_groups_by_density, plot_correlation_with_radii, plot_norm_corr)

here = os.path.dirname(__file__)
results_folder = os.path.join(here, "results")
new_key = "value"  # common name the two modalities' quantities get stored under after loading

h5_path_a = os.path.join(here, "..", "Example_1_AverageMap", "results", "analysis.h5")  # e.g. AFM
h5_path_b = os.path.join(here, "..", "Example_2_ConditionGroups", "results", "analysis.h5")  # e.g. Brillouin

samples_a = load_fused_analysis(h5_path_a, key_quant="modulus", rename_key=new_key, metadata={"modality": "AFM"})
samples_a = merge_keys(samples_a, new_key="group", keys_to_merge=["condition", "modality"])

samples_b = load_fused_analysis(h5_path_b, key_quant="shift", group_field="condition", rename_key=new_key, metadata={"modality": "Brillouin"})
samples_b = merge_keys(samples_b, new_key="group", keys_to_merge=["condition", "modality"])

samples = samples_a + samples_b

FUSION_KWARGS = dict(contour_template="average", outline_averaging="median", contour_interp_n=200,
                     clustering="Mean", group_field="group")
analysis = run_fusion(samples, FUSION_KWARGS, results_path=os.path.join(results_folder, "analysis.h5"))

plot_sample_warps(analysis, results_folder, key_quant=new_key, cmap="hot", cbar_label=new_key)

# Cross-correlate every pair of groups point-by-point. sparse_group is whichever side has fewer points
# (used as the anchor grid); dense_group is radius-averaged down onto it - pass priority=[...] (sparsest
# first) to correlate_groups_by_density if point count alone shouldn't decide the ordering.
for (sparse_group, dense_group), result in correlate_groups_by_density(analysis, "group", new_key).items():
    print(f"{sparse_group} vs {dense_group}: r={result['pearson_correlation']:.3f}, "
         f"p={result['pearson_p_value']:.3g}, n={result['n_points']}")

    contour = result["contour"]
    sparse_grid, dense_grid = result["reference_grid"], result["dense_grid"]
    plot_correlation_with_radii(sparse_grid, dense_grid, contour, result["radii"], result[sparse_group],
                                result["dense_data"], label_a=sparse_group, label_b=dense_group,
                                results_folder=results_folder,
                                results_name=f"CorrelationRadii_{sparse_group}_vs_{dense_group}")

    mask = result["valid_mask"]
    data_a, data_b = result[sparse_group][mask], result[dense_group][mask]
    plot_norm_corr(data_a, data_b, pearson=result["pearson_correlation"], p_value=result["pearson_p_value"],
                  label1=sparse_group, label2=dense_group,
                  output_path=os.path.join(results_folder, f"{sparse_group}_vs_{dense_group}_correlation.png"))
