"""
Case 3: two modalities (e.g. AFM and Brillouin) fused SEPARATELY - each with its own loader, own template,
own results/analysis.h5 (see Example_1/Example_2) - combined afterward by loading both analyses back in as
plain Samples and running them through the regular fuse function again, tagged by modality.

Run an Example_1/Example_2-style analysis for each modality first (or point h5_path_a/b below at existing
results), then run this script.

This shared-grid route assumes the two modalities have comparable point densities (true for AFM vs
Brillouin - both modest regular scans). If one side is much denser than the other (e.g. AFM vs a per-pixel
myelin image), see Example_5 instead: `correlate_around_reference_grid` radius-matches the dense side down
onto the sparse side's own points rather than resampling both onto a third, in-between grid.
"""

import os

from brainfusion import (load_fused_analysis, run_fusion, plot_brainfusion_results, group_average_on_shared_grid,
                         pairwise_correlate_groups, plot_correlation_masks, plot_norm_corr)

here = os.path.dirname(__file__)
results_folder = os.path.join(here, "results")
value_key = "value"  # common name the two modalities' quantities get stored under after loading

h5_path_a = os.path.join(here, "..", "Example_1_AverageMap", "results", "analysis.h5")  # e.g. AFM
h5_path_b = os.path.join(here, "..", "Example_2_ConditionGroups", "results", "analysis.h5")  # e.g. Brillouin

# Each already-fused analysis becomes one Sample carrying its own averaged map. `value_key` gives both
# modalities' quantities the SAME dataset key even if they were named differently in their own analysis
# (e.g. AFM's key_quant='modulus' vs Brillouin's key_quant='brillouin_shift_f_proj') - group_average_on_shared_grid
# needs one common key to average/correlate on. 'modality' is the tag this script groups by; pick a
# group_field= here too if you'd rather keep an original analysis's own sub-conditions instead of collapsing
# it to one average.
samples = (load_fused_analysis(h5_path_a, key_quant="modulus", value_key=value_key, metadata={"modality": "A"})
          + load_fused_analysis(h5_path_b, key_quant="modulus", value_key=value_key, metadata={"modality": "B"}))

FUSION_KWARGS = dict(contour_template="average", outline_averaging="median", contour_interp_n=200,
                     clustering="Mean")
analysis = run_fusion(samples, FUSION_KWARGS, results_path=os.path.join(results_folder, "analysis.h5"))

plot_brainfusion_results(analysis, results_folder, key_quant=value_key, cmap="hot", cbar_label=value_key)

grid, contour, group_maps = group_average_on_shared_grid(analysis, group_field="modality")
for (group_a, group_b), result in pairwise_correlate_groups(group_maps, grid, contour, value_key).items():
    print(f"{group_a} vs {group_b}: r={result['pearson_correlation']:.3f}, p={result['pearson_p_value']:.3g}, "
         f"n={result['n_points']}")

    valid = result["valid_mask"]
    data_a, data_b = group_maps[group_a][value_key], group_maps[group_b][value_key]
    plot_correlation_masks(grid[valid], data_a[valid], data_b[valid], contour, label_a=group_a, label_b=group_b,
                           results_folder=results_folder)
    plot_norm_corr(data_a[valid], data_b[valid], pearson=result["pearson_correlation"],
                  p_value=result["pearson_p_value"], label1=group_a, label2=group_b,
                  output_path=os.path.join(results_folder, f"{group_a}_vs_{group_b}_correlation.png"))
