"""
Case 2: same as Example_1, but each experiment folder's name also carries a condition (e.g. "..._Control_..."
or "..._CS_..."). Fuse everyone onto ONE shared shape as usual, then split the result into per-condition
averages and cross-correlate every pair of conditions.
"""

import os

from brainfusion import (load_batchforce_all, run_fusion, plot_brainfusion_results, plot_group_average_map,
                         list_groups, group_average_on_shared_grid, pairwise_correlate_groups,
                         plot_correlation_masks, plot_norm_corr)

here = os.path.dirname(__file__)
data_folder = os.path.join(here, "data")
results_folder = os.path.join(here, "results")
key_quant = "modulus"

LOADER_KWARGS = dict(
    afm_variables={"modulus": "Reduced apparent elastic modulus"},
    batchforce_filename="data.csv",
    grid_conv_filename="GridInversionMatrix.csv",
    boundary_filename="outline",
    # Named groups here become `Sample.metadata`, e.g. {'animal_number': 1, 'condition': 'Control', 'stage': 37}
    name_pattern=r"#(?P<animal_number>\d+)_.*?_(?P<condition>[A-Za-z]+)_Stage(?P<stage>\d+)_(?P<date>\d+)",
    name_converters={"animal_number": int, "stage": int},
)

FUSION_KWARGS = dict(contour_template="average", outline_averaging="median", contour_interp_n=200,
                     clustering="Mean")

samples = load_batchforce_all(data_folder, **LOADER_KWARGS)
analysis = run_fusion(samples, FUSION_KWARGS, results_path=os.path.join(results_folder, "analysis.h5"))

plot_brainfusion_results(analysis, results_folder, key_quant=key_quant, cmap="hot", vmin=0, vmax=500,
                         cbar_label="Reduced elastic modulus (Pa)")

# Global average, ignoring condition - same as Example_1's single averaged map
plot_group_average_map(analysis, key_quant, cbar_label="Reduced elastic modulus (Pa)", cmap="hot", vmin=0,
                       vmax=500, output_path=os.path.join(results_folder, "Averaged_All.png"))

# One averaged map per condition
groups = list_groups(analysis, group_field="condition")
print(f"Conditions found: {groups}")
for group in groups:
    plot_group_average_map(analysis, key_quant, group_field="condition", group=group,
                           cbar_label="Reduced elastic modulus (Pa)", cmap="hot", vmin=0, vmax=500,
                           output_path=os.path.join(results_folder, f"Averaged_{group}.png"))

# Cross-correlate every pair of conditions point-by-point (2 conditions -> 1 pair, N -> every unordered pair)
grid, contour, group_maps = group_average_on_shared_grid(analysis, group_field="condition")
for (group_a, group_b), result in pairwise_correlate_groups(group_maps, grid, contour, key_quant).items():
    print(f"{group_a} vs {group_b}: r={result['pearson_correlation']:.3f}, p={result['pearson_p_value']:.3g}, "
         f"n={result['n_points']}")

    valid = result["valid_mask"]
    data_a, data_b = group_maps[group_a][key_quant], group_maps[group_b][key_quant]
    plot_correlation_masks(grid[valid], data_a[valid], data_b[valid], contour, label_a=group_a, label_b=group_b,
                           results_folder=results_folder, results_name=f"correlation_masks_{group_a}_{group_b}")
    plot_norm_corr(data_a[valid], data_b[valid], pearson=result["pearson_correlation"],
                  p_value=result["pearson_p_value"], label1=group_a, label2=group_b,
                  output_path=os.path.join(results_folder, f"{group_a}_vs_{group_b}_correlation.png"))
