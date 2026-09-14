"""
Case 2: same as Example_1, but each experiment folder's name also carries a condition (e.g. "..._Control_..."
or "..._CS_..."). Fuse everyone onto ONE shared shape as usual, passing group_field="condition" so
run_fusion also computes and caches each condition's own averaged map right away, on the SAME shared grid as
everyone else - then cross-correlate every pair of conditions on that shared grid.
"""

import os

from brainfusion import (load_batchforce_all, run_fusion, plot_sample_warps, plot_average_map,
                         extract_groups, correlate_groups, plot_correlation_masks, plot_norm_corr)

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
                     clustering="Mean", group_field="condition")

samples = load_batchforce_all(data_folder, **LOADER_KWARGS)
analysis = run_fusion(samples, FUSION_KWARGS, results_path=os.path.join(results_folder, "analysis.h5"))

plot_sample_warps(analysis, results_folder, key_quant=key_quant, cmap="hot", vmin=0, vmax=500,
                         cbar_label="Reduced elastic modulus (Pa)")

# One averaged map per condition
groups = extract_groups(analysis)
print(f"Conditions found: {groups}")
for group in groups:
    plot_average_map(analysis, key_quant, group=group, cbar_label="Reduced elastic modulus (Pa)", cmap="hot",
                     vmin=0, vmax=500, output_path=os.path.join(results_folder, f"Averaged_{group}.png"))

# Cross-correlate every pair of conditions point-by-point.
for (group_a, group_b), result in correlate_groups(analysis, key_quant, groups=groups).items():
    print(f"{group_a} vs {group_b}: r={result['pearson_correlation']:.3f}, p={result['pearson_p_value']:.3g}, "
         f"n={result['n_points']}")

    grid, contour = result["grid"], result["contour"]
    data_a, data_b = result[group_a], result[group_b]
    mask = result["valid_mask"]
    plot_correlation_masks(grid[mask], data_a[mask], data_b[mask], contour, label_a=group_a, label_b=group_b,
                           results_folder=results_folder, results_name=f"correlation_masks_{group_a}_{group_b}")
    plot_norm_corr(data_a, data_b, pearson=result["pearson_correlation"], p_value=result["pearson_p_value"],
                  label1=group_a, label2=group_b,
                  output_path=os.path.join(results_folder, f"{group_a}_vs_{group_b}_correlation.png"))
