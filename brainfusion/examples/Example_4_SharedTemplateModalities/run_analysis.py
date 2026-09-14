"""
Case 4: two modalities fused SEPARATELY (see Example_1/Example_2), but onto the SAME template (e.g. both
used contour_template="first_element" pointing at the same fixed outline) - so, unlike Example_3, their own
averaged maps already live in the same coordinate space. No run_fusion/group_field needed at all here, just
read each file's own grid/data/contour back and correlate directly with `pairwise_correlate_by_density`.

Each file's own averaged grid still comes from its OWN run_fusion call's independent extend_grid spacing, so
even with "comparable" density the two grids are generally NOT the exact same points -
`pairwise_correlate_by_density` handles that correctly regardless (nearest-neighbour radius-matching, not
point-by-point matching).

If the two files do NOT share a template, this doesn't apply - see Example_3 instead, which re-fuses them
together first to get them into one shared coordinate space.
"""

import os

from brainfusion import load_fused_analysis, pairwise_correlate_by_density, plot_correlation_with_radii, \
    plot_norm_corr

here = os.path.dirname(__file__)
results_folder = os.path.join(here, "results")

h5_path_a = os.path.join(here, "..", "Example_1_AverageMap", "results", "analysis.h5")  # e.g. AFM
h5_path_b = os.path.join(here, "..", "Example_2_ConditionGroups", "results", "analysis.h5")  # e.g. Brillouin

# Each file becomes one Sample carrying its own averaged map, own grid, own (shared) contour
sample_a = load_fused_analysis(h5_path_a, key_quant="modulus")[0]
sample_b = load_fused_analysis(h5_path_b, key_quant="modulus")[0]  # swap key_quant for Brillouin's own key

datasets = {"AFM": (sample_a.grid, sample_a.dataset["modulus"]),
           "Brillouin": (sample_b.grid, sample_b.dataset["modulus"])}
contour = sample_a.contour  # both already on the same template, so either file's contour works

# `sparse_group` is whichever side has fewer points (used as the anchor grid); `dense_group` is radius-
# averaged down onto it - pass priority=[...] (sparsest first) instead of relying on point count if needed.
for (sparse_group, dense_group), result in pairwise_correlate_by_density(datasets, contour).items():
    print(f"{sparse_group} vs {dense_group}: r={result['pearson_correlation']:.3f}, "
         f"p={result['pearson_p_value']:.3g}, n={result['n_points']}")

    dense_grid, dense_data = datasets[dense_group]
    plot_correlation_with_radii(result["reference_grid"], dense_grid, contour, result["radii"],
                                result[sparse_group], dense_data, label_a=sparse_group,
                                label_b=dense_group, results_folder=results_folder,
                                results_name=f"CorrelationRadii_{sparse_group}_vs_{dense_group}")

    mask = result["valid_mask"]
    plot_norm_corr(result[sparse_group][mask], result[dense_group][mask],
                  pearson=result["pearson_correlation"], p_value=result["pearson_p_value"],
                  label1=sparse_group, label2=dense_group,
                  output_path=os.path.join(results_folder, f"{sparse_group}_vs_{dense_group}_correlation.png"))
