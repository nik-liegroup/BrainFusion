"""
Case 5: match INDIVIDUAL experiments between two modalities one-to-one (e.g. animal #1's AFM measurement
with animal #1's Brillouin measurement), instead of fusing each modality's whole cohort together first.
This is just the regular fuse function, called once per matched pair - each pair only has 2 samples, so
none of the group_field/pairwise-correlate helpers from Example_2/3 are needed here.
"""

import os

from brainfusion import load_batchforce_all, load_brillouin_all, run_fusion, plot_sample_warps, \
    correlate_on_shared_grid

here = os.path.dirname(__file__)
results_folder = os.path.join(here, "results")
afm_key, brillouin_key = "modulus", "brillouin_shift_f_proj"

afm_samples = load_batchforce_all(
    os.path.join(here, "data", "AFM"),
    afm_variables={afm_key: "Reduced apparent elastic modulus"}, batchforce_filename="data.csv",
    grid_conv_filename="GridInversionMatrix.csv", boundary_filename="outline",
    name_pattern=r"#(?P<animal_number>\d+)_.*", name_converters={"animal_number": int})

brillouin_samples = load_brillouin_all(
    os.path.join(here, "data", "Brillouin"), brillouin_variables=[brillouin_key],
    boundary_filename="brain_outline",
    name_pattern=r"#(?P<animal_number>\d+)_.*", name_converters={"animal_number": int})

afm_by_animal = {s.metadata["animal_number"]: s for s in afm_samples}
brillouin_by_animal = {s.metadata["animal_number"]: s for s in brillouin_samples}

FUSION_KWARGS = dict(contour_template="average", outline_averaging="median", contour_interp_n=200,
                     clustering="Mean")

for animal in sorted(afm_by_animal.keys() & brillouin_by_animal.keys()):
    pair_results = os.path.join(results_folder, f"animal_{animal}")
    analysis = run_fusion([afm_by_animal[animal], brillouin_by_animal[animal]], FUSION_KWARGS,
                          results_path=os.path.join(pair_results, "analysis.h5"))
    plot_sample_warps(analysis, pair_results, key_quant=afm_key, cmap="hot", cbar_label=afm_key)

    grid = analysis["measurement_interpolated_grid"]
    contour = analysis["template_contours"][0]
    afm_data = analysis["measurement_trafo_datasets"][0][afm_key]
    brillouin_data = analysis["measurement_trafo_datasets"][1][brillouin_key]

    result = correlate_on_shared_grid(afm_data, brillouin_data, grid, contour, name_a="AFM", name_b="Brillouin")
    print(f"Animal {animal}: r={result['pearson_correlation']:.3f}, p={result['pearson_p_value']:.3g}, "
         f"n={result['n_points']}")
