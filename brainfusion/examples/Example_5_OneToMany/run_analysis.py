"""
Case 5: one modality has a single averaged map (e.g. one AFM average across several animals), the other has
MANY individual samples of a much DENSER modality (e.g. per-pixel myelin images from many separate animals)
to correlate it against.

Radius-matching (`correlate_around_reference_grid`) rather than the shared-grid route (`Example_3`) because
the two densities are wildly different here - sparse AFM points vs. a dense per-pixel image. Forcing both
onto one intermediate regular grid would upsample AFM and alias the myelin image; radius-averaging myelin
pixels down onto AFM's own sparse points (AFM as the reference - always pick the SPARSER side, see
`correlate_around_reference_grid`'s docstring) avoids both. Everyone still needs to be warped into the same
template coordinate space first, via one shared `run_fusion` call, before radius-matching is meaningful.

Loops per myelin animal rather than pooling myelin first - keeps AFM correlated against each myelin animal
individually. If the myelin images ARE from the same animals as the AFM measurements, this preserves that
per-animal pairing; if not (unpaired cohorts), average the per-animal correlations afterward, or pool myelin
into a single averaged map first via `Example_3`'s `group_average_on_shared_grid` path instead.
"""

import os

from brainfusion import (load_fused_analysis, load_microscopy_all, run_fusion, plot_brainfusion_results,
                         correlate_around_reference_grid, plot_correlation_with_radii)
from brainfusion.sample import replace

here = os.path.dirname(__file__)
results_folder = os.path.join(here, "results")
value_key = "value"

afm_h5 = os.path.join(here, "..", "Example_1_AverageMap", "results", "analysis.h5")
afm_samples = load_fused_analysis(afm_h5, key_quant="modulus", value_key=value_key, metadata={"modality": "AFM"})

myelin_samples = load_microscopy_all(os.path.join(here, "data", "Myelin"), boundary_filename="BrainBoundary")
# load_microscopy_all names channels 'Channel_1', 'Channel_2', ... - pick the one that is myelin staining,
# and rename it to the same `value_key`/'modality' tag the AFM Sample above uses
myelin_samples = [replace(s, dataset={value_key: s.dataset["Channel_3"]},
                          metadata={**s.metadata, "modality": "Myelin"})
                  for s in myelin_samples]

FUSION_KWARGS = dict(contour_template="average", outline_averaging="median", contour_interp_n=200,
                     clustering="Mean")
analysis = run_fusion(afm_samples + myelin_samples, FUSION_KWARGS,
                      results_path=os.path.join(results_folder, "analysis.h5"))

plot_brainfusion_results(analysis, results_folder, key_quant=value_key, cmap="hot", cbar_label=value_key)

# `measurement_datasets`/`measurement_trafo_grids` are each sample's own points, warped into the shared
# template space but NOT resampled onto any shared grid - exactly what radius-matching needs. AFM is
# whichever index it landed at (0, since it was listed first above).
afm_grid = analysis["measurement_trafo_grids"][0]
afm_data = analysis["measurement_datasets"][0][value_key]
contour = analysis["template_contours"][0]

for i, myelin_sample in enumerate(myelin_samples, start=len(afm_samples)):
    myelin_grid = analysis["measurement_trafo_grids"][i]
    myelin_data = analysis["measurement_datasets"][i][value_key]

    result = correlate_around_reference_grid(afm_data, afm_grid, myelin_data, myelin_grid, contour,
                                             name_a="AFM", name_b="Myelin")
    print(f"AFM vs {myelin_sample.filename}: r={result['pearson_correlation']:.3f}, "
         f"p={result['pearson_p_value']:.3g}, n={result['n_points']}")

    plot_correlation_with_radii(result["reference_grid"], myelin_grid, contour, result["radii"],
                                label_a="AFM", label_b="Myelin", results_folder=results_folder,
                                results_name=f"CorrelationRadii_{myelin_sample.filename}",
                                title=f"AFM vs {myelin_sample.filename}")
