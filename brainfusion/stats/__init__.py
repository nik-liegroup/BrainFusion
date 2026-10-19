# brainfusion/stats/__init__.py
#
# Statistics on top of a finished fusion: how consistent are the animals, how sure is a correlation of two average maps
# (animal bootstrap, spatial null model), summaries over animals, correction for noise. None of it changes the fusion; the
# results are saved in their own file (`save_correlation_results`), not in the fusion's.

from .consistency import (animal_maps, average_map, pairwise_map_correlations, split_half_reliability, map_consistency)
from .bootstrap import bootstrap_map_correlation, bootstrap_group_correlation, correlation_vs_animals, bootstrap_map_scalar
from .surrogates import spatial_surrogates, spatial_null_test
from .summary import summarise_correlations, disattenuate
from .association import analyse_correlation_percentile, conditional_probability_table
from .conditions import permutation_scalar_difference, permutation_map_difference
from .results import save_correlation_results, load_correlation_results

__all__ = [
    "animal_maps", "average_map", "pairwise_map_correlations", "split_half_reliability", "map_consistency",
    "bootstrap_map_correlation", "bootstrap_group_correlation", "correlation_vs_animals", "bootstrap_map_scalar",
    "spatial_surrogates", "spatial_null_test",
    "summarise_correlations", "disattenuate",
    "analyse_correlation_percentile", "conditional_probability_table",
    "permutation_scalar_difference", "permutation_map_difference",
    "save_correlation_results", "load_correlation_results",
]
