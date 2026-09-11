"""Small helper for the "compute once, then reuse the cached result" step every analysis script needs."""

import os

from brainfusion.brainfusion import brain_fusion, brain_fusion_correlation
from brainfusion.io import check_parameters, export_analysis, import_analysis


def run_fusion(samples, fusion_kwargs: dict, results_path: str, overwrite: bool = False,
              mode: str = "fusion") -> dict:
    """
    Run `brain_fusion` (or, if `mode="correlation"`, `brain_fusion_correlation`) on `samples` and cache the
    result at `results_path`. If a cached result already exists and `overwrite` is False, it is loaded
    instead and its stored parameters are compared against `fusion_kwargs`.
    """
    if overwrite or not os.path.exists(results_path):
        fuse = brain_fusion_correlation if mode == "correlation" else brain_fusion
        analysis = fuse(samples, **fusion_kwargs)
        export_analysis(results_path, analysis, fusion_kwargs)
    else:
        analysis, loaded_kwargs = import_analysis(results_path)
        check_parameters(fusion_kwargs, loaded_kwargs)

    return analysis
