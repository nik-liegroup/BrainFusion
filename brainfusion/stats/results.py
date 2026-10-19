"""Save and load the results of the statistics (a nested dict) in their own HDF5 file, next to the fusion file they came from."""

import numpy as np

from brainfusion.io import export_analysis, import_analysis


def _plain(obj):
    """Nested dict/list/tuple/numpy values as what the HDF5 writer takes: dicts with string keys, arrays, numbers, strings."""
    if isinstance(obj, dict):
        return {str(key): _plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        if all(isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) for v in obj):
            return np.array(obj, dtype=float)
        if all(isinstance(v, str) for v in obj):
            return list(obj)
        return {str(i): _plain(v) for i, v in enumerate(obj)}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_correlation_results(path, results, params=None):
    """Write `results` (nested dicts of arrays, numbers and strings, e.g. what `bootstrap_group_correlation`,
    `spatial_null_test` and `map_consistency` return, put into one dict) to the HDF5 file `path`; `params` (a flat dict of
    numbers and strings: how it was computed) go into its attributes."""
    export_analysis(str(path), _plain(results), params or {})


def load_correlation_results(path):
    """(results, params) from a file written by `save_correlation_results`."""
    return import_analysis(str(path))
