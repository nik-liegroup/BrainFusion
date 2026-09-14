"""Splitting a fused analysis's samples into named groups (e.g. by condition or modality). Used internally by
`brain_fusion`/`run_fusion`'s `group_field` argument, and directly for grouping by a field that wasn't chosen
at fusion time. `merge_keys` builds a combined group field BEFORE fusion, when the group identity you
actually want is more than one metadata dimension at once (e.g. condition AND modality)."""

import numpy as np

from brainfusion.sample import replace


def merge_keys(samples, new_key, keys_to_merge, separator="_"):
    """
    Return copies of `samples` with `new_key` added to their metadata, combining the values of whichever of
    `keys_to_merge` each Sample actually has - e.g. `merge_keys(samples, "group", ["condition", "modality"])`
    turns `{'condition': 'cond1', 'modality': 'AFM'}` into `{..., 'group': 'cond1_AFM'}`. A Sample missing
    one of `keys_to_merge` (e.g. one modality's Samples were never split by condition) just uses whichever of
    the keys it does have, instead of raising - so the same call works across Samples with different
    metadata schemas.

    Use this when the group identity you actually want to fuse/correlate by is a combination of independent
    metadata dimensions - e.g. combining per-condition Samples from two modalities, where
    `group_field='modality'` alone would wrongly pool every condition of the same modality together.

    Parameters
    ----------
    samples : list of Sample
    new_key : str
        Metadata key to store the merged value under.
    keys_to_merge : list of str
        Existing metadata keys to combine, in order. Keys not present on a given Sample are skipped.
    separator : str, default='_'
        Joins the found values into one string.

    Returns
    -------
    list of Sample
    """
    return [replace(s, metadata={**s.metadata,
                                 new_key: separator.join(str(s.metadata[key]) for key in keys_to_merge
                                                         if key in s.metadata)})
           for s in samples]


def list_groups(analysis, group_field):
    """List the distinct values found under `metadata[group_field]` across an analysis's samples - works for
    ANY metadata field, whether or not it was used as `run_fusion(..., group_field=...)`'s grouping field."""
    return sorted({str(m[group_field]) for m in analysis['measurement_metadata']})


def extract_groups(analysis):
    """
    The groups `run_fusion(..., group_field=...)` computed for this analysis, in the order they were
    computed - just `list(analysis['group_datasets'].keys())`. Raises if this analysis wasn't fused with a
    `group_field` (use `list_groups(analysis, some_field)` to inspect an arbitrary metadata field instead).
    """
    if 'group_datasets' not in analysis:
        raise ValueError("This analysis wasn't fused with a group_field - see run_fusion(..., group_field=...).")
    return list(analysis['group_datasets'].keys())


def group_average_on_shared_grid(analysis, group_field, groups=None):
    """
    Split a pooled `run_fusion`/`brain_fusion` result into per-group averaged maps that all live on the
    analysis's own shared grid, using each sample's `.metadata[group_field]` (e.g. `group_field='condition'`
    to split a pooled Control+CS fusion back into a 'Control' and a 'CS' average).

    Every dataset key present on the samples (e.g. 'modulus' and, if loaded, 'beta_pyforce') is averaged in
    one pass - there is no per-quantity cost to computing them all, so there's no reason to ask for just
    one up front. Pull out whichever key you need downstream, e.g. `group_maps['Control']['modulus']`.

    Because every sample was already resampled onto the same `measurement_interpolated_grid` inside
    `fuse_grids` (stored as `measurement_trafo_datasets`), the returned group averages land on identical
    grid points with no extra spatial matching needed - pass them straight to `correlate_on_shared_grid` or
    `pairwise_correlate_groups`.

    This always (re)computes from `measurement_trafo_datasets` - it's what `run_fusion(...,
    group_field=...)` itself calls to build `analysis['group_datasets']` in the first place. If you already
    fused with this `group_field`, just read `analysis['group_datasets']` directly instead of calling this
    again; use this function for grouping by a field you did NOT fuse with.

    Only works when `clustering` was 'Mean', 'Median' or 'Sum': 'GMM' clustering builds its own
    data-driven grid instead of resampling every sample onto one shared grid, so there is nothing here to
    split by group.

    Parameters
    ----------
    analysis : dict
        A `run_fusion`/`brain_fusion` result (pooled across every group you want to compare).
    group_field : str
        Key into each sample's `.metadata` dict to group by, e.g. 'condition'.
    groups : list, optional
        Which group values to compute, and in what order. Defaults to every unique value found
        (see `list_groups`).

    Returns
    -------
    grid : (P, 2) array
        The shared grid every group average lives on.
    contour : (N, 2) array
        The analysis's template contour.
    group_maps : dict {group_value: {key: (P,) array}}
        `nanmean` of every dataset key across each group's samples.
    """
    per_sample = analysis.get('measurement_trafo_datasets')
    if not per_sample or per_sample[0] is None:
        raise ValueError("This analysis has no per-sample data resampled onto a shared grid - re-run "
                         "run_fusion/brain_fusion with clustering='Mean', 'Median' or 'Sum' (not 'GMM').")

    values = np.array([m[group_field] for m in analysis['measurement_metadata']])
    groups = list_groups(analysis, group_field) if groups is None else groups
    keys = per_sample[0].keys()

    group_maps = {}
    for group in groups:
        sample_idx = np.where(values == group)[0]
        if len(sample_idx) == 0:
            raise ValueError(f"No samples found with metadata['{group_field}'] == '{group}'.")
        group_maps[group] = {key: np.nanmean(np.array([per_sample[i][key] for i in sample_idx]), axis=0)
                             for key in keys}

    return analysis['measurement_interpolated_grid'], analysis['template_contours'][0], group_maps


def extract_group_native_data(analysis, group_field, key_quant, groups=None):
    """
    Pull each group's own per-sample data at its NATIVE point density - every matching sample's
    `measurement_trafo_grids`/`measurement_datasets` points, concatenated together per group - instead of
    resampling everyone onto one shared grid like `group_average_on_shared_grid` does.

    Use this when the groups' point densities are too different to share one grid meaningfully (e.g. sparse
    AFM indentation points vs a dense per-pixel myelin image, or 2+ modalities that were each fused
    separately at their own resolution and combined via `load_fused_analysis`). Every sample must already be
    warped into the SAME template coordinate space, i.e. `analysis` comes from one shared `run_fusion` call
    covering every group - see `brainfusion.correlation.pairwise_correlate_by_density`, which correlates
    groups extracted this way.

    Parameters
    ----------
    analysis : dict
        A `run_fusion`/`brain_fusion` result.
    group_field : str
        Key into each sample's `.metadata` dict to group by, e.g. 'modality'.
    key_quant : str
        Dataset key to pull for each sample, e.g. 'value'.
    groups : list, optional
        Which groups to include. Defaults to every unique value found (see `list_groups`).

    Returns
    -------
    dict {group_value: (grid, data)}
        Every matching sample's own warped grid and data, concatenated together per group.
    """
    values = np.array([m[group_field] for m in analysis['measurement_metadata']])
    groups = list_groups(analysis, group_field) if groups is None else groups

    native = {}
    for group in groups:
        sample_idx = np.where(values == group)[0]
        if len(sample_idx) == 0:
            raise ValueError(f"No samples found with metadata['{group_field}'] == '{group}'.")
        native[group] = (
            np.concatenate([analysis['measurement_trafo_grids'][i] for i in sample_idx]),
            np.concatenate([analysis['measurement_datasets'][i][key_quant] for i in sample_idx]),
        )

    return native
