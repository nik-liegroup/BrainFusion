"""Load a previously-fused BrainFusion result (a `run_fusion`-exported .h5) back in as one or more plain
`Sample`s, so it can be fed into ANOTHER `run_fusion` call - e.g. to combine two modalities that were each
fused separately (their own loader, own template, own samples), or to re-fuse condition-level averages
from one analysis together with another analysis's."""

import os

from brainfusion.io import import_analysis
from brainfusion.sample import Sample
from brainfusion.fusion.grouping import group_average_on_shared_grid


def load_fused_analysis(h5_path, key_quant, group_field=None, value_key=None, metadata=None):
    """
    Load one fused analysis .h5 back in as a list of `Sample`s.

    Without `group_field`, returns a single `Sample` carrying the file's overall (all-samples) average.
    With `group_field`, returns one `Sample` per group found in the ORIGINAL analysis's own sample metadata
    (e.g. one per condition it was fused with), each still tagged with that group under `group_field` - so
    re-fusing several files this way keeps every file's own sub-grouping queryable afterward instead of
    collapsing each file down to one average.

    Parameters
    ----------
    h5_path : str
        Path to a `run_fusion`-exported analysis (see `export_analysis`/`run_fusion(results_path=...)`).
    key_quant : str
        Dataset key to carry over, e.g. 'modulus'.
    group_field : str, optional
        Metadata key the ORIGINAL analysis's samples were grouped by, e.g. 'condition'. Requires that
        analysis to have been fused with `clustering='Mean'/'Median'/'Sum'` (not 'GMM').
    value_key : str, optional
        Dataset key to store the loaded values under on the returned Sample(s). Defaults to `key_quant`.
        Set this when combining two modalities whose quantities are named differently (e.g. AFM's
        'modulus' vs Brillouin's 'brillouin_shift') - give both loads the same `value_key` (e.g. 'value')
        so the re-fused analysis has one common dataset key to group/correlate on instead of two.
    metadata : dict, optional
        Extra metadata to attach to every returned Sample, e.g. {'modality': 'AFM'}. Merged with (and
        taking priority over) any per-group label, so use a different key than `group_field` if you want
        both to survive on the same Sample.

    Returns
    -------
    list of Sample
    """
    value_key = value_key or key_quant
    analysis, _ = import_analysis(h5_path)
    stem = os.path.splitext(os.path.basename(h5_path))[0]

    if group_field is None:
        grid = analysis['measurement_interpolated_grid']
        contour = analysis['template_contours'][0]
        data = analysis['measurement_interpolated_dataset'][key_quant]
        return [Sample(contour=contour, grid=grid, dataset={value_key: data}, filename=stem,
                      metadata=dict(metadata or {}))]

    grid, contour, group_maps = group_average_on_shared_grid(analysis, group_field)
    return [
        Sample(contour=contour, grid=grid, dataset={value_key: data_by_key[key_quant]},
              filename=f"{stem}_{group}", metadata={group_field: group, **(metadata or {})})
        for group, data_by_key in group_maps.items()
    ]
