"""
An atlas: regions drawn once on an averaged outline (e.g. the template of a fusion) and pulled back onto the single samples of
a fusion, using only what the fusion stored - nothing has to be fused again.

  read_regions_file        the regions of a roi-annotator style text file (`regions_brainfusion.txt`): the names of the
                           regions are the section names in the file, nothing is fixed in BrainFusion
  atlas_pullback           regions in the frame of a fusion's template -> the frame of every sample of the fusion (the
                           inverse of the warp the fusion made), added to the analysis dict as 'atlas' (the outline and
                           the regions) and 'sample_regions' (one dict of region outlines per sample)
  save_atlas_pullback      writes those two entries into the fusion's .h5 file
  map_regions_between_outlines   regions drawn on one outline moved onto another (the template of another fusion)
  region_masks             which points of a grid (e.g. a shared grid in the template frame) lie in each region

Regions are polygons (N, 2) in the units of the fusion's samples (AFM: metres; `read_regions_file(scale=...)` converts).
The outlines pulled back are not clipped to the samples' outlines.
"""

import re

import h5py
import numpy as np
from matplotlib.path import Path

from brainfusion.fusion.match_contours import align_contours, boundary_match_contours, interpolate_contour
from brainfusion.fusion.transform_2Dmap import create_rbf_interpolators
from brainfusion.io import write_dict_in_h5

SAME_OUTLINE = 1e-3   # two outlines closer than this (mean distance, in parts of the larger extent) are the same outline


def read_regions_file(path, scale=1.0):
    """({name: polygon (N, 2)} in the order of the file, header {key: value}) of a roi-annotator style regions file:
    header lines '# key: value', then one section '[name]' per region with one 'x<TAB>y' line per point (a first point
    repeated at the end is dropped). `scale` multiplies the coordinates (e.g. 1e-6: um -> m)."""
    regions, header, name, points = {}, {}, None, []

    def close():
        if name is not None and points:
            polygon = np.array(points, dtype=float) * scale
            regions[name] = polygon[:-1] if len(polygon) > 3 and np.array_equal(polygon[0], polygon[-1]) else polygon

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                header[key.strip()] = value.strip()
            elif re.fullmatch(r"\[[^\]]+\]", line):
                close()
                name, points = line[1:-1], []
            elif name is not None:
                points.append([float(v) for v in line.split()])
    close()
    return regions, header


def _closed(outline):
    outline = np.asarray(outline, dtype=float)
    return outline if np.array_equal(outline[0], outline[-1]) else np.vstack([outline, outline[0]])


def _densify(polygon, n):
    """The polygon as `n` points equally spaced along its length."""
    closed = _closed(polygon)
    return interpolate_contour(closed, n + 1)[:-1]


def _apply_matrix(matrix, points):
    return (np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix, dtype=float).T)[:, :2]


def _same_outline(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape and (len(a) < 3 or len(b) < 3):
        return False
    distance = np.linalg.norm(a[:, None] - b[None], axis=2)
    extent = max(np.ptp(a, axis=0).max(), np.ptp(b, axis=0).max())
    return (distance.min(axis=1).mean() + distance.min(axis=0).mean()) / 2 < SAME_OUTLINE * extent


def map_regions_between_outlines(regions, from_outline, to_outline, n_points=400, fit_routine='ellipse', canonical_axes=True,
                                 curvature=0.5, smooth='auto'):
    """The regions drawn on `from_outline` moved onto `to_outline` (both in the same unit; BrainFusion's way of matching two
    outlines: alignment, boundary matching, RBF warp). The same regions if the outlines are the same."""
    if _same_outline(from_outline, to_outline):
        return {name: np.asarray(p, dtype=float) for name, p in regions.items()}
    a = interpolate_contour(_closed(from_outline), 200)
    b = interpolate_contour(_closed(to_outline), 200)
    aligned, _, matrices = align_contours([b, a], [None, None], template_index=0, fit_routine=fit_routine,
                                          canonical_axes=canonical_axes)
    matched, matched_template = boundary_match_contours(aligned, template_index=0, curvature=curvature)
    rbf_x, rbf_y = create_rbf_interpolators(matched[1], matched_template[1], smooth=smooth)
    forward = np.linalg.inv(np.asarray(matrices[1], dtype=float))   # the matrices lead back from the aligned frame
    out = {}
    for name, polygon in regions.items():
        points = _apply_matrix(forward, _densify(polygon, n_points))
        moved = np.column_stack([rbf_x(points[:, 0], points[:, 1]), rbf_y(points[:, 0], points[:, 1])])
        out[name] = _apply_matrix(matrices[0], moved)
    return out


def atlas_pullback(analysis, regions, atlas_outline=None, n_points=400, smooth='auto', **match):
    """
    Pull regions back from the frame of a fusion's template onto every sample of the fusion.

    `analysis`: a fusion result (`brain_fusion`, or `import_analysis` of its .h5). `regions`: {name: polygon (N, 2)} drawn on
    `atlas_outline`; without it they are taken to be drawn on this fusion's template (`analysis['template_contours'][0]`),
    with it they are first moved onto the template (`map_regions_between_outlines`, `match`: its options). Every region
    outline is then carried back through the inverse of the warp the fusion made for that sample (RBF from the matched
    template contour to the matched sample contour, then the stored affine matrix back to the sample's own frame).

    Adds `analysis['atlas']` = {'outline': the template, 'regions': the regions in its frame} and
    `analysis['sample_regions']` = one {name: polygon in the sample's own frame} per sample (the order of
    `analysis['measurement_filenames']`). Returns {filename: {name: polygon}}.
    """
    template = np.asarray(analysis['template_contours'][0], dtype=float)
    if atlas_outline is not None:
        regions = map_regions_between_outlines(regions, atlas_outline, template, n_points=n_points, smooth=smooth, **match)
    regions = {name: _densify(p, n_points) for name, p in regions.items()}
    sample_regions = []
    for index, matrix in enumerate(analysis['affine_matrices'][1:]):   # the first matrix belongs to the template
        rbf_x, rbf_y = create_rbf_interpolators(np.asarray(analysis['template_contours'][index], dtype=float),
                                                np.asarray(analysis['measurement_contours'][index], dtype=float), smooth=smooth)
        sample_regions.append({name: _apply_matrix(matrix, np.column_stack([rbf_x(p[:, 0], p[:, 1]), rbf_y(p[:, 0], p[:, 1])]))
                               for name, p in regions.items()})
    analysis['atlas'] = {'outline': template, 'regions': regions}
    analysis['sample_regions'] = sample_regions
    return dict(zip(analysis['measurement_filenames'], sample_regions))


def save_atlas_pullback(path, analysis):
    """Write the 'atlas' and 'sample_regions' of an analysis (`atlas_pullback`) into the fusion's .h5 file (replacing earlier ones)."""
    with h5py.File(path, 'r+') as h5file:
        for key in ('atlas', 'sample_regions'):
            if key in h5file:
                del h5file[key]
        write_dict_in_h5(h5file, '/', {key: analysis[key] for key in ('atlas', 'sample_regions')})


def region_masks(grid, regions):
    """{name: boolean array} - which points of `grid` (N, 2) lie inside each region polygon (regions may overlap)."""
    grid = np.asarray(grid, dtype=float).reshape(-1, 2)
    return {name: Path(np.asarray(polygon, dtype=float)).contains_points(grid) for name, polygon in regions.items()}
