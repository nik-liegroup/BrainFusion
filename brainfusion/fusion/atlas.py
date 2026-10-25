"""
An atlas: regions drawn once on one outline and pulled back onto every sample of a fusion, from what the fusion stored - nothing
is fused again.

    atlas_pullback(fusion_file, regions_file, drawn_on='average')

reads the regions file (a text file: '# key: value' header lines, one '[name]' section per region with
'x<TAB>y' lines; the names of the regions are the section names), and `drawn_on` says which outline of the fusion they were drawn
on: 'average' (the fusion's template, the default) or one of its samples (its position in the fusion, or its file name). It

  1. brings the regions onto the template (a sample's regions through that sample's own warp),
  2. pulls them back onto every sample of the fusion (`pull_back_regions`),
  3. stores them in the fusion's .h5 file (`analysis['regions']`, below),
  4. plots them on the average outline (`Regions_average.png`) and on the outline of every sample (`Regions_<file name>.png`,
     one picture per sample), next to the fusion file (`plot_regions`).

`analysis['regions']` (after `import_analysis` of that .h5 file):
    names     the names of the regions, in the order of the regions file
    drawn_on  'average', or the file name of the sample they were drawn on
    atlas     {name: polygon (N, 2)} in the frame of the template
    samples   one {name: polygon (N, 2)} per sample (order of `analysis['measurement_filenames']`) in the sample's own frame
              and units; a region that none of the sample's points land in is left out
`region_masks(grid, regions)` tells which points of a grid lie in each region: `analysis['regions']['atlas']` with the sample's
points in the template (`analysis['measurement_trafo_grids'][i]`), or the sample's polygons with its own points
(`measurement_grids` are in the aligned frame: move the polygons there with the inverse of `analysis['affine_matrices'][i + 1]`).

Units: a regions file is in um unless its header says '# unit: m'; a fusion's samples are in metres (AFM) or um (Brillouin) as
their metadata 'length_unit' says, the template in the unit of its first sample.
"""

from pathlib import Path as FilePath

import numpy as np
from matplotlib.path import Path

from brainfusion.fusion.match_contours import interpolate_contour
from brainfusion.fusion.transform_2Dmap import create_rbf_interpolators
from brainfusion.io import import_analysis, write_dict_in_h5

TO_METRES = {"m": 1.0, "um": 1e-6}


def _read_sections(path):
    """({section: [(x, y), ...]}, header {key: value}) of an annotation text file."""
    sections, header, points = {}, {}, None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                header[key.strip()] = value.strip()
            elif line.startswith("[") and line.endswith("]"):
                points = sections.setdefault(line[1:-1], [])
            elif points is not None:
                points.append([float(v) for v in line.split()])
    return {name: np.array(p, dtype=float) for name, p in sections.items()}, header


def read_regions_file(path, scale=1.0):
    """({name: polygon (N, 2)} in the order of the file, header {key: value}) of a regions file; the point repeated at the
    end is dropped. `scale` multiplies the coordinates (e.g. 1e-6: um -> m)."""
    sections, header = _read_sections(path)
    regions = {}
    for name, polygon in sections.items():
        polygon = polygon * scale
        regions[name] = polygon[:-1] if len(polygon) > 3 and np.array_equal(polygon[0], polygon[-1]) else polygon
    return regions, header


def _apply_matrix(matrix, points):
    return (np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix, dtype=float).T)[:, :2]


def _densify(polygon, n=400):
    """The polygon as `n` points equally spaced along its length."""
    polygon = np.asarray(polygon, dtype=float)
    closed = polygon if np.array_equal(polygon[0], polygon[-1]) else np.vstack([polygon, polygon[0]])
    return interpolate_contour(closed, n + 1)[:-1]


def _warp(analysis, index, smooth='auto'):
    """The warp the fusion made for sample `index`: aligned sample frame -> template, (N, 2) -> (N, 2)."""
    fx, fy = create_rbf_interpolators(np.asarray(analysis['measurement_contours'][index], dtype=float),
                                      np.asarray(analysis['template_contours'][index], dtype=float), smooth=smooth)
    return lambda p: np.column_stack([fx(p[:, 0], p[:, 1]), fy(p[:, 0], p[:, 1])])


def _on_average(drawn_on):
    return isinstance(drawn_on, str) and drawn_on == 'average'


def _sample_index(analysis, drawn_on):
    """Position in the fusion of the sample `drawn_on` (an index or a unique file name)."""
    names = list(analysis['measurement_filenames'])
    if isinstance(drawn_on, (int, np.integer)):
        if not 0 <= drawn_on < len(names):
            raise ValueError(f"drawn_on = {drawn_on}: the fusion has {len(names)} samples")
        return int(drawn_on)
    hits = [i for i, name in enumerate(names) if name == drawn_on]
    if len(hits) != 1:
        raise ValueError(f"drawn_on = {drawn_on!r}: {'no sample' if not hits else 'several samples'} of that name in the fusion "
                         f"(use its position)")
    return hits[0]


def _preimage_outline(mask, xs, ys, smoothing=1.0):
    """The outline (N, 2) of the largest connected part of the boolean image `mask` (rows = ys, columns = xs), holes filled,
    smoothed over about `smoothing` pixels; None if the mask is empty."""
    import contourpy
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return None
    largest = ndimage.binary_fill_holes(labels == 1 + np.argmax(ndimage.sum(mask, labels, range(1, count + 1))))
    field = np.pad(ndimage.gaussian_filter(largest.astype(float), smoothing), 1)   # the pad closes outlines at the image edge
    step_x, step_y = xs[1] - xs[0], ys[1] - ys[0]
    lines = contourpy.contour_generator(np.r_[xs[0] - step_x, xs, xs[-1] + step_x], np.r_[ys[0] - step_y, ys, ys[-1] + step_y],
                                        field).lines(0.5)
    return max(lines, key=len)[:-1] if lines else None


def pull_back_regions(analysis, regions, drawn_on='average', resolution=600, smooth='auto'):
    """
    Regions drawn on one outline of a fusion, onto every sample of the fusion.

    `analysis`: a fusion result (`brain_fusion`, or `import_analysis` of its .h5). `regions`: {name: polygon (N, 2)} in the
    units of the outline they were drawn on (`drawn_on`: 'average' = the template, or a sample by position / file name; that
    sample's own frame). Drawn on a sample, they first go onto the template through that sample's own warp.

    The fusion warped each sample into the template by the RBF from the sample's matched contour to the matched template
    contour (`create_rbf_interpolators`; `smooth` as the fusion used it). A region in a sample is what that warp sends into the
    region: the points of the sample's outline that land inside the region, on a raster of `resolution` steps across the
    sample, the largest connected part with its holes filled, as an outline in the sample's own frame (through its affine
    matrix). That is exactly the part of the sample that the fusion counts to the region - also where the warp folds and has no
    inverse (a region point with several sources in the sample, or none).

    Returns ({name: polygon on the template}, [one {name: polygon in the sample's own frame} per sample]).
    """
    matrices = analysis['affine_matrices'][1:]   # the first matrix belongs to the template
    regions = {name: _densify(p) for name, p in regions.items()}
    if not _on_average(drawn_on):
        index = _sample_index(analysis, drawn_on)
        warp = _warp(analysis, index, smooth)
        back = np.linalg.inv(np.asarray(matrices[index], dtype=float))   # own frame -> aligned frame
        regions = {name: warp(_apply_matrix(back, p)) for name, p in regions.items()}
    per_sample = []
    for index, matrix in enumerate(matrices):
        measured = np.asarray(analysis['measurement_contours'][index], dtype=float)
        warp = _warp(analysis, index, smooth)
        lo, hi = measured.min(axis=0), measured.max(axis=0)
        step = (hi - lo).max() / resolution
        xs, ys = np.arange(lo[0] - 2 * step, hi[0] + 2 * step, step), np.arange(lo[1] - 2 * step, hi[1] + 2 * step, step)
        grid = np.column_stack([a.ravel() for a in np.meshgrid(xs, ys)])
        inside = Path(measured).contains_points(grid)
        warped = warp(grid[inside])
        outlines = {}
        for name, polygon in regions.items():
            hit = np.zeros(len(grid), dtype=bool)
            hit[inside] = Path(polygon).contains_points(warped)
            outline = _preimage_outline(hit.reshape(len(ys), len(xs)), xs, ys)
            if outline is not None:
                outlines[name] = _apply_matrix(matrix, outline)
        per_sample.append(outlines)
    return regions, per_sample


def atlas_pullback(fusion_file, regions_file, drawn_on='average', resolution=600, plots_folder=None):
    """
    Pull the regions of `regions_file` back onto every sample of the fusion in `fusion_file` (see the module text for all it does).

    `drawn_on`: the outline the regions were drawn on - 'average' (the fusion's template, default), or a sample of the fusion
    (its position in `measurement_filenames`, or its unique file name; its own frame).

    Stores 'regions' in the fusion's .h5 file (replacing earlier ones) and plots them into the fusion's folder. Returns the
    analysis with 'regions'. `plots_folder`: where the region plots go (default: the fusion's folder).
    """
    import h5py

    analysis, _ = import_analysis(str(fusion_file))
    regions, header = read_regions_file(regions_file)
    if not regions:
        raise ValueError(f"{regions_file}: no regions in it")
    names = list(analysis['measurement_filenames'])
    unit = analysis['measurement_metadata'][0 if _on_average(drawn_on) else _sample_index(analysis, drawn_on)]['length_unit']
    scale = TO_METRES[header.get('unit', 'um')] / TO_METRES[unit]
    atlas, per_sample = pull_back_regions(analysis, {name: p * scale for name, p in regions.items()}, drawn_on, resolution)
    analysis['regions'] = {'names': list(regions), 'atlas': atlas, 'samples': per_sample,
                           'drawn_on': 'average' if _on_average(drawn_on) else names[_sample_index(analysis, drawn_on)]}
    with h5py.File(str(fusion_file), 'r+') as h5file:
        if 'regions' in h5file:
            del h5file['regions']
        write_dict_in_h5(h5file, '/', {'regions': analysis['regions']})
    plot_regions(analysis, FilePath(plots_folder) if plots_folder else FilePath(fusion_file).parent)
    return analysis


def region_masks(grid, regions):
    """{name: boolean array} - which points of `grid` (N, 2) lie inside each region polygon (regions may overlap)."""
    grid = np.asarray(grid, dtype=float).reshape(-1, 2)
    return {name: Path(np.asarray(polygon, dtype=float)).contains_points(grid) for name, polygon in regions.items()}


def plot_regions(analysis, folder):
    """`Regions_average.png` (the template outline with the regions of `analysis['regions']`) and one picture per sample into
    `folder`, next to its `Transformed_<file name>` plots: `Regions_<file name>.png` and .svg (`Regions_<modality>_<file name>`
    where AFM and Brillouin samples share a name, as in a pooled fusion): its outline in the aligned frame with its regions."""
    import matplotlib.pyplot as plt

    found = analysis['regions']
    colours = {str(name): plt.get_cmap('tab10')(i % 10) for i, name in enumerate(found['names'])}
    folder = FilePath(folder)
    folder.mkdir(parents=True, exist_ok=True)

    def draw(ax, outline, polygons, width):
        ax.plot(*np.vstack([outline, outline[:1]]).T, color='k', lw=width)
        for name, polygon in polygons.items():
            closed = np.vstack([polygon, polygon[:1]])
            ax.fill(*closed.T, color=colours[name], alpha=0.35, lw=0)
            ax.plot(*closed.T, color=colours[name], lw=width, label=name)
        ax.set_aspect('equal')

    for old in list(folder.glob('Regions_*.png')) + list(folder.glob('Regions_*.svg')):   # earlier ones of another order or fusion
        old.unlink()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    draw(ax, np.asarray(analysis['template_contours'][0], dtype=float), found['atlas'], 1.0)
    ax.set_title(f"regions on the average outline (drawn on: {found['drawn_on']})", fontsize=7)
    ax.legend(fontsize=6, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.tick_params(labelsize=6)
    fig.savefig(folder / 'Regions_average.png', dpi=250, bbox_inches='tight')
    plt.close(fig)

    names =[str(n) for n in analysis['measurement_filenames']]
    for index, polygons in enumerate(found['samples']):
        inverse = np.linalg.inv(np.asarray(analysis['affine_matrices'][index + 1], dtype=float))   # own frame -> aligned frame
        modality = str(analysis['measurement_metadata'][index].get('modality_condition', '')).split('_')[0]
        label = names[index] if names.count(names[index]) == 1 or not modality else f"{modality}_{names[index]}"   # AFM and Brillouin share names
        fig, ax = plt.subplots(figsize=(2.6, 2.8))
        draw(ax, np.asarray(analysis['measurement_contours'][index], dtype=float),
             {name: _apply_matrix(inverse, polygon) for name, polygon in polygons.items()}, 0.8)
        ax.set_title(label, fontsize=6)
        ax.legend(fontsize=5, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
        ax.axis('off')
        for extension in ('png', 'svg'):
            fig.savefig(folder / f"Regions_{label}.{extension}", dpi=200, bbox_inches='tight')
        plt.close(fig)
