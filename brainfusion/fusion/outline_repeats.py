"""
Outlines drawn several times on the same image (a folder such as
'<outline folder>/outline_repeats/' with 'outline_brainfusion_repeat<n>.txt'): their average and the drawing error -
computed like the fusion's average outline (`average_outline`), with the same settings (`contour_interp_n`,
`outline_averaging` of the fusion kwargs). Nothing is written: the loaders average the folder when they load it.

`average_outline` resamples the outlines (`interpolate_contour`), aligns them (`align_contours`: the fit, then
`circularly_shift_contours` for direction and starting point) and averages them (`find_average_contour`). The repeats
of one animal are all drawn in the same pixel frame, so they skip the fit: a shift, rotation or size difference between
them is drawing error, which the fit would remove. Only `circularly_shift_contours` brings them to corresponding points.

The drawing error per point is the distance of each point of the average outline to the corresponding point of every
outline, averaged over the outlines; its mean is the drawing error of the animal. `find_average_contour`'s own error
(Frechet distance of every outline to the average, as in the fusion) is returned too. `outline_spread` gives the same
numbers for the outlines of several animals around the fusion's average outline, after the fusion's alignment.
"""
import glob
import os

import numpy as np

from brainfusion.fusion.average_contours import find_average_contour
from brainfusion.fusion.core import _close_contour
from brainfusion.fusion.match_contours import align_contours, circularly_shift_contours, interpolate_contour
from brainfusion.io import read_annotation_file


def _settings(fusion_kwargs):
    """Number of outline points and averaging method of the fusion (the defaults of `brain_fusion` otherwise)."""
    fusion_kwargs = fusion_kwargs or {}
    return fusion_kwargs.get('contour_interp_n', 200), fusion_kwargs.get('outline_averaging', 'star_domain')


def _average_and_errors(corresponding, n_points, averaging):
    """`find_average_contour` of corresponding outlines (all n_points long, same start and direction), centred like
    the fusion's aligned outlines (one common shift for all, undone afterwards), and the distances of every average
    point to the corresponding point of each outline."""
    centre = corresponding[0].mean(axis=0)
    average, frechet = find_average_contour([c - centre for c in corresponding], average=averaging,
                                            star_bins=n_points, error_metric='frechet')
    # find_average_contour closes the average (last point = first): its distinct points are all but the last
    open_average = (average + centre)[:-1]
    distances = np.array([np.linalg.norm(np.asarray(c)[:len(open_average)] - open_average, axis=1)
                          for c in corresponding])
    return open_average, frechet, distances


def average_repeated_outlines(contours, fusion_kwargs=None):
    """
    Average and drawing error of outlines drawn in the same frame (see the module docstring). `fusion_kwargs`: the
    fusion's settings (`contour_interp_n`, `outline_averaging`).

    Returns a dict: 'outline' (n, 2) the average (open); 'corresponding' (n_outlines, n, 2); 'error_per_point' (n,) and
    'error' (scalar) in the units of the coordinates; 'error_per_outline' (mean distance of each outline to the
    average); 'frechet' (find_average_contour's error of each outline).
    """
    if len(contours) < 2:
        raise ValueError('At least two outlines are needed for an average and a drawing error.')
    n_points, averaging = _settings(fusion_kwargs)
    resampled = [interpolate_contour(_close_contour(np.asarray(c, dtype=float)), n_points) for c in contours]
    corresponding = circularly_shift_contours(resampled, template_contour=resampled[0],
                                              init_points=[None] * len(resampled))
    average, frechet, distances = _average_and_errors(corresponding, n_points, averaging)
    return {'outline': average, 'corresponding': np.array(corresponding),
            'error_per_point': distances.mean(axis=0), 'error': float(distances.mean()),
            'error_per_outline': distances.mean(axis=1), 'frechet': frechet}


def outline_spread(samples, fusion_kwargs=None):
    """
    The spread of the outlines of several animals around their average after the fusion's alignment - the same numbers
    as the drawing error, between animals: `average_outline`'s steps (with `fit_routine`, `landmark_fit` and the
    outline settings of `fusion_kwargs`), then the distance of each average point to the corresponding point of every
    animal. Returns a dict like `average_repeated_outlines` ('error_per_point', 'error_per_outline' = per animal, ...),
    in the units of the samples' outlines.
    """
    fusion_kwargs = fusion_kwargs or {}
    n_points, averaging = _settings(fusion_kwargs)
    contours = [interpolate_contour(_close_contour(s.contour), n_points) for s in samples]
    aligned, _, _ = align_contours(contours, [s.grid for s in samples], landmarks_list=[s.landmarks for s in samples],
                                   template_index=0, fit_routine=fusion_kwargs.get('fit_routine', 'ellipse'),
                                   landmark_fit=fusion_kwargs.get('landmark_fit', 'similarity'),
                                   canonical_axes=fusion_kwargs.get('canonical_axes', True))
    average, frechet, distances = _average_and_errors(aligned, n_points, averaging)
    return {'outline': average, 'corresponding': np.array(aligned),
            'error_per_point': distances.mean(axis=0), 'error': float(distances.mean()),
            'error_per_outline': distances.mean(axis=1), 'frechet': frechet}


def average_outline_repeats(repeat_dir, fusion_kwargs=None):
    """
    Averages the outlines in a folder of repeated outlines (every '*.txt' with an outline; all must name the same
    image and mirroring). Landmarks are averaged (median) if every repeat has the same number of them.

    Returns the dict of `average_repeated_outlines` (in pixels) plus 'files', 'header' (that of the first repeat),
    'pixels_per_um' (None if unknown), 'error_um' and 'landmarks'.
    """
    files = sorted(glob.glob(os.path.join(repeat_dir, '*.txt')))
    annotations = [(f, read_annotation_file(f)) for f in files]
    annotations = [(f, a) for f, a in annotations if a is not None and a['outline'] is not None]
    if len(annotations) < 2:
        raise ValueError(f"{repeat_dir}: {len(annotations)} outline(s), at least two are needed for an average")
    for key in ('image', 'mirror_horizontal', 'mirror_vertical'):
        values = {a['header'].get(key) for _, a in annotations}
        if len(values) != 1:
            raise ValueError(f"{repeat_dir}: the repeated outlines differ in '{key}' ({sorted(map(str, values))}); "
                             f"they must share one pixel frame to be averaged without alignment")
    result = average_repeated_outlines([a['outline'] for _, a in annotations], fusion_kwargs)
    header = dict(annotations[0][1]['header'])
    try:
        pixels_per_um = float(header.get('pixels_per_um') or 'nan')
    except ValueError:
        pixels_per_um = float('nan')
    pixels_per_um = pixels_per_um if np.isfinite(pixels_per_um) and pixels_per_um > 0 else None
    landmarks = [a['landmarks'] for _, a in annotations]
    result.update(files=[f for f, _ in annotations], header=header, pixels_per_um=pixels_per_um,
                  error_um=result['error'] / pixels_per_um if pixels_per_um else None,
                  landmarks=(np.median(np.array(landmarks), axis=0) if all(l is not None for l in landmarks)
                             and len({len(l) for l in landmarks}) == 1 else None))
    return result


def read_outline(folder, outline, fusion_kwargs=None):
    """
    The outline of an experiment for the loaders. `outline` is a name (or path) relative to `folder`, the folder the
    annotation files are in (AFM 'Pics/calibration', Brillouin 'Plots'):
      ending in '.txt'  that single outline file, e.g. 'outline_brainfusion.txt'
      otherwise         a folder of repeated outlines, e.g. 'outline_repeats', averaged (`average_outline_repeats`
                        with the outline settings of `fusion_kwargs`); a folder with a single outline gives that
                        outline as it is (nothing to average, no drawing error)
    Returns the dict of `brainfusion.io.read_annotation_file` ('header', 'outline', 'landmarks') plus 'path' and
    'image_dir' (the folder the image named in the header is relative to: the folder of the annotation files for both kinds).
    Raises FileNotFoundError if there is no such file or folder, ValueError if it has no outline.
    """
    path = os.path.join(folder, outline)
    if outline.lower().endswith('.txt'):
        annotation = read_annotation_file(path)
        if annotation is None:
            raise FileNotFoundError(f"No annotation file at {path}")
        if annotation['outline'] is None:
            raise ValueError(f"No [outline] in {path}")
        return {**annotation, 'path': path, 'image_dir': os.path.dirname(path)}
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No folder of repeated outlines at {path}")
    image_dir = os.path.dirname(os.path.normpath(path))
    drawn = [a for a in (read_annotation_file(f) for f in sorted(glob.glob(os.path.join(path, '*.txt'))))
             if a is not None and a['outline'] is not None]
    if not drawn:
        raise ValueError(f"No outline in {path}")
    if len(drawn) == 1:
        return {**drawn[0], 'path': path, 'image_dir': image_dir}
    average = average_outline_repeats(path, fusion_kwargs)
    return {'header': average['header'], 'outline': average['outline'], 'landmarks': average['landmarks'],
            'path': path, 'image_dir': image_dir}
