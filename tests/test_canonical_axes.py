"""Canonical orientation of the ellipse alignment: outlines in a similar orientation in the image (within 90 degrees) end with
the major axis along y and the same end up."""
import numpy as np

from brainfusion.fusion.match_contours import align_contours, interpolate_contour


def egg(angle, shift=(0.0, 0.0), scale=1.0, flip=False):
    """An egg-shaped outline (narrower at the top, long axis along y), turned by `angle` degrees."""
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    y = 300 * np.sin(t)
    x = 150 * np.cos(t) * (1 - 0.3 * np.sin(t))
    points = np.column_stack([x, y])
    if flip:
        points = -points
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    points = scale * points @ rotation.T + shift
    return np.vstack([points, points[:1]])


def width_at(contour, top):
    y = contour[:, 1]
    selected = contour[y > np.percentile(y, 80)] if top else contour[y < np.percentile(y, 20)]
    return np.ptp(selected[:, 0])


def test_every_outline_ends_with_the_major_axis_along_y_and_the_same_end_up():
    angles = [0, 35, -40, 70, -65, 20]
    contours = [interpolate_contour(egg(a, shift=(40 * i, -20 * i), scale=1 + 0.1 * i, flip=False), 200) for i, a in enumerate(angles)]
    aligned, _, _ = align_contours(contours, [None] * len(contours), template_index=0, fit_routine='ellipse', canonical_axes=True)
    for contour in aligned:
        assert np.ptp(contour[:, 1]) > 1.5 * np.ptp(contour[:, 0])                 # major axis along y
    narrow_end_up = [width_at(c, top=True) < width_at(c, top=False) for c in aligned]
    assert len(set(narrow_end_up)) == 1                                            # the same end up in all
    first = aligned[0]
    for contour in aligned[1:]:
        assert np.abs(np.sort(contour[:, 1])[[0, -1]] - np.sort(first[:, 1])[[0, -1]]).max() < 30   # same extent as the first


def test_the_canonical_template_is_the_same_whatever_the_first_outlines_angle():
    templates = []
    for angle in (0, 40, -60):
        contours = [interpolate_contour(egg(angle), 200), interpolate_contour(egg(angle + 40, scale=1.1), 200)]
        aligned, _, _ = align_contours(contours, [None, None], template_index=0, fit_routine='ellipse', canonical_axes=True)
        templates.append(aligned[0])
    for other in templates[1:]:
        assert np.abs(other - templates[0]).max() < 10        # the same outline in the same orientation (points may start elsewhere)
