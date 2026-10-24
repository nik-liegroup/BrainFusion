"""The DTW matching of two outlines must not depend on the unit they are given in."""
import numpy as np

from brainfusion.fusion import dtw
from brainfusion.utils import closed_xy


def outline(radius_x, radius_y, bumps, n=120):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1 + 0.12 * np.sin(bumps * t)
    return np.column_stack([radius_x * r * np.cos(t), radius_y * r * np.sin(t)])


def test_the_match_is_the_same_in_metres_and_micrometres():
    a, b = outline(300.0, 200.0, 3), outline(280.0, 230.0, 4)          # micrometres
    in_um = dtw.segmented_contour_dtw(a, b)
    in_m = dtw.segmented_contour_dtw(a * 1e-6, b * 1e-6)
    for um, m in zip(in_um, in_m):
        np.testing.assert_allclose(m * 1e6, um, rtol=1e-6, atol=1e-6)


def test_the_default_weight_is_the_tested_one():
    assert dtw.DTW_CURVATURE == 0.3
    from brainfusion.fusion.core import fuse_boundaries
    import inspect
    assert inspect.signature(fuse_boundaries).parameters['curvature'].default == 0.3


def test_an_outline_is_drawn_closed():
    x, y = closed_xy(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))
    assert (x[0], y[0]) == (x[-1], y[-1]) and len(x) == 4
    x, y = closed_xy(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]))     # already closed: not closed twice
    assert len(x) == 3


def _seg(start, end):
    return {"start": start, "end": end, "straight": True}


def test_straight_stretches_are_paired_by_place_not_by_number():
    circle = outline(1.0, 1.0, 0, n=200)
    seg_indices = lambda s, e, n: np.arange(s, e + 1)   # noqa: E731
    # the outline has two straight stretches, the other only one, at the place of the first: only that pair is a cut
    cuts1, cuts2 = dtw._paired_cuts(circle, circle, [_seg(10, 30), _seg(100, 120)], [_seg(12, 31)], seg_indices)
    assert (cuts1, cuts2) == ([10, 30], [12, 31])
    # a single stretch of the other outline at the place of the second one of this outline pairs with that one
    cuts1, cuts2 = dtw._paired_cuts(circle, circle, [_seg(10, 30), _seg(100, 120)], [_seg(102, 121)], seg_indices)
    assert (cuts1, cuts2) == ([100, 120], [102, 121])
    # stretches at different places do not correspond
    assert dtw._paired_cuts(circle, circle, [_seg(10, 30)], [_seg(100, 120)], seg_indices) == ([], [])

