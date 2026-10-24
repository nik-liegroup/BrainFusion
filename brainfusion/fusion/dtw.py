import numpy as np
import scipy.ndimage as ndi
from typing import Tuple
from collections import defaultdict

# The matching does not depend on the unit of the outlines: they are put into units in which a circle of their length has radius 1
# (`segmented_contour_dtw`), `dtw_curvature` is the weight of the curvature against the distance, and a stretch is flat if its curvature is
# below FLAT_THRESH times the mean curvature of the outline. (Before, a squared length was added to a length, the curvature was
# regularised with a fixed 1e-10 and a fixed curvature of 0.001 per unit length marked a flat stretch: AFM outlines in metres were
# matched by their curvature almost alone, Brillouin outlines in micrometres by their distance alone.)
DTW_CURVATURE = 0.3   # tested on 31 AFM and 17 Brillouin outlines: 0.3 gives the least folds in the warp with moderate offsets
FLAT_THRESH = 0.15


def segmented_contour_dtw(contour1: np.ndarray, contour2: np.ndarray,
                          dtw_curvature: float = DTW_CURVATURE, smoothing: float = 0.05,
                          flat_thresh: float = FLAT_THRESH, chord_arc_thresh: float = 0.99,
                          min_frac: float = 0.05, ) -> Tuple[np.ndarray, np.ndarray]:
    """
    `_segmented_contour_dtw` of two outlines put into units of their own size (a circle as long as the outlines has radius 1), so the
    result is the same in metres and in micrometres; `flat_thresh` is a curvature in units of the mean curvature.
    """
    contour1, contour2 = np.asarray(contour1, dtype=float), np.asarray(contour2, dtype=float)
    length = sum(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)) for c in (contour1, contour2))
    unit = length / (4 * np.pi)   # the radius of a circle as long as the outlines
    out1, out2 = _segmented_contour_dtw(contour1 / unit, contour2 / unit, dtw_curvature, smoothing, flat_thresh, chord_arc_thresh, min_frac)
    return out1 * unit, out2 * unit


def _segmented_contour_dtw(contour1: np.ndarray, contour2: np.ndarray,
                           dtw_curvature: float, smoothing: float, flat_thresh: float, chord_arc_thresh: float,
                           min_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform piecewise alignment of two closed 2D contours using straight segments as cut points.

    Segmentation is based on curvature thresholding + chord/arc length ratio.
    Each pair of consecutive cut points defines a compartment along the contour.

    - If both contours are straight between the same cut points:
        Arc-length-based alignment is performed.
    - If either contour is curved between the cut points:
        Dynamic Time Warping (DTW) with a curvature-based penalty is used.
    """
    def compute_contour_length(contour):
        return np.sum(np.linalg.norm(np.diff(contour, axis=0), axis=1))

    def gaussian_smooth_contour(contour, sigma):
        smoothed_x = ndi.gaussian_filter1d(contour[:, 0], sigma, mode='wrap')
        smoothed_y = ndi.gaussian_filter1d(contour[:, 1], sigma, mode='wrap')
        return np.column_stack((smoothed_x, smoothed_y))

    def compute_css_curvature(contour, sigma, eps=1e-10):
        smoothed = gaussian_smooth_contour(contour, sigma)

        def circ_1st_gradient(c): return 0.5 * (np.roll(c, -1) - np.roll(c, 1))

        def circ_2nd_gradient(c): return np.roll(c, -1) - 2.0 * c + np.roll(c, 1)

        dx, dy = circ_1st_gradient(smoothed[:, 0]), circ_1st_gradient(smoothed[:, 1])
        ddx, ddy = circ_2nd_gradient(smoothed[:, 0]), circ_2nd_gradient(smoothed[:, 1])
        return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2 + eps) ** 1.5

    def seg_indices(start, end, N):
        return np.arange(start, end + 1) if start <= end else np.r_[np.arange(start, N), np.arange(0, end + 1)]

    # Calculate curvature
    n1, n2 = len(contour1), len(contour2)
    sigma1 = smoothing * (compute_contour_length(contour1) / max(n1, 1))
    sigma2 = smoothing * (compute_contour_length(contour2) / max(n2, 1))
    curv1 = compute_css_curvature(contour1, sigma1)
    curv2 = compute_css_curvature(contour2, sigma2)

    # Detect straight segments
    straight_mask1, seg_info1, _ = detect_linear_segments_from_curvature(
        contour1, curv1, flat_thresh=flat_thresh, min_frac=min_frac, chord_arc_thresh=chord_arc_thresh
    )
    straight_mask2, seg_info2, _ = detect_linear_segments_from_curvature(
        contour2, curv2, flat_thresh=flat_thresh, min_frac=min_frac, chord_arc_thresh=chord_arc_thresh
    )

    # Cut points: the ends of straight stretches that both outlines have at the same place
    cuts1, cuts2 = _paired_cuts(contour1, contour2, seg_info1, seg_info2, seg_indices)

    # --- If not enough segments, fall back to DTW ---
    num_segs = len(cuts1)
    if num_segs < 2:
        return dtw_with_curvature_penalty(contour1, contour2, curv1, curv2, dtw_curvature)

    # --- Align segments ---
    out1, out2 = [], []
    for k in range(num_segs):
        start1, end1 = cuts1[k], cuts1[(k + 1) % num_segs]
        start2, end2 = cuts2[k], cuts2[(k + 1) % num_segs]

        seg_idx1 = seg_indices(start1, end1, n1)
        seg_idx2 = seg_indices(start2, end2, n2)

        seg1 = contour1[seg_idx1]
        seg2 = contour2[seg_idx2]
        seg_curv1 = curv1[seg_idx1]
        seg_curv2 = curv2[seg_idx2]

        s1_straight = any(s["straight"] and s["start"] == start1 and s["end"] == end1 for s in seg_info1)
        s2_straight = any(s["straight"] and s["start"] == start2 and s["end"] == end2 for s in seg_info2)

        if s1_straight and s2_straight:
            seg_out1, seg_out2 = align_straight_segments(seg1, seg2)
        else:
            seg_out1, seg_out2 = dtw_with_curvature_penalty(seg1, seg2, seg_curv1, seg_curv2, dtw_curvature)

        if k > 0:
            seg_out1 = seg_out1[1:]
            seg_out2 = seg_out2[1:]
        out1.append(seg_out1)
        out2.append(seg_out2)

    return np.vstack(out1), np.vstack(out2)


MAX_STRAIGHT_OFFSET = 0.3   # straight stretches of two outlines correspond if their middles are closer than this (units: radius of a circle as long as the outlines)


def _paired_cuts(contour1, contour2, seg_info1, seg_info2, seg_indices):
    """
    The cut points (indices into contour1 and contour2, in the same order) at the ends of the straight stretches the two outlines
    have in common: a stretch of one outline is paired with the stretch of the other whose middle is nearest if that is also the
    nearest the other way round and closer than MAX_STRAIGHT_OFFSET, and the pairs must keep their order around the outlines.
    (Pairing the n-th stretch of one outline with the n-th of the other matched different parts when the outlines had different numbers
    of straight stretches.) Returns two empty lists if there is no consistent pairing.
    """
    straight1 = [s for s in seg_info1 if s["straight"]]
    straight2 = [s for s in seg_info2 if s["straight"]]
    if not straight1 or not straight2:
        return [], []

    def middle(contour, s):
        idx = seg_indices(s["start"], s["end"], len(contour))
        return contour[idx[len(idx) // 2]]

    mid1 = np.array([middle(contour1, s) for s in straight1])
    mid2 = np.array([middle(contour2, s) for s in straight2])
    distance = np.linalg.norm(mid1[:, None, :] - mid2[None, :, :], axis=2)
    pairs = [(i, int(np.argmin(distance[i]))) for i in range(len(straight1))]
    pairs = [(i, j) for i, j in pairs if int(np.argmin(distance[:, j])) == i and distance[i, j] < MAX_STRAIGHT_OFFSET]
    if not pairs:
        return [], []
    cuts = sorted((straight1[i][end], straight2[j][end]) for i, j in pairs for end in ("start", "end"))
    cuts1, cuts2 = [c[0] for c in cuts], [c[1] for c in cuts]
    descents = sum(cuts2[k] > cuts2[(k + 1) % len(cuts2)] for k in range(len(cuts2)))
    if len(set(cuts1)) < len(cuts1) or len(set(cuts2)) < len(cuts2) or descents != 1:
        return [], []
    return cuts1, cuts2


def detect_linear_segments_from_curvature(contour, curvature,
                                          flat_thresh=0.001,
                                          min_frac=0.05,
                                          chord_arc_thresh=0.99):
    N = len(contour)
    min_points = max(3, int(np.ceil(min_frac * N)))

    flat_mask = np.abs(curvature) < flat_thresh
    seg_mask = np.zeros(N, dtype=bool)
    seg_ids = np.full(N, -1, dtype=int)
    seg_info = []
    seg_id_counter = 0

    visited = np.zeros(N, dtype=bool)
    for i in range(N):
        if flat_mask[i] and not visited[i]:
            idx_range = []
            j = i
            while flat_mask[j] and not visited[j]:
                idx_range.append(j)
                visited[j] = True
                j = (j + 1) % N
                if j == i:  # wrapped around full loop
                    break

            if len(idx_range) >= min_points:
                pts = contour[idx_range]

                # Arc/chord ratio test
                arc_len = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
                chord_len = np.linalg.norm(pts[0] - pts[-1])
                straightness_ratio = chord_len / arc_len  # ≤ 1
                straight = straightness_ratio >= chord_arc_thresh

                if straight:
                    seg_mask[idx_range] = True
                    seg_ids[idx_range] = seg_id_counter
                    seg_info.append({
                        "start": idx_range[0],
                        "end": idx_range[-1],
                        "straight": True
                    })
                    seg_id_counter += 1

    return seg_mask, seg_info, seg_ids


def dtw_with_curvature_penalty(contour1: np.ndarray, contour2: np.ndarray,
                               curvatures1: np.ndarray, curvatures2: np.ndarray,
                               dtw_curvature: float = DTW_CURVATURE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two 2D contours using Dynamic Time Warping (DTW) with a curvature-based penalty.

    This function computes a soft point-to-point alignment between `contour1` and `contour2` using DTW.
    The cost function includes both spatial distance and a curvature mismatch penalty.

    Parameters
    ----------
    contour1 : np.ndarray of shape (N, 2)
        First contour to be aligned (reference).
    contour2 : np.ndarray of shape (M, 2)
        Second contour to be matched against.
    curvatures1 : np.ndarray of shape (N,)
        Local curvature of reference contour.
    curvatures2 : np.ndarray of shape (M,)
        Local curvature of second contour
    dtw_curvature : float, optional (keyword-only)
        Weighting factor for curvature mismatch penalty in the DTW cost function.

    Returns
    -------
    warped_contour1 : np.ndarray of shape (K, 2)
        Subsampled or repeated version of `contour1` that participated in the alignment.
    warped_contour2 : np.ndarray of shape (K, 2)
        Aligned and averaged version of `contour2` matched to `warped_contour1`.

    Notes
    -----
    - The resulting aligned points may contain duplicates or be averaged where `contour2` maps multiple points to the same location.
    """
    if not isinstance(contour1, np.ndarray) or contour1.ndim != 2 or contour1.shape[1] != 2:
        raise ValueError("contour1 must be a (N, 2) numpy array")
    if not isinstance(contour2, np.ndarray) or contour2.ndim != 2 or contour2.shape[1] != 2:
        raise ValueError("contour2 must be a (N, 2) numpy array")
    if not isinstance(curvatures1, np.ndarray) or curvatures1.ndim != 1 or curvatures1.shape[0] != len(contour1):
        raise ValueError(f"curvature1 must be a ({len(contour1)},) numpy array")
    if not isinstance(curvatures2, np.ndarray) or curvatures2.ndim != 1 or curvatures2.shape[0] != len(contour2):
        raise ValueError(f"curvature2 must be a ({len(contour2)},) numpy array")

    # Cost matrix: squared distance of the points and weighted curvature difference, each relative to its median over all pairs
    n, m = len(contour1), len(contour2)
    spatial = np.linalg.norm(contour1[:, None, :] - contour2[None, :, :], axis=2)
    curvature_diff = np.abs(curvatures1[:, None] - curvatures2[None, :])
    cost_matrix = (spatial / (np.median(spatial) + 1e-300)) ** 2 + dtw_curvature * curvature_diff / (np.median(curvature_diff) + 1e-11)

    # Initialize DP matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Compute DTW
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],  # Insertion
                                          dtw_matrix[i, j - 1],  # Deletion
                                          dtw_matrix[i - 1, j - 1])  # Match

    # Backtrack to find the optimal path
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1  # Move diagonally
            j -= 1

    path.reverse()

    idx1, idx2 = np.array(list(zip(*path)))

    # Dictionary to store values corresponding to idx1
    contour2_mapping = defaultdict(list)

    # Store values of contour2 based on alignment
    for i1, i2 in zip(idx1, idx2):
        contour2_mapping[i1].append(contour2[i2])  # Store actual contour2 values, not indices

    # Compute averaged values
    idx1_unique = np.array(list(contour2_mapping.keys()))
    contour2_averaged = np.array(
        [np.mean(values, axis=0) for values in contour2_mapping.values()])  # Averaging the points

    return contour1[idx1_unique], contour2_averaged

def align_straight_segments(seg1, seg2):
    def interp_along_arc(src_pts, t_targets):
        ds = np.r_[0.0, np.linalg.norm(np.diff(src_pts, axis=0), axis=1)]
        s = np.cumsum(ds)
        if s[-1] == 0:
            return np.repeat(src_pts[:1], len(t_targets), axis=0)
        u = s / s[-1]
        idx = np.searchsorted(u, t_targets, side='right')
        idx = np.clip(idx, 1, len(u) - 1)
        u0, u1 = u[idx - 1], u[idx]
        w = (t_targets - u0) / np.maximum(u1 - u0, 1e-12)
        p0, p1 = src_pts[idx - 1], src_pts[idx]
        return (1 - w)[:, None] * p0 + w[:, None] * p1

    t_uniform = np.linspace(0, 1, max(len(seg1), len(seg2)))
    return interp_along_arc(seg1, t_uniform), interp_along_arc(seg2, t_uniform)
