# brainfusion/fusion/__init__.py
#
# The alignment/warping/averaging math: contour matching (match_contours, average_contours, dtw), grid
# warping (transform_2Dmap, interpolation), and the top-level orchestration (core) that ties them together
# via a Sample list. Self-contained - nothing outside this subpackage is needed to run it besides the shared
# `Sample` type, `brainfusion.utils`, and `brainfusion.io` (for `run_fusion`'s cache step).

from .core import brain_fusion, run_fusion, fuse_boundaries, fuse_grids, fuse_measurement_datasets
from .grouping import (list_groups, extract_groups, group_average_on_shared_grid, extract_group_native_data,
                       merge_keys)
