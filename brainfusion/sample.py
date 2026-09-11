"""Common data container passed between loaders and the fusion pipeline.

A `Sample` bundles everything that belongs to one measurement (grid, data, contour, ...) so that
these pieces of information can never drift out of sync with each other, which used to happen when
they were passed around as several separately-indexed lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

__all__ = ["Sample", "replace"]


@dataclass
class Sample:
    contour: np.ndarray
    grid: Optional[np.ndarray] = None
    dataset: Optional[dict] = None
    scale: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None  # (K, 2) points, matched by position against the template's own
    affine: Optional[np.ndarray] = None
    grid_shape: Optional[np.ndarray] = None
    filename: str = ""
    bg_image: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)  # Free-form info parsed from the folder/file name, e.g. condition
