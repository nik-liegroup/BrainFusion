import numpy as np
import matplotlib.path as mpath


def mask_contour(contour: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask indicating which points in `grid` lie inside the polygon defined by `contour` (inclusive).

    Parameters
    ----------
    contour : np.ndarray of shape (N, 2)
        Array of polygon vertices as (x, y) points.
    grid : np.ndarray of shape (M, 2)
        Array of points to test as (x, y) coordinates.

    Returns
    -------
    mask : np.ndarray of shape (M,)
        Boolean array where True indicates the corresponding point in `grid` is inside the polygon.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError(f"Expected contour shape (N, 2), got {contour.shape}")
    if grid.ndim != 2 or grid.shape[1] != 2:
        raise ValueError(f"Expected grid shape (M, 2), got {grid.shape}")

    # Create a path object from the contour
    path = mpath.Path(contour)

    # Compute bounding box diagonal length of contour
    bbox_min = contour.min(axis=0)
    bbox_max = contour.max(axis=0)
    bbox_diag = np.linalg.norm(bbox_max - bbox_min)

    # Set inclusion radius as a small fraction of bbox diagonal
    radius = bbox_diag * 1e-6

    # Find which points are inside the contour
    mask = path.contains_points(grid, radius=radius)

    return mask


def regular_grid_on_bbox(contour: np.ndarray, axis_points: int = 50) -> np.ndarray:
    """
    Generate a regular grid of points covering the bounding box of a contour.

    Parameters
    ----------
    contour : np.ndarray of shape (N, 2)
        Array of polygon vertices as (x, y) points.
    axis_points : int, optional
        Number of grid points along each axis (default is 50).

    Returns
    -------
    grid : np.ndarray of shape (axis_points * axis_points, 2)
        Array of grid points (x, y) covering the bounding box of the contour.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("Contour must be of shape (N, 2)")

    # Get min and max x, y coordinates
    min_x, min_y = np.min(contour, axis=0)
    max_x, max_y = np.max(contour, axis=0)

    # Generate linearly spaced grid points
    x_values = np.linspace(min_x, max_x, axis_points)
    y_values = np.linspace(min_y, max_y, axis_points)

    # Create meshgrid
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    return grid


def bin_2D_image(img: np.ndarray, bin_size: int, method: str = 'mean', crop: bool = True) -> np.ndarray:
    """
    Bin a 2D image into larger pixels by aggregating blocks of size (bin_size x bin_size).

    Parameters
    ----------
    img : np.ndarray of shape (n, m)
        Input 2D image array.
    bin_size : int
        Size of the bin along each axis.
    method : str, optional
        Aggregation method: 'mean', 'sum', or 'max'. Default is 'mean'.
    crop : bool, optional
        If True, crop the image to make it divisible by bin_size. If False, pad with zeros. Default is True.

    Returns
    -------
    binned_img : np.ndarray
        Binned image of shape (n//bin_size, m//bin_size) if crop=True, or padded version accordingly.
    """
    if bin_size <= 0:
        raise ValueError("bin_size must be a positive integer")

    if img.ndim != 2:
        raise ValueError("img must be a 2D array (single image channel)")

    n, m = img.shape

    if crop:
        new_n = n - (n % bin_size)
        new_m = m - (m % bin_size)
        img = img[:new_n, :new_m]
    else:
        # Pad to make divisible
        pad_n = (bin_size - n % bin_size) % bin_size
        pad_m = (bin_size - m % bin_size) % bin_size
        img = np.pad(img, ((0, pad_n), (0, pad_m)), mode='constant')

    # Bin the image
    n_bins_n = img.shape[0] // bin_size
    n_bins_m = img.shape[1] // bin_size

    reshaped = img.reshape(n_bins_n, bin_size, n_bins_m, bin_size)

    if method == 'mean':
        return reshaped.mean(axis=(1, 3))
    elif method == 'sum':
        return reshaped.sum(axis=(1, 3))
    elif method == 'max':
        return reshaped.max(axis=(1, 3))
    else:
        raise ValueError("Invalid method: choose 'mean', 'sum', or 'max'")


def bin_outline(outline: np.ndarray, bin_size: int, crop: bool = False, original_shape: tuple[int, int] = None) -> np.ndarray:
    """
    Transform outline coordinates to match the binned image coordinate system.

    Parameters
    ----------
    outline : np.ndarray of shape (n, 2)
        Array of (x, y) coordinates outlining a region.
    bin_size : int
        Binning factor (how many pixels per binned pixel).
    crop : bool, optional
        Whether to crop the outline points to the cropped image size before scaling. Default is False.
    original_shape : tuple[int, int], optional
        The original (height, width) shape of the image before cropping. Required if crop=True.

    Returns
    -------
    np.ndarray
        Transformed outline coordinates scaled down by the bin size.
    """
    outline = np.asarray(outline, dtype=float)
    if outline.ndim != 2 or outline.shape[1] != 2:
        raise ValueError("outline must be an (n, 2) array of coordinates")
    bin_size = int(bin_size)
    if bin_size <= 0:
        raise ValueError("bin_size must be a positive integer")

    if crop:
        if original_shape is None:
            raise ValueError("original_shape must be provided when crop=True")
        n, m = original_shape
        new_n = n - (n % bin_size)
        new_m = m - (m % bin_size)
        # Filter out points that are outside the cropped region
        keep = (outline[:, 0] < new_m) & (outline[:, 1] < new_n)
        outline = outline[keep]

    # Scale down coordinates
    outline = (outline - 0.5) / bin_size + 0.5
    return outline
