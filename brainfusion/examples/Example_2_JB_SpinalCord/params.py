# Parameters for AFM spinal cord data
afm_params = {
    "overwrite_analysis": True,  # Whether to overwrite a pre-existing .h5 analysis file
    "boundary_filename": "whitematter_outline",  # Choose the .txt files containing the outline coordinates
    "key_point_filename": "midline_closing_points",  # Choose .txt file containing a common starting point on the outline; can be None
    "rot_axis_filename": "midline_axis",  # Choose .txt file containing two points along which the contours will be aligned (rotation); can be None
    "contour_interp_n": 200,  # Choose the number of points the contours will be interpolated to
    "sampling_size": 100  # Randomly sample given number of points from myelin dataset; should be used for testing only and otherwise None
}
