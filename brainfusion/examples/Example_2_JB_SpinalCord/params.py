# Parameters for myelin spinal cord data
myelin_params = {
    "overwrite_analysis": True,  # Whether to overwrite a pre-existing .h5 analysis file
    "template": "first_element",  # "first_element": Use first element in imported lists as template; "average": Calculate average contour and use as template
    "afm_variables": ['modulus', 'beta_pyforce'],  # Choose AFM variables to analyse
    "data_filename": "data.csv",  # Filename of AFM batchforce analysis files
    "key_point_filename": "midline_closing_points",  # Choose .txt file containing a common starting point on the outline; can be "None"
    "rot_axis_filename": "midline_axis",  # Choose .txt file containing two points along which the contours will be aligned (rotation); can be "None"
    "grid_conv_filename": "GridInversionMatrix.csv",  # Name of file containing grid conversion variables
    "boundary_filename": "whitematter_outline",  # Choose the .txt files containing the outline coordinates
    "sampling_size": 100,  # Randomly sample given number of points from myelin dataset; should be used for testing only and otherwise "None",
    "contour_interp_n": 500,  # Choose the number of points the contours will be interpolated to
    "clustering": "None"  # "GMM": Use Gaussian Mixture Model to cluster data points and create average map; "None": Use regular grid interpolation and average
}
