# Parameters for AFM brain tissue stiffness maps
afm_params = {
    "overwrite_analysis": True,  # Whether to overwrite a pre-existing .h5 analysis file
    "template": "average",  # "first_element": Use first element in imported lists as template; "average": Calculate average contour and use as template
    "afm_variables": ['modulus', 'beta_pyforce'],  # Choose AFM variables to analyse
    "data_filename": "data.csv",  # Filename of AFM batchforce analysis files
    "key_point_filename": "None",  # Choose .txt file containing a common starting point on the outline; can be "None"
    "rot_axis_filename": "None",  # Choose .txt file containing two points along which the contours will be aligned (rotation); can be "None"
    "grid_conv_filename": "GridInversionMatrix.csv",  # Name of file containing grid conversion variables
    "boundary_filename": "brain_outline",  # Choose the .txt files containing the outline coordinates
    "sampling_size": 100,  # Randomly sample given number of points from myelin dataset; should be used for testing only and otherwise "None",
    "contour_interp_n": 200,  # Choose the number of points the contours will be interpolated to
    "clustering": "None"  # "GMM": Use Gaussian Mixture Model to cluster data points and create average map; "None": Use regular grid interpolation and average
}

# Parameters for HCR brain tissue mRNA expression maps
hcr_params = {
    "overwrite_analysis": True,  # Whether to overwrite a pre-existing .h5 analysis file
    "template": "average",  # "first_element": Use first element in imported lists as template; "average": Calculate average contour and use as template
    "hcr_variables": ['Channel_2', 'Channel_3'],  # Choose image channels to analyse
    "data_filename": "data.csv",  # Filename of AFM batchforce analysis files
    "key_point_filename": "None",  # Choose .txt file containing a common starting point on the outline; can be "None"
    "rot_axis_filename": "midline",  # Choose .txt file containing two points along which the contours will be aligned (rotation); can be "None"
    "grid_conv_filename": "None",  # Name of file containing grid conversion variables
    "boundary_filename": "brain_outline",  # Choose the .txt files containing the outline coordinates
    "sampling_size": 100,  # Randomly sample given number of points from myelin dataset; should be used for testing only and otherwise "None",
    "contour_interp_n": 200,  # Choose the number of points the contours will be interpolated to
    "clustering": "None"  # "GMM": Use Gaussian Mixture Model to cluster data points and create average map; "None": Use regular grid interpolation and average
}
