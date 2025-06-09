# BrainFusion: Average tissue boundaries to create global contour and merge measurement points to new shape.

Did you ever wonder, how an average frog brain looks like?
Consider five embryonic brains from different animals, no brain is exactly like the other and this of course also applies to its shape.
Using this Python package, a global tissue shape can be calculated by averaging the outlines of individual brains.
Given imaging data (e.g. local mRNA expression) or gridded spatial maps (e.g. local tissue stiffness) for the single brains, these can additionally be transformed to the new global shape.
Therefore, paths enclosing the shapes are aligned using DTW boundary matching with an adjustable curvature penalty.
The transformation between contours is modelled via radial basis functions and applied to the grid coordinates within the contour.

This concept is of course applicable to all different sorts of tissues and enables the extraction of general patterns found in the measurement data.
Measurement data can for example be averaged after interpolation to a common regular grid or using a Gaussian Mixture Model to find clusters.

This makes spatial cross-correlations of any arbitrary measurement signal easily possible!

[![codecov](https://codecov.io/gh/nik-liegroup/BrainFusion/branch/main/graph/badge.svg)](https://codecov.io/gh/nik-liegroup/BrainFusion)

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Functionality](#functionality)
4. [Contributing](#contributing)
5. [License](#license)

## Installation
1. Clone the Repository:
   ```bash
   cd <repository-directory>
   git clone https://github.com/nik-liegroup/BrainFusion
   ```
   
2. Create Virtual Conda Environment and Activate
   ```bash
   conda create --name brainfusion-env python=3.12
   conda activate brainfusion-env
   ```
   
3. Install Brainfusion Package with Dependencies:
    ```bash
   cd <repository-directory>\BrainFusion
   pip install .
   ```

## Usage
To analyze AFM and Brillouin data, modify the parameters in the script. You can specify the base folder containing your data, the results folder for saving analysis, and any specific parameters needed for the experiments. For other datasets, you need to create your own function.

### Example
#### Define parameters for Brillouin data analysis
```bash
brillouin_params = {
    "experiment": 'brillouin',
    "load_experiment_func": load_brillouin_experiment,
    "base_folder": 'path/to/brillouin/data',
    "results_folder": 'path/to/brillouin/results',
    "raw_data_key": 'brillouin_shift_f_proj',
    "label": 'Brillouin shift (GHz)',
    "cmap": 'viridis',
    "marker_size": 35
}
```

#### Process the experiment
```
brillouin_analysis = process_experiment(**brillouin_params)
```

![ContourDomainTrafo](https://github.com/user-attachments/assets/78436b3f-f4a1-4016-81ab-77465ef5f1e7)

## Functionality

    - Load AFM and Brillouin datasets.
    - Calculate average contours and corresponding grids.
    - Transform 3D/2D maps to median contours.
    - Plot and visualize results, including correlation maps between AFM and Brillouin data.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or suggestions.


## License
This project is licensed under the MIT License - see the LICENSE file for details.

**Author:** Niklas Gampl\
**Email:** niklas.gampl@mpzpm.mpg.de
