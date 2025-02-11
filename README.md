# AFM and Brillouin Data Analysis

Python package for transforming measurement grids (AFM, Brillouin, in situ HCR, ...) defined on individual embryonic brains to a common coordinate system and averaging the corresponding datasets. Maps can be cross-correlated using a Gaussian Mixture Model to cluster and average regions with high point densities.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Functionality](#functionality)
4. [Contributing](#contributing)
5. [License](#license)

## Installation
1. Clone the repository:
   ```bash
   cd <repository-directory>
   git clone https://github.com/nik-liegroup/BrainFusion
   ```

2. Install required packages:
    ```bash
   pip install -r requirements.txt
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
