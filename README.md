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

A measurement is a `Sample`: an outline (`contour`), the measurement positions (`grid`), the values at the positions (`dataset`, a dict of arrays),
optional landmarks and metadata. Loaders build samples from AFM (`load_batchforce_single`) and Brillouin (`load_brillouin_experiment`) experiments;
outlines and landmarks come from annotation files (`outline_brainfusion.txt`), and `attach_metadata` adds fields such as the animal number or the
condition (`parse_name` reads them from folder names). Brillouin samples can be loaded in the frame of the overview image (`image_frame=True`, y up),
which is what they need to be fused with AFM samples.

```python
from brainfusion import (load_batchforce_single, load_brillouin_experiment, run_fusion, plot_average_map, animal_maps,
                         map_consistency, bootstrap_map_correlation, correlate_group_maps, spatial_null_test, save_correlation_results)

samples = [load_brillouin_experiment(folder, brillouin_variables=["brillouin_shift_f_stokes_anti_stokes"], image_frame=True)
           for folder in experiment_folders]
analysis = run_fusion(samples, dict(contour_template="average", outline_averaging="median", clustering=["Median", "GMM"],
                                    group_field="condition"), results_path="fusion.h5")   # cached; overwrite=True recomputes
plot_average_map(analysis, "brillouin_shift_f_stokes_anti_stokes", method="Median", output_path="Median_Averaged_Map.png")
```

### Fusion
`brain_fusion` (and `run_fusion`, which adds the cache in an `.h5` file) aligns the outlines affinely (ellipse fit, or landmarks if the samples have
them), matches them with DTW, warps the grids onto the template outline with radial basis functions and averages.
- **Template** (`contour_template`): `"average"` (average outline of all samples), `"first_element"`, or an outline you give (an `(N, 2)` array or a
  `Sample`, e.g. with landmarks): every sample is aligned and warped onto it and the template takes no part in the averages. `average_outline(samples)`
  makes the average outline of a set of samples, so several fusions (one per modality, say) can share one outline.
- **Clustering** (`clustering`, one name or a list; all use the same alignment and warp): `"Mean"`, `"Median"`, `"Sum"` resample every sample onto one
  shared regular grid and reduce per grid point; `"GMM"` pools the warped points of all samples, clusters them with a Gaussian mixture and takes the
  median of a component, on a grid of its own (the components). Every method stores grid, values, number of samples per point, spread and standard error in
  `analysis["averaged"][method]`.
- **Groups** (`group_field`): additionally average per value of a metadata field (e.g. condition). Every method is applied per group,
  `analysis["group_averaged"][method][group]`; every group lives on the same grid (for GMM: the components of one clustering of all samples, a group's value is the median of its own points in a component).
  Maps on different grids (separate fusions) are compared by resampling (`resample_points`: Gaussian kernel, radius or nearest neighbour).
- **Pooling everything** in one fusion (e.g. all AFM and all Brillouin animals, the modality being the group) gives one outline, one warp and, for the
  pooled average, one grid; per-modality fusions with a shared template (`average_outline`) share the outline only.

### Correlating maps
`correlate_on_shared_grid`, `correlate_group_maps` (two groups of one fusion, any clustering, resampling where the grids differ),
`pairwise_correlate_groups`, `correlate_around_reference_grid` (radius averaging), `plot_norm_corr`, `plot_correlation_masks`.

### Statistics (`brainfusion.stats`)
Statistics are a separate step on top of a finished fusion; they never change it and are saved in their own file (`save_correlation_results`,
`correlation.h5`), not in `fusion.h5`.
- **Consistency of the animals**: `animal_maps` (the maps of the samples on the shared grid), `pairwise_map_correlations`, `split_half_reliability`
  (Spearman-Brown corrected reproducibility of an average map), `map_consistency`.
- **Uncertainty of a correlation of two average maps**: `bootstrap_map_correlation` (animals resampled, maps averaged again; interval of r),
  `correlation_vs_animals`, `bootstrap_map_scalar` (interval of the median of an average map).
- **Spatial null model**: `spatial_surrogates`, `spatial_null_test` (surrogate maps with the same spatial autocorrelation; neighbouring grid points are
  not independent, so the p-value of a plain correlation test is far too small).
- **Summaries and corrections**: `summarise_correlations` (Fisher z mean and interval over animals), `disattenuate` (r corrected for the noise of the maps).
- **Conditions**: `permutation_scalar_difference` (one number per animal: difference of medians with bootstrap interval, permutation p-value, Cliff's
  delta), `permutation_map_difference` (difference map with Welch t per point and p-values corrected for all points by the maximum statistic).
- **Association of high values**: `analyse_correlation_percentile`, `conditional_probability_table` (Fisher exact test on thresholded points).

## Functionality

    - Load AFM and Brillouin datasets (outline, landmarks, mirroring, quality filter, frame of the overview image).
    - Average outlines and warp measurements onto a template outline (average, first sample or given).
    - Average with Mean, Median, Sum or a Gaussian mixture, overall and per group.
    - Correlate maps of different modalities, also from different grids, and plot them.
    - Statistics for correlations and condition differences on top of a fusion.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or suggestions.


## License
This project is licensed under the MIT License - see the LICENSE file for details.

**Author:** Niklas Gampl\
**Email:** niklas.gampl@mpzpm.mpg.de
