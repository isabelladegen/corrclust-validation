Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

# corrclust-validation

Code for evaluating distance measures and internal validation indices for correlation-based clustering. Includes synthetic dataset generation and validation required for this evaluation.

## Overview

This repository provides:
1. Code for generating synthetic datasets with known correlation patterns and validation of the generation
2. Code for evaluating distance measures and internal validation indices for correlation-based clustering
3. Tools for assessing robustness to common data quality variations

## Directory Structure

```
corrclust-validation/
├── data/                                       # Not in git - add yourself
│   ├── synthetic-data/                         # Available on Zenodo
│   │    ├── normal/                            # 30 ground truth datasets
│   │    │   ├── wandb-run-name-1-data.csv      # ts data for ds 1
│   │    │   ├── wandb-run-name-1-labels.csv    # ground truth labels for ds 1
│   │    │   ├── wandb-run-name-2-data.csv      # ts data for ds 2
│   │    │   ├── wandb-run-name-2-labels.csv    # ground truth labels for ds 2
│   │    │   └── ...
│   │    │   └── bad_partitions/                # 66 bad partitions for each of the 30 datasets
│   │    │   │    └── wandb-run-name-1-wrong-cluster-1-labels.csv # just the labels file as that defines the segmentation and clustering
│   │    │   │    └── wandb-run-name-1-wrong-cluster-2-labels.csv
│   │    │   │    └── ...
│   │    ├── non_normal/                        # Distribution-shifted versions
│   │    ├── irregular_p30/                     # Irregular 30% randomly dropped observations, non-normal  
│   │    ├── irregular_p90/                     # Irregular 90% randomly dropped observations, non-normal
│   │    ├── downsampled/                       # 1-min intervals downsampled and aggregated, non-normal
│   │    ├── min_max_scaled/                    # min-max scaled 0-10
│   │    └── wandb-30ds-generation-summary.csv  # Dataset generation parameters
├── src/                    
│   ├── data_generation/                        # Synthetic data generation
│   ├── evaluation/                             # Core evaluation methodology
│   ├── visualisation/                          # Results visualisation
│   └── utils/                                  # Helper functions
├── notebooks/                                  # Usage examples & analysis
├── tests/                                      # Unit tests
├── conda.yml                                   # Conda environment specification
└── README.md                                   # This file
```

## Dataset Generation & Tracking

We use Weights & Biases (wandb) to track the normal and non-normal generation of the data. Each dataset gets a unique wandb run name (e.g. "delicate-forest-42") and complete parameter logging. You can:

- View all generation runs: [wandb project link]
- See exact parameters used for each dataset
- Track data quality metrics
- Compare dataset variations

## Validation Datasets on Zenodo

The generated datasets are available on Zenodo [DOI link] organized by data quality variation:
- `normal/`: Original normally distributed data
- `non_normal/`: Distribution-shifted data
- `irregular_p30/`: 30% randomly dropped observations
- `irregular_p90/`: 90% randomly dropped observations
- `downsampled/`: Downsampled to 1-minute intervals
- `min_max_scaled/`: Min/Max scaled 0-10

Each variation contains 30 datasets with unique wandb run identifiers for full reproducibility.

## Getting Started

1. Clone this repository
2. Create conda environment
3. Download datasets from Zenodo [DOI link]
4. Extract maintaining the folder structure shown above
5. See notebooks for usage examples of the tools

## Environment Setup

The code uses a conda environment specified in conda.yml. To set up:

```
conda env create -f conda.yml
conda activate corrclust-validation
```

## Using With Your Own Data

The evaluation methodology accepts any dataset that:
1. Contains multivariate time series data in the same format
2. A lables file for the segmentation and clustering in the same format

To use your own data you need to:
1. Add your data as a new folder under data following the structure of the synthetic data
2. Extend the loading code to be able to deal with your folder names (unless you use exactly the same folder name

## Citation

If you use this code or the validation datasets, please cite:
[Paper citation]
