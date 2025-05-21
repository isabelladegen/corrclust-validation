# CSTS - Correlation Structures in Time Series
[![CC BY 4.0][cc-by-shield]][cc-by]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isabelladegen/corrclust-validation/blob/main/src/utils/hf_tooling/CSTS_HuggingFace_UsageExample.ipynb)

## Overview

This repository contains the code for generating, validating, and evaluating the CSTS (Correlation Structures in Time Series) benchmark dataset. CSTS is a comprehensive synthetic benchmark for evaluating the discovery of correlation structures in multivariate time series data.

Key features of CSTS:
- Synthetic time series data with 23 distinct correlation structures
- Systematic variation of data conditions (distribution shifts, sparsification, downsampling)
- Ground truth segmentation and clustering labels
- Controlled degraded clustering results for validation method evaluation
- Extensible data generation framework

**For quick access to the dataset without setting up this repository, use our [Hugging Face dataset](https://huggingface.co/datasets/idegen/csts) and [Google Colab notebook](https://colab.research.google.com/github/isabelladegen/corrclust-validation/blob/main/src/utils/hf_tooling/CSTS_HuggingFace_UsageExample.ipynb).**

## Repository Contents

This repository provides:
1. **Data Generation**: Code for generating synthetic datasets with known correlation structures
2. **Data Validation**: Tools for validating the preservation of correlation structures
3. **Evaluation Framework**: Methods for assessing clustering algorithms and validation indices
4. **Case Study Implementation**: Code for reproducing the TICC algorithm evaluation

## Directory Structure

```
corrclust-validation/
├── csts/                                  # Data directory (not in git, see clone HF data)
│   ├── exploratory/                       # Training/exploration subjects
│   │   ├── irregular_p30/                 # Partial data variants (70% observations)
│   │   ├── irregular_p90/                 # Sparse data variant (10% observations)
│   │   ├── raw/                           # Complete raw data variant
│   │   ├── normal/                        # Complete correlated data variant
│   │   ├── non_normal/                    # Complete non-normal data variant
│   │   └── downsampled_1min/              # Complete downsampled data variant
│   └── confirmatory/                      # Validation data
│       └── ...                            # Same structure as exploratory
├── src/                    
│   ├── data_generation/                   # Synthetic data generation
│   ├── evaluation/                        # Evaluation Methods
│   ├── experiments/                       # Run scripts for expriments
│   ├── use_case/                          # TICC example case study
│   ├── visualisation/                     # Results visualisation
│   └── utils/                             # Helper functions
├── tests/                                 # Unit and Integration Tests
├── conda-exact.yml                        # Conda environment exact versions for CSTS
├── private-yaml-template.yml              # Template file to create private.yaml for WANDB config
└── README.md                              # This file
```

## Getting Started

### Accessing the Data
The complete dataset is available on [Hugging Face](https://huggingface.co/datasets/idegen/csts). This is the easiest way to access the data if you just want to evaluate your algorithms.

```python
from datasets import load_dataset

# Load data for the exploratory split, complete correlated variant
data = load_dataset("idegen/csts", name="correlated_complete_data", split="exploratory")

# Load corresponding ground truth labels
labels = load_dataset("idegen/csts", name="correlated_complete_labels", split="exploratory")
```

### Use this Repository (For Evaluation/Development/Extension)

#### 1. Clone the Code Repository
```bash
# Clone via SSH
git clone git@github.com:isabelladegen/corrclust-validation.git
cd corrclust-validation
```

#### 2. Clone the Hugging Face Data
```bash
# Make sure Git LFS is installed
git lfs install

# Clone the dataset into the corrclust-validation directory
git clone https://huggingface.co/datasets/idegen/csts
```

#### 3. Create conda environment
```bash
conda env create -f conda-exact.yml
conda activate corr-24
```
## Key Applications

This codebase supports several research applications:

1. **Evaluating Clustering Algorithms**: Test how well algorithms discover correlation structures across data variants, see [Algorithm Evaluation Guide](docs/algorithm_evaluation.md)
2. **Assessing Validation Methods**: Evaluate internal and external validation indices for correlation-based clustering
3. **Analyzing Preprocessing Effects**: Investigate how techniques like downsampling affect correlation structures
4. **Extending the Benchmark**: Generate custom data variants with different properties, see [Data Generation Guide](docs/data_generation.md)

## Citation

If you use this code, the CSTS dataset or our benchmark findings in your research, please cite our paper accordingly. 
This is the arXiv preprint version that describes the benchmark, check back for updates:

```bibtex
@misc{degen2025csts,
      title={CSTS: A Benchmark for the Discovery of Correlation Structures in Time Series Clustering}, 
      author={Isabella Degen and Zahraa S Abdallah and Henry W J Reeve and Kate Robson Brown},
      year={2025},
      eprint={2505.14596},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.14596}, 
}
```

If you use our validated validation method or findings please cite our (soon to be published) paper:

```bibtex
@misc{degen2025,
  author       = {Degen, I and Abdallah, Z S and Robson Brown, K and Reeve, H W J},
  title        = {Validating Clustering Validation: An Empirical Evaluation for Time Series Correlation Structure Discovery},
  year         = {2025},
  note         = {forthcoming},
}
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: https://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://licensebuttons.net/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg