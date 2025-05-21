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

1. **Evaluating Clustering Algorithms**: Test how well algorithms discover correlation structures across data variants
2. **Assessing Validation Methods**: Evaluate internal and external validation indices for correlation-based clustering
3. **Analyzing Preprocessing Effects**: Investigate how techniques like downsampling affect correlation structures
4. **Extending the Benchmark**: Generate custom data variants with different properties

## Algorithm Evaluation Guide

This section provides a step-by-step guide for evaluating time series clustering algorithms using the CSTS benchmark, 
following the recommended evaluation protocol from our paper. By following this structure, you can systematically 
evaluate your time series clustering algorithm using our benchmark and compare your results directly with our findings.

### 1. Selecting Data Variants

Choose which data variants to evaluate your algorithm against. This is a combination of what we call in the code
`data_type` and `completeness`. `DataCompleteness.irregular_p30` is the partial completeness level with 70% of observations, 
`DataCompleteness.irregular_p0` is the sparse completeness levels with 10% of the observation. From case study:
```python
 # run on normal and non-normal data variant
dataset_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated]
# use exploratory data: run for all three completeness level
completeness_levels = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]
```

Load all the exploratory subject names:
```python
run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
```

Load the data and labels dataframes for a subject and a data variant:
```python
data_df, gt_labels_df = load_synthetic_data(subject_name, data_type, data_dir)
```

Complete details how to get to data dir see [Ticc run](https://github.com/isabelladegen/corrclust-validation/blob/main/src/use_case/wandb_run_ticc.py)

### 2. Generating Clustering Results

Train and apply your clustering algorithm as you usually would. See example in [TICC run](https://github.com/isabelladegen/corrclust-validation/blob/main/src/use_case/wandb_run_ticc.py).

Translate the results into a labels format this codebase understands which means there needs to be a row
for each segment and each segment has the following columns, see `to_labels_df()` in [TICC result](https://github.com/isabelladegen/corrclust-validation/blob/main/src/use_case/ticc_result.py):
```python
SyntheticDataSegmentCols.segment_id
SyntheticDataSegmentCols.start_idx
SyntheticDataSegmentCols.end_idx
SyntheticDataSegmentCols.length
SyntheticDataSegmentCols.pattern_id
SyntheticDataSegmentCols.actual_correlation
```

### 3. Calculating Evaluation Measures

To map an algorithm's clusters to ground truth and calculate metrics you can use our class [algorithm_evaluation.py](https://github.com/isabelladegen/corrclust-validation/blob/main/src/use_case/algorithm_evaluation.py)
```python
evaluate = AlgorithmEvaluation(result_labels_df, gt_labels_df, data_df, subject_name, data_dir, data_type)
evaluate.silhouette_score()
evaluate.dbi()
evaluate.jaccard_index()
evaluate.pattern_discovery_percentage()
evaluate.pattern_specificity_percentage()
evaluate.segmentation_ratio()
evaluate.segmentation_length_ratio()
evaluate.pattern_not_discovered() # list of patterns in gt that the algorithm missed
evaluate.mae_stats_mapped_resulting_patterns_relaxed() # pandas describe df, access values using e.g. ['mean']
evaluate.mae_stats_mapped_gt_patterns_relaxed() # pandas describe df, access values using e.g. ['mean']
```

### 4. Results Interpretation

Contextualise your results using our benchmark reference values provide in our paper: [CSTS: A Benchmark for the Discovery of Correlation Structures in Time Series Clustering](https://arxiv.org/html/2505.14596v1).


### 5. Statistical Validation

Use Wilcoxon signed rank test to evaluate if the difference e.g. between performance measure for the complete and partial
variants are statistically significant. Ensure you pair the values on variant type and subject.
```python
from src.utils.stats import calculate_wilcox_signed_rank
values_complete_normal = [] # list or numpy array of results for a measure for the complete normal data variant
values_complete_partial = [] # list or numpy array of results for a measure for the partial normal data variant

alternative = "two-sided" # see scipy scipy.stats.wilcoxon for valid alternatives
alpha = 0.05 # set unadjusted alpha level
bonferroni_adjust = 3 # put to number of comparisons you make or do your own multiplicity adjustment

# handy wrapper ensuring differences are only used if > than non_zero
wilcox_result = calculate_wilcox_signed_rank(values_complete_normal, values_complete_partial, non_zero=1e-8, 
                                           alternative=alternative)

# wilcox_result is an instance of WilcoxResult class providing convenience functions
p = wilcox_result.p_value 
is_sig = wilcox_result.is_significant(alpha=alpha, bonferroni_adjust=bonferroni_adjust)
es = wilcox_result.effect_size(alternative) # calculated as r = ± Z / √N, N are the none zero pairs
# the power achieved for the given effect size and adjusted alpha
power = wilcox_result.achieved_power(alpha=alpha, bonferroni_adjust=bonferroni_adjust, alternative=alternative)
# how many samples you need for the achieved p value and effect size to reach a power of 80%
n_for_80_power = wilcox_result.sample_size_for_power(target_power=0.8, alpha=alpha, bonferroni_adjust=bonferroni_adjust)

```


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