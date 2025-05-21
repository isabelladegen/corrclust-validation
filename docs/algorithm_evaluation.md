# Algorithm Evaluation Guide

This section provides a step-by-step guide for evaluating time series clustering algorithms using the CSTS benchmark, 
following the recommended evaluation protocol from our paper. By following this structure, you can systematically 
evaluate your time series clustering algorithm using our benchmark and compare your results directly with our findings.

## 1. Selecting Data Variants

Choose which data variants to evaluate your algorithm against. This is a combination of what we call in the code
`data_type` and `completeness`. `DataCompleteness.irregular_p30` is the partial completeness level with 70% of observations, 
`DataCompleteness.irregular_p0` is the sparse completeness levels with 10% of the observation. From case study:
```python
from src.utils.configurations import  DataCompleteness
from src.utils.load_synthetic_data import SyntheticDataType

 # run on normal and non-normal data variant
dataset_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated]
# use exploratory data: run for all three completeness level
completeness_levels = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]
```

Load all the exploratory subject names:
```python
import pandas as pd
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH

run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
```

Load the data and labels dataframes for a subject and a data variant:
```python
from src.utils.load_synthetic_data import load_synthetic_data
data_df, gt_labels_df = load_synthetic_data(subject_name, data_type, data_dir)
```

Complete details how to get to data dir see [Ticc run](https://github.com/isabelladegen/corrclust-validation/blob/main/src/use_case/wandb_run_ticc.py)

## 2. Generating Clustering Results

Train and apply your clustering algorithm as you usually would. See example in [TICC run](https://github.com/isabelladegen/corrclust-validation/blob/main/src/use_case/wandb_run_ticc.py).

Translate the results into a labels format this codebase understands which means there needs to be a row
for each segment and each segment has the following columns, see `to_labels_df()` in [TICC result](https://github.com/isabelladegen/corrclust-validation/blob/main/src/use_case/ticc_result.py):
```python
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols

SyntheticDataSegmentCols.segment_id
SyntheticDataSegmentCols.start_idx
SyntheticDataSegmentCols.end_idx
SyntheticDataSegmentCols.length
SyntheticDataSegmentCols.pattern_id
SyntheticDataSegmentCols.actual_correlation
```

## 3. Calculating Evaluation Measures

To map an algorithm's clusters to ground truth and calculate metrics you can use our class [algorithm_evaluation.py](https://github.com/isabelladegen/corrclust-validation/blob/main/src/use_case/algorithm_evaluation.py)
```python
from src.use_case.algorithm_evaluation import AlgorithmEvaluation

evaluate = AlgorithmEvaluation(result_labels_df, gt_labels_df, data_df, subject_name, data_dir, data_type)
evaluate.silhouette_score()
evaluate.dbi()
evaluate.jaccard_index()
evaluate.pattern_discovery_percentage()
evaluate.pattern_specificity_percentage()
evaluate.segmentation_ratio()
evaluate.segmentation_length_ratio()
evaluate.pattern_not_discovered() # list of patterns in gt that the algorithm missed
# MAE between achieved patterns and their mapped ground truth relaxed pattern
evaluate.mae_stats_mapped_resulting_patterns_relaxed() # pandas describe df, access values using e.g. ['mean']
# MAE between the mapped ground truth patterns and their relaxed target pattern, for reference
evaluate.mae_stats_mapped_gt_patterns_relaxed() # pandas describe df, access values using e.g. ['mean']
```

## 4. Results Interpretation

Contextualise your results using our benchmark reference values provide in our paper: [CSTS: A Benchmark for the Discovery of Correlation Structures in Time Series Clustering](https://arxiv.org/html/2505.14596v1).


## 5. Statistical Validation

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
