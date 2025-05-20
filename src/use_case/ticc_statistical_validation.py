from os import path

import pandas as pd

from src.utils.configurations import DataCompleteness, ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, \
    get_algorithm_use_case_result_dir
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.stats import calculate_wilcox_signed_rank, WilcoxResult


def load_ticc_results(ticc_results_dir):
    # read csv
    file_name = path.join(ticc_results_dir, 'ticc_use_case_results.csv')
    df = pd.read_csv(file_name)

    # add run name column
    df['subject_id'] = df['Name'].apply(lambda x: x.split(':')[0])
    return df


def print_stats(wilcox_result: WilcoxResult, alpha:float, bonferroni_adjust:int, alternative:str):
    print(f"p-value: {wilcox_result.p_value}")
    print(f"Effect size: {wilcox_result.effect_size(alternative)}")
    print(
        f"Power: {wilcox_result.achieved_power(alpha=alpha, bonferroni_adjust=bonferroni_adjust, alternative=alternative)}")
    print(
        f"N subject needed for 80% power: {wilcox_result.sample_size_for_power(target_power=0.8, alpha=alpha, bonferroni_adjust=bonferroni_adjust)}")
    print(f"Alpha: {wilcox_result.adjusted_alpha(alpha=alpha, bonferroni_adjust=bonferroni_adjust)}")
    print(f"Alternative: {alternative}")
    print("-----")


if __name__ == "__main__":
    """ 
    Script to run TICC statistical validation
    """
    # run on normal and non-normal data variant
    dataset_types = [SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated]
    # use exploratory data: run for all three completeness level
    completeness_levels = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]

    data_dir = SYNTHETIC_DATA_DIR
    ticc_results_dir = get_algorithm_use_case_result_dir(root_results_dir=ROOT_RESULTS_DIR,
                                                         algorithm_id='ticc-statistical-validation')

    # load results
    results_df = load_ticc_results(ticc_results_dir)

    # test statistical difference in mean MAE between normal and non-normal (pairing subjects by type)
    normal_results = results_df[results_df['data_type'] == SyntheticDataType.normal_correlated]
    # sort by completeness levels then subject
    normal_results = normal_results.sort_values(by=['completeness_level', 'subject_id'])

    # same for non-normal results
    non_normal_results = results_df[results_df['data_type'] == SyntheticDataType.non_normal_correlated]
    non_normal_results = non_normal_results.sort_values(by=['completeness_level', 'subject_id'])

    # statistical evaluation for SWC between normal and non/normal
    measure_name = 'SWC'
    non_zero = 0.00000001  # differences smaller than non-zero are considered equivalent
    alpha = 0.05
    bonferroni_adjust = 3  # Bonferroni corrected for the three tests
    alternative = "two-sided"
    n_vs_nn_result = calculate_wilcox_signed_rank(normal_results[measure_name], non_normal_results[measure_name],
                                                  non_zero, alternative=alternative)
    print("Evaluate statistical difference between normal and non-normal results in SWC:")
    print_stats(n_vs_nn_result, alpha, bonferroni_adjust, alternative)

    # Difference between complete and partial normal
    complete_normal = normal_results[normal_results['completeness_level'].isna()]
    partial_normal = normal_results[normal_results['completeness_level'] == DataCompleteness.irregular_p30]
    complete_vs_partial_result = calculate_wilcox_signed_rank(complete_normal[measure_name],
                                                              partial_normal[measure_name],
                                                              non_zero, alternative=alternative)
    print("Evaluate statistical difference between complete and partial results in SWC for the normal variant:")
    print_stats(complete_vs_partial_result, alpha, bonferroni_adjust, alternative)

    # Difference between complete and sparse normal
    sparse_normal = normal_results[normal_results['completeness_level'] == DataCompleteness.irregular_p90]
    complete_vs_sparse_result = calculate_wilcox_signed_rank(complete_normal[measure_name],
                                                              sparse_normal[measure_name],
                                                              non_zero, alternative=alternative)
    print("Evaluate statistical difference between complete and sparse results in SWC for the normal variant:")
    print_stats(complete_vs_sparse_result, alpha, bonferroni_adjust, alternative)
