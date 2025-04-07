from itertools import chain
from os import path

import pandas as pd
from scipy.stats import spearmanr

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.evaluation.describe_subjects_for_data_variant import DescribeSubjectsForDataVariant
from src.utils.configurations import ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, CONFIRMATORY_DATASETS_FILE_PATH, \
    DataCompleteness, SYNTHETIC_DATA_DIR, CONFIRMATORY_SYNTHETIC_DATA_DIR, get_data_dir, \
    get_clustering_quality_multiple_data_variants_result_folder, ResultsType
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    # calculate correlations and descriptive statistics to show that the two datasets are independent but
    # equivalent
    root_result_dir = ROOT_RESULTS_DIR
    exploratory_ds = GENERATED_DATASETS_FILE_PATH
    exp_data_dir = SYNTHETIC_DATA_DIR
    confirmatory_ds = CONFIRMATORY_DATASETS_FILE_PATH
    conf_data_dir = CONFIRMATORY_SYNTHETIC_DATA_DIR

    overall_ds_name = "n30"

    normal = SyntheticDataType.normal_correlated

    # load all required data
    # normal complete variants for exploratory and confirmatory
    complete = DataCompleteness.complete
    exp_nc_100 = DescribeSubjectsForDataVariant(wandb_run_file=exploratory_ds, overall_ds_name=overall_ds_name,
                                                data_type=normal, data_dir=get_data_dir(exp_data_dir, complete),
                                                load_data=True)
    conf_nc_100 = DescribeSubjectsForDataVariant(wandb_run_file=confirmatory_ds, overall_ds_name=overall_ds_name,
                                                 data_type=normal, data_dir=get_data_dir(conf_data_dir, complete),
                                                 load_data=True)

    # normal partial variants for exploratory and confirmatory
    partial = DataCompleteness.irregular_p30
    exp_nc_70 = DescribeSubjectsForDataVariant(wandb_run_file=exploratory_ds, overall_ds_name=overall_ds_name,
                                               data_type=normal, data_dir=get_data_dir(exp_data_dir, partial),
                                               load_data=True)
    conf_nc_70 = DescribeSubjectsForDataVariant(wandb_run_file=confirmatory_ds, overall_ds_name=overall_ds_name,
                                                data_type=normal, data_dir=get_data_dir(conf_data_dir, partial),
                                                load_data=True)

    # normal sparse variants for exploratory and confirmatory
    sparse = DataCompleteness.irregular_p90
    exp_nc_10 = DescribeSubjectsForDataVariant(wandb_run_file=exploratory_ds, overall_ds_name=overall_ds_name,
                                               data_type=normal, data_dir=get_data_dir(exp_data_dir, sparse),
                                               load_data=True)
    conf_nc_10 = DescribeSubjectsForDataVariant(wandb_run_file=confirmatory_ds, overall_ds_name=overall_ds_name,
                                                data_type=normal, data_dir=get_data_dir(conf_data_dir, sparse),
                                                load_data=True)

    # run calculations
    # correlation for various measures per segment per individual to show independence (low correlation)
    # MAE
    exp_mae = exp_nc_100.all_mae_values(SyntheticDataSegmentCols.relaxed_mae)
    exp_mae = list(chain.from_iterable(exp_mae))
    conf_mae = conf_nc_100.all_mae_values(SyntheticDataSegmentCols.relaxed_mae)
    conf_mae = list(chain.from_iterable(conf_mae))
    #  The p-value roughly indicates the probability of an uncorrelated system
    # producing datasets that have a Spearman correlation at least as extreme
    # as the one computed from these datasets
    mae_result = spearmanr(a=exp_mae, b=conf_mae, alternative="two-sided")

    # Segment length
    exp_sl = exp_nc_100.all_segment_lengths_values()
    exp_sl = list(chain.from_iterable(exp_sl))
    conf_sl = conf_nc_100.all_segment_lengths_values()
    conf_sl = list(chain.from_iterable(conf_sl))
    sl_result = spearmanr(a=exp_sl, b=conf_sl, alternative="two-sided")

    # Pattern ID
    exp_pid = exp_nc_100.all_pattern_id_values()
    exp_pid = list(chain.from_iterable(exp_pid))
    conf_pid = conf_nc_100.all_pattern_id_values()
    conf_pid = list(chain.from_iterable(conf_pid))
    pid_result = spearmanr(a=exp_pid, b=conf_pid, alternative="two-sided")

    # gap - for partial
    exp_gap70 = exp_nc_70.all_time_gaps_in_seconds()
    conf_gap70 = conf_nc_70.all_time_gaps_in_seconds()
    # we only use the observations that we have (some segments might disappear so these are not equal length)
    min_length = min(len(exp_gap70), len(conf_gap70))
    gap70_result = spearmanr(a=exp_gap70[:min_length], b=conf_gap70[:min_length], alternative="two-sided")

    # gap - for sparse
    exp_gap10 = exp_nc_10.all_time_gaps_in_seconds()
    conf_gap10 = conf_nc_10.all_time_gaps_in_seconds()
    # we only use the observations that we have (some segments might disappear so these are not equal length)
    min_length = min(len(exp_gap10), len(conf_gap10))
    gap10_result = spearmanr(a=exp_gap10[:min_length], b=conf_gap10[:min_length], alternative="two-sided")

    # run statistics
    exp_mae_stats = exp_nc_100.overall_mae_stats(SyntheticDataSegmentCols.relaxed_mae)
    conf_mae_stats = conf_nc_100.overall_mae_stats(SyntheticDataSegmentCols.relaxed_mae)
    exp_sl_stats = exp_nc_100.overall_segment_length_stats()
    conf_sl_stats = conf_nc_100.overall_segment_length_stats()
    exp_pcount_stats = exp_nc_100.overall_pattern_id_count_stats()
    conf_pcount_stats = conf_nc_100.overall_pattern_id_count_stats()
    exp_gap70_stats = exp_nc_70.overall_time_gap_stats()
    conf_gap70_stats = conf_nc_70.overall_time_gap_stats()
    exp_gap10_stats = exp_nc_10.overall_time_gap_stats()
    conf_gap10_stats = conf_nc_10.overall_time_gap_stats()
    results_dict = {
        "Measure": ["relaxed maes", "segment lengths", "pattern ids (counts for stats)", "time gaps (s)",
                    "time gaps (s)"],
        "DataVariant": [exp_nc_100.variant_name, exp_nc_100.variant_name, exp_nc_100.variant_name,
                        exp_nc_70.variant_name, exp_nc_10.variant_name],
        "Spearman r": [round(mae_result.statistic, 3), round(sl_result.statistic, 3), round(pid_result.statistic, 3),
                       round(gap70_result.statistic, 3), round(gap10_result.statistic, 3)],
        "p-value": [round(mae_result.pvalue, 3), round(sl_result.pvalue, 3), round(pid_result.pvalue, 3),
                    round(gap70_result.pvalue, 3), round(gap10_result.pvalue, 3)],
        "Exploratory Mean": [exp_mae_stats["mean"], exp_sl_stats["mean"], exp_pcount_stats["mean"],
                             exp_gap70_stats["mean"], exp_gap10_stats["mean"]],
        "Confirmatory Mean": [conf_mae_stats["mean"], conf_sl_stats["mean"], conf_pcount_stats["mean"],
                              conf_gap70_stats["mean"], conf_gap10_stats["mean"]],
        "Exploratory Median": [exp_mae_stats["50%"], exp_sl_stats["50%"], exp_pcount_stats["50%"],
                               exp_gap70_stats["50%"], exp_gap10_stats["50%"]],
        "Confirmatory Median": [conf_mae_stats["50%"], conf_sl_stats["50%"], conf_pcount_stats["50%"],
                                conf_gap70_stats["50%"], conf_gap10_stats["50%"]],
        "Exploratory 25%": [exp_mae_stats["25%"], exp_sl_stats["25%"], exp_pcount_stats["25%"], exp_gap70_stats["25%"],
                            exp_gap10_stats["25%"]],
        "Confirmatory 25%": [conf_mae_stats["25%"], conf_sl_stats["25%"], conf_pcount_stats["25%"],
                             conf_gap70_stats["25%"], conf_gap10_stats["25%"]],
        "Exploratory 75%": [exp_mae_stats["75%"], exp_sl_stats["75%"], exp_pcount_stats["75%"], exp_gap70_stats["75%"],
                            exp_gap10_stats["75%"]],
        "Confirmatory 75%": [conf_mae_stats["75%"], conf_sl_stats["75%"], conf_pcount_stats["75%"],
                             conf_gap70_stats["75%"], conf_gap10_stats["75%"]],
    }

    # create df
    results_df = pd.DataFrame(results_dict)

    # safe df
    folder_name = get_clustering_quality_multiple_data_variants_result_folder(
        results_type=ResultsType.dataset_description,
        overall_dataset_name=overall_ds_name,
        results_dir=exp_data_dir,
        distance_measure="")
    results_df.to_csv(path.join(folder_name, "dataset_independence_and_equivalence_description.csv"))
