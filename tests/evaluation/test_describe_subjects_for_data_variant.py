from os import path

import numpy as np
import pandas as pd
from hamcrest import *
import scipy.stats as stats

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols, CorrType
from src.evaluation.describe_subjects_for_data_variant import DescribeSubjectsForDataVariant, SummaryStatistics, \
    combine_all_ds_variations_multiple_description_summary_dfs, DistParams
from src.utils.configurations import SYNTHETIC_DATA_DIR, GENERATED_DATASETS_FILE_PATH, IRREGULAR_P90_DATA_DIR, \
    dataset_description_dir, MULTIPLE_DS_SUMMARY_FILE, IRREGULAR_P30_DATA_DIR, ROOT_RESULTS_DIR, SyntheticDataVariates
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_ROOT_RESULTS_DIR

run_file = GENERATED_DATASETS_FILE_PATH
data_dir = SYNTHETIC_DATA_DIR
raw = SyntheticDataType.raw
overall_ds_name = "test_stuff"
results_dir = TEST_ROOT_RESULTS_DIR
ds_raw = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=raw,
                                        data_dir=data_dir)
ds_nn = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                       data_type=SyntheticDataType.non_normal_correlated,
                                       data_dir=data_dir, load_data=True)


# these tests read real data, but they save results in a test result folder!
def test_can_load_base_raw_datasets_and_return_ds_variation_mae():
    assert_that(len(ds_raw.run_names), is_(30))  # 30 files
    assert_that(len(ds_raw.label_dfs), is_(30))  # 30 files
    assert_that(ds_raw.label_dfs[ds_raw.run_names[0]].shape[0], is_(100))  # loaded the data properly

    mae_stats = ds_raw.mae_stats(SyntheticDataSegmentCols.mae)
    assert_that(mae_stats["mean"], is_(0.613))
    assert_that(mae_stats["std"], is_(0.006))


def test_calculates_various_stats_on_across_the_datasets():
    column_name = SyntheticDataSegmentCols.mae
    n_outside = ds_raw.n_segments_outside_tolerance_stats()
    assert_that(n_outside["mean"], is_(95.6))
    assert_that(n_outside["std"], is_(0.498))
    assert_that(n_outside["min"], is_(95))

    assert_that(ds_raw.overall_mae_stats(column_name)["mean"], is_(0.613))
    assert_that(ds_raw.overall_mae_stats(column_name)["min"], is_(0.001))
    assert_that(ds_raw.mae_stats(column_name)["min"], is_(0.602))

    assert_that(ds_raw.observations_stats()["mean"], is_(1264010.0))
    assert_that(ds_raw.n_patterns_stats()["mean"], is_(23.0))
    assert_that(ds_raw.segment_length_stats()["mean"], is_(12640.1))
    assert_that(ds_raw.segment_length_stats()["min"], is_(12438.0))  # across ds
    assert_that(ds_raw.overall_segment_length_stats()["mean"], is_(12640.1))
    assert_that(ds_raw.overall_segment_length_stats()["min"], is_(900))  # considering all segment lengths
    assert_that(ds_raw.overall_pattern_id_count_stats()["mean"], is_(4.348))


def test_can_load_base_non_normal_datasets():
    assert_that(len(ds_nn.run_names), is_(30))  # 30 files
    assert_that(len(ds_nn.label_dfs), is_(30))  # 30 files
    assert_that(ds_nn.label_dfs[ds_nn.run_names[0]].shape[0], is_(100))  # loaded the data properly

    mae_stats = ds_nn.mae_stats(SyntheticDataSegmentCols.mae)
    assert_that(mae_stats["mean"], is_(0.116))
    assert_that(mae_stats["min"], is_(0.11))


def test_can_calculate_time_gaps_for_irregular_p90_non_normal_dataset():
    ds_irr_p90nn = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                                  data_type=SyntheticDataType.non_normal_correlated,
                                                  data_dir=IRREGULAR_P90_DATA_DIR,
                                                  load_data=True)

    assert_that(len(ds_irr_p90nn.run_names), is_(30))  # 30 files
    assert_that(len(ds_irr_p90nn.label_dfs), is_(30))  # 30 files
    assert_that(ds_irr_p90nn.label_dfs[ds_irr_p90nn.run_names[0]].shape[0], is_(100))  # loaded the data properly

    mae_stats = ds_irr_p90nn.mae_stats(SyntheticDataSegmentCols.mae)
    assert_that(mae_stats["mean"], is_(0.123))

    time_gaps = ds_irr_p90nn.all_time_gaps_in_seconds()
    assert_that(len(set(time_gaps)), is_(126))

    time_gaps_stats = ds_irr_p90nn.overall_time_gap_stats()
    assert_that(time_gaps_stats["mean"], is_(10.0))
    assert_that(time_gaps_stats["max"], is_(135.0))


def test_creates_summary_df_of_statistics():
    df = ds_raw.summary()

    assert_that(df[SummaryStatistics.mae]["count"], is_(30))
    assert_that(df[SummaryStatistics.overall_mae]["count"], is_(30 * 100))
    assert_that(df[SummaryStatistics.seg_outside_tol]["count"], is_(30))
    assert_that(df[SummaryStatistics.observations]["count"], is_(30))
    assert_that(df[SummaryStatistics.segments]["count"], is_(30))
    assert_that(df[SummaryStatistics.patterns]["count"], is_(30))
    assert_that(df[SummaryStatistics.segment_lengths]["count"], is_(30))
    assert_that(df[SummaryStatistics.overall_segment_lengths]["count"], is_(30 * 100))


def test_saves_summary_df_of_statistics_in_provide_results_root_using_ds_description():
    # Save standard raw
    ds_raw.save_summary(root_results_dir=results_dir)

    # save p30 raw
    ds_p30_raw = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=raw,
                                                data_dir=IRREGULAR_P30_DATA_DIR)
    ds_p30_raw.save_summary(root_results_dir=results_dir)

    # read standard raw
    res_dir = dataset_description_dir(overall_ds_name, raw, results_dir, data_dir)
    file_name = path.join(res_dir, MULTIPLE_DS_SUMMARY_FILE)
    df_raw = pd.read_csv(str(file_name), index_col=0)
    assert_that(df_raw[SummaryStatistics.overall_segment_lengths]["min"], is_(900))

    # read p30 version
    res_dir = dataset_description_dir(overall_ds_name, raw, results_dir, IRREGULAR_P30_DATA_DIR)
    p30_file_name = path.join(res_dir, MULTIPLE_DS_SUMMARY_FILE)
    df_raw_p30 = pd.read_csv(str(p30_file_name), index_col=0)
    assert_that(df_raw_p30[SummaryStatistics.overall_segment_lengths]["min"], is_(592))


def test_combines_all_datasets_into_one_table():
    # given we read from the real result folder we don't want to save the result!
    df = combine_all_ds_variations_multiple_description_summary_dfs(result_root_dir=ROOT_RESULTS_DIR,
                                                                    overall_ds_name="n30",
                                                                    dataset_types=[SyntheticDataType.raw,
                                                                                   SyntheticDataType.normal_correlated,
                                                                                   SyntheticDataType.non_normal_correlated,
                                                                                   SyntheticDataType.rs_1min],
                                                                    data_dirs=[SYNTHETIC_DATA_DIR,
                                                                               IRREGULAR_P30_DATA_DIR,
                                                                               IRREGULAR_P90_DATA_DIR],
                                                                    save_combined_results=False)

    # a few of the datasets
    nn_ds = SyntheticDataType.get_dataset_variation_name(SyntheticDataType.non_normal_correlated, '')
    nn_ds_rs = SyntheticDataType.get_dataset_variation_name(SyntheticDataType.rs_1min, '')
    nn_ds_irr_p30 = SyntheticDataType.get_dataset_variation_name(SyntheticDataType.non_normal_correlated, 'p30')
    nn_ds_irr_p90 = SyntheticDataType.get_dataset_variation_name(SyntheticDataType.non_normal_correlated, 'p90')
    nn_ds_rs_irr_p30 = SyntheticDataType.get_dataset_variation_name(SyntheticDataType.rs_1min, 'p30')
    nn_ds_rs_irr_p90 = SyntheticDataType.get_dataset_variation_name(SyntheticDataType.rs_1min, 'p90')

    assert_that(df[nn_ds][SummaryStatistics.overall_segment_lengths]['min'], is_(900))
    assert_that(df[nn_ds_rs][SummaryStatistics.overall_segment_lengths]['min'], is_(15))
    assert_that(df[nn_ds_irr_p30][SummaryStatistics.overall_segment_lengths]['min'], is_(592))
    assert_that(df[nn_ds_irr_p90][SummaryStatistics.overall_segment_lengths]['min'], is_(58))
    assert_that(df[nn_ds_rs_irr_p30][SummaryStatistics.overall_segment_lengths]['min'], is_(15))
    assert_that(df[nn_ds_rs_irr_p90][SummaryStatistics.overall_segment_lengths]['min'], is_(13))


def test_returns_name_of_dataset_with_worst_mae():
    partial_nn = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                                data_type=SyntheticDataType.non_normal_correlated,
                                                data_dir=IRREGULAR_P30_DATA_DIR)
    name, mae = partial_nn.name_for_worst_relaxed_mae()
    assert_that(name, is_('trim-fire-24'))
    assert_that(mae, is_(0.034))


def test_loads_distribution_parameters_from_run_file():
    dist_params = ds_raw.get_median_min_max_distribution_parameters()

    # check it is "iob", "cob", "ig"
    assert_that(list(dist_params.keys()), contains_exactly(*SyntheticDataVariates.columns()))

    # check iob params
    iob_params = dist_params[SyntheticDataVariates.columns()[0]]
    assert_that(iob_params[DistParams.method], is_(stats.genextreme))
    assert_that(iob_params[DistParams.median_args], is_((-0.2237,)))
    assert_that(iob_params[DistParams.min_args], is_((-0.5206,)))
    assert_that(iob_params[DistParams.max_args], is_((0.0664,)))

    assert_that(iob_params[DistParams.median_kwargs].keys(), contains_exactly("loc", "scale"))
    assert_that(iob_params[DistParams.min_kwargs].keys(), contains_exactly("loc", "scale"))
    assert_that(iob_params[DistParams.min_kwargs].keys(), contains_exactly("loc", "scale"))

    assert_that(iob_params[DistParams.median_kwargs]["loc"], is_(0.49035))
    assert_that(iob_params[DistParams.max_kwargs]["loc"], is_(1.489))
    assert_that(iob_params[DistParams.min_kwargs]["loc"], is_(0.0955))

    assert_that(iob_params[DistParams.median_kwargs]["scale"], is_(1.0751))
    assert_that(iob_params[DistParams.max_kwargs]["scale"], is_(3.219))
    assert_that(iob_params[DistParams.min_kwargs]["scale"], is_(0.3602))

    # check cob params
    cob_params = dist_params[SyntheticDataVariates.columns()[1]]
    assert_that(cob_params[DistParams.method], is_(stats.nbinom))
    assert_that(cob_params[DistParams.median_args], is_((1.0, 0.10264999999999999)))
    assert_that(cob_params[DistParams.min_args], is_((1.0, 0.0464)))
    assert_that(cob_params[DistParams.max_args], is_((1.0, 0.4031)))

    assert_that(len(cob_params[DistParams.median_kwargs].keys()), is_(0))
    assert_that(len(cob_params[DistParams.min_kwargs].keys()), is_(0))
    assert_that(len(cob_params[DistParams.min_kwargs].keys()), is_(0))

    # check ig params
    ig_params = dist_params[SyntheticDataVariates.columns()[2]]
    assert_that(ig_params[DistParams.method], is_(stats.genextreme))
    assert_that(ig_params[DistParams.median_args], is_((0.0,)))
    assert_that(ig_params[DistParams.min_args], is_((0.0,)))
    assert_that(ig_params[DistParams.max_args], is_((0.0782,)))

    assert_that(ig_params[DistParams.median_kwargs].keys(), contains_exactly("loc", "scale"))
    assert_that(ig_params[DistParams.min_kwargs].keys(), contains_exactly("loc", "scale"))
    assert_that(ig_params[DistParams.min_kwargs].keys(), contains_exactly("loc", "scale"))

    assert_that(ig_params[DistParams.median_kwargs]["loc"], is_(116.64075))
    assert_that(ig_params[DistParams.max_kwargs]["loc"], is_(131.9869))
    assert_that(ig_params[DistParams.min_kwargs]["loc"], is_(88.7941))

    assert_that(ig_params[DistParams.median_kwargs]["scale"], is_(33.59955))
    assert_that(ig_params[DistParams.max_kwargs]["scale"], is_(53.5276))
    assert_that(ig_params[DistParams.min_kwargs]["scale"], is_(17.8245))


def test_can_load_data_if_required():
    full_data_ds = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                                  data_type=raw,
                                                  data_dir=data_dir, load_data=True)
    assert_that(len(full_data_ds.data_dfs), is_(30))
    assert_that(len(full_data_ds.label_dfs), is_(30))
    assert_that(full_data_ds.data_dfs.keys(), is_(full_data_ds.label_dfs.keys()))

    last_df_name = list(full_data_ds.data_dfs.keys())[-1]
    last_data = full_data_ds.get_data_as_xtrain(ds_name=last_df_name)
    assert_that(last_data.shape[1], is_(3))  # 3 timeseries
    assert_that(last_data.shape[0], greater_than(1240000))  # the big version

    all_datasets = full_data_ds.get_list_of_xtrain_of_all_datasets()
    assert_that(len(all_datasets), is_(30))
    assert_that(np.array_equal(all_datasets[-1], last_data))


def test_get_overall_correlation_stat_for_pattern_id():
    pattern_id = 19
    result = ds_nn.achieved_correlation_stats_for_pattern(pattern_id)

    assert_that(result.loc['50%'].tolist(), is_([-0.717, -0.087, 0.718]))
    assert_that(result.loc['mean'].tolist(), is_([-0.716, -0.087, 0.716]))
    assert_that(result.loc['max'].tolist(), is_([-0.669, -0.018, 0.745]))


def test_per_pattern_overall_mae_stats():
    stats_df = ds_nn.overall_per_pattern_mae_stats(SyntheticDataSegmentCols.relaxed_mae)

    assert_that(stats_df.loc[0, 'mean'], is_(0.01))
    assert_that(stats_df.loc[25, 'mean'], is_(0.004))

    # check stats df is sorted by median mae
    assert_that(stats_df.index[0], is_(7))  # worst pattern
    assert_that(stats_df.index[22], is_(25))  # best pattern


def test_per_pattern_n_segment_outside_of_tolerance():
    stats_df = ds_nn.per_pattern_n_segments_outside_tolerance_stats()

    assert_that(stats_df.loc[0, 'mean'], is_(0.0))
    assert_that(stats_df.loc[25, 'mean'], is_(0.0))

    # check stats df is sorted by median mae
    assert_that(stats_df.index[0], is_(0))  # worst pattern
    assert_that(stats_df.index[22], is_(25))  # best pattern


def test_calculate_mae_for_different_segment_lengths():
    ds_nn_cor = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                               data_type=SyntheticDataType.non_normal_correlated,
                                               data_dir=data_dir, load_data=True, additional_corr=[CorrType.pearson])
    lengths = [10, 15, 30]
    maes_results_spear = ds_nn_cor.mean_mae_for_segment_lengths(lengths, cor_type=CorrType.spearman)
    maes_results_pears = ds_nn_cor.mean_mae_for_segment_lengths(lengths, cor_type=CorrType.pearson)

    means_spear = maes_results_spear['mean']
    assert_that(len(means_spear), is_(len(lengths)))
    assert_that(means_spear[0], greater_than(means_spear[1]))


def test_calculate_overall_mae_and_stats_for_different_correlations():
    # load downsampled data
    correlations = [CorrType.pearson, CorrType.kendall]
    ds_ds = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                           data_type=SyntheticDataType.rs_1min,
                                           data_dir=data_dir, load_data=True, additional_corr=correlations)

    spearman_stats = ds_ds.overall_mae_stats(SyntheticDataSegmentCols.relaxed_mae)
    spearman_seg_tol = ds_ds.n_segments_outside_tolerance_stats()
    pearsons_stats = ds_ds.overall_mae_stats(SyntheticDataSegmentCols.relaxed_mae, corr_type=correlations[0])
    pearsons_seg_tol = ds_ds.n_segments_outside_tolerance_stats(corr_type=correlations[0])
    kendall_stats = ds_ds.overall_mae_stats(SyntheticDataSegmentCols.relaxed_mae, corr_type=correlations[1])
    kendall_seg_tol = ds_ds.n_segments_outside_tolerance_stats(corr_type=correlations[1])

    assert_that(spearman_stats["count"], is_(3000))  # stats overall not averaged per subject
    assert_that(spearman_stats["mean"], is_(0.131))
    assert_that(pearsons_stats["mean"], is_(0.132))  # downsampling turns data back to normal
    assert_that(kendall_stats["mean"], is_(0.206))

    assert_that(spearman_seg_tol["count"], is_(30))  # stats overall but this is a number per subject
    assert_that(spearman_seg_tol["mean"], is_(67.6))
    assert_that(pearsons_seg_tol["mean"], is_(66))
    assert_that(kendall_seg_tol["mean"], is_(79.7))
