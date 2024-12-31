import os

import pytest
from hamcrest import *

from src.evaluation.describe_bad_partitions import DescribeBadPartitions, DescribeBadPartCols
from src.utils.configurations import internal_measure_assessment_dir_for
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.stats import ConfidenceIntervalCols
from src.evaluation.distance_metric_assessment import DistanceMeasureCols
from src.evaluation.internal_measure_assessment import InternalMeasureAssessment, InternalMeasureCols, \
    run_internal_measure_assessment__datasets, get_internal_measures_summary_file_name, \
    get_full_filename_for_results_csv, IAResultsCSV
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR, TEST_GENERATED_DATASETS_FILE_PATH, \
    TEST_ROOT_RESULTS_DIR

ds1_name = "misty-forest-56"
ds2_name = "twilight-fog-55"
ds3_name = "playful-thunder-52"
distance_measure = DistanceMeasureCols.l1_cor_dist
internal_measures = [DescribeBadPartCols.silhouette_score, DescribeBadPartCols.pmb]
test_data_dir = TEST_DATA_DIR

bp1 = DescribeBadPartitions(ds1_name, distance_measure=distance_measure, internal_measures=internal_measures,
                            data_dir=test_data_dir)
bp2 = DescribeBadPartitions(ds2_name, distance_measure=distance_measure, internal_measures=internal_measures,
                            data_dir=test_data_dir)
bp3 = DescribeBadPartitions(ds3_name, distance_measure=distance_measure, internal_measures=internal_measures,
                            data_dir=test_data_dir)

ds = [bp1.summary_df, bp2.summary_df, bp3.summary_df]
ia = InternalMeasureAssessment(distance_measure=distance_measure, internal_measures=internal_measures,
                               dataset_results=ds)


def test_calculate_correlation_between_internal_and_external_measures_for_each_dataset():
    df = ia.correlation_summary

    assert_that(df.shape[0], is_(len(ds)))
    assert_that(df.iloc[0][InternalMeasureCols.name], is_(ds1_name))
    assert_that(df.iloc[1][InternalMeasureCols.name], is_(ds2_name))
    assert_that(df.iloc[2][InternalMeasureCols.name], is_(ds3_name))
    assert_that(df.iloc[0][InternalMeasureCols.partitions], is_(len(bp1.partitions) + 1))
    assert_that(df.iloc[1][InternalMeasureCols.partitions], is_(len(bp2.partitions) + 1))
    assert_that(df.iloc[2][InternalMeasureCols.partitions], is_(len(bp3.partitions) + 1))
    # assert sil score corr
    r_col_name = ia.measures_corr_col_names[0]
    assert_that(df.iloc[0][r_col_name], is_(0.98))
    assert_that(df.iloc[1][r_col_name], is_(0.828))
    assert_that(df.iloc[2][r_col_name], is_(0.899))
    p_col_name = ia.measures_p_col_names[0]
    assert_that(df.iloc[0][p_col_name], is_(0.003))
    assert_that(df.iloc[1][p_col_name], is_(0.084))
    assert_that(df.iloc[2][p_col_name], is_(0.038))

    # assert pmb score corr
    r_col_name = ia.measures_corr_col_names[1]
    assert_that(df.iloc[0][r_col_name], is_(0.491))
    assert_that(df.iloc[1][r_col_name], is_(0.603))
    assert_that(df.iloc[2][r_col_name], is_(0.444))
    p_col_name = ia.measures_p_col_names[1]
    assert_that(df.iloc[0][p_col_name], is_(0.401))
    assert_that(df.iloc[1][p_col_name], is_(0.282))
    assert_that(df.iloc[2][p_col_name], is_(0.454))


def test_effect_size_d_of_difference_in_means_between_gt_and_worst_partition_for_sil():
    effect_size, lo_ci, hi_ci, standard_error = ia.effect_size_and_ci_of_difference_of_means_gt_and_worst_partition(
        internal_measure=DescribeBadPartCols.silhouette_score,
        worst_ranked_by=DescribeBadPartCols.jaccard_index)

    assert_that(round(effect_size, 3), is_(81.825))
    assert_that(round(lo_ci, 3), is_(1.289))
    assert_that(round(hi_ci, 3), is_(1.352))
    assert_that(round(standard_error, 3), is_(0.016))


def test_effect_size_d_of_difference_in_means_between_gt_and_worst_partition_for_pmb():
    effect_size, lo_ci, hi_ci, standard_error = ia.effect_size_and_ci_of_difference_of_means_gt_and_worst_partition(
        internal_measure=DescribeBadPartCols.pmb,
        worst_ranked_by=DescribeBadPartCols.jaccard_index)

    assert_that(round(effect_size, 3), is_(14.141))
    assert_that(round(lo_ci, 3), is_(10.352))
    assert_that(round(hi_ci, 3), is_(13.683))
    assert_that(round(standard_error, 3), is_(0.85))


def test_creates_df_of_effect_sizes():
    df = ia.differences_between_worst_and_best_partition()

    assert_that(df.shape[0], is_(len(internal_measures)))
    assert_that(df.iloc[0][InternalMeasureCols.effect_size], is_(81.825))
    assert_that(df.iloc[1][InternalMeasureCols.effect_size], is_(14.141))
    assert_that(df.iloc[0][ConfidenceIntervalCols.ci_96lo], is_(1.289))
    assert_that(df.iloc[0][ConfidenceIntervalCols.ci_96hi], is_(1.352))
    assert_that(df.iloc[0][ConfidenceIntervalCols.standard_error], is_(0.016))


def test_calculate_mean_sd_count_for_each_internal_measures_correlation_with_external_measure():
    df = ia.descriptive_statistics_for_internal_measures_correlation()
    sil_col = ia.measures_corr_col_names[0]
    pmb_col = ia.measures_corr_col_names[1]

    mean = df.loc['mean']
    std = df.loc['std']
    minim = df.loc['min']

    assert_that(mean[sil_col], is_(0.9))
    assert_that(mean[pmb_col], is_(0.51))

    assert_that(std[sil_col], is_(0.08))
    assert_that(std[pmb_col], is_(0.08))

    assert_that(minim[sil_col], is_(0.83))
    assert_that(minim[pmb_col], is_(0.44))


def test_calculate_ci_of_differences_between_mean_correlation_between_internal_measures():
    df = ia.ci_of_differences_between_internal_measure_correlations()

    col_name = ia.compare_internal_measures_cols[0]

    lo_ci = df.loc[ConfidenceIntervalCols.ci_96lo]
    hi_ci = df.loc[ConfidenceIntervalCols.ci_96hi]
    se = df.loc[ConfidenceIntervalCols.standard_error]

    assert_that(lo_ci[col_name], is_(0.262))
    assert_that(hi_ci[col_name], is_(0.518))
    assert_that(se[col_name], is_(0.065))


def test_can_assess_different_distance_measures():
    l2 = DistanceMeasureCols.l2_cor_dist
    bp1l2 = DescribeBadPartitions(ds1_name, distance_measure=l2, internal_measures=internal_measures,
                                  data_dir=test_data_dir)
    bp2l2 = DescribeBadPartitions(ds2_name, distance_measure=l2, internal_measures=internal_measures,
                                  data_dir=test_data_dir)
    ial2 = InternalMeasureAssessment(distance_measure=l2,
                                     internal_measures=internal_measures,
                                     dataset_results=[bp1l2.summary_df, bp2l2.summary_df])

    df = ial2.correlation_summary

    assert_that(df.shape[0], is_(2))
    assert_that(df.iloc[0][InternalMeasureCols.name], is_(ds1_name))
    assert_that(df.iloc[1][InternalMeasureCols.name], is_(ds2_name))
    assert_that(df.iloc[0][InternalMeasureCols.partitions], is_(len(bp1l2.partitions) + 1))
    assert_that(df.iloc[1][InternalMeasureCols.partitions], is_(len(bp2l2.partitions) + 1))

    # assert sil score corr
    r_col_name = ial2.measures_corr_col_names[0]
    assert_that(df.iloc[0][r_col_name], is_(0.981))
    assert_that(df.iloc[1][r_col_name], is_(0.829))
    p_col_name = ial2.measures_p_col_names[0]
    assert_that(df.iloc[0][p_col_name], is_(0.003))
    assert_that(df.iloc[1][p_col_name], is_(0.082))

    # assert pmb score corr
    r_col_name = ial2.measures_corr_col_names[1]
    p_col_name = ial2.measures_p_col_names[1]
    assert_that(df.iloc[0][r_col_name], is_(0.494))
    assert_that(df.iloc[1][r_col_name], is_(0.603))
    assert_that(df.iloc[0][p_col_name], is_(0.398))
    assert_that(df.iloc[1][p_col_name], is_(0.282))


def test_can_run_assessment_on_full_dataset_and_store_results_for_runs_with_all_clusters():
    # run test_wandb_create_bad_partitions to create bad partitions if they don't exist for your configuration
    overall_ds_name = "test_stuff"
    # distance_measure = DistanceMeasures.l1_with_ref
    # distance_measure = DistanceMeasures.l2_cor_dist
    distance_measure = DistanceMeasures.l1_cor_dist
    data_type = SyntheticDataType.normal_correlated
    test_results_dir = TEST_ROOT_RESULTS_DIR
    run_internal_measure_assessment__datasets(overall_ds_name=overall_ds_name,
                                              generated_ds_csv=TEST_GENERATED_DATASETS_FILE_PATH,
                                              distance_measure=distance_measure,
                                              data_type=data_type,
                                              data_dir=test_data_dir,
                                              results_dir=test_results_dir,
                                              internal_measures=[DescribeBadPartCols.silhouette_score,
                                                                 DescribeBadPartCols.pmb],
                                              n_dropped_clusters=[],
                                              n_dropped_segments=[],
                                              )

    # check if the files have been created
    results_folder = internal_measure_assessment_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type=data_type,
        results_dir=test_results_dir,
        distance_measure=distance_measure)

    # IA assessment results
    assert_that(os.path.exists(get_full_filename_for_results_csv(results_folder, IAResultsCSV.correlation_summary)))
    assert_that(os.path.exists(
        get_full_filename_for_results_csv(results_folder, IAResultsCSV.effect_size_difference_worst_best)))
    assert_that(os.path.exists(
        get_full_filename_for_results_csv(results_folder, IAResultsCSV.descriptive_statistics_measure_summary)))
    assert_that(os.path.exists(
        get_full_filename_for_results_csv(results_folder, IAResultsCSV.ci_of_differences_between_measures)))


def test_can_run_assessment_and_store_results_for_runs_with_dropping_clusters():
    # run test_wandb_create_bad_partitions to create bad partitions if they don't exist for your configuration
    overall_ds_name = "test_stuff"
    distance_measure = DistanceMeasures.l1_cor_dist
    data_type = SyntheticDataType.normal_correlated
    test_results_dir = TEST_ROOT_RESULTS_DIR
    run_internal_measure_assessment__datasets(overall_ds_name=overall_ds_name,
                                              generated_ds_csv=TEST_GENERATED_DATASETS_FILE_PATH,
                                              distance_measure=distance_measure,
                                              data_type=data_type,
                                              data_dir=test_data_dir,
                                              results_dir=test_results_dir,
                                              internal_measures=[DescribeBadPartCols.silhouette_score,
                                                                 DescribeBadPartCols.pmb],
                                              n_dropped_clusters=[5, 15],
                                              n_dropped_segments=[],
                                              )

    # check if the files have been created
    results_folder_cl15 = internal_measure_assessment_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type=data_type,
        results_dir=test_results_dir,
        distance_measure=distance_measure,
        drop_clusters=15)

    # IA assessment results
    assert_that(
        os.path.exists(get_full_filename_for_results_csv(results_folder_cl15, IAResultsCSV.correlation_summary)))
    assert_that(os.path.exists(
        get_full_filename_for_results_csv(results_folder_cl15, IAResultsCSV.effect_size_difference_worst_best)))
    assert_that(os.path.exists(
        get_full_filename_for_results_csv(results_folder_cl15, IAResultsCSV.descriptive_statistics_measure_summary)))
    assert_that(os.path.exists(
        get_full_filename_for_results_csv(results_folder_cl15, IAResultsCSV.ci_of_differences_between_measures)))


def test_can_run_assessment_and_store_results_for_runs_with_dropping_segments():
    # run test_wandb_create_bad_partitions to create bad partitions if they don't exist for your configuration
    overall_ds_name = "test_stuff"
    distance_measure = DistanceMeasures.l1_cor_dist
    data_type = SyntheticDataType.normal_correlated
    test_results_dir = TEST_ROOT_RESULTS_DIR
    run_internal_measure_assessment__datasets(overall_ds_name=overall_ds_name,
                                              generated_ds_csv=TEST_GENERATED_DATASETS_FILE_PATH,
                                              distance_measure=distance_measure,
                                              data_type=data_type,
                                              data_dir=test_data_dir,
                                              results_dir=test_results_dir,
                                              internal_measures=[DescribeBadPartCols.silhouette_score,
                                                                 DescribeBadPartCols.pmb],
                                              n_dropped_clusters=[],
                                              n_dropped_segments=[50],
                                              )

    # check if the files have been created
    results_folder_seg50 = internal_measure_assessment_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type=data_type,
        results_dir=test_results_dir,
        distance_measure=distance_measure,
        drop_segments=50)

    # IA assessment results
    assert_that(
        os.path.exists(get_full_filename_for_results_csv(results_folder_seg50, IAResultsCSV.correlation_summary)))
    assert_that(os.path.exists(
        get_full_filename_for_results_csv(results_folder_seg50, IAResultsCSV.effect_size_difference_worst_best)))
    assert_that(os.path.exists(
        get_full_filename_for_results_csv(results_folder_seg50, IAResultsCSV.descriptive_statistics_measure_summary)))
    assert_that(os.path.exists(
        get_full_filename_for_results_csv(results_folder_seg50, IAResultsCSV.ci_of_differences_between_measures)))
