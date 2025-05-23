import pandas as pd
from hamcrest import *

import pandas.testing as tm

from src.evaluation.describe_bad_partitions import DescribeBadPartitions
from src.experiments.run_cluster_quality_measures_calculation import read_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import internal_measure_evaluation_dir_for, ROOT_RESULTS_DIR, \
    GENERATED_DATASETS_FILE_PATH, IRREGULAR_P30_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.stats import ConfidenceIntervalCols, StatsCols
from src.evaluation.internal_measure_assessment import InternalMeasureAssessment, InternalMeasureCols, \
    get_full_filename_for_results_csv, IAResultsCSV, read_internal_assessment_result_for
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR, TEST_GENERATED_DATASETS_FILE_PATH

ds1_name = "misty-forest-56"
ds2_name = "twilight-fog-55"
ds3_name = "playful-thunder-52"
distance_measure = DistanceMeasures.l1_cor_dist
test_data_dir = TEST_DATA_DIR
run_names = pd.read_csv(TEST_GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
calculate_results_index = internal_measures = [ClusteringQualityMeasures.silhouette_score,
                                               ClusteringQualityMeasures.pmb, ClusteringQualityMeasures.dbi]
data_type = SyntheticDataType.non_normal_correlated
bp1 = DescribeBadPartitions(ds1_name, distance_measure=distance_measure, data_type=data_type,
                            internal_measures=calculate_results_index, data_dir=test_data_dir)
bp2 = DescribeBadPartitions(ds2_name, distance_measure=distance_measure, data_type=data_type,
                            internal_measures=calculate_results_index, data_dir=test_data_dir)
bp3 = DescribeBadPartitions(ds3_name, distance_measure=distance_measure, data_type=data_type,
                            internal_measures=calculate_results_index, data_dir=test_data_dir)

ds = [bp1.summary_df, bp2.summary_df, bp3.summary_df]
internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb]
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
    assert_that(df.iloc[0][r_col_name], is_(0.996))
    assert_that(df.iloc[1][r_col_name], is_(0.909))
    assert_that(df.iloc[2][r_col_name], is_(0.974))
    p_col_name = ia.measures_p_col_names[0]
    assert_that(df.iloc[0][p_col_name], is_(0.0))
    assert_that(df.iloc[1][p_col_name], is_(0.033))
    assert_that(df.iloc[2][p_col_name], is_(0.005))

    # assert pmb score corr
    r_col_name = ia.measures_corr_col_names[1]
    assert_that(df.iloc[0][r_col_name], is_(0.495))
    assert_that(df.iloc[1][r_col_name], is_(0.603))
    assert_that(df.iloc[2][r_col_name], is_(0.446))
    p_col_name = ia.measures_p_col_names[1]
    assert_that(df.iloc[0][p_col_name], is_(0.397))
    assert_that(df.iloc[1][p_col_name], is_(0.282))
    assert_that(df.iloc[2][p_col_name], is_(0.451))


def test_effect_size_d_of_difference_in_means_between_gt_and_worst_partition_for_sil():
    effect_size, lo_ci, hi_ci, standard_error = ia.effect_size_and_ci_of_difference_of_means_gt_and_worst_partition(
        internal_measure=ClusteringQualityMeasures.silhouette_score,
        worst_ranked_by=ClusteringQualityMeasures.jaccard_index)

    assert_that(round(effect_size, 3), is_(72.963))
    assert_that(round(lo_ci, 3), is_(1.363))
    assert_that(round(hi_ci, 3), is_(1.438))
    assert_that(round(standard_error, 3), is_(0.019))


def test_effect_size_d_of_difference_in_means_between_gt_and_worst_partition_for_pmb():
    effect_size, lo_ci, hi_ci, standard_error = ia.effect_size_and_ci_of_difference_of_means_gt_and_worst_partition(
        internal_measure=ClusteringQualityMeasures.pmb,
        worst_ranked_by=ClusteringQualityMeasures.jaccard_index)

    assert_that(round(effect_size, 3), is_(13.051))
    assert_that(round(lo_ci, 3), is_(12.951))
    assert_that(round(hi_ci, 3), is_(17.528))
    assert_that(round(standard_error, 3), is_(1.168))


def test_creates_df_of_effect_sizes():
    df = ia.differences_between_worst_and_best_partition()

    assert_that(df.shape[0], is_(len(internal_measures)))
    assert_that(df.iloc[0][InternalMeasureCols.effect_size], is_(72.963))
    assert_that(df.iloc[1][InternalMeasureCols.effect_size], is_(13.051))
    assert_that(df.iloc[0][ConfidenceIntervalCols.ci_96lo], is_(1.363))
    assert_that(df.iloc[0][ConfidenceIntervalCols.ci_96hi], is_(1.438))
    assert_that(df.iloc[0][ConfidenceIntervalCols.standard_error], is_(0.019))


def test_calculate_mean_sd_count_for_each_internal_measures_correlation_with_external_measure():
    df = ia.descriptive_statistics_for_internal_measures_correlation()
    sil_col = ia.measures_corr_col_names[0]
    pmb_col = ia.measures_corr_col_names[1]

    mean = df.loc['mean']
    std = df.loc['std']
    minim = df.loc['min']

    assert_that(mean[sil_col], is_(0.96))
    assert_that(mean[pmb_col], is_(0.51))

    assert_that(std[sil_col], is_(0.05))
    assert_that(std[pmb_col], is_(0.08))

    assert_that(minim[sil_col], is_(0.91))
    assert_that(minim[pmb_col], is_(0.45))


def test_calculate_ci_of_differences_between_mean_correlation_between_internal_measures():
    df = ia.ci_of_differences_between_internal_measure_correlations()

    col_name = ia.compare_internal_measures_cols[0]

    assert_that(col_name, contains_string('PMB'))
    assert_that(col_name, contains_string('SCW'))

    lo_ci = df.loc[ConfidenceIntervalCols.ci_96lo]
    hi_ci = df.loc[ConfidenceIntervalCols.ci_96hi]
    se = df.loc[ConfidenceIntervalCols.standard_error]
    effect_sizes = df.loc[StatsCols.effect_size]

    assert_that(lo_ci[col_name], is_(0.343))
    assert_that(hi_ci[col_name], is_(0.557))
    assert_that(se[col_name], is_(0.054))
    assert_that(effect_sizes[col_name], is_(6.746))


def test_calculate_ci_of_differences_for_internal_measures_can_handle_inverted_measures():
    ia_dbi = InternalMeasureAssessment(distance_measure=distance_measure,
                                       internal_measures=[ClusteringQualityMeasures.silhouette_score,
                                                          ClusteringQualityMeasures.dbi],
                                       dataset_results=ds)

    df = ia_dbi.ci_of_differences_between_internal_measure_correlations()

    col_name = ia_dbi.compare_internal_measures_cols[0]

    assert_that(col_name, contains_string('DBI'))
    assert_that(col_name, contains_string('SCW'))

    lo_ci = df.loc[ConfidenceIntervalCols.ci_96lo]
    hi_ci = df.loc[ConfidenceIntervalCols.ci_96hi]
    se = df.loc[ConfidenceIntervalCols.standard_error]
    effect_sizes = df.loc[StatsCols.effect_size]

    assert_that(lo_ci[col_name], is_(-0.08))
    assert_that(hi_ci[col_name], is_(0.08))
    assert_that(se[col_name], is_(0.041))
    assert_that(effect_sizes[col_name], is_(0.0))


def test_paired_samples_t_test_on_fisher_transformed_correlation_coefficients_between_internal_measures():
    ia_dbi = InternalMeasureAssessment(distance_measure=distance_measure,
                                       internal_measures=[ClusteringQualityMeasures.silhouette_score,
                                                          ClusteringQualityMeasures.dbi],
                                       dataset_results=ds)

    df = ia_dbi.paired_samples_t_test_on_fisher_transformed_correlation_coefficients()

    col_name = ia_dbi.compare_internal_measures_cols[0]

    assert_that(col_name, contains_string('DBI'))
    assert_that(col_name, contains_string('SCW'))

    ps = df.loc[StatsCols.p_value]
    statistics = df.loc[StatsCols.statistic]
    effect_sizes = df.loc[StatsCols.effect_size]
    powers = df.loc[StatsCols.achieved_power]

    assert_that(ps[col_name], is_(0.274))
    assert_that(statistics[col_name], is_(-1.491))
    assert_that(effect_sizes[col_name], is_(-0.861))
    assert_that(powers[col_name], is_(0.148))


def test_can_assess_different_distance_measures():
    l2 = DistanceMeasures.l2_cor_dist
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
    assert_that(df.iloc[0][r_col_name], is_(0.996))
    assert_that(df.iloc[1][r_col_name], is_(0.908))
    p_col_name = ial2.measures_p_col_names[0]
    assert_that(df.iloc[0][p_col_name], is_(0.0))
    assert_that(df.iloc[1][p_col_name], is_(0.033))

    # assert pmb score corr
    r_col_name = ial2.measures_corr_col_names[1]
    p_col_name = ial2.measures_p_col_names[1]
    assert_that(df.iloc[0][r_col_name], is_(0.492))
    assert_that(df.iloc[1][r_col_name], is_(0.603))
    assert_that(df.iloc[0][p_col_name], is_(0.4))
    assert_that(df.iloc[1][p_col_name], is_(0.282))


def test_can_read_result_for_internal_measure_assessment_summary_df(tmp_path):
    root_result_dir = str(tmp_path)
    overall_dataset_name = "test"
    summary_df = ia.correlation_summary

    # save correlation summary
    store_results_in = internal_measure_evaluation_dir_for(overall_dataset_name=overall_dataset_name,
                                                           results_dir=root_result_dir, data_type=data_type,
                                                           data_dir=test_data_dir,
                                                           distance_measure=distance_measure)
    summary_df.to_csv(get_full_filename_for_results_csv(store_results_in, IAResultsCSV.correlation_summary))

    # read correlation summary
    read_df = read_internal_assessment_result_for(result_type=IAResultsCSV.correlation_summary,
                                                  overall_dataset_name=overall_dataset_name,
                                                  results_dir=root_result_dir, data_type=data_type,
                                                  data_dir=test_data_dir,
                                                  distance_measure=distance_measure)

    # check saved and reloaded dataframe are the same
    tm.assert_frame_equal(read_df, summary_df)


def test_names_summary_files_right():
    # this is on production data as the test data does not create the mistake
    runs = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    partitions = read_clustering_quality_measures(overall_ds_name="n30", data_type=SyntheticDataType.normal_correlated,
                                                  root_results_dir=ROOT_RESULTS_DIR, data_dir=IRREGULAR_P30_DATA_DIR,
                                                  distance_measure=DistanceMeasures.l1_cor_dist, run_names=runs)
    assessment = InternalMeasureAssessment(distance_measure=distance_measure, dataset_results=partitions,
                                           internal_measures=[ClusteringQualityMeasures.vrc])
    df = assessment.correlation_summary
    assert_that(df[InternalMeasureCols.name], contains_exactly(*runs))
