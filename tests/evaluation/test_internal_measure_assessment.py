from hamcrest import *

from src.evaluation.describe_bad_partitions import DescribeBadPartitions, DescribeBadPartCols
from src.utils.stats import ConfidenceIntervalCols
from src.evaluation.distance_metric_assessment import DistanceMeasureCols
from src.evaluation.internal_measure_assessment import InternalMeasureAssessment, InternalMeasureCols

ds1_name = "misty-forest-56"
ds2_name = "twilight-fog-55"
ds3_name = "playful-thunder-52"
distance_measure = DistanceMeasureCols.l1_cor_dist
internal_measures = [DescribeBadPartCols.silhouette_score, DescribeBadPartCols.pmb]

bp1 = DescribeBadPartitions(ds1_name, distance_measure=distance_measure,
                            internal_measures=internal_measures).summary_df.copy()
bp2 = DescribeBadPartitions(ds2_name, distance_measure=distance_measure,
                            internal_measures=internal_measures).summary_df.copy()
bp3 = DescribeBadPartitions(ds3_name, distance_measure=distance_measure,
                            internal_measures=internal_measures).summary_df.copy()

ds = [bp1, bp2, bp3]
ia = InternalMeasureAssessment(distance_measure=distance_measure, internal_measures=internal_measures,
                               dataset_results=ds)


def test_calculate_correlation_between_internal_and_external_measures_for_each_dataset():
    df = ia.correlation_summary

    assert_that(df.shape[0], is_(len(ds)))
    assert_that(df.iloc[0][InternalMeasureCols.name], is_(ds1_name))
    assert_that(df.iloc[1][InternalMeasureCols.name], is_(ds2_name))
    assert_that(df.iloc[2][InternalMeasureCols.name], is_(ds3_name))
    assert_that(df.iloc[0][InternalMeasureCols.partitions], is_(4))
    assert_that(df.iloc[1][InternalMeasureCols.partitions], is_(4))
    assert_that(df.iloc[2][InternalMeasureCols.partitions], is_(4))
    # assert sil score corr
    r_col_name = ia.measures_corr_col_names[0]
    assert_that(df.iloc[0][r_col_name], is_(0.837))
    assert_that(df.iloc[1][r_col_name], is_(0.944))
    assert_that(df.iloc[2][r_col_name], is_(0.984))
    p_col_name = ia.measures_p_col_names[0]
    assert_that(df.iloc[0][p_col_name], is_(0.163))
    assert_that(df.iloc[1][p_col_name], is_(0.056))
    assert_that(df.iloc[2][p_col_name], is_(0.016))

    # assert pmb score corr
    r_col_name = ia.measures_corr_col_names[1]
    assert_that(df.iloc[0][r_col_name], is_(0.477))
    assert_that(df.iloc[1][r_col_name], is_(0.398))
    assert_that(df.iloc[2][r_col_name], is_(0.614))
    p_col_name = ia.measures_p_col_names[1]
    assert_that(df.iloc[0][p_col_name], is_(0.523))
    assert_that(df.iloc[1][p_col_name], is_(0.602))
    assert_that(df.iloc[2][p_col_name], is_(0.386))


def test_effect_size_d_of_difference_in_means_between_gt_and_worst_partition_for_sil():
    effect_size, lo_ci, hi_ci, standard_error = ia.effect_size_and_ci_of_difference_of_means_gt_and_worst_partition(
        internal_measure=DescribeBadPartCols.silhouette_score,
        worst_ranked_by=DescribeBadPartCols.jaccard_index)

    assert_that(round(effect_size, 3), is_(5.741))
    assert_that(round(lo_ci, 3), is_(0.697))
    assert_that(round(hi_ci, 3), is_(1.420))
    assert_that(round(standard_error, 3), is_(0.184))


def test_effect_size_d_of_difference_in_means_between_gt_and_worst_partition_for_pmb():
    effect_size, lo_ci, hi_ci, standard_error = ia.effect_size_and_ci_of_difference_of_means_gt_and_worst_partition(
        internal_measure=DescribeBadPartCols.pmb,
        worst_ranked_by=DescribeBadPartCols.jaccard_index)

    assert_that(round(effect_size, 3), is_(10.822))
    assert_that(round(lo_ci, 3), is_(228.854))
    assert_that(round(hi_ci, 3), is_(330.086))
    assert_that(round(standard_error, 3), is_(25.824))


def test_creates_df_of_effect_sizes():
    df = ia.differences_between_worst_and_best_partition()

    assert_that(df.shape[0], is_(len(internal_measures)))
    assert_that(df.iloc[0][InternalMeasureCols.effect_size], is_(5.741))
    assert_that(df.iloc[1][InternalMeasureCols.effect_size], is_(10.822))
    assert_that(df.iloc[0][ConfidenceIntervalCols.ci_96lo], is_(0.697))
    assert_that(df.iloc[0][ConfidenceIntervalCols.ci_96hi], is_(1.420))
    assert_that(df.iloc[0][ConfidenceIntervalCols.standard_error], is_(0.184))


def test_calculate_mean_sd_count_for_each_internal_measures_correlation_with_external_measure():
    df = ia.descriptive_statistics_for_internal_measures_correlation()
    sil_col = ia.measures_corr_col_names[0]
    pmb_col = ia.measures_corr_col_names[1]

    mean = df.loc['mean']
    std = df.loc['std']
    minim = df.loc['min']

    assert_that(mean[sil_col], is_(0.92))
    assert_that(mean[pmb_col], is_(0.5))

    assert_that(std[sil_col], is_(0.08))
    assert_that(std[pmb_col], is_(0.11))

    assert_that(minim[sil_col], is_(0.84))
    assert_that(minim[pmb_col], is_(0.4))


def test_calculate_ci_of_differences_between_mean_correlation_between_internal_measures():
    df = ia.ci_of_differences_between_internal_measure_correlations()

    col_name = ia.compare_internal_measures_cols[0]

    lo_ci = df.loc[ConfidenceIntervalCols.ci_96lo]
    hi_ci = df.loc[ConfidenceIntervalCols.ci_96hi]
    se = df.loc[ConfidenceIntervalCols.standard_error]

    assert_that(lo_ci[col_name], is_(0.266))
    assert_that(hi_ci[col_name], is_(0.574))
    assert_that(se[col_name], is_(0.079))


def test_can_assess_different_distance_measures():
    l2 = DistanceMeasureCols.l2_cor_dist
    bp1l2 = DescribeBadPartitions(ds1_name, distance_measure=l2, internal_measures=internal_measures).summary_df.copy()
    bp2l2 = DescribeBadPartitions(ds2_name, distance_measure=l2,
                                  internal_measures=internal_measures, ).summary_df.copy()
    ial2 = InternalMeasureAssessment(distance_measure=l2,
                                     internal_measures=internal_measures,
                                     dataset_results=[bp1l2, bp2l2])

    df = ial2.correlation_summary

    assert_that(df.shape[0], is_(2))
    assert_that(df.iloc[0][InternalMeasureCols.name], is_(ds1_name))
    assert_that(df.iloc[1][InternalMeasureCols.name], is_(ds2_name))
    assert_that(df.iloc[0][InternalMeasureCols.partitions], is_(4))
    assert_that(df.iloc[1][InternalMeasureCols.partitions], is_(4))

    # assert sil score corr
    r_col_name = ial2.measures_corr_col_names[0]
    assert_that(df.iloc[0][r_col_name], is_(0.814))
    assert_that(df.iloc[1][r_col_name], is_(0.938))
    p_col_name = ial2.measures_p_col_names[0]
    assert_that(df.iloc[0][p_col_name], is_(0.186))
    assert_that(df.iloc[1][p_col_name], is_(0.062))

    # assert pmb score corr
    r_col_name = ial2.measures_corr_col_names[1]
    p_col_name = ial2.measures_p_col_names[1]
    assert_that(df.iloc[0][r_col_name], is_(0.478))
    assert_that(df.iloc[1][r_col_name], is_(0.403))
    assert_that(df.iloc[0][p_col_name], is_(0.522))
    assert_that(df.iloc[1][p_col_name], is_(0.597))
