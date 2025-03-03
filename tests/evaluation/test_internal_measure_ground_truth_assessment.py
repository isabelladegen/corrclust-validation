from hamcrest import *

from src.evaluation.internal_measure_ground_truth_assessment import InternalMeasureGroundTruthAssessment, \
    GroupAssessmentCols
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi]
distance_measures = [DistanceMeasures.l1_cor_dist, DistanceMeasures.l3_cor_dist,
                     DistanceMeasures.foerstner_cor_dist]
data_dir = SYNTHETIC_DATA_DIR
data_type = SyntheticDataType.normal_correlated
name = "n30"
results_dir = ROOT_RESULTS_DIR

ga = InternalMeasureGroundTruthAssessment(overall_ds_name=name, internal_measures=internal_measures,
                                          distance_measures=distance_measures, data_dir=data_dir,
                                          data_type=data_type, root_results_dir=results_dir)


def test_internal_measure_ground_truth_assessment_loads_all_distance_measures():
    dfs = ga.ground_truth_calculation_dfs

    assert_that(len(dfs), is_(len(distance_measures)))
    l1_results = dfs[DistanceMeasures.l1_cor_dist]
    # one ground truth result for all runs
    assert_that(l1_results.shape[0], is_(30))
    assert_that(l1_results[internal_measures[0]].isna().sum(), is_(0))


def test_reshapes_raw_data_into_raw_scores_df_per_internal_measure():
    raw_scores = ga.raw_scores_for_each_internal_measure()

    assert_that(len(raw_scores), is_(len(internal_measures)))

    rank_df = raw_scores[internal_measures[0]]
    # columns are now distance measures
    assert_that(rank_df.columns, contains_exactly(*distance_measures))
    # rows are run
    assert_that(rank_df.shape[0], is_(30))
    assert_that(raw_scores[internal_measures[1]].shape[0], is_(30))


def test_rank_distance_measures_for_internal_measures():
    ranks = ga.rank_distance_measures_for_each_internal_measure()
    raw_values = ga.raw_scores_for_each_internal_measure()

    assert_that(len(ranks), is_(len(internal_measures)))

    scw_df = ranks[ClusteringQualityMeasures.silhouette_score]
    assert_that(scw_df.shape[0], is_(30))
    # SCW = the higher the better
    highest_raw_value = raw_values[ClusteringQualityMeasures.silhouette_score].iloc[0].idxmax()
    lowest_rank = scw_df.iloc[0].idxmin()
    assert_that(lowest_rank, is_(highest_raw_value))

    dbi_df = ranks[ClusteringQualityMeasures.dbi]
    assert_that(dbi_df.shape[0], is_(30))
    # DBI = the lower the better
    lowest_raw_value = raw_values[ClusteringQualityMeasures.dbi].iloc[0].idxmin()
    lowest_rank = dbi_df.iloc[0].idxmin()
    assert_that(lowest_rank, is_(lowest_raw_value))


def test_calculate_stats_for_ranking_and_raw_values():
    rank_stats = ga.stats_for_ranks_across_all_runs()
    raw_values = ga.stats_for_raw_values_across_all_runs()

    scw_ranks = rank_stats[ClusteringQualityMeasures.silhouette_score]
    assert_that(scw_ranks.loc['mean', DistanceMeasures.l1_cor_dist], is_(2.0))
    assert_that(scw_ranks.loc['50%', DistanceMeasures.l1_cor_dist], is_(2.0))

    scw_values = raw_values[ClusteringQualityMeasures.silhouette_score]
    assert_that(scw_values.loc['mean', DistanceMeasures.l1_cor_dist], is_(0.971))
    assert_that(scw_values.loc['50%', DistanceMeasures.l1_cor_dist], is_(0.972))


def test_calculate_grouping_for_distance_measure_comparisons():
    grouping = ga.grouping_for_each_internal_measure(stats_value='50%')

    # has grouping for all internal measure
    assert_that(len(grouping.keys()), is_(len(internal_measures)))

    scw_groupings = grouping[ClusteringQualityMeasures.silhouette_score]
    pmb_groupings = grouping[ClusteringQualityMeasures.dbi]
    # DistanceMeasures.l1_cor_dist, DistanceMeasures.l3_cor_dist,
    # DistanceMeasures.foerstner_cor_dist
    assert_that(len(scw_groupings), is_(3))
    assert_that(scw_groupings[1], contains_exactly(DistanceMeasures.l3_cor_dist))
    assert_that(scw_groupings[2], contains_exactly(DistanceMeasures.l1_cor_dist))
    assert_that(scw_groupings[3], contains_exactly(DistanceMeasures.foerstner_cor_dist))

    assert_that(len(pmb_groupings), is_(3))
    assert_that(pmb_groupings[1], contains_exactly(DistanceMeasures.l3_cor_dist))
    assert_that(pmb_groupings[2], contains_exactly(DistanceMeasures.l1_cor_dist))
    assert_that(pmb_groupings[3], contains_exactly(DistanceMeasures.foerstner_cor_dist))


def test_wilcoxon_signed_rank_until_significant():
    result = ga.wilcoxons_signed_rank_until_all_significant()

    # result for each internal measure
    assert_that(len(result), is_(len(internal_measures)))

    # silhouette results
    scw_df = result[ClusteringQualityMeasures.silhouette_score]
    assert_that(scw_df.shape[0], is_(1))
    assert_that(scw_df[GroupAssessmentCols.alpha].iloc[0], is_(0.05))
    assert_that(scw_df[GroupAssessmentCols.statistic].iloc[0], is_(0.0))
    assert_that(round(scw_df[GroupAssessmentCols.p_value].iloc[0], 2), is_(0.0))
    assert_that(scw_df[GroupAssessmentCols.effect_size].iloc[0], is_(1.097))
    assert_that(scw_df[GroupAssessmentCols.achieved_power].iloc[0], is_(1.0))
    assert_that(scw_df[GroupAssessmentCols.non_zero_pairs].iloc[0], is_(30))
    assert_that(scw_df[GroupAssessmentCols.is_significat].iloc[0], is_(True))
    assert_that(scw_df[GroupAssessmentCols.group].iloc[0], is_((1, 2)))
    assert_that(scw_df[GroupAssessmentCols.distance_measures_in_group].iloc[0],
                contains_exactly(DistanceMeasures.l1_cor_dist))
    assert_that(scw_df[GroupAssessmentCols.compared_distance_measures].iloc[0],
                is_((DistanceMeasures.l3_cor_dist, DistanceMeasures.l1_cor_dist)))

    # dbi results
    dbi_df = result[ClusteringQualityMeasures.dbi]
    assert_that(dbi_df.shape[0], is_(1))
    assert_that(dbi_df[GroupAssessmentCols.alpha].iloc[0], is_(0.05))
    assert_that(dbi_df[GroupAssessmentCols.statistic].iloc[0], is_(0.0))
    assert_that(round(dbi_df[GroupAssessmentCols.p_value].iloc[0], 2), is_(0.0))
    assert_that(dbi_df[GroupAssessmentCols.effect_size].iloc[0], is_(1.097))
    assert_that(dbi_df[GroupAssessmentCols.achieved_power].iloc[0], is_(1.0))
    assert_that(dbi_df[GroupAssessmentCols.non_zero_pairs].iloc[0], is_(30))
    assert_that(dbi_df[GroupAssessmentCols.is_significat].iloc[0], is_(True))
    assert_that(dbi_df[GroupAssessmentCols.group].iloc[0], is_((1, 2)))
    assert_that(dbi_df[GroupAssessmentCols.distance_measures_in_group].iloc[0],
                contains_exactly(DistanceMeasures.l1_cor_dist))
    assert_that(dbi_df[GroupAssessmentCols.compared_distance_measures].iloc[0],
                is_((DistanceMeasures.l3_cor_dist, DistanceMeasures.l1_cor_dist)))
