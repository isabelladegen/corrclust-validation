from hamcrest import *

from src.evaluation.internal_measure_ground_truth_assessment import InternalMeasureGroundTruthAssessment
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
