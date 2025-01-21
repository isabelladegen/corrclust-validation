from hamcrest import *

from src.evaluation.distance_metric_evaluation import read_csv_of_raw_values_for_all_criteria, EvaluationCriteria
from src.evaluation.distance_metric_ranking import DistanceMetricRanking
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_ROOT_RESULTS_DIR, TEST_IRREGULAR_P90_DATA_DIR

test_data_dir = TEST_IRREGULAR_P90_DATA_DIR  # so we have much less data for speed
distance_measures = [DistanceMeasures.l2_cor_dist, DistanceMeasures.log_frob_cor_dist,
                     DistanceMeasures.foerstner_cor_dist]
data_type = SyntheticDataType.non_normal_correlated
base_results_dir = TEST_ROOT_RESULTS_DIR

# this is a modified df for testing
ds_name1 = "test-ranking"
raw1 = read_csv_of_raw_values_for_all_criteria(ds_name1, data_type, test_data_dir, base_results_dir)
ds_name2 = "test-ranking2"
raw2 = read_csv_of_raw_values_for_all_criteria(ds_name2, data_type, test_data_dir, base_results_dir)
raw_criteria_data = {ds_name1: raw1, ds_name2: raw2}
ranker = DistanceMetricRanking(raw_criteria_data, distance_measures)


def test_ranks_each_distance_metric_criteria():
    ranked = ranker.ranking_df_for_ds(ds_name1)

    # test rank 1 = best for each measure
    highest_rank = len(distance_measures)  # worst
    # lower raw value is better
    assert_that(ranked.loc[EvaluationCriteria.inter_i, DistanceMeasures.l2_cor_dist], is_(1))
    assert_that(ranked.loc[EvaluationCriteria.inter_i, DistanceMeasures.log_frob_cor_dist], is_(highest_rank))
    # Pass/Fail - averaged
    assert_that(ranked.loc[EvaluationCriteria.inter_ii, DistanceMeasures.l2_cor_dist], is_(1.5))
    assert_that(ranked.loc[EvaluationCriteria.inter_ii, DistanceMeasures.foerstner_cor_dist], is_(1.5))
    assert_that(ranked.loc[EvaluationCriteria.inter_ii, DistanceMeasures.log_frob_cor_dist], is_(highest_rank))
    # higher raw value is better
    assert_that(ranked.loc[EvaluationCriteria.inter_iii, DistanceMeasures.log_frob_cor_dist], is_(1))
    assert_that(ranked.loc[EvaluationCriteria.inter_iii, DistanceMeasures.l2_cor_dist], is_(highest_rank))
    assert_that(ranked.loc[EvaluationCriteria.disc_i, DistanceMeasures.l2_cor_dist], is_(1))
    assert_that(ranked.loc[EvaluationCriteria.disc_ii, DistanceMeasures.foerstner_cor_dist], is_(1))
    # tied outcome for highest rank
    assert_that(ranked.loc[EvaluationCriteria.disc_iii, DistanceMeasures.l2_cor_dist], is_(1))
    assert_that(ranked.loc[EvaluationCriteria.disc_iii, DistanceMeasures.log_frob_cor_dist], is_(highest_rank / 2 + 1))
    assert_that(ranked.loc[EvaluationCriteria.disc_iii, DistanceMeasures.foerstner_cor_dist], is_(highest_rank / 2 + 1))
    # lower is better
    assert_that(ranked.loc[EvaluationCriteria.stab_ii, DistanceMeasures.l2_cor_dist], is_(1))
    assert_that(ranked.loc[EvaluationCriteria.stab_ii, DistanceMeasures.foerstner_cor_dist], is_(highest_rank))


def test_calculates_average_rank_per_distance_level_and_ds():
    ranked = ranker.calculate_overall_rank()

    assert_that(ranked.shape, is_((2, len(distance_measures))))
    mean_rank = ranked.mean().round(3)
    assert_that(mean_rank.loc[DistanceMeasures.l2_cor_dist], is_(1.535))
    assert_that(mean_rank.loc[DistanceMeasures.log_frob_cor_dist], is_(2.286))
    assert_that(mean_rank.loc[DistanceMeasures.foerstner_cor_dist], is_(2.178))


