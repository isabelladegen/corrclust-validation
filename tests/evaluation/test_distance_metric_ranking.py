from hamcrest import *

from src.evaluation.distance_metric_evaluation import read_csv_of_raw_values_for_all_criteria, EvaluationCriteria
from src.evaluation.distance_metric_ranking import DistanceMetricRanking
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_ROOT_RESULTS_DIR, TEST_IRREGULAR_P90_DATA_DIR

a_ds_name = "misty-forest-56"
test_data_dir = TEST_IRREGULAR_P90_DATA_DIR  # so we have much less data for speed
distance_measures = [DistanceMeasures.l2_cor_dist, DistanceMeasures.log_frob_cor_dist,
                     DistanceMeasures.foerstner_cor_dist]
data_type = SyntheticDataType.non_normal_correlated
base_results_dir = TEST_ROOT_RESULTS_DIR


def test_ranks_each_distance_metric_criteria():
    ds_name = "test-ranking"
    # this is a modified df for testing
    raw1 = read_csv_of_raw_values_for_all_criteria(ds_name, data_type, test_data_dir, base_results_dir)
    raw_criteria_data = {a_ds_name: raw1}
    ranker = DistanceMetricRanking(raw_criteria_data, distance_measures)
    ranked = ranker.ranking_df_for_ds(a_ds_name)

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
