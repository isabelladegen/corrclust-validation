from hamcrest import *
import pandas.testing as tm

from src.evaluation.distance_metric_evaluation import read_csv_of_raw_values_for_all_criteria, EvaluationCriteria
from src.evaluation.distance_metric_ranking import DistanceMetricRanking, read_csv_of_ranks_for_all_criteria, \
    read_csv_of_overall_rank_per_dataset, RankingStats, read_csv_of_average_criteria_across_datasets
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
    # lower raw value is better
    assert_that(ranked.loc[EvaluationCriteria.inter_i, DistanceMeasures.l2_cor_dist], is_(1))
    assert_that(ranked.loc[EvaluationCriteria.inter_i, DistanceMeasures.log_frob_cor_dist], is_(3))
    # Pass/Fail - dense
    assert_that(ranked.loc[EvaluationCriteria.inter_ii, DistanceMeasures.l2_cor_dist], is_(1))
    assert_that(ranked.loc[EvaluationCriteria.inter_ii, DistanceMeasures.foerstner_cor_dist], is_(1))
    assert_that(ranked.loc[EvaluationCriteria.inter_ii, DistanceMeasures.log_frob_cor_dist], is_(2))
    # higher raw value is better
    assert_that(ranked.loc[EvaluationCriteria.inter_iii, DistanceMeasures.log_frob_cor_dist], is_(1))
    assert_that(ranked.loc[EvaluationCriteria.inter_iii, DistanceMeasures.l2_cor_dist], is_(3))
    assert_that(ranked.loc[EvaluationCriteria.disc_i, DistanceMeasures.l2_cor_dist], is_(1))
    assert_that(ranked.loc[EvaluationCriteria.disc_ii, DistanceMeasures.foerstner_cor_dist], is_(1))
    # tied outcome for highest rank
    assert_that(ranked.loc[EvaluationCriteria.disc_iii, DistanceMeasures.l2_cor_dist], is_(1))
    assert_that(ranked.loc[EvaluationCriteria.disc_iii, DistanceMeasures.log_frob_cor_dist], is_(2))
    assert_that(ranked.loc[EvaluationCriteria.disc_iii, DistanceMeasures.foerstner_cor_dist], is_(2))


def test_calculates_average_rank_per_distance_level_and_ds():
    ranked = ranker.calculate_overall_rank()

    assert_that(ranked.shape, is_((2, len(distance_measures))))
    mean_rank = ranked.mean().round(3)
    assert_that(mean_rank.loc[DistanceMeasures.l2_cor_dist], is_(1.5))
    assert_that(mean_rank.loc[DistanceMeasures.log_frob_cor_dist], is_(2.167))
    assert_that(mean_rank.loc[DistanceMeasures.foerstner_cor_dist], is_(2))


def test_calculates_average_rank_per_criteria_and_distance_measures_across_runs():
    ranked = ranker.calculate_criteria_level_average_rank()

    # row for each criterion, column for each distance measure
    assert_that(ranked.shape, is_((6, len(distance_measures) + 1)))
    assert_that(ranked.loc[EvaluationCriteria.inter_i, DistanceMeasures.l2_cor_dist], is_(1))
    assert_that(ranked.loc[EvaluationCriteria.inter_i, DistanceMeasures.log_frob_cor_dist], is_(3))
    assert_that(ranked.loc[EvaluationCriteria.inter_i, DistanceMeasures.foerstner_cor_dist], is_(2))
    assert_that(ranked.loc[EvaluationCriteria.inter_ii, DistanceMeasures.l2_cor_dist], is_(1))
    assert_that(ranked.loc[EvaluationCriteria.inter_iii, DistanceMeasures.l2_cor_dist], is_(3))
    assert_that(ranked.loc[EvaluationCriteria.disc_i, DistanceMeasures.l2_cor_dist], is_(1))
    assert_that(ranked.loc[EvaluationCriteria.disc_ii, DistanceMeasures.l2_cor_dist], is_(2))
    assert_that(ranked.loc[EvaluationCriteria.disc_iii, DistanceMeasures.l2_cor_dist], is_(1))

    # check we calculated the best measure for each row
    assert_that(ranked.loc[EvaluationCriteria.inter_i, RankingStats.best], is_(DistanceMeasures.l2_cor_dist))
    assert_that(ranked.loc[EvaluationCriteria.inter_ii, RankingStats.best], is_(DistanceMeasures.l2_cor_dist))
    assert_that(ranked.loc[EvaluationCriteria.inter_iii, RankingStats.best], is_(DistanceMeasures.log_frob_cor_dist))
    assert_that(ranked.loc[EvaluationCriteria.disc_i, RankingStats.best], is_(DistanceMeasures.l2_cor_dist))
    assert_that(ranked.loc[EvaluationCriteria.disc_ii, RankingStats.best], is_(DistanceMeasures.foerstner_cor_dist))
    assert_that(ranked.loc[EvaluationCriteria.disc_iii, RankingStats.best], is_(DistanceMeasures.l2_cor_dist))


def test_saves_results_if_result_dir_given(tmp_path):
    # base_results_dir = TEST_ROOT_RESULTS_DIR
    base_results_dir = str(tmp_path)
    ds_1_criteria_rank = ranker.ranking_df_for_ds(ds_name1)  # not saving here just reading so can compare
    overall_ds_name = "test"
    overall_rank = ranker.calculate_overall_rank(overall_ds_name=overall_ds_name, root_results_dir=base_results_dir,
                                                 data_type=data_type, data_dir=test_data_dir)

    average_criteria_rank = ranker.calculate_criteria_level_average_rank(overall_ds_name=overall_ds_name,
                                                                         root_results_dir=base_results_dir,
                                                                         data_type=data_type, data_dir=test_data_dir)

    # read csv
    loaded_ds1_criteria_rank = read_csv_of_ranks_for_all_criteria(ds_name1, data_type, test_data_dir, base_results_dir)
    loaded_overall_rank = read_csv_of_overall_rank_per_dataset(overall_ds_name, data_type, test_data_dir,
                                                               base_results_dir)
    loaded_average_criteria_rank = read_csv_of_average_criteria_across_datasets(overall_ds_name, data_type,
                                                                                test_data_dir, base_results_dir)

    # check calculated and reloaded dfs are the same
    tm.assert_frame_equal(loaded_ds1_criteria_rank, ds_1_criteria_rank)
    tm.assert_frame_equal(loaded_overall_rank, overall_rank)
    tm.assert_frame_equal(loaded_average_criteria_rank, average_criteria_rank)
