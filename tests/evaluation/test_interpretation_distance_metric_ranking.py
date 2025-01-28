import pandas as pd
from hamcrest import *

from src.evaluation.distance_metric_assessment import DistanceMeasureCols
from src.evaluation.distance_metric_evaluation import EvaluationCriteria
from src.evaluation.interpretation_distance_metric_ranking import DistanceMetricInterpretation, DistanceInterpretation, \
    read_top_bottom_distance_measure_result
from src.utils.configurations import RunInformationCols
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_ROOT_RESULTS_DIR, \
    TEST_IRREGULAR_P90_DATA_DIR, TEST_GENERATED_DATASETS_FILE_PATH

test_data_dir = TEST_IRREGULAR_P90_DATA_DIR
data_type = SyntheticDataType.non_normal_correlated
root_results_dir = TEST_ROOT_RESULTS_DIR
measures = [DistanceMeasures.l1_cor_dist, DistanceMeasures.l2_cor_dist, DistanceMeasures.log_frob_cor_dist]
overall_ds_name = "n2"
number_of_criteria = 6
run_names = pd.read_csv(TEST_GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

inter = DistanceMetricInterpretation(run_names=run_names, overall_ds_name=overall_ds_name, data_type=data_type,
                                     data_dir=test_data_dir,
                                     root_results_dir=root_results_dir, measures=measures)


def test_loads_df_average_overall_criterion_per_run_df():
    overall_ranks = inter.average_rank_per_run

    assert_that(overall_ranks.shape, is_((len(run_names), len(measures))))


def test_calculates_statistics_df_for_raw_ranks_across_all_runs():
    df = inter.stats_for_raw_criteria_ranks_across_all_runs()

    assert_that(df.loc['mean', DistanceMeasures.l1_cor_dist], is_(1.833))
    assert_that(df.loc['std', DistanceMeasures.l1_cor_dist], is_(0.937))


def test_calculates_statistics_df_for_average_ranks():
    df = inter.stats_for_average_ranks_across_all_runs()

    assert_that(df.loc['mean', DistanceMeasures.l1_cor_dist], is_(1.833))
    assert_that(df.loc['std', DistanceMeasures.l1_cor_dist], is_(0.0))


def test_loads_criteria_average_run_df():
    df = inter.criteria_average_run_df

    assert_that(df.shape, is_((number_of_criteria, len(measures))))


def test_loads_all_criteria_raw_ranks_for_each_run_into_one_df():
    df = inter.raw_criteria_ranks_df
    n_runs = len(run_names)
    n_measures = len(measures)
    n_rows = n_runs * number_of_criteria * n_measures

    assert_that(df.shape[0], is_(n_rows))
    assert_that(df.columns,
                contains_exactly(RunInformationCols.ds_name, DistanceMeasureCols.criterion, DistanceMeasureCols.type,
                                 DistanceMeasureCols.rank))

    # select for one criterion
    filtered = df[df[DistanceMeasureCols.criterion] == EvaluationCriteria.inter_i]
    assert_that(filtered.shape[0], is_(n_runs * n_measures))


def test_calculates_statistics_df_per_criterion():
    results = inter.stats_per_criterion_raw_ranks()

    df = results[EvaluationCriteria.inter_i]

    assert_that(df.loc['mean', DistanceMeasures.l1_cor_dist], is_(2))
    assert_that(df.loc['std', DistanceMeasures.l1_cor_dist], is_(0))


def test_returns_the_average_top_x_bottom_x_distance_measure_by_various_statistics():
    x = 2
    df = inter.top_and_bottom_x_distance_measures_ranks(x=x, save_results=True)

    assert_that(df.shape[1], is_(16))  # column for top and bottom avg and raw and all 6 criterion
    assert_that(df.loc['mean', DistanceInterpretation.raw_top_rank], is_('L1, L2'))

    # read results from disk
    read_top_bottom_distance_measure_result(x=x, overall_ds_name=overall_ds_name, data_type=data_type,
                                            base_results_dir=root_results_dir, data_dir=test_data_dir)
