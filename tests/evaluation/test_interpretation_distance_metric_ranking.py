import pandas as pd
from hamcrest import *

from src.evaluation.distance_metric_assessment import DistanceMeasureCols
from src.evaluation.distance_metric_evaluation import EvaluationCriteria
from src.evaluation.interpretation_distance_metric_ranking import DistanceMetricInterpretation
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


def test_loads_criteria_average_run_df():
    df = inter.criteria_average_run_df

    assert_that(df.shape, is_((number_of_criteria, len(measures))))


def test_loads_all_criteria_level_ranks_for_each_run_into_one_df():
    df = inter.all_criteria_ranks_df
    n_runs = len(run_names)
    n_measures = len(measures)
    n_rows = n_runs * number_of_criteria * n_measures

    assert_that(df.shape[0], is_(n_rows))
    assert_that(df.columns,
                contains_exactly(RunInformationCols.ds_name, DistanceMeasureCols.criterion, DistanceMeasureCols.type,
                                 DistanceMeasureCols.rank))

    # select for one criteria
    fitlered = df[df[DistanceMeasureCols.criterion] == EvaluationCriteria.inter_i]
    assert_that(fitlered.shape[0], is_(n_runs * n_measures))
