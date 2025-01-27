import pandas as pd
from hamcrest import *

from src.evaluation.interpretation_distance_metric_ranking import DistanceMetricInterpretation
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_ROOT_RESULTS_DIR, \
    TEST_IRREGULAR_P90_DATA_DIR, TEST_GENERATED_DATASETS_FILE_PATH

test_data_dir = TEST_IRREGULAR_P90_DATA_DIR
data_type = SyntheticDataType.non_normal_correlated
root_results_dir = TEST_ROOT_RESULTS_DIR
measures = [DistanceMeasures.l1_cor_dist, DistanceMeasures.l2_cor_dist, DistanceMeasures.log_frob_cor_dist]
overall_ds_name = "n2"
run_names = pd.read_csv(TEST_GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

inter = DistanceMetricInterpretation(run_names=run_names, overall_ds_name=overall_ds_name, data_type=data_type,
                                     data_dir=test_data_dir,
                                     root_results_dir=root_results_dir, measures=measures)


def test_loads_df_average_overall_criterion_per_run_df():
    overall_ranks = inter.average_rank_per_run

    assert_that(overall_ranks.shape, is_((len(run_names), len(measures))))


def test_loads_criteria_average_run_df():
    df = inter.criteria_average_run_df

    number_of_criteria = 6
    assert_that(df.shape, is_((number_of_criteria, len(measures))))

