import pandas as pd
from hamcrest import *

from src.evaluation.distance_metric_evaluation import read_csv_of_raw_values_for_all_criteria
from src.evaluation.run_distance_evaluation_raw_criteria import run_distance_evaluation_raw_criteria_for_ds
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_IRREGULAR_P30_DATA_DIR, TEST_IRREGULAR_P90_DATA_DIR, \
    TEST_GENERATED_DATASETS_FILE_PATH


def test_calculates_the_raw_criteria_for_the_specified_runs(tmp_path):
    root_result_dir = str(tmp_path)
    dataset_types = [SyntheticDataType.normal_correlated, SyntheticDataType.rs_1min]
    data_dirs = [TEST_IRREGULAR_P30_DATA_DIR,
                 TEST_IRREGULAR_P90_DATA_DIR]
    run_names = pd.read_csv(TEST_GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    distance_measures = [DistanceMeasures.l3_cor_dist, DistanceMeasures.l3_with_ref]

    run_distance_evaluation_raw_criteria_for_ds(data_dirs=data_dirs, dataset_types=dataset_types, run_names=run_names,
                                                root_result_dir=root_result_dir, distance_measures=distance_measures)

    # read csv that have been saved
    for data_dir in data_dirs:
        for data_type in dataset_types:
            for run_name in run_names:
                raw_criteria = read_csv_of_raw_values_for_all_criteria(run_name=run_name, data_type=data_type,
                                                                       data_dir=data_dir,
                                                                       base_results_dir=root_result_dir)
                # calculated all distance measures
                assert_that(raw_criteria.columns.tolist(), contains_exactly(*distance_measures))
                # all criteria for all distance measures have been calculated
                assert_that(raw_criteria.isna().sum().sum(), is_(0))
