import pandas as pd
from hamcrest import *

from src.evaluation.distance_metric_ranking import read_csv_of_ranks_for_all_criteria, \
    read_csv_of_overall_rank_per_dataset
from src.evaluation.run_distance_distance_metric_ranking import run_ranking_for
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
    distance_measures = [DistanceMeasures.l3_with_ref, DistanceMeasures.l10_cor_dist]
    overall_ds_name = "run-test"
    # setup: given we work with temp directories, we need to first create the criteria, in reality do this
    # separately so we can change the ranking independently from having to recalculate the criteria
    run_distance_evaluation_raw_criteria_for_ds(data_dirs=data_dirs, dataset_types=dataset_types, run_names=run_names,
                                                root_result_dir=root_result_dir, distance_measures=distance_measures)

    # do the ranking
    run_ranking_for(data_dirs=data_dirs, dataset_types=dataset_types, run_names=run_names,
                    root_result_dir=root_result_dir, distance_measures=distance_measures,
                    overall_ds_name=overall_ds_name)

    # check the ranking
    for data_dir in data_dirs:
        for data_type in dataset_types:
            for run_name in run_names:
                # check ranking for each run_name
                ranked_criteria = read_csv_of_ranks_for_all_criteria(run_name, data_type, data_dir, root_result_dir)
                # calculated all distance measures
                assert_that(ranked_criteria.columns.tolist(), contains_exactly(*distance_measures))
                # all criteria for all distance measures have been calculated
                assert_that(ranked_criteria.isna().sum().sum(), is_(0))

            overall_rank = read_csv_of_overall_rank_per_dataset(overall_ds_name, data_type, data_dir, root_result_dir)
            # calculated all distance measures
            assert_that(overall_rank.columns.tolist(), contains_exactly(*distance_measures))
            # all criteria for all distance measures have been calculated
            assert_that(overall_rank.isna().sum().sum(), is_(0))
            # calculated all runs specified
            assert_that(overall_rank.shape[0], is_(len(run_names)))
