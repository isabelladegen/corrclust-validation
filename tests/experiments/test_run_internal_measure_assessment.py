import os

import pandas as pd
from hamcrest import *

from src.evaluation.internal_measure_assessment import get_full_filename_for_results_csv, IAResultsCSV
from src.experiments.run_internal_measure_assessment import run_internal_measure_assessment_datasets
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import internal_measure_evaluation_dir_for
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_ROOT_RESULTS_DIR, TEST_DATA_DIR, \
    TEST_GENERATED_DATASETS_FILE_PATH

test_data_dir = TEST_DATA_DIR
run_names = pd.read_csv(TEST_GENERATED_DATASETS_FILE_PATH)['Name'].tolist()


def test_can_run_assessment_on_full_dataset_and_store_results_for_runs_with_all_clusters():
    # run test_wandb_create_bad_partitions to create bad partitions if they don't exist for your configuration
    overall_ds_name = "test_stuff"
    # distance_measure = DistanceMeasures.l1_with_ref
    # distance_measure = DistanceMeasures.l2_cor_dist
    distance_measure = DistanceMeasures.l1_cor_dist
    data_type = SyntheticDataType.normal_correlated
    test_results_dir = TEST_ROOT_RESULTS_DIR
    run_internal_measure_assessment_datasets(overall_ds_name=overall_ds_name, run_names=run_names,
                                             distance_measure=distance_measure, data_type=data_type,
                                             data_dir=test_data_dir, results_dir=test_results_dir,
                                             internal_measures=[ClusteringQualityMeasures.silhouette_score,
                                                                ClusteringQualityMeasures.pmb])

    # check if the files have been created
    results_folder = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type=data_type,
        results_dir=test_results_dir, data_dir=test_data_dir,
        distance_measure=distance_measure)

    # IA assessment results
    assert_that(os.path.exists(get_full_filename_for_results_csv(results_folder, IAResultsCSV.correlation_summary)))
    assert_that(os.path.exists(
        get_full_filename_for_results_csv(results_folder, IAResultsCSV.effect_size_difference_worst_best)))
    assert_that(os.path.exists(
        get_full_filename_for_results_csv(results_folder, IAResultsCSV.descriptive_statistics_measure_summary)))
    assert_that(os.path.exists(
        get_full_filename_for_results_csv(results_folder, IAResultsCSV.ci_of_differences_between_measures)))
