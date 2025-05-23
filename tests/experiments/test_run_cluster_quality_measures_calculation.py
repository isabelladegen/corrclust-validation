import pandas as pd
from hamcrest import *

from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.experiments.run_cluster_quality_measures_calculation import read_clustering_quality_measures, \
    run_internal_measure_calculation_for_dataset
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_GENERATED_DATASETS_FILE_PATH, \
    TEST_DATA_DIR

test_data_dir = TEST_DATA_DIR
runs = pd.read_csv(TEST_GENERATED_DATASETS_FILE_PATH)['Name'].tolist()


# the commented out lines create test data for other tests!
def test_can_run_calculation_for_internal_measures_on_all_datasets(tmp_path):
    # run test_wandb_create_bad_partitions to create bad partitions if they don't exist for your configuration
    overall_ds_name = "test_stuff"
    # distance_measure = DistanceMeasures.l2_cor_dist
    # distance_measure = DistanceMeasures.l1_with_ref
    distance_measure = DistanceMeasures.l1_cor_dist
    data_type = SyntheticDataType.normal_correlated
    # test_results_dir = TEST_ROOT_RESULTS_DIR
    test_results_dir = str(tmp_path)
    run_internal_measure_calculation_for_dataset(overall_ds_name=overall_ds_name, run_names=runs,
                                                 distance_measure=distance_measure, data_type=data_type,
                                                 data_dir=test_data_dir, results_dir=test_results_dir,
                                                 internal_measures=[ClusteringQualityMeasures.silhouette_score,
                                                                    ClusteringQualityMeasures.pmb,
                                                                    ClusteringQualityMeasures.dbi,
                                                                    ClusteringQualityMeasures.vrc])

    # read the files from disk
    datasets = read_clustering_quality_measures(overall_ds_name=overall_ds_name, data_type=data_type,
                                                root_results_dir=test_results_dir, data_dir=test_data_dir,
                                                distance_measure=distance_measure, run_names=runs)

    assert_that(len(datasets), is_(2))
    assert_that(datasets[0][DescribeBadPartCols.name], has_item("misty-forest-56"))
    assert_that(datasets[1][DescribeBadPartCols.name], has_item("splendid-sunset-12"))
