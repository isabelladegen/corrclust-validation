from os import path

import pandas as pd
from hamcrest import *

from src.utils.configurations import internal_measure_evaluation_dir_for, \
    get_image_name_based_on_data_dir_and_data_type, PARTITIONS_QUALITY_DESCRIPTION
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.run_partition_visualisations import partition_visualisation
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR, \
    TEST_GENERATED_DATASETS_FILE_PATH, TEST_ROOT_RESULTS_DIR

backend = Backends.none.value


def test_runs_partition_visualisations_for_multiple_data_variants():
    data_dirs = [TEST_DATA_DIR]
    dataset_types = [SyntheticDataType.normal_correlated]
    overall_ds_name = "test_stuff"
    run_names = pd.read_csv(TEST_GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    root_results_dir = TEST_ROOT_RESULTS_DIR
    distance_measure = DistanceMeasures.l1_with_ref
    partition_visualisation(data_dirs, dataset_types, run_names, root_results_dir, distance_measure, overall_ds_name,
                            backend, save_fig=True)

    saved_img_folder = internal_measure_evaluation_dir_for(overall_dataset_name=overall_ds_name,
                                                           data_type=dataset_types[0], results_dir=root_results_dir,
                                                           data_dir=data_dirs[0], distance_measure=distance_measure)
    saved_img_folder = path.join(saved_img_folder, "images")
    image_name = get_image_name_based_on_data_dir_and_data_type(PARTITIONS_QUALITY_DESCRIPTION, data_dirs[0],
                                                                dataset_types[0])
    assert_that(path.exists(path.join(saved_img_folder, image_name)))
