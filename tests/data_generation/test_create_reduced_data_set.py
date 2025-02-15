import pandas as pd
from hamcrest import *

from src.data_generation.create_reduced_datasets import CreateReducedDatasets
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR, TEST_GENERATED_DATASETS_FILE_PATH

test_data_dir = TEST_DATA_DIR
run_names = pd.read_csv(TEST_GENERATED_DATASETS_FILE_PATH)['Name'].tolist()


def test_keeps_n_clusters_at_random():
    drop_n_clusters = [11, 17]
    drop_n_segments = [50, 75]
    seed = 100
    rd = CreateReducedDatasets(run_names=run_names, data_type=SyntheticDataType.normal_correlated,
                               data_dir=test_data_dir, drop_n_clusters=drop_n_clusters,
                               drop_n_segments=drop_n_segments, base_seed=seed)

    # check number of resulting datasets
    assert_that(len(rd.selected_patterns[drop_n_clusters[0]][run_names[0]]), is_(12))
    assert_that(len(rd.selected_patterns[drop_n_clusters[1]][run_names[0]]), is_(6))
    assert_that(len(rd.selected_segments[drop_n_segments[0]][run_names[0]]), is_(50))
    assert_that(len(rd.selected_segments[drop_n_segments[1]][run_names[0]]), is_(25))
    assert_that(len(rd.reduced_labels_patterns), is_(2))
    assert_that(len(rd.reduced_data_patterns), is_(2))
    assert_that(len(rd.reduced_labels_segments), is_(2))
    assert_that(len(rd.reduced_data_segments), is_(2))

    # check that the number of patterns in the labels file is correct
    pattern_ids = rd.reduced_labels_patterns[drop_n_clusters[0]][run_names[0]][
        SyntheticDataSegmentCols.pattern_id].unique()
    assert_that(len(pattern_ids), is_(12))
    pattern_ids = rd.reduced_labels_patterns[drop_n_clusters[1]][run_names[0]][
        SyntheticDataSegmentCols.pattern_id].unique()
    assert_that(len(pattern_ids), is_(6))
    assert_that(rd.reduced_labels_segments[drop_n_segments[0]][run_names[0]].shape[0], is_(50))
    assert_that(rd.reduced_labels_segments[drop_n_segments[1]][run_names[0]].shape[0], is_(25))
