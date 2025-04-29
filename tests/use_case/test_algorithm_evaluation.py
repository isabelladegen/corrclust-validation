from os import path
from pathlib import Path

import pandas as pd
import pytest
from hamcrest import *

from src.use_case.algorithm_evaluation import AlgorithmEvaluation, EvalMappingCols
from src.use_case.ticc.TICC_solver import TICC
from src.use_case.wandb_run_ticc import TICCDefaultSettings
from src.utils.configurations import SyntheticDataVariates, ROOT_DIR
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data, load_labels_file_for
from tests.test_utils.configurations_for_testing import TEST_GENERATED_DATASETS_FILE_PATH, \
    TEST_IRREGULAR_P90_DATA_DIR

data_dir = TEST_IRREGULAR_P90_DATA_DIR
run_name = pd.read_csv(TEST_GENERATED_DATASETS_FILE_PATH)['Name'].tolist()[0]
data_type = SyntheticDataType.normal_correlated
data, gt_labels = load_synthetic_data(run_name, data_type, data_dir)
root_path_to_result = path.join(ROOT_DIR, 'tests', 'use_case', 'ticc_test_result_labels')
result_1 = Path(path.join(root_path_to_result, 'ds_nn_p90_untuned_ticc_result_23_clusters.parquet'))
result_2 = Path(path.join(root_path_to_result, 'nn_p90_untuned_ticc_result_8clusters.parquet'))
result_3 = Path(path.join(root_path_to_result, 'nn_p90_untuned_ticc_result_23_clusters.parquet'))
result_labels_df_1 = load_labels_file_for(Path(result_1))
result_labels_df_2 = load_labels_file_for(Path(result_2))
result_labels_df_3 = load_labels_file_for(Path(result_3))

eval1 = AlgorithmEvaluation(result_labels_df_1, gt_labels, data, run_name, data_dir, data_type)
eval2 = AlgorithmEvaluation(result_labels_df_2, gt_labels, data, run_name, data_dir, data_type)
eval3 = AlgorithmEvaluation(result_labels_df_3, gt_labels, data, run_name, data_dir, data_type)


def create_a_ticc_result(name: str):
    """
    Method used to create a test result from TICC to test our evaluation
    """
    ticc_settings = TICCDefaultSettings()
    ticc = TICC(window_size=ticc_settings.window_size, number_of_clusters=ticc_settings.number_of_clusters,
                lambda_parameter=ticc_settings.lambda_var, beta=ticc_settings.switch_penalty,
                max_iters=ticc_settings.max_iter,
                threshold=ticc_settings.threshold,
                biased=ticc_settings.biased,
                allow_zero_cluster_inbetween=ticc_settings.allow_zero_cluster_inbetween,
                do_training_split=ticc_settings.do_training_split,
                cluster_reassignment=ticc_settings.cluster_reassignment,
                keep_track_of_assignments=ticc_settings.keep_track_of_assignments,
                backend=ticc_settings.backend)

    result = ticc.fit(data=data[SyntheticDataVariates.columns()].to_numpy(),
                      use_gmm_initialisation=ticc_settings.use_gmm_initialisation,
                      reassign_points_to_zero_clusters=ticc_settings.reassign_points_to_zero_clusters)
    result.print_info()
    ticc_result = result.to_labels_df()
    ticc_result.to_csv(name)


@pytest.mark.skip(reason="takes a long time and we need it just to create test data so no need to run")
def test_calculates_jaccard_index_from_ticc_result():
    csv_name = 'ds_nn_p90_untuned_ticc_result_23_clusters.csv'
    create_a_ticc_result(csv_name)


def test_calculates_segmentation_ratio():
    assert_that(eval1.segmentation_ratio(), is_(0.89))  # algorithm undersamples
    assert_that(eval2.segmentation_ratio(), is_(0.68))  # algorithm undersamples by 1/3
    assert_that(eval3.segmentation_ratio(), is_(0.82))  # algorithm undersamples


def test_calculates_segment_length_ratio():
    assert_that(eval1.segmentation_length_ratio(), is_(1))  # algorithm same median segment length
    assert_that(eval1.segmentation_length_ratio(stats='max'), is_(1.319))  # algorithm max segments 30% longer than gt
    assert_that(eval1.segmentation_length_ratio(stats='mean'), is_(1.124))  # algorithm makes longer segments
    assert_that(eval2.segmentation_length_ratio(), is_(1.033))  # algorithm similar median segment length
    assert_that(eval3.segmentation_length_ratio(), is_(1.015))  # algorithm similar median segment length


def test_map_resulting_clusters_to_ground_truth():
    map1 = eval1.map_clusters()

    # test all values for row 1 for map1
    assert_that(map1.shape[0], is_(23))
    row1 = map1.iloc[0]
    assert_that(row1[EvalMappingCols.result_cluster_id], is_(1))
    assert_that(row1[EvalMappingCols.distance], is_(0.013))
    assert_that(row1[EvalMappingCols.result_mean_cluster_cor], contains_exactly(*[-0.001, -0.005, 0.011]))
    assert_that(row1[EvalMappingCols.closest_gt_ids], contains_exactly(0))
    assert_that(row1[EvalMappingCols.closest_gt_mean_cors][0], contains_exactly(*[-0.008, -0.009, 0.013]))
    assert_that(row1[EvalMappingCols.mae_result_and_relaxed_pattern][0], is_(0.006))
    assert_that(row1[EvalMappingCols.mae_gt_and_relaxed_pattern][0], is_(0.01))
    assert_that(row1[EvalMappingCols.result_mean_cluster_cor_within_tolerance_of_gt][0], is_(True))
    assert_that(row1[EvalMappingCols.gt_within_tolerance_of_relaxed_pattern][0], is_(True))

    map2 = eval2.map_clusters()
    assert_that(map2.shape[0], is_(8))


def test_calculates_pattern_discovery_rate():
    assert_that(eval1.pattern_not_discovered(), contains_exactly(3, 6, 18, 20))
    assert_that(eval1.pattern_discovery_percentage(), is_(82.609))

    assert_that(eval2.patterns_discovered(), contains_exactly(2, 6, 10, 12, 13, 15, 18, 19))
    assert_that(eval2.pattern_discovery_percentage(), is_(34.783))

    assert_that(eval3.pattern_not_discovered(), contains_exactly(3, 6, 9, 12))
    assert_that(eval3.pattern_discovery_percentage(), is_(82.609))


def test_calculates_pattern_specificity():
    assert_that(eval1.pattern_specificity_percentage(), is_(69.565))
    assert_that(eval2.pattern_specificity_percentage(), is_(100.0))
    assert_that(eval3.pattern_specificity_percentage(), is_(65.217))


def test_calculates_silhouette_score():
    # all of these result show that the clustering results are poor
    assert_that(eval1.silhouette_score(), is_(0.598))
    assert_that(eval2.silhouette_score(), is_(0.373))
    assert_that(eval3.silhouette_score(), is_(0.544))


def test_calculates_dbi():
    # all of these results indicate poor clustering results (dbi should be < 0.05)
    assert_that(eval1.dbi(), is_(1.749))
    assert_that(eval2.dbi(), is_(0.98))
    assert_that(eval3.dbi(), is_(0.728))


def test_calculates_jaccard_index():
    assert_that(eval1.jaccard_index(), is_(0.829))
    assert_that(eval2.jaccard_index(), is_(0.352))
    assert_that(eval3.jaccard_index(), is_(0.835))

def test_calculates_mae_stats_resulting_patterns_relaxed_patterns_and_the_same_for_gt():
    assert_that(eval1.mae_stats_mapped_resulting_patterns_relaxed()['mean'], is_(0.045))
    # for the patterns that did get matched gt has much lower mean MAE than TICC had
    assert_that(eval1.mae_stats_mapped_gt_patterns_relaxed()['mean'], is_(0.02))
    assert_that(eval2.mae_stats_mapped_resulting_patterns_relaxed()['mean'], is_(0.21))
    assert_that(eval2.mae_stats_mapped_gt_patterns_relaxed()['mean'], is_(0.02))
    assert_that(eval3.mae_stats_mapped_resulting_patterns_relaxed()['mean'], is_(0.054))
    assert_that(eval3.mae_stats_mapped_gt_patterns_relaxed()['mean'], is_(0.023))

