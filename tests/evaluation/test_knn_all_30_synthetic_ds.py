from hamcrest import *

from src.evaluation.distance_metric_assessment import DistanceMeasures
from src.evaluation.knn_all_synthetic_datasets import KnnAllSyntheticDatasets, AssessSynthCols
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_GENERATED_DATASETS_FILE_PATH_1, TEST_ROOT_RESULTS_DIR, \
    TEST_DATA_DIR, TEST_GENERATED_DATASETS_FILE_PATH, TEST_IRREGULAR_P30_DATA_DIR, TEST_IRREGULAR_P90_DATA_DIR

overall_ds_name = "test-1"
run_csv = TEST_GENERATED_DATASETS_FILE_PATH_1
root_results_dir = TEST_ROOT_RESULTS_DIR
data_dir = TEST_DATA_DIR


def test_calculates_scores_for_log_frob():
    measures = [DistanceMeasures.log_frob_cor_dist]
    knn = KnnAllSyntheticDatasets(measures, overall_ds_name=overall_ds_name, save_confusion_matrix=True,
                                  run_csv=run_csv,
                                  root_results_dir=root_results_dir, data_dir=data_dir)
    scores = knn.scores_for_all_measures
    assert_that(scores.shape[0], is_(len(measures) * 2))
    assert_that(scores[AssessSynthCols.f1][0], is_(0.44))


def test_calculates_scores_for_all_measures_on_aid_like_ds():
    measures = [DistanceMeasures.foerstner_cor_dist]
    knn = KnnAllSyntheticDatasets(measures, save_confusion_matrix=False, run_csv=run_csv,
                                  root_results_dir=root_results_dir, data_dir=data_dir)
    scores = knn.scores_for_all_measures
    assert_that(scores.shape[0], is_(len(measures) * 2))
    assert_that(scores[AssessSynthCols.f1][0], is_(0.02))


def test_calculates_scores_some_measures_downsampled_and_irregular_datasets():
    measures = [DistanceMeasures.l1_with_ref, DistanceMeasures.linf_cor_dist]

    # don't save cm
    knn = KnnAllSyntheticDatasets(measures, overall_ds_name='test_misty', save_confusion_matrix=True,
                                  data_type=SyntheticDataType.rs_1min,
                                  run_csv=TEST_GENERATED_DATASETS_FILE_PATH, root_results_dir=root_results_dir,
                                  data_dir=TEST_IRREGULAR_P30_DATA_DIR)
    scores = knn.scores_for_all_measures
    assert_that(scores.shape[0], is_(len(measures) * 2))
    assert_that(scores[AssessSynthCols.f1][0], is_(0.00))


def test_calculates_scores_for_all_measures_normaldistribution_correlated_datasets():
    measures = [DistanceMeasures.l2_cor_dist]

    knn = KnnAllSyntheticDatasets(measures, overall_ds_name=overall_ds_name, save_confusion_matrix=False,
                                  data_type=SyntheticDataType.normal_correlated,
                                  run_csv=run_csv, root_results_dir=root_results_dir, data_dir=data_dir)
    scores = knn.scores_for_all_measures
    assert_that(scores.shape[0], is_(len(measures) * 2))
    assert_that(scores[AssessSynthCols.f1][0], is_(1))


def test_calculates_scores_for_all_measures_0_10_min_max_scaled_datasets():
    measures = [DistanceMeasures.l1_with_ref]

    knn = KnnAllSyntheticDatasets(measures, overall_ds_name=overall_ds_name, save_confusion_matrix=False,
                                  value_range=(0., 10.), run_csv=run_csv,
                                  root_results_dir=root_results_dir, data_dir=data_dir)
    scores = knn.scores_for_all_measures
    assert_that(scores.shape[0], is_(len(measures) * 2))
    assert_that(scores[AssessSynthCols.f1][0], is_(1))


def test_calculates_scores_for_all_measures_irregular_p_0_3_datasets():
    measures = [DistanceMeasures.linf_with_ref, DistanceMeasures.linf_cor_dist]

    # don't save cm
    knn = KnnAllSyntheticDatasets(measures, overall_ds_name='test_misty', save_confusion_matrix=False,
                                  data_type=SyntheticDataType.normal_correlated,
                                  run_csv=TEST_GENERATED_DATASETS_FILE_PATH, root_results_dir=root_results_dir,
                                  data_dir=TEST_IRREGULAR_P30_DATA_DIR)
    scores = knn.scores_for_all_measures
    assert_that(scores.shape[0], is_(len(measures) * 2))
    assert_that(scores[AssessSynthCols.f1][0], is_(1))


def test_calculates_scores_for_all_measures_irregular_p_0_9_datasets():
    measures = [DistanceMeasures.l1_cor_dist, DistanceMeasures.l2_with_ref]

    knn = KnnAllSyntheticDatasets(measures, overall_ds_name='test_misty', save_confusion_matrix=False,
                                  data_type=SyntheticDataType.non_normal_correlated,
                                  run_csv=TEST_GENERATED_DATASETS_FILE_PATH, root_results_dir=root_results_dir,
                                  data_dir=TEST_IRREGULAR_P90_DATA_DIR)
    scores = knn.scores_for_all_measures
    assert_that(scores.shape[0], is_(len(measures) * 2))
    assert_that(scores[AssessSynthCols.f1][0], is_(1))
