from pathlib import Path

from hamcrest import *

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.use_case.wandb_run_ticc import TICCWandbUseCaseConfig, run_ticc_on_a_data_variant
from src.utils.configurations import DataCompleteness, get_algorithm_use_case_result_dir, get_data_dir
from src.utils.load_synthetic_data import SyntheticDataType, load_labels_file_for, SyntheticFileTypes
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR, TEST_ROOT_RESULTS_DIR


def test_wandb_logs_parameters_and_evaluates_ticc_run():
    config = TICCWandbUseCaseConfig()
    # put wandb into offline mode for testing
    config.wandb_mode = "offline"  # don't log test runs
    config.wandb_notes = "unit testing"

    # for test run on artificially small data for speed
    config.root_data_dir = TEST_DATA_DIR
    config.data_type = SyntheticDataType.normal_correlated
    config.completeness_level = DataCompleteness.irregular_p90
    config.number_of_clusters = 11  # we deleted the rest of data in the labels
    config.max_iter = 5  # will  speed up training
    config.root_results_dir = get_algorithm_use_case_result_dir(root_results_dir=TEST_ROOT_RESULTS_DIR,
                                                                algorithm_id='ticc-test')

    run_names = ['short-splendid-sunset-12', 'short-misty-forest-56']
    training_subject_name = run_names[1]
    save_results = True

    # run
    evaluates, wandb_summaries = run_ticc_on_a_data_variant(config=config, run_names=run_names,
                                                            training_subject_name=training_subject_name,
                                                            save_results=save_results)

    # assert test subject results
    eval_test = evaluates[run_names[0]]
    test_summary = wandb_summaries[run_names[0]]
    assert_that(test_summary["Has converged"], is_(False))
    assert_that(test_summary["Jaccard Index"], is_(eval_test.jaccard_index()))
    assert_that(test_summary["SWC"], is_(eval_test.silhouette_score()))
    assert_that(test_summary["DBI"], is_(eval_test.dbi()))
    assert_that(test_summary["Pattern Discovery "], is_(eval_test.pattern_discovery_percentage()))
    assert_that(test_summary["Pattern Specificity"], is_(eval_test.pattern_specificity_percentage()))
    assert_that(test_summary["Segmentation Ratio"], is_(eval_test.segmentation_ratio()))
    assert_that(test_summary["Segment Length Ratio"], is_(eval_test.segmentation_length_ratio()))
    assert_that(test_summary["Undiscovered Patterns"], is_(eval_test.pattern_not_discovered()))
    assert_that(test_summary["mean MAE TICC result - relaxed pattern"], is_(eval_test.mae_stats_mapped_resulting_patterns_relaxed()['mean']))
    assert_that(test_summary["mean MAE ground truth - relaxed pattern"], is_(eval_test.mae_stats_mapped_gt_patterns_relaxed()['mean']))
    assert_that(test_summary["n Clusters TICC"], is_(8))
    assert_that(test_summary["n Clusters ground truth"], is_(11))

    # assert training subject results
    eval_train = evaluates[training_subject_name]
    training_summary = wandb_summaries[training_subject_name]

    assert_that(training_summary["Has converged"], is_(False))
    assert_that(training_summary["Jaccard Index"], is_(eval_train.jaccard_index()))
    assert_that(training_summary["SWC"], is_(eval_train.silhouette_score()))
    assert_that(training_summary["DBI"], is_(eval_train.dbi()))
    assert_that(training_summary["Pattern Discovery "], is_(eval_train.pattern_discovery_percentage()))
    assert_that(training_summary["Pattern Specificity"], is_(eval_train.pattern_specificity_percentage()))
    assert_that(training_summary["Segmentation Ratio"], is_(eval_train.segmentation_ratio()))
    assert_that(training_summary["Segment Length Ratio"], is_(eval_train.segmentation_length_ratio()))
    assert_that(training_summary["Undiscovered Patterns"], is_(eval_train.pattern_not_discovered()))
    assert_that(training_summary["mean MAE TICC result - relaxed pattern"], is_(eval_train.mae_stats_mapped_resulting_patterns_relaxed()['mean']))
    assert_that(training_summary["mean MAE ground truth - relaxed pattern"], is_(eval_train.mae_stats_mapped_gt_patterns_relaxed()['mean']))
    assert_that(training_summary["n Clusters TICC"], is_(11))
    assert_that(training_summary["n Clusters ground truth"], is_(11))

    # assert we can load resulting labels files using standard method
    folder = get_data_dir(root_data_dir=config.root_results_dir,
                          extension_type=config.completeness_level)
    path_first_subject_result = Path(folder, run_names[0] + SyntheticFileTypes.labels)
    first_subject_labels_df = load_labels_file_for(path_first_subject_result)
    assert_that(first_subject_labels_df.shape, is_((27, 6)))
    assert_that(len(first_subject_labels_df[SyntheticDataSegmentCols.pattern_id].unique()), is_(8)) # didn't find 11 clusters
