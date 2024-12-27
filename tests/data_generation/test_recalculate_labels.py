from pathlib import Path

import pytest

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols, \
    recalculate_labels_df_from_data
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.utils.configurations import dir_for_data_type, bad_partition_dir_for_data_type
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data, SyntheticFileTypes, \
    load_synthetic_data_and_labels_for_bad_partitions
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR

test_data_dir = TEST_DATA_DIR


@pytest.mark.skip(reason="this is a once off calculation to bring old labels files into new format")
def test_this_is_temporary_to_create_correct_labels_files():
    run_name = "misty-forest-56"
    # run_name = "splendid-sunset-12"

    data_type = SyntheticDataType.non_normal_correlated
    data_df, labels_df = load_synthetic_data(run_name, data_type=data_type, data_dir=test_data_dir)
    # drop columns:
    keep_columns = [SyntheticDataSegmentCols.segment_id, SyntheticDataSegmentCols.start_idx,
                    SyntheticDataSegmentCols.end_idx, SyntheticDataSegmentCols.length,
                    SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.correlation_to_model,
                    SyntheticDataSegmentCols.actual_correlation, SyntheticDataSegmentCols.actual_within_tolerance]

    recalculated_labels_df = recalculate_labels_df_from_data(data_df, labels_df[keep_columns])

    labels_file = SyntheticFileTypes.labels
    file_dir = dir_for_data_type(data_type, test_data_dir)
    labels_file_name = Path(file_dir, run_name + labels_file)
    recalculated_labels_df.to_csv(labels_file_name)


@pytest.mark.skip(reason="this is a once off calculation to bring old labels files into new format")
def test_this_is_temporary_to_create_correct_labels_files_for_bad_partitions():
    run_name = "misty-forest-56"
    # run_name = "splendid-sunset-12"

    data_type = SyntheticDataType.non_normal_correlated
    data, gt_label, partitions = load_synthetic_data_and_labels_for_bad_partitions(run_name,
                                                                                   data_type,
                                                                                   test_data_dir)
    bad_partition_dir = bad_partition_dir_for_data_type(data_type, test_data_dir)

    # drop columns:
    keep_columns = [SyntheticDataSegmentCols.segment_id, SyntheticDataSegmentCols.start_idx,
                    SyntheticDataSegmentCols.end_idx, SyntheticDataSegmentCols.length,
                    SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.correlation_to_model,
                    SyntheticDataSegmentCols.actual_correlation, SyntheticDataSegmentCols.actual_within_tolerance]

    # dictionary with key being pattern_id and value being the ideal correlations
    patterns_to_model = ModelCorrelationPatterns().ideal_correlations()

    for file_name, p_label in partitions.items():
        # to make them consistent with other csv where we save the index as a unnamed column
        p_label.reset_index(inplace=True)

        # update pattern to model to actual pattern id
        p_label[SyntheticDataSegmentCols.correlation_to_model] = p_label[SyntheticDataSegmentCols.pattern_id].map(
            patterns_to_model)

        # now recalculate
        recalculated_labels_df = recalculate_labels_df_from_data(data, p_label[keep_columns])

        labels_file_name = Path(bad_partition_dir, file_name)
        recalculated_labels_df.to_csv(labels_file_name)
