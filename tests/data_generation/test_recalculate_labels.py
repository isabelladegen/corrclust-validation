from pathlib import Path

import pytest

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.data_generation.recalculate_labels import recalculate_labels_df_from_data
from src.utils.configurations import dir_for_data_type
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data, SyntheticFileTypes
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR

test_data_dir = TEST_DATA_DIR


@pytest.mark.skip(reason="this is a once off calculation to bring old labels files into new format")
def test_this_is_temporary_to_create_correct_labels_files():
    run_name = "misty-forest-56"
    # run_name = "splendid-sunset-12"

    data_type = SyntheticDataType.downsampled_1min
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
