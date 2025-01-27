from dataclasses import dataclass
from os import path
from pathlib import Path

import pandas as pd
import pytest

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols, \
    recalculate_labels_df_from_data
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.evaluation.distance_metric_evaluation import read_csv_of_raw_values_for_all_criteria, EvaluationCriteria
from src.utils.configurations import dir_for_data_type, bad_partition_dir_for_data_type, SYNTHETIC_DATA_DIR, \
    IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, ROOT_RESULTS_DIR, \
    distance_measure_evaluation_results_dir_for, DISTANCE_MEASURE_EVALUATION_CRITERIA_RESULTS_FILE
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data, SyntheticFileTypes, \
    load_synthetic_data_and_labels_for_bad_partitions
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR, TEST_IRREGULAR_P90_DATA_DIR, \
    TEST_IRREGULAR_P30_DATA_DIR

test_data_dir = TEST_DATA_DIR


@pytest.mark.skip(reason="this is a once off calculation to bring old labels files into new format")
def test_this_is_temporary_to_create_correct_labels_files():
    twilight_fog = "twilight-fog-55"
    playful_thun = "playful-thunder-52"
    misty_for = "misty-forest-56"
    sp_sunset = "splendid-sunset-12"
    amber_glade = "amber-glade-10"
    # data_dir = TEST_IRREGULAR_P30_DATA_DIR
    data_dir = TEST_DATA_DIR

    # labels_files = [amber_glade, "gallant-galaxy-1", "glorious-shadow-2", "playful-thunder-52", "twilight-fog-55"]
    labels_files = ["gallant-galaxy-1", "glorious-shadow-2", misty_for]
    # labels_files = [misty_for, sp_sunset]

    for run_name in labels_files:
        data_type = SyntheticDataType.rs_1min
        data_df, labels_df = load_synthetic_data(run_name, data_type=data_type, data_dir=data_dir)
        # drop columns:
        keep_columns = [SyntheticDataSegmentCols.segment_id, SyntheticDataSegmentCols.start_idx,
                        SyntheticDataSegmentCols.end_idx, SyntheticDataSegmentCols.length,
                        SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.correlation_to_model,
                        SyntheticDataSegmentCols.actual_correlation, SyntheticDataSegmentCols.actual_within_tolerance]

        recalculated_labels_df = recalculate_labels_df_from_data(data_df, labels_df[keep_columns])

        labels_file = SyntheticFileTypes.labels
        file_dir = dir_for_data_type(data_type, data_dir)
        labels_file_name = Path(file_dir, run_name + labels_file)
        recalculated_labels_df.to_csv(labels_file_name)


@pytest.mark.skip(reason="this is a once off calculation to bring old labels files into new format")
def test_this_is_temporary_to_create_correct_labels_files_for_bad_partitions():
    run_name = "misty-forest-56"
    # run_name = "twilight-fog-55"
    # run_name = "playful-thunder-52"
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
    patterns_to_model = ModelCorrelationPatterns().canonical_patterns()

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


@pytest.mark.skip(reason="this is a once off calculation to update correlations in labels files")
def test_recalculate_all_labels_files_to_allow_for_distance_measure_calculation():
    """
    This is a once off to recalculate all the labels files to allow for distance calculation
    We change the correlation 0.99 to not get rounded to 1 and keep 3 decimals for correlations
    We also now calculate the relaxed MAE
    This does not include bad partitions!
    """
    dataset_types = [SyntheticDataType.raw,
                     SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    generated_ds = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    for data_dir in data_dirs:
        for data_type in dataset_types:
            for run_name in generated_ds:
                data_df, labels_df = load_synthetic_data(run_name, data_type=data_type, data_dir=data_dir)
                recalculated_labels_df = recalculate_labels_df_from_data(data_df, labels_df)
                labels_file = SyntheticFileTypes.labels
                file_dir = dir_for_data_type(data_type, data_dir)
                labels_file_name = Path(file_dir, run_name + labels_file)
                recalculated_labels_df.to_csv(labels_file_name)


@dataclass
class OldEvaluationCriteria:
    inter_i: str = "Interpretability: L_0 close to zero"
    inter_ii: str = "Interpretability: proper level sets ordering"
    inter_iii: str = "Interpretability: rate of increase between level sets"
    disc_i: str = "Discriminative Power: overall RC"
    disc_ii: str = "Discriminative Power: overall CV"
    disc_iii: str = "Discriminative Power: macro F1 score"
    # stab_i: str = "Stability: completed" -> this is kind of a not catchable one
    stab_ii: str = "Stability: count of nan and inf distances"


@pytest.mark.skip(reason="this is a once off calculation to update index name for calculated raw criteria")
def test_rename_criteria():
    name_mapping = {
        OldEvaluationCriteria.inter_i: EvaluationCriteria.inter_i,
        OldEvaluationCriteria.inter_ii: EvaluationCriteria.inter_ii,
        OldEvaluationCriteria.inter_iii: EvaluationCriteria.inter_iii,
        OldEvaluationCriteria.disc_i: EvaluationCriteria.disc_i,
        OldEvaluationCriteria.disc_ii: EvaluationCriteria.disc_ii,
        OldEvaluationCriteria.disc_iii: EvaluationCriteria.disc_iii,
        OldEvaluationCriteria.stab_ii: EvaluationCriteria.stab_ii
    }

    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.raw,
                     SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    generated_ds = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    for data_dir in data_dirs:
        for data_type in dataset_types:
            for run_name in generated_ds:
                # read file
                raw_criteria_df = read_csv_of_raw_values_for_all_criteria(run_name=run_name, data_type=data_type,
                                                                          data_dir=data_dir,
                                                                          base_results_dir=root_result_dir)
                result_dir = distance_measure_evaluation_results_dir_for(run_name=run_name,
                                                                         data_type=data_type,
                                                                         base_results_dir=root_result_dir,
                                                                         data_dir=data_dir)

                # update index
                raw_criteria_df.index = raw_criteria_df.index.map(lambda x: name_mapping.get(x, x))

                # save back
                file_name = DISTANCE_MEASURE_EVALUATION_CRITERIA_RESULTS_FILE
                full_path = path.join(result_dir, file_name)
                raw_criteria_df.to_csv(full_path)
