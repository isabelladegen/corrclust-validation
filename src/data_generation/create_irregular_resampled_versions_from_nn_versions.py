import pandas as pd

from src.data_generation.create_irregular_datasets import find_best_matching_segment_id, create_new_labels_df_from
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols, \
    recalculate_labels_df_from_data
from src.data_generation.wandb_create_synthetic_data import save_data_labels_to_file
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, \
    GeneralisedCols, SyntheticDataVariates, CONF_IRREGULAR_P30_DATA_DIR, CONF_IRREGULAR_P90_DATA_DIR, \
    CONFIRMATORY_DATASETS_FILE_PATH
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data


def resample_data(nn_data, nn_labels, rule: str = "1min", data_cols: [] = SyntheticDataVariates.columns()):
    """ This is copied from the irregular wandb generation code where originally we resampled the wrong (raw instead of
     nn correlated) data"""
    data_to_resample = nn_data.copy()
    data_to_resample.drop(SyntheticDataSegmentCols.old_regular_id, axis=1, inplace=True, errors='ignore')

    # map the indices of the irregular data to the new time stamps
    data_to_resample['original_index'] = data_to_resample.index
    index_mapping = data_to_resample.resample(rule, on=GeneralisedCols.datetime)['original_index'].agg(list)
    # this gives us a number based index, index_mapping is a df with datatime column and original_index which
    # includes what original data indices go into the new resampled index
    index_mapping = index_mapping.reset_index()

    # actual resample observations
    subject_id = data_to_resample[SyntheticDataSegmentCols.subject_id][0]
    cols_to_resample = [GeneralisedCols.datetime, 'original_index'] + data_cols
    resampled = data_to_resample[cols_to_resample].resample(rule, on=GeneralisedCols.datetime).mean().round(3)
    # to make it consistent with the other df that the datetime is not automatically the index we reset the index
    resampled.reset_index(inplace=True)
    resampled.insert(0, SyntheticDataSegmentCols.subject_id, subject_id)

    # drop all samples where there are 0 original_indices
    zero_orig_indices = [idx for idx, n in enumerate(index_mapping['original_index'].apply(lambda x: len(x))) if n < 1]
    index_mapping = index_mapping.drop(index=zero_orig_indices).reset_index(drop=True)
    resampled = resampled.drop(index=zero_orig_indices).reset_index(drop=True)

    # build resampled labels df from index mapping
    # first find the best matching segment id (based on index overlap for each new index
    index_mapping[SyntheticDataSegmentCols.segment_id] = index_mapping['original_index'].apply(
        lambda x: find_best_matching_segment_id(x, nn_labels))

    # build a new labels df using the segments ids
    new_labels_df = create_new_labels_df_from(index_mapping)
    # get segment id information from given labels file
    lengths = []
    pattern_ids = []
    patterns_to_model = []
    for _, row in new_labels_df.iterrows():
        start_idx = row[SyntheticDataSegmentCols.start_idx]
        end_idx = row[SyntheticDataSegmentCols.end_idx]
        segment_id = row[SyntheticDataSegmentCols.segment_id]
        row = nn_labels.loc[nn_labels[SyntheticDataSegmentCols.segment_id] == segment_id]

        lengths.append((end_idx - start_idx) + 1)
        pattern_ids.append(row[SyntheticDataSegmentCols.pattern_id].values[0])
        patterns_to_model.append(row[SyntheticDataSegmentCols.correlation_to_model].values[0])
    # updated new labels with this information
    new_labels_df[SyntheticDataSegmentCols.length] = lengths
    new_labels_df[SyntheticDataSegmentCols.pattern_id] = pattern_ids
    new_labels_df[SyntheticDataSegmentCols.correlation_to_model] = patterns_to_model

    # recalculate the rest of the labels df
    new_labels_df = recalculate_labels_df_from_data(resampled, new_labels_df)
    new_labels_df.insert(0, SyntheticDataSegmentCols.subject_id, subject_id)
    return resampled, new_labels_df


if __name__ == '__main__':
    # create summary for a dataset variation
    # run_file = GENERATED_DATASETS_FILE_PATH
    run_file = CONFIRMATORY_DATASETS_FILE_PATH
    run_ids = run_names = pd.read_csv(run_file)['Name'].tolist()
    data_type = SyntheticDataType.normal_correlated
    overall_ds_name = "n30"
    # data_dirs = [IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR]
    data_dirs = [CONF_IRREGULAR_P30_DATA_DIR, CONF_IRREGULAR_P90_DATA_DIR]

    for data_dir in data_dirs:
        for run_name in run_ids:
            nn_data_df, nn_labels_df = load_synthetic_data(run_id=run_name, data_type=data_type, data_dir=data_dir)
            # load nn irregular data and resample to 1min
            rs_data, rs_labels = resample_data(nn_data_df, nn_labels_df)
            # safe rs_data and label
            save_data_labels_to_file(data_dir, SyntheticDataType.rs_1min, rs_data, rs_labels, run_name)
            # # # load to check
            # reloaded_rs_data, reloaded_rs_labels_df = load_synthetic_data(run_id=run_name, data_type=SyntheticDataType.rs_1min, data_dir=data_dir)



