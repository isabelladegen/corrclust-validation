from os import path
from pathlib import Path

import pandas as pd

from src.data_generation.generate_synthetic_segmented_dataset import min_max_scaled_df, SyntheticDataSegmentCols, \
    recalculate_labels_df_from_data
from src.utils.configurations import SyntheticDataVariates, SYNTHETIC_DATA_DIR, bad_partition_dir_for_data_type, \
    GeneralisedCols
from src.utils.load_synthetic_data import load_synthetic_data, load_labels_file_for, SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends


def find_best_matching_segment_id(original_indices: [int], labels_before_resampling: pd.DataFrame):
    """
    Find segment_id from labels_before_resampling with the best indices overlap for original indices
    """
    df = labels_before_resampling.copy()
    lowest_idx = original_indices[0]
    highest_idx = original_indices[-1]

    # from labels df select only rows where start_idx is >= lowest_idx and end_idx <= highest_idx
    sub_df = df[
        (df[SyntheticDataSegmentCols.start_idx] <= highest_idx) &
        (df[SyntheticDataSegmentCols.end_idx] >= lowest_idx)]
    if sub_df.shape[0] == 1:
        return sub_df[SyntheticDataSegmentCols.segment_id].tolist()[0]
    else:
        max_overlap = 0
        best_segment_id = None

        # return id with most overlap
        for _, row in sub_df.iterrows():
            indices_in_row = set(
                # +1 to include the end idx in the range
                range(row[SyntheticDataSegmentCols.start_idx], row[SyntheticDataSegmentCols.end_idx] + 1))
            overlap = len(set(original_indices) & indices_in_row)

            if overlap > max_overlap:
                max_overlap = overlap
                best_segment_id = row[SyntheticDataSegmentCols.segment_id]

        return best_segment_id


def create_new_labels_df_from(index_mapping_df: pd.DataFrame):
    """
    Create df with segment_id, start_idx and end_idx column from the index mapping
    """
    df = index_mapping_df.copy()
    # Get indices of rows where segment id changes
    changes = df[SyntheticDataSegmentCols.segment_id] != df[SyntheticDataSegmentCols.segment_id].shift()
    start_indices = [i for i, val in enumerate(changes.tolist()) if val]
    end_indices = [x - 1 for x in start_indices[1:]]
    # add last index to end indices
    end_indices.append(df.shape[0] - 1)
    # segment_ids without repetition
    segment_ids = df[SyntheticDataSegmentCols.segment_id].unique()

    new_labels_df = pd.DataFrame({
        SyntheticDataSegmentCols.segment_id: segment_ids,
        SyntheticDataSegmentCols.start_idx: start_indices,
        SyntheticDataSegmentCols.end_idx: end_indices
    })
    return new_labels_df


class CreateIrregularDataset:
    def __init__(self, run_name: str, data_type: str = SyntheticDataType.non_normal_correlated,
                 data_dir: str = SYNTHETIC_DATA_DIR, data_cols: [str] = SyntheticDataVariates.columns(),
                 bad_partition_name: str = "", value_range: (float, float) = None, backend: str = Backends.none.value,
                 seed: int = 1661):
        self.__backend = backend
        self.__run_name = run_name
        self.__data_type = data_type
        self.__data_dir = data_dir
        self.__data_cols = data_cols
        self.__value_range = value_range
        self.__bad_partition_name = bad_partition_name
        self.__seed = seed
        # load data and labels
        self.orig_data, self.orig_labels = load_synthetic_data(self.__run_name, self.__data_type,
                                                               data_dir=self.__data_dir)

        # if loading bad partitions
        if bad_partition_name is not "":
            # load labels for bad partition and overwrites the label for good partition!!
            bad_partitions_dir = bad_partition_dir_for_data_type(self.__data_type, self.__data_dir)
            bad_file_full_path = path.join(bad_partitions_dir, self.__bad_partition_name)
            self.orig_labels = load_labels_file_for(Path(bad_file_full_path))

        if self.__value_range is not None:  # needs to scale data first
            self.data = min_max_scaled_df(self.orig_data, scale_range=self.__value_range, columns=self.__data_cols)

    def drop_observation_with_likelihood(self, p: float):
        """
        Randomly select 1-p rows from the original dataset
        :param p: percentage of rows to drop, needs to be between 0.1-0.9
        :return: new data and labels df to be saved
        """
        assert 0 < p < 1.0, "p must be between (0,1)"
        # from data select 1-p rows to keep
        frac = 1 - p
        irr_data_df = self.orig_data.copy().sample(frac=frac, axis=0, random_state=self.__seed, ignore_index=False)
        irr_data_df.sort_index(inplace=True)
        # the irregular data has a new consecutive index but it no longer matches the old index
        irr_data_df = irr_data_df.reset_index(drop=False, names=SyntheticDataSegmentCols.old_regular_id)

        # create irregular labels file start and end idx and pattern and correlation to model
        subject_ids = []
        segment_ids = []  # keep original segment id for new irregular label
        start_indices = []
        end_indices = []
        lengths = []
        pattern_ids = []
        correlations_to_model = []

        # iterate over each segment in original labels and update the indices
        for idx, row in self.orig_labels.iterrows():
            # select left over data for old idx
            orig_start_idx = row[SyntheticDataSegmentCols.start_idx]
            orig_end_idx = row[SyntheticDataSegmentCols.end_idx]
            data = irr_data_df[(irr_data_df[SyntheticDataSegmentCols.old_regular_id] >= orig_start_idx) & (
                    irr_data_df[SyntheticDataSegmentCols.old_regular_id] <= orig_end_idx)]

            # we might not get to keep all segments, if we have sufficient observations to calculate
            # a correlation then calculate the correlation
            if data.shape[0] >= len(self.__data_cols):
                # get indices from new irregular data
                start_index = data.index.values[0]
                end_index = data.index.values[-1]
                length = data.shape[0]

                # copy original segment id and pattern data for this segment
                subject_id = row[SyntheticDataSegmentCols.subject_id]
                segment_id = row[SyntheticDataSegmentCols.segment_id]
                pattern_id = row[SyntheticDataSegmentCols.pattern_id]
                correlation_to_model = row[SyntheticDataSegmentCols.correlation_to_model]

                # add to list for new irregular labels df
                subject_ids.append(subject_id)
                segment_ids.append(segment_id)
                start_indices.append(start_index)
                end_indices.append(end_index)
                lengths.append(length)
                pattern_ids.append(pattern_id)
                correlations_to_model.append(correlation_to_model)
            elif data.shape[0] > 0:  # throw the observations away as we cannot calculate a segment
                irr_data_df = irr_data_df.drop(data.index).reset_index(drop=True)

        irr_labels_df = pd.DataFrame({
            SyntheticDataSegmentCols.subject_id: subject_ids,
            SyntheticDataSegmentCols.segment_id: segment_ids,
            SyntheticDataSegmentCols.start_idx: start_indices,
            SyntheticDataSegmentCols.end_idx: end_indices,
            SyntheticDataSegmentCols.length: lengths,
            SyntheticDataSegmentCols.pattern_id: pattern_ids,
            SyntheticDataSegmentCols.correlation_to_model: correlations_to_model,
        })

        # calculate the correlation achieved and errors
        irr_labels_df = recalculate_labels_df_from_data(irr_data_df, irr_labels_df)

        return irr_data_df, irr_labels_df

    def irregular_version_for_data_type(self, data_type: str, given_irr_data: pd.DataFrame,
                                        given_irr_labels: pd.DataFrame):
        """ Creates same irregular version for the given data type from the
        :param data_type: see SyntheticDataType other than what we already created
        :param given_irr_data: use the old indices in this data df to select the required data
        :param given_irr_labels: use this labels indexing to create the irregular data
        :return irregular data and labels for that data type
        """
        assert data_type != self.__data_type, "Already created versions for data type " + data_type

        if not data_type.startswith(SyntheticDataType.resample('')):
            # 1. Load data
            full_data, full_labels = load_synthetic_data(self.__run_name, data_type, data_dir=self.__data_dir)

            # 2. From full data df select all indices in the given data df's old index column
            new_irr_data = full_data[full_data.index.isin(given_irr_data[SyntheticDataSegmentCols.old_regular_id])]
            new_irr_data.reset_index(drop=False, names=SyntheticDataSegmentCols.old_regular_id, inplace=True)

            # 3. Creat new Labels file with indexing from (we start with the given labels which already has the right
            # segment indices and lengths and recalculate the correlations from the new_irr_data of this new data type
            new_irr_labels = recalculate_labels_df_from_data(new_irr_data, given_irr_labels)
            return new_irr_data, new_irr_labels
        else:
            rule = SyntheticDataType.rule_from_resample_type(data_type)
            data_to_resample = given_irr_data.copy()
            data_to_resample.drop(SyntheticDataSegmentCols.old_regular_id, axis=1, inplace=True, errors='ignore')

            # map the indices of the irregular data to the new time stamps
            data_to_resample['original_index'] = data_to_resample.index
            index_mapping = data_to_resample.resample(rule, on=GeneralisedCols.datetime)['original_index'].agg(list)
            # this gives us a number based index, index_mapping is a df with datatime column and original_index which
            # includes what original data indices go into the new resampled index
            index_mapping = index_mapping.reset_index()

            # actual resample observations
            subject_id = data_to_resample[SyntheticDataSegmentCols.subject_id][0]
            cols_to_resample = [GeneralisedCols.datetime, 'original_index'] + self.__data_cols
            resampled = data_to_resample[cols_to_resample].resample(rule, on=GeneralisedCols.datetime).mean()
            # to make it consistent with the other df that the datetime is not automatically the index we reset the index
            resampled.reset_index(inplace=True)
            resampled.insert(0, SyntheticDataSegmentCols.subject_id, subject_id)

            # drop all samples where there are 0 original_indices
            zero_orig_indices = [idx for idx, n in enumerate(index_mapping['original_index'].apply(lambda x: len(x))) if n < 1]
            index_mapping = index_mapping.drop(index=zero_orig_indices).reset_index(drop=True)
            resampled =resampled.drop(index=zero_orig_indices).reset_index(drop=True)

            # build resampled labels df from index mapping
            # first find the best matching segment id (based on index overlap for each new index
            index_mapping[SyntheticDataSegmentCols.segment_id] = index_mapping['original_index'].apply(
                lambda x: find_best_matching_segment_id(x, given_irr_labels))

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
                row = given_irr_labels.loc[given_irr_labels[SyntheticDataSegmentCols.segment_id] == segment_id]

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
