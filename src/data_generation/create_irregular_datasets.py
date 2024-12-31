from os import path
from pathlib import Path

import pandas as pd

from src.data_generation.generate_synthetic_segmented_dataset import min_max_scaled_df, SyntheticDataSegmentCols, \
    recalculate_labels_df_from_data
from src.utils.configurations import SyntheticDataVariates, SYNTHETIC_DATA_DIR, bad_partition_dir_for_data_type
from src.utils.load_synthetic_data import load_synthetic_data, load_labels_file_for, SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends


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
        self.orig_data, self.orig_labels = load_synthetic_data(self.__run_name, self.__data_type, data_dir=data_dir)

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
                segment_id = row[SyntheticDataSegmentCols.segment_id]
                pattern_id = row[SyntheticDataSegmentCols.pattern_id]
                correlation_to_model = row[SyntheticDataSegmentCols.correlation_to_model]

                # add to list for new irregular labels df
                segment_ids.append(segment_id)
                start_indices.append(start_index)
                end_indices.append(end_index)
                lengths.append(length)
                pattern_ids.append(pattern_id)
                correlations_to_model.append(correlation_to_model)
            elif data.shape[0] > 0:  # throw the observations away as we cannot calculate a segment
                irr_data_df = irr_data_df.drop(data.index).reset_index(drop=True)

        irr_labels_df = pd.DataFrame({
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

# TODO move to wandb
# def main(ds_name: str, p: float, add_to_seed: int = 0):
#     """Used to create the irregular versions of the dataset"""
#     # with ds_name only we read the AID like versions of the datasets
#     # we drop the same amount of data but not the same observations
#     seed = 1661 + add_to_seed
#     irds = CreateIrregularDataset(run_name=ds_name, seed=seed)
#     data, labels = irds.drop_observation_with_likelihood(p)
#
#     # save csv
#     dir = path.join(ROOT_DIR, 'benchmark/Synthetic')
#     p_string = str(p).replace('.', '_')
#     file_prefix = 'irregular_p_' + p_string + '_' + ds_name
#     data_file_name = file_prefix + SyntheticFileTypes.data
#     label_file_name = file_prefix + SyntheticFileTypes.labels
#
#     print('Saving data to ' + data_file_name)
#     data.to_csv(path.join(dir, data_file_name))
#     print('Saving labels to ' + label_file_name)
#     labels.to_csv(path.join(dir, label_file_name))
#
#
# if __name__ == "__main__":
#     # load 30 ds
#     csv_file = path.join(ROOT_DIR, 'experiments/evaluate/csv/synthetic-data/wandb_export_30_ds-creation.csv')
#     generated_ds = pd.read_csv(csv_file)['Name'].tolist()
#     p = 0.9  # this means 90% of the data will be dropped
#     for idx, ds_name in enumerate(generated_ds):
#         main(ds_name=ds_name, p=p, add_to_seed=idx)
