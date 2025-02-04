import random

import numpy as np

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols, \
    recalculate_labels_df_from_data
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.utils.configurations import SyntheticDataVariates, SYNTHETIC_DATA_DIR
from src.utils.load_synthetic_data import load_synthetic_data, SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends


class CreateBadSyntheticPartitions:
    def __init__(self, run_name: str, data_type: str = SyntheticDataType.non_normal_correlated,
                 data_cols: [str] = SyntheticDataVariates.columns(), data_dir: str = SYNTHETIC_DATA_DIR,
                 backend: str = Backends.none.value, seed: int = 666):
        """
        :param run_name: the name of the wandb run that generated the dataset
        :param data_type: which data type to load, by default it will be non normal and correlated,
        see SyntheticDataType for more types
        :param data_cols: the list of column names for the time series variates
        :param data_dir: the directory to load the original data set from
        before the distribution was shifted and correlation patterns implied

        """
        self.__backend = backend
        self.__run_name = run_name
        self.__data_type = data_type
        self.__data_cols = data_cols
        self.__data_dir = data_dir
        self.__data, self.labels = load_synthetic_data(self.__run_name, self.__data_type, data_dir=self.__data_dir)
        self.__cluster_ids = self.labels[SyntheticDataSegmentCols.pattern_id].unique().tolist()
        self.__segments = self.labels[SyntheticDataSegmentCols.segment_id].unique().tolist()
        self.__seed = seed
        self.__patterns_to_model_lookup = ModelCorrelationPatterns().canonical_patterns()

    def randomly_assign_wrong_cluster(self, n_partitions: int, n_segments: [int]):
        """Creates new partitions by assigning n_segments to a randomly picked wrong cluster

        :param n_partitions: number of new partitions to create
        :param n_segments: list of segments to reassign to a random cluster
        :return: a list of labels dataframes for the new partitions"""
        new_labels = []

        for p in range(n_partitions):
            n = n_segments[p]
            # randomly choose n segments from all the segments
            random.seed(self.__seed)
            # list of segment id's to change cluster for
            reassign_segments = random.sample(self.__segments, n)
            # new labels df
            new_label_df = self.labels.copy()
            for seg_id in reassign_segments:
                gt_cluster_id = \
                    self.labels.loc[self.labels[SyntheticDataSegmentCols.segment_id] == seg_id]['cluster_id'].values[0]

                # remove ground truth cluster from the list of clusters
                clusters = self.__cluster_ids.copy()
                clusters.remove(gt_cluster_id)

                # pick random new cluster id
                random.seed(self.__seed + p + seg_id)
                new_cluster_id = random.choice(clusters)

                # update new labels df
                idx_to_update = new_label_df.loc[new_label_df[SyntheticDataSegmentCols.segment_id] == seg_id].index[0]
                new_label_df.loc[idx_to_update, SyntheticDataSegmentCols.pattern_id] = new_cluster_id

            # update patterns to model
            new_label_df[SyntheticDataSegmentCols.correlation_to_model] = new_label_df[
                SyntheticDataSegmentCols.pattern_id].map(self.__patterns_to_model_lookup)

            # recalculate labels df
            updated_new_label = recalculate_labels_df_from_data(self.__data, new_label_df)

            new_labels.append(updated_new_label)

        return new_labels

    def shift_segments_end_index(self, n_partitions, n_observations):
        """
        Creates new partitions by shifting all segments end index by n_observations. The correlations get updated
        but the cluster ids stay the same

           :param n_partitions: number of new partitions to create
           :param n_observations: list number of observations to shift end index by for each of n_partition
           :return: a list of labels dataframes for the new partitions with updated correlations
        """
        new_labels = []

        for p in range(n_partitions):
            print(p)
            labels_df = self.labels.copy()
            add_obs = n_observations[p]
            # add obs to each start idx other than the first
            labels_df.loc[1:, SyntheticDataSegmentCols.start_idx] = labels_df.loc[1:,
                                                                    SyntheticDataSegmentCols.start_idx].apply(
                lambda x: x + add_obs)
            # add obs to each end idx other than the last
            labels_df.loc[0:labels_df.index[-2], SyntheticDataSegmentCols.end_idx] = labels_df.loc[
                                                                                     0:labels_df.index[-2],
                                                                                     SyntheticDataSegmentCols.end_idx].apply(
                lambda x: x + add_obs)
            # add obs to first segment length
            labels_df.loc[0, SyntheticDataSegmentCols.length] = labels_df.loc[
                                                                    0, SyntheticDataSegmentCols.length] + add_obs
            # remove add_obs from last segment
            labels_df.loc[labels_df.index[-1], SyntheticDataSegmentCols.length] = labels_df.loc[
                                                                                      labels_df.index[
                                                                                          -1], SyntheticDataSegmentCols.length] - add_obs

            # drop segments that have been shifted outside of data length or less than n-variates long
            max_end_idx = self.__data.shape[0] - 1 - len(self.__data_cols)
            first_exceed_idx = labels_df[labels_df[SyntheticDataSegmentCols.end_idx] > max_end_idx].index.min()
            if not np.isnan(first_exceed_idx):
                labels_df = labels_df.loc[:first_exceed_idx]
                # in sure last row ends on data end (we might have dropped a segment with just 1 obs)
                # -1 due to zero based indexing
                labels_df.loc[labels_df.index[-1], SyntheticDataSegmentCols.end_idx] = self.__data.shape[0] - 1
                # make length correct, +1 due to end_idx being selected too
                length = labels_df.loc[labels_df.index[-1], SyntheticDataSegmentCols.end_idx] - labels_df.loc[
                    labels_df.index[-1], SyntheticDataSegmentCols.start_idx] + 1
                labels_df.loc[labels_df.index[-1], SyntheticDataSegmentCols.length] = length

            # check lengths in labels_df match data length
            assert labels_df[SyntheticDataSegmentCols.length].sum() == self.__data.shape[
                0], "Length mismatch for shifting segments by p " + str(p)

            # recalculate labels df (will recalculate correlations)
            updated_new_label = recalculate_labels_df_from_data(self.__data, labels_df)

            new_labels.append(updated_new_label)

        return new_labels

    def shift_segments_end_index_and_assign_wrong_clusters(self, n_partitions: int, n_observations: [int],
                                                           n_segments: [int]):
        """
        Creates new partitions by shifting all segments end index by n_observations. The correlations get updated
        according to the new correlations. The cluster ids of n_segments get randomly changed.The labels file
        gets updated according to these changes

        :param n_partitions: number of new partitions to create
        :param n_observations: number of observations to shift end index by
        :param n_segments: list of segments to reassign to a random cluster
        :return: a list of updated labels dataframes for the new partitions
        """

        shifted_labels = self.shift_segments_end_index(n_partitions, n_observations)
        new_cluster_labels = self.randomly_assign_wrong_cluster(n_partitions, n_segments)

        # merge the labels
        new_labels = []
        for idx, new_label in enumerate(shifted_labels):
            new_cluster = new_cluster_labels[idx]
            new_label[SyntheticDataSegmentCols.pattern_id] = new_cluster[SyntheticDataSegmentCols.pattern_id]

            # update patterns to model
            new_label[SyntheticDataSegmentCols.correlation_to_model] = new_label[
                SyntheticDataSegmentCols.pattern_id].map(
                self.__patterns_to_model_lookup)

            # recalculate labels df
            updated_new_label = recalculate_labels_df_from_data(self.__data, new_label)

            new_labels.append(updated_new_label)

        return new_labels
