import random
from os import path

import pandas as pd

from src.data_generation.generate_synthetic_correlated_data import calculate_spearman_correlation
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.utils.configurations import ROOT_DIR, SyntheticDataVariates
from src.utils.load_synthetic_data import SyntheticFileTypes, load_synthetic_data
from src.utils.plots.matplotlib_helper_functions import Backends


class CreateBadSyntheticPartitions:
    def __init__(self, run_name: str, data_type: str = SyntheticFileTypes.data,
                 data_cols: [str] = SyntheticDataVariates.columns(), backend: str = Backends.none.value,
                 seed: int = 666):
        """
        :param run_name: the name of the wandb run that generated the dataset
        :param data_type: data is the correlated distribution shifted version, you can also access versions
        before the distribution was shifted and correlation patterns implied

        """
        self.__backend = backend
        self.__run_name = run_name
        self.__data_type = data_type
        self.__data_cols = data_cols
        self.__data, self.labels, gt_labels = load_synthetic_data(self.__run_name, self.__data_type)
        self.__cluster_ids = self.labels[SyntheticDataSegmentCols.pattern_id].unique().tolist()
        self.__segments = self.labels[SyntheticDataSegmentCols.segment_id].unique().tolist()
        self.__seed = seed
        self.__patterns_to_model_lookup = ModelCorrelationPatterns().ideal_correlations()

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
                new_cor_to_model = self.__patterns_to_model_lookup[new_cluster_id]
                new_label_df.loc[idx_to_update, SyntheticDataSegmentCols.correlation_to_model] = str(new_cor_to_model)

            new_labels.append(new_label_df)

        return new_labels

    def shift_segments_end_index(self, n_partitions, n_observations):
        """
        Creates new partitions by shifting all segments end index by n_observations. The correlations get updated
        but the cluster ids stay the same

           :param n_partitions: number of new partitions to create
           :param n_observations: number of observations to shift end index by
           :return: a list of labels dataframes for the new partitions with updated correlations
        """
        new_labels = []

        for p in range(n_partitions):
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

            # update correlations achieved
            new_start_indices = labels_df[SyntheticDataSegmentCols.start_idx].tolist()
            new_end_indices = labels_df[SyntheticDataSegmentCols.end_idx].tolist()
            new_correlations = []
            for seg_id in range(len(new_start_indices)):
                data = self.__data.loc[new_start_indices[seg_id]:new_end_indices[seg_id]][self.__data_cols].to_numpy()
                new_correlations.append(calculate_spearman_correlation(data))
            labels_df[SyntheticDataSegmentCols.actual_correlation] = new_correlations

            new_labels.append(labels_df)

        return new_labels

    def shift_segments_end_index_and_assign_wrong_clusters(self, n_partitions: int, n_observations: [int],
                                                           n_segments: [int]):
        """
        Creates new partitions by shifting all segments end index by n_observations. The correlations get updated
        according to the new correlations. The cluster ids of n_segments get randomly changed.

        :param n_partitions: number of new partitions to create
        :param n_observations: number of observations to shift end index by
        :param n_segments: list of segments to reassign to a random cluster
        :return: a list of labels dataframes for the new partitions with updated correlations
        """

        shifted_labels = self.shift_segments_end_index(n_partitions, n_observations)
        new_cluster_labels = self.randomly_assign_wrong_cluster(n_partitions, n_segments)

        # merge the labels
        new_labels = []
        for idx, new_label in enumerate(shifted_labels):
            new_cluster = new_cluster_labels[idx]
            new_label[SyntheticDataSegmentCols.pattern_id] = new_cluster[SyntheticDataSegmentCols.pattern_id]
            new_labels.append(new_label)
        return new_labels


# todo: move to wandb so it gets logged too
def main(ds_name: str, add_to_seed: int = 0):
    """Used to create the actual bad partitions for evaluation of internal measures"""
    max_seg = 100
    max_obs = 800  # this is 100 less than the shortest segment
    n_partitions = 22
    bp = CreateBadSyntheticPartitions(run_name=ds_name)

    # First create 22 partitions where we assign a wrong cluster
    # select random number of segments but ensure one partition changes all 100 segments
    possible_n = list(range(1, max_seg + 1))
    random.seed(66 + add_to_seed)
    n_segments = random.sample(possible_n, n_partitions - 1)
    n_segments.append(max_seg)
    n_segments.sort()  # so that lower partitions have fewer errors than higher ones
    print("CREATING WRONG CLUSTER...")
    print("Number of segments to change:")
    print(n_segments)
    wrong_clusters = bp.randomly_assign_wrong_cluster(n_partitions=n_partitions, n_segments=n_segments)
    # save csv
    for p in range(n_partitions):
        file_name = "bad_partitions/" + ds_name + "-wrong-clusters-" + str(p) + "-labels.csv"
        wrong_clusters[p].to_csv(file_name, index=False)

    # Second create 22 partitions with shifted index
    # ensure one partition will shift by the max of 800 observations
    possible_obs = list(range(1, max_obs + 1))
    random.seed(101 + add_to_seed)
    n_observations = random.sample(possible_obs, n_partitions - 1)
    n_observations.append(max_obs)
    n_observations.sort()  # so that lower partitions have fewer errors than higher ones
    print("\n")
    print("CREATING SHIFTED END IDX...")
    print("Number of observations to shift:")
    print(n_observations)
    shifted_end_idx = bp.shift_segments_end_index(n_partitions=n_partitions, n_observations=n_observations)
    # save csv
    for p in range(n_partitions):
        file_name = "bad_partitions/" + ds_name + "-shifted-end-idx-" + str(p) + "-labels.csv"
        shifted_end_idx[p].to_csv(file_name, index=False)

    # Third create 22 partitions with both changes
    random.seed(6306 + add_to_seed)
    n_segments_both = random.sample(possible_n, n_partitions - 1)
    n_segments_both.append(max_seg)
    n_segments_both.sort()
    n_observations_both = random.sample(possible_obs, n_partitions - 1)
    n_observations_both.append(max_obs)
    n_observations_both.sort()
    print("\n")
    print("CREATING SHIFTED END IDX AND WRONG CLUSTERS...")
    print("Number of observations to shift:")
    print(n_observations_both)
    print("Number of segments to change:")
    print(n_segments_both)
    shift_and_wrong_clusters = bp.shift_segments_end_index_and_assign_wrong_clusters(n_partitions=n_partitions,
                                                                                     n_observations=n_observations_both,
                                                                                     n_segments=n_segments_both)
    # save csv
    for p in range(n_partitions):
        file_name = "bad_partitions/" + ds_name + "-shifted-and-wrong-cluster-" + str(p) + "-labels.csv"
        shift_and_wrong_clusters[p].to_csv(file_name, index=False)


if __name__ == "__main__":
    # load 30 ds
    csv_file = path.join(ROOT_DIR, 'experiments/evaluate/csv/synthetic-data/wandb_export_30_ds-creation.csv')
    generated_ds = pd.read_csv(csv_file)['Name'].tolist()
    for idx, ds_name in enumerate(generated_ds):
        main(ds_name=ds_name, add_to_seed=idx)
