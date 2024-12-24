import ast
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.configurations import SYNTHETIC_DATA_DIR, SyntheticDataVariates
from src.data_generation.generate_synthetic_correlated_data import calculate_spearman_correlation, \
    check_correlations_are_within_original_strength
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols, \
    recalculate_labels_df_from_data
from src.utils.internal_measures import silhouette_avg_from_distances, calculate_pmb, clustering_jaccard_coeff
from src.utils.labels_utils import calculate_overall_data_correlation, \
    calculate_distance_between_segment_and_data_centroid, calculate_cluster_centroids, \
    calculate_distances_between_each_segment_and_its_cluster_centroid, calculate_distances_between_cluster_centroids, \
    calculate_y_pred_from, calculate_distance_matrix_for
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data_and_labels_for_bad_partitions


@dataclass
class DescribeBadPartCols:
    pmb: str = "PMB"
    silhouette_score: str = "SCW"
    jaccard_index: str = "Jaccard"
    name: str = "file name"
    n_patterns: str = "n patterns"
    n_segments: str = "n segments"
    n_observations: str = "n observations"
    errors: str = "sum corr errors"
    n_wrong_clusters: str = "n wrong clusters"
    n_obs_shifted: str = "n obs shifted"


default_internal_measures = [DescribeBadPartCols.silhouette_score]
default_external_measures = [DescribeBadPartCols.jaccard_index]


# todo: delete
def recalculate_correlation_columns_in_labels(data_df, labels_df):
    achieved_corrs = []
    within_tols = []
    # iterate through all segments and recalculate achieved correlation, keep data ordered by pattern_id
    for idx, row in labels_df.iterrows():
        start_idx = row[SyntheticDataSegmentCols.start_idx]
        end_idx = row[SyntheticDataSegmentCols.end_idx]
        length = row[SyntheticDataSegmentCols.length]
        original_cor = ast.literal_eval(row[SyntheticDataSegmentCols.correlation_to_model])

        # select data
        segment_df = data_df.iloc[start_idx:end_idx + 1]
        segment = segment_df.to_numpy()
        if segment.shape[0] != length:
            print("Help")
        assert segment.shape[0] == length, "Mistake with indexing dataframe"

        # calculated
        achieved_cor = calculate_spearman_correlation(segment)
        within_tol = check_correlations_are_within_original_strength(original_cor, achieved_cor)

        # store results
        achieved_corrs.append(str(achieved_cor))
        within_tols.append(str(within_tol))

    # update df
    labels_df[SyntheticDataSegmentCols.actual_correlation] = achieved_corrs
    labels_df[SyntheticDataSegmentCols.actual_within_tolerance] = within_tols
    return labels_df


def select_data_from_df(data: pd.DataFrame, label: pd.DataFrame):
    """
    Selects only the observations that are actually indexed in the label df
    :param data: pd.DataFrame of the data (might have more columns than just observations
    :param label: pd.DataFrame of the labels that should be used to select rows from the data
    """
    n_segments = label.shape[0]

    # select all observations still referenced in labels_df
    updated_data = None
    for seg in range(n_segments):
        start_idx = label.loc[seg, SyntheticDataSegmentCols.start_idx]
        end_idx = label.loc[seg, SyntheticDataSegmentCols.end_idx]
        length = label.loc[seg, SyntheticDataSegmentCols.length]

        seg_data = data[start_idx:end_idx + 1, :]
        assert seg_data.shape[0] == length, "Selected wrong indices"

        if updated_data is None:
            updated_data = seg_data
        else:
            updated_data = np.concatenate((updated_data, seg_data), axis=0)
    return updated_data


# todo: get rid of test run and instead just provide file with fewer datasets to load
class DescribeBadPartitions:
    def __init__(self, ds_name, distance_measure: str, data_type: str = SyntheticDataType.non_normal_correlated,
                 internal_measures: [] = default_internal_measures, external_measures: [] = default_external_measures,
                 data_cols: [str] = SyntheticDataVariates.columns(), n_clusters=None, n_segments=None, seed: int = 600,
                 data_dir: str = SYNTHETIC_DATA_DIR, round_to: int = 3):
        self.ds_name = ds_name
        self.distance_measure = distance_measure
        self.__internal_measures = internal_measures
        self.__external_measures = external_measures
        self.__seed = seed
        self.__n_clusters = n_clusters
        self.__n_segments = n_segments
        self.__data_dir = data_dir
        self.__data_type = data_type
        self.__cols = data_cols

        # lazy loaded fields
        self.__segment_correlations = None
        self.__overall_corr = None
        self.__cluster_correlations = None

        # load data, ground truth labels and all other partition for given ds_name
        data, gt_label, partitions = load_synthetic_data_and_labels_for_bad_partitions(self.ds_name,
                                                                                       data_type=self.__data_type,
                                                                                       data_dir=self.__data_dir)

        self.data = data  # time series data df
        self.gt_label = gt_label  # ground truth labels df
        self.partitions = partitions  # dictionary of filename as key and bad partition labels_df as value

        # To reduce the number of clusters and segments in the data we can give a number of clusters or segments
        # to select. In that case clusters and segments get dropped. The data remains unchanged
        if n_clusters is not None or n_segments is not None:
            self.gt_label, self.partitions, selected_patterns, selected_segs = self.__drop_clusters_or_segments_from_data(
                self.data, self.gt_label, self.partitions)

        # Calculate distance matrix for ground truth
        # 1D np array of length n_segments
        self.gt_patterns = self.gt_label[SyntheticDataSegmentCols.pattern_id].to_numpy()
        # 2D np array of dimension n_segments x n_segments with 0 diagonal and symmetric
        self.gt_distance_matrix = calculate_distance_matrix_for(self.gt_label, self.distance_measure)

        # Calculate various correlation based distances for ground truth:
        self.gt_updated_data = select_data_from_df(self.data, gt_label)
        data_np = self.gt_updated_data[self.__cols].to_numpy()
        self.data_centroid = calculate_overall_data_correlation(data_np)
        self.gt_cluster_centroids = calculate_cluster_centroids(self.gt_label, data_np)
        # distances between each segment to overall correlation of data
        self.gt_dist_seg_overall_data = calculate_distance_between_segment_and_data_centroid(self.gt_label,
                                                                                             self.data_centroid,
                                                                                             self.distance_measure)
        # distances between each segment to their cluster centroid
        self.gt_dist_seg_cluster = calculate_distances_between_each_segment_and_its_cluster_centroid(self.gt_label,
                                                                                                     self.gt_cluster_centroids,
                                                                                                     self.distance_measure)
        # distances between all cluster centroids
        self.gt_dist_between_clusters = calculate_distances_between_cluster_centroids(self.gt_cluster_centroids,
                                                                                      self.distance_measure)

        # Create overview description df that includes ground truth and all bad partitions
        file_names = []
        n_patterns = []
        n_segments = []
        n_observations = []
        mean_mae = []
        n_wrong_clusters = []
        n_obs_shifted = []
        jaccards = []
        sils = []
        pmbs = []

        # 1. add ground truth information to summary df
        file_names.append(ds_name)
        n_patterns.append(len(self.gt_label[SyntheticDataSegmentCols.pattern_id].unique()))
        n_segments.append(self.gt_label.shape[0])
        n_observations.append(self.data.shape[0])
        mean_mae.append(round(self.gt_label[SyntheticDataSegmentCols.mae].mean(), round_to))
        n_wrong_clusters.append(0)
        n_obs_shifted.append(0)

        if DescribeBadPartCols.jaccard_index in self.__external_measures:
            jaccards.append(1)

        if DescribeBadPartCols.silhouette_score in self.__internal_measures:
            sil_avg = silhouette_avg_from_distances(self.gt_distance_matrix, self.gt_patterns)
            sils.append(sil_avg)

        if DescribeBadPartCols.pmb in self.__internal_measures:
            pmb = calculate_pmb(self.gt_dist_seg_overall_data, self.gt_dist_seg_cluster, self.gt_dist_between_clusters)
            pmbs.append(pmb)

        # calculate y pred for ground truth
        gt_y_pred = calculate_y_pred_from(self.gt_label)

        # to calculate the shift
        gt_first_seg_end_idx = self.gt_label.loc[0, SyntheticDataSegmentCols.end_idx]

        for file_name, p_label in self.partitions:
            p_updated_data = select_data_from_df(self.data, p_label)
            p_data_np = p_updated_data[self.__cols].to_numpy()
            p_mean_mae_error = round(p_label[SyntheticDataSegmentCols.mae].mean(), round_to)
            p_patterns = p_label[SyntheticDataSegmentCols.pattern_id].to_numpy()

            # calculate and add info to new df for this partition
            file_names.append(file_name)
            n_patterns.append(len(p_patterns))
            n_segments.append(p_label.shape[0])
            n_observations.append(p_updated_data.shape[0])
            mean_mae.append(p_mean_mae_error)

            # calculate how many patterns were changed and how many observations shifted for the partition
            n_wrong_clusters.append(sum(i != j for i, j in zip(self.gt_patterns, p_patterns)))
            p_first_seg_end_idx = p_label.loc[0, SyntheticDataSegmentCols.end_idx]
            n_obs_shifted.append(p_first_seg_end_idx - gt_first_seg_end_idx)

            # calculate external and internal measures
            if DescribeBadPartCols.jaccard_index in self.__external_measures:
                p_y_pred = calculate_y_pred_from(p_label)
                p_jacc = clustering_jaccard_coeff(p_y_pred, gt_y_pred, round_to)
                jaccards.append(p_jacc)

            if DescribeBadPartCols.silhouette_score in self.__internal_measures:
                # 2D np array of dimension n_segments x n_segments with 0 diagonal and symmetric
                p_distance_matrix = calculate_distance_matrix_for(p_label, self.distance_measure)
                sil_avg = silhouette_avg_from_distances(p_distance_matrix, p_label)
                sils.append(sil_avg)

            if DescribeBadPartCols.pmb in self.__internal_measures:
                p_data_centroid = calculate_overall_data_correlation(p_data_np)
                p_cluster_centroids = calculate_cluster_centroids(p_label, p_data_np)
                # distances between each segment to overall correlation of data
                p_dist_seg_overall_data = calculate_distance_between_segment_and_data_centroid(p_label,
                                                                                               p_data_centroid,
                                                                                               self.distance_measure)
                # distances between each segment to their cluster centroid
                p_dist_seg_cluster = calculate_distances_between_each_segment_and_its_cluster_centroid(
                    p_label,
                    p_cluster_centroids,
                    self.distance_measure)
                # distances between all cluster centroids
                p_dist_between_clusters = calculate_distances_between_cluster_centroids(p_cluster_centroids,
                                                                                        self.distance_measure)
                pmb = calculate_pmb(p_dist_seg_overall_data, p_dist_seg_cluster, p_dist_between_clusters)
                pmbs.append(pmb)

        # put summary df together
        self.summary_df = pd.DataFrame({
            DescribeBadPartCols.name: file_names,
            DescribeBadPartCols.n_patterns: n_patterns,
            DescribeBadPartCols.n_segments: n_segments,
            DescribeBadPartCols.n_observations: n_observations,
            DescribeBadPartCols.errors: mean_mae,
            DescribeBadPartCols.n_wrong_clusters: n_wrong_clusters,
            DescribeBadPartCols.n_obs_shifted: n_obs_shifted,
        })

        if DescribeBadPartCols.jaccard_index in self.__external_measures:
            self.summary_df[DescribeBadPartCols.jaccard_index] = jaccards

        if DescribeBadPartCols.silhouette_score in self.__internal_measures:
            self.summary_df[DescribeBadPartCols.silhouette_score] = sils

        if DescribeBadPartCols.pmb in self.__internal_measures:
            self.summary_df[DescribeBadPartCols.pmb] = pmbs

    def __drop_clusters_or_segments_from_data(self, data: pd.DataFrame, gt_labels: pd.DataFrame, part_labels: {}):
        """
            Selects at random a specific number of clusters and segments
            - you need at least 2 clusters
            :param data: pd.DataFrame of the observations (timeseries data)
            :param gt_labels: pd.DataFrame of the ground truth labels df
            :param part_labels: dict of all the partitions labels df as values and filenames as keys
            :return
        """
        data_ = data.copy()
        gt_labels_ = gt_labels.copy()

        # select at random but set seed
        random.seed(self.__seed)

        n_clusters = self.__n_clusters
        n_segment = self.__n_segments
        if self.__n_clusters is None:
            n_clusters = len(gt_labels_[SyntheticDataSegmentCols.pattern_id].unique())

        # from pattern ids select n at random
        clusters = gt_labels_[SyntheticDataSegmentCols.pattern_id].unique().tolist()
        selected_patterns = random.sample(clusters, n_clusters)

        # from the remaining segments select n segments
        if self.__n_segments is None:
            n_segment = gt_labels_.shape[0]

        # from the leftover segment ids select n at random
        segments = gt_labels_[gt_labels_[SyntheticDataSegmentCols.pattern_id].isin(selected_patterns)][
            SyntheticDataSegmentCols.segment_id].unique().tolist()
        # if there are still sufficient segment lefts select n_segments from them
        if n_segment < len(segments):
            selected_segs = random.sample(segments, n_segment)
        else:
            selected_segs = segments

        # update data and labels df accordingly
        gt_labels_, p_labels_ = self.__select_segments_and_patterns_from(data_, gt_labels_, part_labels,
                                                                         selected_patterns, selected_segs)

        return gt_labels_, p_labels_, selected_patterns, selected_segs

    @staticmethod
    def __select_segments_and_patterns_from(data_, gt_labels_, partitions: {}, selected_patterns, selected_segs):
        """
        This method updates all the labels df and recalculates their data, the data remains unchanged as the labels df
        is used to access the right observations in the data
        :param data_: pd.DataFrame of timeseries data (observations)
        :param gt_labels_: pd.DataFrame of ground truth labels
        :param partitions: dictionary of key filename and value pd.DataFrame of partitions labels
        :param selected_patterns: patterns to keep
        :param selected_segs: segments to keep
        :return: pd.DataFrame of for ground truth, dictionary of filename aas key and pd.DataFrame as partition
        labels df
        """
        # 1. Update all labels df to only contain selected segments and patterns
        gt_labels_ = update_labels_df(gt_labels_, selected_patterns, selected_segs)
        partitions_ = {file_name: update_labels_df(label, selected_patterns, selected_segs) for file_name, label in
                       partitions}

        # 2. Recalculate the actual correlations and mae
        gt_labels_ = recalculate_labels_df_from_data(data_, gt_labels_)
        updated_partitions = {file_name: recalculate_labels_df_from_data(data_, label) for file_name, label in
                              partitions_}

        return gt_labels_, updated_partitions


def update_labels_df(df, patterns, segments):
    df = df[df[SyntheticDataSegmentCols.pattern_id].isin(patterns) & df[SyntheticDataSegmentCols.segment_id].isin(
        segments)]
    df = df.reset_index(drop=True)
    return df
