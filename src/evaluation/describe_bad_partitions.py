import os
import random
from dataclasses import dataclass

import pandas as pd

from src.utils.clustering_quality_measures import silhouette_avg_from_distances, calculate_pmb, clustering_jaccard_coeff
from src.utils.configurations import SYNTHETIC_DATA_DIR, SyntheticDataVariates, \
    get_internal_measures_summary_file_name, GENERATED_DATASETS_FILE_PATH, ROOT_RESULTS_DIR, \
    internal_measure_calculation_dir_for
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols, \
    recalculate_labels_df_from_data
from src.utils.distance_measures import DistanceMeasures
from src.utils.labels_utils import calculate_overall_data_correlation, \
    calculate_distance_between_segment_and_data_centroid, calculate_cluster_centroids, \
    calculate_distances_between_each_segment_and_its_cluster_centroid, calculate_distances_between_cluster_centroids, \
    calculate_y_pred_from, calculate_distance_matrix_for, calculate_y_pred_and_updated_gt_y_pred_from, \
    calculate_n_observations_for, calculate_n_segments_within_tolerance_for, calculate_n_segments_outside_tolerance_for
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data_and_labels_for_bad_partitions


@dataclass
class DescribeBadPartCols:
    n_seg_outside_tol = "n segments outside tolerance"
    pmb: str = "PMB"
    silhouette_score: str = "SCW"
    jaccard_index: str = "Jaccard"
    name: str = "file name"
    n_patterns: str = "n patterns"
    n_segments: str = "n segments"
    n_observations: str = "n observations"
    errors: str = "mean mae"
    n_wrong_clusters: str = "n wrong clusters"
    n_obs_shifted: str = "n obs shifted"


default_internal_measures = [DescribeBadPartCols.silhouette_score]
default_external_measures = [DescribeBadPartCols.jaccard_index]


def select_data_from_df(data: pd.DataFrame, label: pd.DataFrame):
    """
    Selects only the observations that are actually indexed in the label df
    :param data: pd.DataFrame of the full dataset
    :param label: pd.DataFrame of the labels that should be used to select rows from the data
    """
    n_segments = label.shape[0]

    # select all observations still referenced in labels_df
    segments_data = []
    for seg in range(n_segments):
        start_idx = label.loc[seg, SyntheticDataSegmentCols.start_idx]
        end_idx = label.loc[seg, SyntheticDataSegmentCols.end_idx]
        length = label.loc[seg, SyntheticDataSegmentCols.length]

        seg_data = data.iloc[start_idx:end_idx + 1, :]
        assert seg_data.shape[0] == length, "Selected wrong indices"
        segments_data.append(seg_data)

    # we're leaving the index original to be able to know which bits of the datasets were selected
    updated_data = pd.concat(segments_data, axis=0)
    return updated_data


class DescribeBadPartitions:
    def __init__(self, ds_name, distance_measure: str, data_type: str = SyntheticDataType.non_normal_correlated,
                 internal_measures: [] = default_internal_measures, external_measures: [] = default_external_measures,
                 data_cols: [str] = SyntheticDataVariates.columns(), drop_n_clusters: int = 0, drop_n_segments: int = 0,
                 seed: int = 600,
                 data_dir: str = SYNTHETIC_DATA_DIR, round_to: int = 3):
        self.ds_name = ds_name
        self.distance_measure = distance_measure
        self.__internal_measures = internal_measures
        self.__external_measures = external_measures
        self.__seed = seed
        # number of clusters to drop, if None this is 0
        self.__drop_n_clusters = drop_n_clusters
        # number of segments to drop, if None this is 0
        self.__drop_n_segments = drop_n_segments
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

        self.data = data  # time series data df this will not be modified if we drop clusters or segments
        self.gt_label = gt_label  # ground truth labels df this will be modified if we drop clusters or segments
        self.partitions = partitions  # dictionary of filename as key and bad partition labels_df as value

        # calculate full y pred for ground truth before dropping segments or clusters, this is required for
        # jaccard index of partitions where the indices have shifted we need to select the to original pattern
        # of the new indices after some segments might have been dropped
        full_gt_y_pred = calculate_y_pred_from(self.gt_label)

        # To reduce the number of clusters and segments in the data we can give a number of clusters or segments
        # to select. In that case clusters and segments get dropped. The data remains unchanged
        if self.__drop_clusters_or_segments():
            self.gt_label, self.partitions, selected_patterns, selected_segs = self.__drop_clusters_or_segments_from_data(
                self.data, self.gt_label, self.partitions)

        # 1D np array of length n_segments
        self.gt_patterns = self.gt_label[SyntheticDataSegmentCols.pattern_id].to_numpy()

        self.gt_data_np = self.data[self.__cols].to_numpy()

        # Create overview description df that includes ground truth and all bad partitions
        file_names = []
        patterns_count = []
        segments_count = []
        n_observations = []
        mean_mae = []
        n_segments_outside_tolerance = []
        n_wrong_clusters = []
        n_obs_shifted = []
        jaccards = []
        sils = []
        pmbs = []

        # 1. add ground truth information to summary df
        file_names.append(ds_name)
        patterns_count.append(len(self.gt_label[SyntheticDataSegmentCols.pattern_id].unique()))
        segments_count.append(self.gt_label.shape[0])
        n_observations.append(calculate_n_observations_for(self.gt_label))
        mean_mae.append(round(self.gt_label[SyntheticDataSegmentCols.mae].mean(), round_to))
        n_segments_outside_tolerance.append(calculate_n_segments_outside_tolerance_for(self.gt_label))
        n_wrong_clusters.append(0)
        n_obs_shifted.append(0)

        if DescribeBadPartCols.jaccard_index in self.__external_measures:
            jaccards.append(1)

        if DescribeBadPartCols.silhouette_score in self.__internal_measures:
            # 2D np array of dimension n_segments x n_segments with 0 diagonal and symmetric
            self.gt_distance_matrix = calculate_distance_matrix_for(self.gt_label, self.distance_measure)
            sil_avg = silhouette_avg_from_distances(self.gt_distance_matrix, self.gt_patterns)
            sils.append(sil_avg)

        if DescribeBadPartCols.pmb in self.__internal_measures:
            self.data_centroid = calculate_overall_data_correlation(self.gt_data_np)
            self.gt_cluster_centroids = calculate_cluster_centroids(self.gt_label, self.gt_data_np)

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
            pmb = calculate_pmb(self.gt_dist_seg_overall_data, self.gt_dist_seg_cluster,
                                self.gt_dist_between_clusters.values())
            pmbs.append(pmb)

        # to calculate the shift
        gt_first_seg_end_idx = self.gt_label.loc[0, SyntheticDataSegmentCols.end_idx]

        for file_name, p_label in self.partitions.items():
            p_data_np = self.data[self.__cols].to_numpy()
            p_mean_mae_error = round(p_label[SyntheticDataSegmentCols.mae].mean(), round_to)
            p_patterns = p_label[SyntheticDataSegmentCols.pattern_id].to_numpy()

            # calculate and add info to new df for this partition
            file_names.append(file_name)
            patterns_count.append(len(set(p_patterns)))
            segments_count.append(p_label.shape[0])
            n_observations.append(calculate_n_observations_for(p_label))
            mean_mae.append(p_mean_mae_error)
            n_segments_outside_tolerance.append(calculate_n_segments_outside_tolerance_for(p_label))

            # calculate how many patterns were changed and how many observations shifted for the partition
            n_wrong_clusters.append(sum(i != j for i, j in zip(self.gt_patterns, p_patterns)))
            p_first_seg_end_idx = p_label.loc[0, SyntheticDataSegmentCols.end_idx]
            n_obs_shifted.append(p_first_seg_end_idx - gt_first_seg_end_idx)

            # calculate external and internal measures
            if DescribeBadPartCols.jaccard_index in self.__external_measures:
                p_y_pred, p_y_pred_gt = calculate_y_pred_and_updated_gt_y_pred_from(p_label, full_gt_y_pred)
                p_jacc = clustering_jaccard_coeff(p_y_pred, p_y_pred_gt, round_to)
                jaccards.append(p_jacc)

            if DescribeBadPartCols.silhouette_score in self.__internal_measures:
                # 2D np array of dimension n_segments x n_segments with 0 diagonal and symmetric
                p_distance_matrix = calculate_distance_matrix_for(p_label, self.distance_measure)
                sil_avg = silhouette_avg_from_distances(p_distance_matrix, p_patterns)
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
                pmb = calculate_pmb(p_dist_seg_overall_data, p_dist_seg_cluster, p_dist_between_clusters.values())
                pmbs.append(pmb)

        # put summary df together
        self.summary_df = pd.DataFrame({
            DescribeBadPartCols.name: file_names,
            DescribeBadPartCols.n_patterns: patterns_count,
            DescribeBadPartCols.n_segments: segments_count,
            DescribeBadPartCols.n_observations: n_observations,
            DescribeBadPartCols.errors: mean_mae,
            DescribeBadPartCols.n_seg_outside_tol: n_segments_outside_tolerance,
            DescribeBadPartCols.n_wrong_clusters: n_wrong_clusters,
            DescribeBadPartCols.n_obs_shifted: n_obs_shifted,
        })

        if DescribeBadPartCols.jaccard_index in self.__external_measures:
            self.summary_df[DescribeBadPartCols.jaccard_index] = jaccards

        if DescribeBadPartCols.silhouette_score in self.__internal_measures:
            self.summary_df[DescribeBadPartCols.silhouette_score] = sils

        if DescribeBadPartCols.pmb in self.__internal_measures:
            self.summary_df[DescribeBadPartCols.pmb] = pmbs

    def __drop_clusters_or_segments(self):
        """
        Check if we need to drop clusters or segments from the data
        :return: True if clusters, segments or both need to be dropped, otherwise False
        """
        return self.__drop_n_clusters > 0 or self.__drop_n_segments > 0

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
        total_clusters = len(gt_labels_[SyntheticDataSegmentCols.pattern_id].unique())
        total_segments = gt_labels_.shape[0]

        # clusters to select
        n_clusters = total_clusters - self.__drop_n_clusters
        assert n_clusters >= 2, "Min number of clusters to keep is 2 otherwise internal measures are not valid"

        # from pattern ids select n at random
        clusters = gt_labels_[SyntheticDataSegmentCols.pattern_id].unique().tolist()
        selected_patterns = random.sample(clusters, n_clusters)

        # segments to select
        n_segment = total_segments - self.__drop_n_segments
        # check how many segments that are left
        segments_left = gt_labels_[gt_labels_[SyntheticDataSegmentCols.pattern_id].isin(selected_patterns)][
            SyntheticDataSegmentCols.segment_id].unique().tolist()

        # if there are still sufficient segment lefts select n_segments from them (haven't been dropped by
        # dropping clusters
        if n_segment < len(segments_left):
            selected_segs = random.sample(segments_left, n_segment)
        else:
            selected_segs = segments_left

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
                       partitions.items()}

        # ensure we have two clusters and 2 left
        assert len(gt_labels_[SyntheticDataSegmentCols.pattern_id].unique()) > 1, "We need at least two clusters"
        for file_name, p_labels in partitions_.items():
            error_msg = "We need at least two clusters. Not the case for " + file_name
            assert len(p_labels[SyntheticDataSegmentCols.pattern_id].unique()) > 1, error_msg

        # 2. Recalculate the actual correlations and mae
        gt_labels_ = recalculate_labels_df_from_data(data_, gt_labels_)
        updated_partitions = {file_name: recalculate_labels_df_from_data(data_, label) for file_name, label in
                              partitions_.items()}

        return gt_labels_, updated_partitions

    def n_segment_within_tolerance_stats(self, round_to: int = 3):
        """ Calculate and return stats df across the partitions for the dataset"""
        labels_dfs = list(self.partitions.values())
        labels_dfs.append(self.gt_label)  # add ground truth
        n_within_tolerance = [calculate_n_segments_within_tolerance_for(df) for df in labels_dfs]

        return pd.Series(n_within_tolerance).describe().round(round_to)

    def n_segment_outside_tolerance_stats(self, round_to: int = 3):
        """ Calculate and return stats df across the partitions for the dataset"""
        return self.summary_df[DescribeBadPartCols.n_seg_outside_tol].describe().round(round_to)

    def mae_stats(self, round_to: int = 3):
        """
        Calculate and return mae stats df across the partitions for the dataset. We use the mean
        value per partition to match what happens with the Jaccard Index that is already a per
        partition measure.
        """
        labels_dfs = list(self.partitions.values())
        labels_dfs.append(self.gt_label)  # add ground truth
        partition_means = [df[SyntheticDataSegmentCols.mae].mean() for df in labels_dfs]
        return pd.Series(partition_means).describe().round(round_to)

    def segment_length_stats(self, round_to: int = 3):
        """
        Calculate and return segment length stats df across the partitions for the dataset. We use the mean
        value per partition to match what happens with the Jaccard Index that is already a per
        partition measure.
        """
        labels_dfs = list(self.partitions.values())
        labels_dfs.append(self.gt_label)  # add ground truth
        partition_means = [df[SyntheticDataSegmentCols.length].mean() for df in labels_dfs]
        return pd.Series(partition_means).describe().round(round_to)

    def jaccard_stats(self, round_to: int = 3):
        """
            Calculate and return jaccard stats df across the partitions for the dataset
        """
        return self.summary_df[DescribeBadPartCols.jaccard_index].describe().round(round_to)

    def n_wrong_cluster_stats(self, round_to: int = 3):
        """
           Calculate and return n_wrong_clusters stats df across the partitions for the dataset
        """
        return self.summary_df[DescribeBadPartCols.n_wrong_clusters].describe().round(round_to)

    def n_obs_shifted_stats(self, round_to: int = 3):
        """
           Calculate and return n obs shifted stats df across the partitions for the dataset
        """
        return self.summary_df[DescribeBadPartCols.n_obs_shifted].describe().round(round_to)


def update_labels_df(df, patterns, segments):
    # first select patterns
    df = df[df[SyntheticDataSegmentCols.pattern_id].isin(patterns)]
    # from the left over df select the segments
    df = df[df[SyntheticDataSegmentCols.segment_id].isin(segments)]
    df = df.reset_index(drop=True)
    return df


def read_internal_measures_calculation(overall_ds_name: str, data_type: str, root_results_dir: str, data_dir: str,
                                       distance_measure: str, n_dropped_clusters=0, n_dropped_segments=0,
                                       generated_ds_csv: str = GENERATED_DATASETS_FILE_PATH):
    """ Reads the summary results files from the internal measure calculation for the datasets in the overall_ds_name
    :param overall_ds_name: a name for the dataset we're using e.g. n30 or n2
    :param data_type: which datatype to use see SyntheticDataType
    :param root_results_dir: directory where to store the results, it will use a subdirectory based on
    the distance measure, and the data type and other information
    :param n_dropped_clusters: number of clusters that were dropped if 0 it reads results from full ds
    :param n_dropped_segments: number of segments that were dropped if 0 it reads results from full ds
    :param generated_ds_csv: csv file with a list of all the run names
    """
    results_dir = internal_measure_calculation_dir_for(overall_ds_name,
                                                       data_type,
                                                       root_results_dir,
                                                       data_dir,
                                                       distance_measure,
                                                       n_dropped_clusters,
                                                       n_dropped_segments)
    datasets = pd.read_csv(generated_ds_csv)['Name'].tolist()

    results_for_run = []
    for ds_name in datasets:
        file_name = get_internal_measures_summary_file_name(ds_name)
        file_to_read = os.path.join(results_dir, file_name)
        df = pd.read_csv(file_to_read, index_col=0)
        # sort by Jaccard index
        df = df.sort_values(by=DescribeBadPartCols.jaccard_index)
        df.reset_index(drop=True, inplace=True)
        results_for_run.append(df)
    return results_for_run


def calculate_internal_measures_for_all_partitions(overall_ds_name: str, dataset_names: [str], data_type: str,
                                                   data_dir: str,
                                                   results_dir: str,
                                                   distance_measure: str,
                                                   internal_measures: [str], drop_clusters=0, drop_segments=0):
    store_results_in = internal_measure_calculation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type=data_type,
        results_dir=results_dir,
        data_dir=data_dir,
        distance_measure=distance_measure,
        drop_clusters=drop_clusters,
        drop_segments=drop_segments)
    partitions = []
    for ds_name in dataset_names:
        print(ds_name)
        # we don't vary the seed so all datasets will select the same clusters and segments
        sum_df = DescribeBadPartitions(ds_name=ds_name, data_type=data_type, data_dir=data_dir,
                                       distance_measure=distance_measure,
                                       internal_measures=internal_measures, drop_n_segments=drop_segments,
                                       drop_n_clusters=drop_clusters).summary_df.copy()
        file_name = get_internal_measures_summary_file_name(ds_name)
        sum_df.to_csv(os.path.join(store_results_in, file_name))
        partitions.append(sum_df)


def run_internal_measure_calculation_for_dataset(overall_ds_name: str,
                                                 generated_ds_csv: str = GENERATED_DATASETS_FILE_PATH,
                                                 distance_measure: str = DistanceMeasures.l1_cor_dist,
                                                 data_type: str = SyntheticDataType.non_normal_correlated,
                                                 data_dir: str = SYNTHETIC_DATA_DIR,
                                                 results_dir: str = ROOT_RESULTS_DIR,
                                                 internal_measures: [str] = [DescribeBadPartCols.silhouette_score,
                                                                             DescribeBadPartCols.pmb],
                                                 n_dropped_clusters: [int] = [],
                                                 n_dropped_segments: [int] = [],
                                                 ):
    """ Calculates the internal measure assessment for all ds in the csv files of the generated runs
    :param overall_ds_name: a name for the dataset we're using e.g. n30 or n2
    :param generated_ds_csv: full path to the csv file of the generated runs, using the names to locate the data
    :param distance_measure: name of distance measure to run assessment for
    :param data_type: which datatype to use see SyntheticDataType
    :param data_dir: where to read the data from
    :param results_dir: directory where to store the results, it will use a subdirectory based on the distance measure,
    and the data type
    :param internal_measures: list of internal measures to assess
    :param n_dropped_clusters: list of the number of clusters to drop in each run, if empty then we run the assessment
    on all the cluster
    :param n_dropped_segments: list of the number of segments to drop in each run, if empty then we run the assessment
    using all segments
    """
    # read file of which datasets to assess
    dataset_names = pd.read_csv(generated_ds_csv)['Name'].tolist()

    # decide which assessment to run
    if len(n_dropped_clusters) == 0 and len(n_dropped_segments) == 0:
        calculate_internal_measures_for_all_partitions(overall_ds_name, dataset_names, data_type, data_dir, results_dir,
                                                       distance_measure,
                                                       internal_measures)
    else:
        # run evaluation for all dropped clusters and for all dropped segments separately
        # for this we just do clusters first
        for n_clus in n_dropped_clusters:
            calculate_internal_measures_for_all_partitions(overall_ds_name, dataset_names, data_type, data_dir,
                                                           results_dir, distance_measure,
                                                           internal_measures, drop_clusters=n_clus)
        # and second we do segments
        for n_seg in n_dropped_segments:
            calculate_internal_measures_for_all_partitions(overall_ds_name, dataset_names, data_type, data_dir,
                                                           results_dir, distance_measure,
                                                           internal_measures, drop_segments=n_seg)


if __name__ == "__main__":
    overall_dataset_name = "n2"
    run_internal_measure_calculation_for_dataset(overall_dataset_name)
