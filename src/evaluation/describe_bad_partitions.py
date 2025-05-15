from dataclasses import dataclass

import pandas as pd

from src.utils.clustering_quality_measures import silhouette_avg_from_distances, calculate_pmb, \
    clustering_jaccard_coeff, ClusteringQualityMeasures, calculate_dbi, calculate_vrc
from src.utils.configurations import SYNTHETIC_DATA_DIR, SyntheticDataVariates
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.labels_utils import calculate_overall_data_correlation, \
    calculate_distance_between_segment_and_data_centroid, calculate_cluster_centroids, \
    calculate_distances_between_each_segment_and_its_cluster_centroid, calculate_distances_between_cluster_centroids, \
    calculate_y_pred_from, calculate_distance_matrix_for, calculate_y_pred_and_updated_gt_y_pred_from, \
    calculate_n_observations_for, calculate_n_segments_within_tolerance_for, calculate_n_segments_outside_tolerance_for, \
    calculate_distance_between_cluster_centroids_and_data
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data_and_labels_for_bad_partitions


@dataclass
class DescribeBadPartCols:
    n_seg_outside_tol = "n segments outside tolerance"
    name: str = "file name"
    n_patterns: str = "n patterns"
    n_segments: str = "n segments"
    n_observations: str = "n observations"
    errors: str = "mean relaxed mae"  # using relaxed mae now
    n_wrong_clusters: str = "n wrong clusters"
    n_obs_shifted: str = "n obs shifted"


default_internal_measures = [ClusteringQualityMeasures.silhouette_score]
default_external_measures = [ClusteringQualityMeasures.jaccard_index]


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
                 data_cols: [str] = SyntheticDataVariates.columns(), data_dir: str = SYNTHETIC_DATA_DIR,
                 round_to: int = 3, data: pd.DataFrame = None, gt_label: pd.DataFrame = None,
                 partitions: pd.DataFrame = None):
        self.ds_name = ds_name
        self.distance_measure = distance_measure
        self.__internal_measures = internal_measures
        self.__external_measures = external_measures
        self.__data_dir = data_dir
        self.__data_type = data_type
        self.__cols = data_cols

        # lazy loaded fields
        self.__segment_correlations = None
        self.__overall_corr = None
        self.__cluster_correlations = None

        # load data if it has not been loaded yet otherwise use loaded data
        if data is None or gt_label is None or partitions is None:
            data, gt_label, partitions = load_synthetic_data_and_labels_for_bad_partitions(ds_name,
                                                                                           data_type=data_type,
                                                                                           data_dir=data_dir)

        self.data = data  # time series data df
        self.gt_label = gt_label  # ground truth labels df
        self.partitions = partitions  # dictionary of filename as key and bad partition labels_df as value

        # calculate full y pred for ground truth
        full_gt_y_pred = calculate_y_pred_from(self.gt_label)

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
        dbis = []
        vrcs = []

        # 1. add ground truth information to summary df
        file_names.append(ds_name)
        patterns_count.append(len(self.gt_label[SyntheticDataSegmentCols.pattern_id].unique()))
        segments_count.append(self.gt_label.shape[0])
        n_observations.append(calculate_n_observations_for(self.gt_label))
        mean_mae.append(round(self.gt_label[SyntheticDataSegmentCols.relaxed_mae].mean(), round_to))
        n_segments_outside_tolerance.append(calculate_n_segments_outside_tolerance_for(self.gt_label))
        n_wrong_clusters.append(0)
        n_obs_shifted.append(0)

        if ClusteringQualityMeasures.jaccard_index in self.__external_measures:
            jaccards.append(1)

        if ClusteringQualityMeasures.silhouette_score in self.__internal_measures:
            # 2D np array of dimension n_segments x n_segments with 0 diagonal and symmetric
            self.gt_distance_matrix = calculate_distance_matrix_for(self.gt_label, self.distance_measure)
            sil_avg = silhouette_avg_from_distances(self.gt_distance_matrix, self.gt_patterns)
            sils.append(sil_avg)

        # set calculations to None and only calculate if measures need it
        self.data_centroid = None
        # VRC, DBI, PMB
        self.gt_cluster_centroids = None
        # distance between segments and their cluster's centroid - VRC, DBI, PMB
        self.gt_dist_seg_cluster_centroid = None
        self.gt_dist_cluster_centroids = None  # distance between the cluster centroids - PMB, DBI

        if ClusteringQualityMeasures.dbi in self.__internal_measures:
            if self.gt_cluster_centroids is None:
                self.gt_cluster_centroids = calculate_cluster_centroids(self.gt_label, self.gt_data_np)
            if self.gt_dist_cluster_centroids is None:
                self.gt_dist_cluster_centroids = calculate_distances_between_cluster_centroids(
                    self.gt_cluster_centroids,
                    self.distance_measure)
            if self.gt_dist_seg_cluster_centroid is None:
                self.gt_dist_seg_cluster_centroid = calculate_distances_between_each_segment_and_its_cluster_centroid(
                    self.gt_label,
                    self.gt_cluster_centroids,
                    self.distance_measure)

            dbi = calculate_dbi(self.gt_dist_seg_cluster_centroid, self.gt_dist_cluster_centroids)
            dbis.append(dbi)

        if ClusteringQualityMeasures.vrc in self.__internal_measures:
            if self.data_centroid is None:
                self.data_centroid = calculate_overall_data_correlation(self.gt_data_np)
            if self.gt_cluster_centroids is None:
                self.gt_dist_cluster_centroids = calculate_cluster_centroids(self.gt_label, self.gt_data_np)
            self.gt_dist_cluster_centroids_data = calculate_distance_between_cluster_centroids_and_data(
                self.gt_cluster_centroids, self.data_centroid, self.distance_measure)

            if self.gt_dist_seg_cluster_centroid is None:
                self.gt_dist_seg_cluster_centroid = calculate_distances_between_each_segment_and_its_cluster_centroid(
                    self.gt_label,
                    self.gt_cluster_centroids,
                    self.distance_measure)

            vrc = calculate_vrc(self.gt_dist_seg_cluster_centroid, self.gt_dist_cluster_centroids_data)
            vrcs.append(vrc)

        if ClusteringQualityMeasures.pmb in self.__internal_measures:
            if self.data_centroid is None:
                self.data_centroid = calculate_overall_data_correlation(self.gt_data_np)
            if self.gt_cluster_centroids is None:
                self.gt_cluster_centroids = calculate_cluster_centroids(self.gt_label, self.gt_data_np)

            # distances between each segment to overall correlation of data
            self.gt_dist_seg_overall_data = calculate_distance_between_segment_and_data_centroid(self.gt_label,
                                                                                                 self.data_centroid,
                                                                                                 self.distance_measure)
            # distances between each segment to their cluster centroid
            if self.gt_dist_seg_cluster_centroid is None:
                self.gt_dist_seg_cluster_centroid = calculate_distances_between_each_segment_and_its_cluster_centroid(
                    self.gt_label,
                    self.gt_cluster_centroids,
                    self.distance_measure)
            # distances between all cluster centroids
            if self.gt_dist_cluster_centroids is None:
                self.gt_dist_cluster_centroids = calculate_distances_between_cluster_centroids(
                    self.gt_cluster_centroids,
                    self.distance_measure)
            flat_gt_seg_cluster_centroid = [item for clusterslist in self.gt_dist_seg_cluster_centroid.values() for item
                                            in clusterslist]
            pmb = calculate_pmb(self.gt_dist_seg_overall_data, flat_gt_seg_cluster_centroid,
                                self.gt_dist_cluster_centroids.values())
            pmbs.append(pmb)

        # to calculate the shift
        gt_first_seg_end_idx = self.gt_label.loc[0, SyntheticDataSegmentCols.end_idx]

        for file_name, p_label in self.partitions.items():
            p_data_np = self.data[self.__cols].to_numpy()
            p_mean_mae_error = round(p_label[SyntheticDataSegmentCols.relaxed_mae].mean(), round_to)
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
            if ClusteringQualityMeasures.jaccard_index in self.__external_measures:
                p_y_pred, p_y_pred_gt = calculate_y_pred_and_updated_gt_y_pred_from(p_label, full_gt_y_pred)
                p_jacc = clustering_jaccard_coeff(p_y_pred, p_y_pred_gt, round_to)
                jaccards.append(p_jacc)

            if ClusteringQualityMeasures.silhouette_score in self.__internal_measures:
                # 2D np array of dimension n_segments x n_segments with 0 diagonal and symmetric
                p_distance_matrix = calculate_distance_matrix_for(p_label, self.distance_measure)
                sil_avg = silhouette_avg_from_distances(p_distance_matrix, p_patterns)
                sils.append(sil_avg)

            p_data_centroid = None
            # VRC, DBI, PMB
            p_cluster_centroids = None
            # distance between segments and their cluster's centroid - VRC, DBI, PMB
            p_dist_seg_cluster_centroid = None
            p_dist_cluster_centroids = None  # distance between the cluster centroids - PMB, DBI

            if ClusteringQualityMeasures.dbi in self.__internal_measures:
                if p_cluster_centroids is None:
                    p_cluster_centroids = calculate_cluster_centroids(p_label, p_data_np)
                if p_dist_cluster_centroids is None:
                    p_dist_cluster_centroids = calculate_distances_between_cluster_centroids(
                        p_cluster_centroids,
                        self.distance_measure)
                if p_dist_seg_cluster_centroid is None:
                    p_dist_seg_cluster_centroid = calculate_distances_between_each_segment_and_its_cluster_centroid(
                        p_label,
                        p_cluster_centroids,
                        self.distance_measure)

                dbi = calculate_dbi(p_dist_seg_cluster_centroid, p_dist_cluster_centroids)
                dbis.append(dbi)

            if ClusteringQualityMeasures.vrc in self.__internal_measures:
                if p_data_centroid is None:
                    p_data_centroid = calculate_overall_data_correlation(p_data_np)
                if p_cluster_centroids is None:
                    p_cluster_centroids = calculate_cluster_centroids(p_label, p_data_np)
                p_dist_cluster_centroids_data = calculate_distance_between_cluster_centroids_and_data(
                    p_cluster_centroids, p_data_centroid, self.distance_measure)

                if p_dist_seg_cluster_centroid is None:
                    p_dist_seg_cluster_centroid = calculate_distances_between_each_segment_and_its_cluster_centroid(
                        p_label,
                        p_cluster_centroids,
                        self.distance_measure)

                vrc = calculate_vrc(p_dist_seg_cluster_centroid, p_dist_cluster_centroids_data)
                vrcs.append(vrc)

            if ClusteringQualityMeasures.pmb in self.__internal_measures:
                if p_data_centroid is None:
                    p_data_centroid = calculate_overall_data_correlation(p_data_np)
                if p_cluster_centroids is None:
                    p_cluster_centroids = calculate_cluster_centroids(p_label, p_data_np)
                # distances between each segment to overall correlation of data
                p_dist_seg_overall_data = calculate_distance_between_segment_and_data_centroid(p_label,
                                                                                               p_data_centroid,
                                                                                               self.distance_measure)
                # distances between each segment to their cluster centroid
                if p_dist_seg_cluster_centroid is None:
                    p_dist_seg_cluster_centroid = calculate_distances_between_each_segment_and_its_cluster_centroid(
                        p_label,
                        p_cluster_centroids,
                        self.distance_measure)
                flat_p_dist_seg_cluster_centroid = [item for clusterslist in p_dist_seg_cluster_centroid.values() for
                                                    item in
                                                    clusterslist]
                # distances between all cluster centroids
                p_dist_between_clusters = calculate_distances_between_cluster_centroids(p_cluster_centroids,
                                                                                        self.distance_measure)
                pmb = calculate_pmb(p_dist_seg_overall_data, flat_p_dist_seg_cluster_centroid,
                                    p_dist_between_clusters.values())
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

        if ClusteringQualityMeasures.jaccard_index in self.__external_measures:
            self.summary_df[ClusteringQualityMeasures.jaccard_index] = jaccards

        if ClusteringQualityMeasures.silhouette_score in self.__internal_measures:
            self.summary_df[ClusteringQualityMeasures.silhouette_score] = sils

        if ClusteringQualityMeasures.dbi in self.__internal_measures:
            self.summary_df[ClusteringQualityMeasures.dbi] = dbis

        if ClusteringQualityMeasures.vrc in self.__internal_measures:
            self.summary_df[ClusteringQualityMeasures.vrc] = vrcs

        if ClusteringQualityMeasures.pmb in self.__internal_measures:
            self.summary_df[ClusteringQualityMeasures.pmb] = pmbs

        # sort summary df
        self.summary_df.sort_values(by=ClusteringQualityMeasures.jaccard_index, ascending=False, inplace=True)

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
        partition_means = [df[SyntheticDataSegmentCols.relaxed_mae].mean() for df in labels_dfs]
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
        return self.summary_df[ClusteringQualityMeasures.jaccard_index].describe().round(round_to)

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
