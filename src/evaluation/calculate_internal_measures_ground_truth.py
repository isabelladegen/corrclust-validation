import pandas as pd

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.utils.clustering_quality_measures import ClusteringQualityMeasures, silhouette_avg_from_distances, \
    calculate_dbi, calculate_vrc, calculate_pmb
from src.utils.configurations import SyntheticDataVariates
from src.utils.labels_utils import calculate_n_observations_for, calculate_n_segments_outside_tolerance_for, \
    calculate_distance_matrix_for, calculate_cluster_centroids, calculate_distances_between_cluster_centroids, \
    calculate_distances_between_each_segment_and_its_cluster_centroid, calculate_overall_data_correlation, \
    calculate_distance_between_cluster_centroids_and_data, calculate_distance_between_segment_and_data_centroid
from src.utils.load_synthetic_data import load_synthetic_data


class CalculateInternalMeasuresGroundTruth:
    """
    Calculates the provided internal measures for the provided distance measure, and data variant for the runs
    """

    def __init__(self, run_names: [str], internal_measures: [str], distance_measure: str, data_type: str, data_dir: str,
                 data_cols: [str] = SyntheticDataVariates.columns(), round_to: int = 3):
        self.run_names = run_names
        self.distance_measure = distance_measure
        self.__internal_measures = internal_measures
        self.__data_dir = data_dir
        self.__data_type = data_type
        self.__cols = data_cols
        self.__round_to = round_to
        # key= run name value = data
        self.datas = {}
        # key= run name value = label
        self.labels = {}

        # load all ground truth data for run names
        for run_name in run_names:
            data, labels = load_synthetic_data(run_name, self.__data_type, self.__data_dir)
            self.datas[run_name] = data
            self.labels[run_name] = labels

        # calculate each internal measure for each dataset
        file_names = []
        internal_measure_results = {internal_measure: [] for internal_measure in self.__internal_measures}
        patterns_count = []
        segments_count = []
        n_observations = []
        mean_mae = []
        n_segments_outside_tolerance = []
        for run_name in run_names:
            data = self.datas[run_name]
            label = self.labels[run_name]
            data_np = data[self.__cols].to_numpy()
            y_pred = label[SyntheticDataSegmentCols.pattern_id].to_numpy()

            file_names.append(run_name)
            patterns_count.append(len(label[SyntheticDataSegmentCols.pattern_id].unique()))
            segments_count.append(label.shape[0])
            n_observations.append(calculate_n_observations_for(label))
            mean_mae.append(round(label[SyntheticDataSegmentCols.relaxed_mae].mean(), round_to))
            n_segments_outside_tolerance.append(calculate_n_segments_outside_tolerance_for(label))

            if ClusteringQualityMeasures.silhouette_score in self.__internal_measures:
                # 2D np array of dimension n_segments x n_segments with 0 diagonal and symmetric
                distance_matrix = calculate_distance_matrix_for(label, self.distance_measure)
                sil_avg = silhouette_avg_from_distances(distance_matrix, y_pred)
                internal_measure_results[ClusteringQualityMeasures.silhouette_score].append(sil_avg)

            # set calculations to None and only calculate if measures need it
            data_centroid = None
            # VRC, DBI, PMB
            cluster_centroids = None
            # distance between segments and their cluster's centroid - VRC, DBI, PMB
            dist_seg_cluster_centroid = None
            #  distance between the cluster centroids - PMB, DBI
            dist_cluster_centroids = None

            if ClusteringQualityMeasures.dbi in self.__internal_measures:
                if cluster_centroids is None:
                    cluster_centroids = calculate_cluster_centroids(label, data_np)
                if dist_cluster_centroids is None:
                    dist_cluster_centroids = calculate_distances_between_cluster_centroids(
                        cluster_centroids,
                        self.distance_measure)
                if dist_seg_cluster_centroid is None:
                    dist_seg_cluster_centroid = calculate_distances_between_each_segment_and_its_cluster_centroid(
                        label,
                        cluster_centroids,
                        self.distance_measure)

                dbi = calculate_dbi(dist_seg_cluster_centroid, dist_cluster_centroids)
                internal_measure_results[ClusteringQualityMeasures.dbi].append(dbi)

            if ClusteringQualityMeasures.vrc in self.__internal_measures:
                if data_centroid is None:
                    data_centroid = calculate_overall_data_correlation(data_np)
                if cluster_centroids is None:
                    dist_cluster_centroids = calculate_cluster_centroids(label, data_np)
                dist_cluster_centroids_data = calculate_distance_between_cluster_centroids_and_data(
                    cluster_centroids, data_centroid, self.distance_measure)

                if dist_seg_cluster_centroid is None:
                    dist_seg_cluster_centroid = calculate_distances_between_each_segment_and_its_cluster_centroid(
                        label,
                        cluster_centroids,
                        self.distance_measure)

                vrc = calculate_vrc(dist_seg_cluster_centroid, dist_cluster_centroids_data)
                internal_measure_results[ClusteringQualityMeasures.vrc].append(vrc)

            if ClusteringQualityMeasures.pmb in self.__internal_measures:
                if data_centroid is None:
                    data_centroid = calculate_overall_data_correlation(data_np)
                if cluster_centroids is None:
                    cluster_centroids = calculate_cluster_centroids(label, data_np)

                # distances between each segment to overall correlation of data
                dist_seg_overall_data = calculate_distance_between_segment_and_data_centroid(label,
                                                                                             data_centroid,
                                                                                             self.distance_measure)
                # distances between each segment to their cluster centroid
                if dist_seg_cluster_centroid is None:
                    dist_seg_cluster_centroid = calculate_distances_between_each_segment_and_its_cluster_centroid(
                        label,
                        cluster_centroids,
                        self.distance_measure)
                # distances between all cluster centroids
                if dist_cluster_centroids is None:
                    dist_cluster_centroids = calculate_distances_between_cluster_centroids(
                        cluster_centroids,
                        self.distance_measure)
                flat_gt_seg_cluster_centroid = [item for clusterslist in dist_seg_cluster_centroid.values() for
                                                item
                                                in clusterslist]
                pmb = calculate_pmb(dist_seg_overall_data, flat_gt_seg_cluster_centroid,
                                    dist_cluster_centroids.values())
                internal_measure_results[ClusteringQualityMeasures.pmb].append(pmb)

        #assemble df
        self.ground_truth_summary_df = pd.DataFrame({
            DescribeBadPartCols.name: file_names,
            DescribeBadPartCols.n_patterns: patterns_count,
            DescribeBadPartCols.n_segments: segments_count,
            DescribeBadPartCols.n_observations: n_observations,
            DescribeBadPartCols.errors: mean_mae,
            DescribeBadPartCols.n_seg_outside_tol: n_segments_outside_tolerance,
        })
        for measure, values in internal_measure_results.items():
            self.ground_truth_summary_df[measure] = values

