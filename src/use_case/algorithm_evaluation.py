from dataclasses import dataclass

import pandas as pd

from src.data_generation.generate_synthetic_correlated_data import calculate_spearman_correlation
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import SyntheticDataVariates
from src.utils.distance_measures import distance_calculation_method_for, DistanceMeasures


def calculate_overall_cluster_correlations(labels_df: pd.DataFrame, data: pd.DataFrame, round_to: int):
    resulting_clusters = {}  # key cluster id, value overall cluster correlation
    resulting_ids = labels_df[SyntheticDataSegmentCols.pattern_id].unique().tolist()
    for cluster_id in resulting_ids:
        cluster_df = labels_df[labels_df[SyntheticDataSegmentCols.pattern_id] == cluster_id]
        # select all data
        segments = []
        for _, row in cluster_df.iterrows():
            start = row[SyntheticDataSegmentCols.start_idx]
            end = row[SyntheticDataSegmentCols.end_idx]
            length = row[SyntheticDataSegmentCols.length]  # just as check

            # Extract data from start to end (inclusive)
            segment = data.iloc[start:end + 1].copy()

            assert segment.shape[0] == length, "Length and start/end idx did not match"
            segments.append(segment)
        cluster_data = pd.concat(segments, ignore_index=True)
        cluster_cor = calculate_spearman_correlation(cluster_data, round_to=round_to)
        # store results
        resulting_clusters[cluster_id] = cluster_cor
    return resulting_clusters


@dataclass
class EvalMappingCols:
    result_cluster_id = 'result_cluster_id'
    result_overall_cluster_cor = 'result_overall_cluster_cor'
    closest_gt_ids = 'closest_gt_ids'
    closest_gt_overall_cors = 'closest_gt_overall_cors'
    distance = 'distance'


class AlgorithmEvaluation:
    """
    This class can be used to evaluate the results of any algorithm and compare it to the ground truth
    """

    def __init__(self, result_labels_df: pd.DataFrame, gt_labels_df: pd.DataFrame, data: pd.DataFrame, run_name: str,
                 data_dir: str, data_type: str, data_columns: [str] = SyntheticDataVariates.columns(),
                 round_to: int = 3):
        """
        :param result_labels_df: pandas dataframe containing the results of a clustering algorithm, expected columns
        are  SyntheticDataSegmentCols.segment_id, SyntheticDataSegmentCols.start_idx, SyntheticDataSegmentCols.end_idx,
            SyntheticDataSegmentCols.length, SyntheticDataSegmentCols.pattern_id,
            SyntheticDataSegmentCols.actual_correlation. Note that start_idx is 0 based and end_idx is expected to be
            selected!
        :param gt_labels_df: labels_file for matching ground truth
        :param data: data for matching ground truth and algorithms results
        :param run_name: subject name  the files are from
        :param data_dir: data completeness but given as a dir to allow for testing
        :param data_type: generation stage, see SyntheticDataType for options
        :param data_columns: columns of data frame to calculate correlation over (to exclude time stamp)
        :param round_to: decimals results get rounded to
        """
        self._result_labels_df = result_labels_df
        self._gt_labels = gt_labels_df
        self._data = data
        self._run_name = run_name
        self._data_dir = data_dir
        self._data_type = data_type
        self._data_columns = data_columns
        self._round_to = round_to

    def segmentation_ratio(self) -> float:
        """
        Calculates ratio between resulting number of segments and ground truth number of segments
        Interpretation:
        - 1 -> same number of segments
        - > 1 -> algorithm over segments (produces more segments than the ground truth has)
        - < <1 -> algorithm under segments
        :return: ratio
        """
        return round(self._result_labels_df.shape[0] / self._gt_labels.shape[0], self._round_to)

    def segmentation_length_ratio(self, stats: str = '50%'):
        """
        Calculates ratio between resulting stats length of segments and ground truth
        Interpretation:
        - 1 -> same segment length
        - > 1 -> algorithm produces longer segments
        - < <1 -> algorithm produces shorter segments
        :param stats: pandas describe stats to use for length, default median
        :return: ratio
        """
        results_stats = self._result_labels_df[SyntheticDataSegmentCols.length].describe()
        gt_stats = self._gt_labels[SyntheticDataSegmentCols.length].describe()
        return round(results_stats[stats] / gt_stats[stats], self._round_to)

    def map_clusters(self, distance_measure: str = DistanceMeasures.l1_cor_dist):
        """ Method to map resulting clusters to their closest ground truth clusters
        We first calculate the correlations for all observations in the cluster and then
        map the resulting cluster correlations to their closest ground truth cluster correlations
        :param distance_measure: options form DistanceMeasures, defaulted to L1 as it evaluated best for comparing
        correlation matrices -- see our validation paper
        """
        # calculate correlations for all observations in a cluster
        overall_result_corr = calculate_overall_cluster_correlations(self._result_labels_df,
                                                                     self._data[self._data_columns], self._round_to)
        overall_gt_corr = calculate_overall_cluster_correlations(self._gt_labels, self._data[self._data_columns],
                                                                 self._round_to)

        # calculate mapping
        result_cluster_ids = []
        result_overall_cluster_cors = []
        closest_gt_idss = []
        closest_gt_overall_corss = []
        min_distances = []
        for result_id, result_corr in overall_result_corr.items():
            # calculate distances to all ground truth cluster correlations
            distances = {}
            overall_cors = {}
            for gt_id, gt_corr in overall_gt_corr.items():
                calc_distance = distance_calculation_method_for(distance_measure)
                dist = round(calc_distance(result_corr, gt_corr), self._round_to)
                distances[gt_id] = dist
                overall_cors[gt_id] = gt_corr

            # find all ground truth clusters with min dist
            min_dist = min(distances.values())
            closest_gt_clusters = [gt_id for gt_id, dist in distances.items() if dist == min_dist]
            closest_gt_cors = [overall_cors[gt_id] for gt_id in closest_gt_clusters]

            # add results
            result_cluster_ids.append(result_id)
            result_overall_cluster_cors.append(result_corr)  # resulting overall cluster correlation
            closest_gt_idss.append(closest_gt_clusters)  # list of closest gt cluster ids
            closest_gt_overall_corss.append(closest_gt_cors)  # list of gt overall cors
            min_distances.append(min_dist)  # distance between resulting corr and closest

        results = {
            EvalMappingCols.result_cluster_id: result_cluster_ids,
            EvalMappingCols.result_overall_cluster_cor: result_overall_cluster_cors,
            EvalMappingCols.closest_gt_ids: closest_gt_idss,
            EvalMappingCols.closest_gt_overall_cors: closest_gt_overall_corss,
            EvalMappingCols.distance: min_distances
        }
        return pd.DataFrame(results)
