from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_generation.generate_synthetic_correlated_data import calculate_spearman_correlation
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols, calculate_mae
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.utils.configurations import SyntheticDataVariates
from src.utils.distance_measures import distance_calculation_method_for, DistanceMeasures


def calculate_mean_cluster_correlations(labels_df: pd.DataFrame, round_to: int):
    resulting_clusters = {}  # key cluster id, value mean cluster correlation
    resulting_ids = labels_df[SyntheticDataSegmentCols.pattern_id].unique().tolist()
    for cluster_id in resulting_ids:
        cluster_df = labels_df[labels_df[SyntheticDataSegmentCols.pattern_id] == cluster_id]
        correlations_np = np.array(cluster_df[SyntheticDataSegmentCols.actual_correlation].tolist())

        # calculate mean for each column
        resulting_clusters[cluster_id] = np.round(np.mean(correlations_np, axis=0), round_to)

    return resulting_clusters


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


def check_within_tolerance(list1, list2, tolerance: float):
    """ Returns true if all items in list1 are =/- within tolerance of same
    indexed item in list 2
    """
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    assert arr1.shape == arr2.shape
    diff = np.abs(arr1 - arr2)
    return np.all(diff <= 1 + tolerance)


@dataclass
class EvalMappingCols:
    result_cluster_id = 'result_cluster_id'
    closest_gt_ids = 'closest_gt_ids'
    result_mean_cluster_cor = 'result_mean_cluster_cor'
    result_mean_cluster_cor_within_tolerance_of_gt = 'result_mean_cluster_cor_within_tolerance_of_gt'
    mae_result_and_relaxed_pattern = 'mae_mean_result_and_relaxed_pattern'
    distance = 'distance'
    closest_gt_mean_cors = 'closest_gt_mean_cors'
    gt_within_tolerance_of_relaxed_pattern = 'gt_within_tolerance_of_relaxed_pattern'
    mae_gt_and_relaxed_pattern = 'mae_mean_gt_and_relaxed_pattern'


class AlgorithmEvaluation:
    """
    This class can be used to evaluate the results of any algorithm and compare it to the ground truth
    """

    def __init__(self, result_labels_df: pd.DataFrame, gt_labels_df: pd.DataFrame, data: pd.DataFrame, run_name: str,
                 data_dir: str, data_type: str, data_columns: [str] = SyntheticDataVariates.columns(),
                 tolerance: float = 0.1, distance_measure: str = DistanceMeasures.l1_cor_dist, round_to: int = 3):
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
        :param tolerance: tolerated errors for each correlation coefficient either between resulting pattern and
        ground truth, or ground truth and relaxed pattern
        :param distance_measure to calculate distance between resulting correlation structure and ground truth
        relaxed pattern, defaults to L1 as validated in our research
        :param round_to: decimals results get rounded to
        """
        self._result_labels_df = result_labels_df
        self._gt_labels = gt_labels_df
        self._data = data
        self._run_name = run_name
        self._data_dir = data_dir
        self._data_type = data_type
        self._data_columns = data_columns
        self._tolerance = tolerance
        self._distance_measure = distance_measure
        self._round_to = round_to
        self.model_correlation_patterns = ModelCorrelationPatterns()
        self.relaxed_patterns_lookup = self.model_correlation_patterns.relaxed_patterns()
        self.__map_clusters = None  # lazy loaded (only calculated once)

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

    def map_clusters(self, ):
        if self.__map_clusters is None:
            self.__map_clusters = self.__calculate_cluster_mapping(self._distance_measure, self._tolerance)
        return self.__map_clusters

    def pattern_discovery_percentage(self):
        """
        Percentage of ground truth patterns that have been identified within given tolerance in the result
        """
        all_gt_ids = self._gt_labels[SyntheticDataSegmentCols.pattern_id].unique().tolist()

        percentage = (len(self.patterns_discovered()) / len(all_gt_ids)) * 100
        return round(percentage, self._round_to)

    def patterns_discovered(self):
        """
        Returns a unique list of ground truth correlation structures discovered within tolerance
        in the result. Note a pattern can be discovered multiple times.
        :return: list of ids
        """
        # get gt_ids that are within tolerance
        gt_ids_matched_within_tol = set()
        for _, row in self.map_clusters().iterrows():
            gt_ids = row[EvalMappingCols.closest_gt_ids]  # list of matching gt ids
            tolerance_flags = row[EvalMappingCols.result_mean_cluster_cor_within_tolerance_of_gt]
            gt_ids_matched_within_tol.update([an_id for idx, an_id in enumerate(gt_ids) if tolerance_flags[idx]])
        result = list(gt_ids_matched_within_tol)
        result.sort()
        return result

    def pattern_not_discovered(self):
        """
        Returns a list of ground truth correlation structure ids that have not been discovered
        in the results within the tolerance
        :return: list of ids not discovered
        """
        ids_found = set(self.patterns_discovered())
        all_gt_ids = set(self._gt_labels[SyntheticDataSegmentCols.pattern_id].unique())
        result = list(all_gt_ids - ids_found)
        result.sort()
        return result

    def __calculate_cluster_mapping(self, distance_measure, tolerance):
        """ Method to map resulting clusters to their closest ground truth clusters
            We first calculate the correlations for all observations in the cluster and then
            map the resulting cluster correlations to their closest ground truth cluster correlations
            :param tolerance: float that specifies how far the mean resulting correlation coefficient can be from the
            ground truth coefficient
            :param distance_measure: options form DistanceMeasures, defaulted to L1 as it evaluated best for comparing
            correlation matrices -- see our validation paper
            """
        # calculate mean correlations for all segments in a cluster
        mean_result_corr = calculate_mean_cluster_correlations(self._result_labels_df, self._round_to)
        mean_gt_corr = calculate_mean_cluster_correlations(self._gt_labels, self._round_to)
        result_mean_cluster_cor_within_tolerances = []
        mae_result_and_patterns = []
        gt_within_pattern_ground_truths = []
        mae_gt_and_patterns = []
        # calculate mapping
        result_cluster_ids = []
        result_mean_cluster_cors = []
        closest_gt_idss = []
        closest_gt_mean_corss = []
        min_distances = []
        for result_id, result_corr in mean_result_corr.items():
            # calculate distances to all ground truth cluster correlations
            distances = {}
            overall_cors = {}
            for gt_id, gt_corr in mean_gt_corr.items():
                calc_distance = distance_calculation_method_for(distance_measure)
                dist = round(calc_distance(result_corr, gt_corr), self._round_to)
                distances[gt_id] = dist
                overall_cors[gt_id] = gt_corr

            # find all ground truth clusters with min dist
            min_dist = min(distances.values())
            closest_gt_clusters = [gt_id for gt_id, dist in distances.items() if dist == min_dist]
            closest_gt_cors = [overall_cors[gt_id] for gt_id in closest_gt_clusters]

            # calculate mae result corr - each relaxed pattern
            wt_rc = [check_within_tolerance(result_corr, gt_cor, tolerance) for gt_cor in closest_gt_cors]
            wt_gt_rp = []
            rp_s = []
            for idx, gt_id in enumerate(closest_gt_clusters):
                relaxed_pattern = self.relaxed_patterns_lookup[gt_id]
                rp_s.append(relaxed_pattern)
                gt_cor = closest_gt_cors[idx]
                wt_gt_rp.append(check_within_tolerance(gt_cor, relaxed_pattern, tolerance))

            mae_rc_rp = calculate_mae([result_corr] * len(rp_s), rp_s, self._round_to)
            mae_gt_rp = calculate_mae(closest_gt_cors, rp_s, self._round_to)

            # add results
            result_cluster_ids.append(result_id)
            result_mean_cluster_cors.append(result_corr)  # resulting overall cluster correlation
            closest_gt_idss.append(closest_gt_clusters)  # list of closest gt cluster ids
            closest_gt_mean_corss.append(closest_gt_cors)  # list of gt overall cors
            min_distances.append(min_dist)  # distance between resulting corr and closest
            result_mean_cluster_cor_within_tolerances.append(wt_rc)
            gt_within_pattern_ground_truths.append(wt_gt_rp)
            mae_result_and_patterns.append(mae_rc_rp)
            mae_gt_and_patterns.append(mae_gt_rp)
        results = {
            EvalMappingCols.result_cluster_id: result_cluster_ids,
            EvalMappingCols.result_mean_cluster_cor: result_mean_cluster_cors,
            EvalMappingCols.closest_gt_ids: closest_gt_idss,
            EvalMappingCols.closest_gt_mean_cors: closest_gt_mean_corss,
            EvalMappingCols.distance: min_distances,
            EvalMappingCols.mae_result_and_relaxed_pattern: mae_result_and_patterns,
            EvalMappingCols.mae_gt_and_relaxed_pattern: mae_gt_and_patterns,
            EvalMappingCols.result_mean_cluster_cor_within_tolerance_of_gt: result_mean_cluster_cor_within_tolerances,
            EvalMappingCols.gt_within_tolerance_of_relaxed_pattern: gt_within_pattern_ground_truths,
        }
        return pd.DataFrame(results)
