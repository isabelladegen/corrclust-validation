from itertools import combinations

import numpy as np
import pandas as pd

from src.data_generation.generate_synthetic_correlated_data import calculate_spearman_correlation
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.distance_measures import distance_calculation_method_for


def calculate_n_observations_for(labels_df: pd.DataFrame):
    """Calculates the numbers of observations in the data based on the given labels df
    :param labels_df: a labels dataframe
    :return: number of observations
    """
    return sum(labels_df[SyntheticDataSegmentCols.length])


def calculate_y_pred_from(labels_df: pd.DataFrame):
    """
    Calculates the y pred from the labels df. For this each pattern id occurs for each observation in the data
    :param labels_df: a labels dataframe
    :return: np.array of y pred
    """
    patterns = labels_df[SyntheticDataSegmentCols.pattern_id].to_numpy()
    lengths = labels_df[SyntheticDataSegmentCols.length].to_numpy()
    return np.repeat(patterns, lengths)


def calculate_y_pred_and_updated_gt_y_pred_from(labels_df: pd.DataFrame, gt_y_pred: np.ndarray):
    """
    Calculates the y pred from the labels df for the partition and selects these indices from gt_y_pred.
    For this each pattern id occurs for each observation in the data
    :param labels_df: a labels dataframe for a partition
    :param gt_y_pred: a np.array of the full gt_y_pred (ensure gt_y_pred has not been shortened)
    :return: np.array of y pred and updated gt_y_pred for the partition indices
    """
    # 1. get y_pred from labels
    y_pred = calculate_y_pred_from(labels_df)

    # 2. select all of gt_y pred that have indices in the labels_df
    # create mask for selection
    mask = np.zeros(len(gt_y_pred), dtype=bool)
    for start, end in zip(labels_df[SyntheticDataSegmentCols.start_idx], labels_df[SyntheticDataSegmentCols.end_idx]):
        # end index needs to be included therefore +1
        mask[start:end + 1] = True
    updated_y_pred = gt_y_pred[mask]

    return y_pred, updated_y_pred


def calculate_overall_data_correlation(data: np.array, round_to: int = 3):
    """
    Calculates the overall correlation in the data
    :param data: np.array of the data to calculate the correlation for
    :param round_to: number of decimals to round to
    :return: [] upper triangular correlations
    """
    return calculate_spearman_correlation(data, round_to)


def calculate_cluster_centroids(labels_df: pd.DataFrame, data: np.array, round_to: int = 3):
    """
    Calculates the cluster centroids for each pattern.
    :param labels_df: a labels df that has the correlation matrices for each segment
    :param data: np.array 2D np array of the observation data
    :param round_to: number of decimals to calculate correlations for
    :return:
    """
    patterns = labels_df[SyntheticDataSegmentCols.pattern_id].unique().tolist()
    cluster_centroids = {}

    # calculated correlation for all data belonging to the same pattern
    for pattern in patterns:
        # select segments for this pattern
        segment_ids = labels_df[labels_df[SyntheticDataSegmentCols.pattern_id] == pattern][
            SyntheticDataSegmentCols.segment_id]

        # select all observations for this pattern
        pattern_data = None
        for seg_id in segment_ids:
            start_idx = labels_df[labels_df[SyntheticDataSegmentCols.segment_id] == seg_id][
                SyntheticDataSegmentCols.start_idx].values[0]
            end_idx = labels_df[labels_df[SyntheticDataSegmentCols.segment_id] == seg_id][
                SyntheticDataSegmentCols.end_idx].values[0]
            length = labels_df[labels_df[SyntheticDataSegmentCols.segment_id] == seg_id][
                SyntheticDataSegmentCols.length].values[0]

            seg_data = data[start_idx:end_idx + 1, :]
            assert seg_data.shape[0] == length, "Selected wrong indices"

            if pattern_data is None:
                pattern_data = seg_data
            else:
                pattern_data = np.concatenate((pattern_data, seg_data), axis=0)

        # calculate correlation for this pattern
        cluster_corr = calculate_spearman_correlation(pattern_data, round_to=round_to)
        cluster_centroids[pattern] = cluster_corr

    return cluster_centroids


def calculate_distance_between_segment_and_data_centroid(labels_df: pd.DataFrame, overall_data_centroid: [],
                                                         distance_measure: str):
    """
    Calculates the distance between each segment and the data centroid.
    :param labels_df: a labels df that has the correlation matrices for each segment
    :param overall_data_centroid: m2 for the distance calculation
    :param distance_measure: name of a distance measure that takes to matrices as argument m1, m2
    :return:
    """
    distance_calc = distance_calculation_method_for(distance_measure)
    m1s = labels_df[SyntheticDataSegmentCols.actual_correlation].to_list()
    return [distance_calc(m1, overall_data_centroid) for m1 in m1s]


def calculate_distances_between_each_segment_and_its_cluster_centroid(labels_df: pd.DataFrame, cluster_centroids: {},
                                                                      distance_measure: str):
    """
    Calculates the distance between each segment and its cluster centroid.
    :param labels_df: a labels df that has the correlation matrices for each segment
    :param cluster_centroids: dictionary with key being the pattern_id and value the upper triu correlation vector for
    the cluster
    :param distance_measure: name of a distance measure that takes to matrices as argument m1, m2
    :return: list of all distances ordered by segment id
    """
    distance_calc = distance_calculation_method_for(distance_measure)
    df = labels_df[[SyntheticDataSegmentCols.actual_correlation, SyntheticDataSegmentCols.pattern_id]]
    segment_correlations_and_pattern = df.apply(tuple, axis=1).tolist()
    # t[0] is the segments correlation, t[1] is the segments pattern_id (=cluster)
    return [distance_calc(t[0], cluster_centroids[t[1]]) for t in segment_correlations_and_pattern]


def calculate_distances_between_cluster_centroids(cluster_centroids: {}, distance_measure: str):
    """
    Calculates the distance between all cluster centroids
    :param cluster_centroids: dictionary with key being the pattern_id and value the upper triu correlation vector for
    the cluster
    :param distance_measure: name of a distance measure that takes to matrices as argument m1, m2
    :return: dictionary of keys being the two clusters id being compared and value being the distance between these two
    clusters
    """
    distance_calc = distance_calculation_method_for(distance_measure)
    patterns_combinations = list(combinations(list(cluster_centroids.keys()), 2))
    # cluster_centroids[t[0]] is the correlation for cluster t[0]
    return {t: distance_calc(cluster_centroids[t[0]], cluster_centroids[t[1]]) for t in patterns_combinations}


def calculate_distance_matrix_for(labels_df: pd.DataFrame, distance_measure: str):
    """
    Calculates the distance matrix of shape n_segments x n_segments
    :param labels_df: a labels datafile with uptodate correlations
    :param distance_measure: the distance measure to use for this calculation
    :return distance_matrix: 2D np.arrray
    """
    distance_calc = distance_calculation_method_for(distance_measure)
    df = labels_df
    n_seg = df.shape[0]

    # calculate all distances between all segment pairs
    distances = np.zeros((n_seg, n_seg))
    segment_ids = df.index.tolist()
    segment_pairs = list(combinations(segment_ids, 2))
    seg_correlations = np.array(df[SyntheticDataSegmentCols.actual_correlation].to_list())
    for pair in segment_pairs:
        seg1 = pair[0]
        seg2 = pair[1]
        corr1 = seg_correlations[seg1]
        corr2 = seg_correlations[seg2]
        dist = distance_calc(corr1, corr2)
        distances[seg1][seg2] = dist
        distances[seg2][seg1] = dist
    return distances
