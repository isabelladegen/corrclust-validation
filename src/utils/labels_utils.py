import itertools
from itertools import combinations

import numpy as np
import pandas as pd

from src.data_generation.generate_synthetic_correlated_data import calculate_spearman_correlation
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.distance_measures import distance_calculation_method_for


def calculate_n_segments_within_tolerance_for(labels_df: pd.DataFrame):
    # turn into a single true false per segment
    counts = labels_df[SyntheticDataSegmentCols.actual_within_tolerance].apply(lambda x: all(x)).value_counts()
    return counts[True] if True in counts else 0


def calculate_n_segments_outside_tolerance_for(labels_df: pd.DataFrame):
    # turn into a single true false per segment
    counts = labels_df[SyntheticDataSegmentCols.actual_within_tolerance].apply(lambda x: all(x)).value_counts()
    return counts[False] if False in counts else 0


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
    :return: dictionary with key being pattern_id and value being cluster centroid
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


def calculate_distance_between_cluster_centroids_and_data(cluster_centroids: {}, data_centroid: [],
                                                          distance_measure: str):
    """
    Calculates the distance between each cluster centroid to the data centroid for the given distance meausre
    :param cluster_centroids: dictionary with key being cluster_id and value being the centroid of the cluster
    :param data_centroid: list of correlation coefficients for data centroid
    :param distance_measure: name of distance measure to use
    :return: dictionary with key being cluster_id and value being the distance between cluster centroid and data centroid
    """
    distance_calc = distance_calculation_method_for(distance_measure)
    return {cluster_id: distance_calc(cluster_centroid, data_centroid) for cluster_id, cluster_centroid in
            cluster_centroids.items()}


def calculate_distances_between_each_segment_and_its_cluster_centroid(labels_df: pd.DataFrame, cluster_centroids: {},
                                                                      distance_measure: str):
    """
    Calculates the distance between each segment and its cluster centroid.
    :param labels_df: a labels df that has the correlation matrices for each segment
    :param cluster_centroids: dictionary with key being the pattern_id and value the upper triu correlation vector for
    the cluster
    :param distance_measure: name of a distance measure that takes to matrices as argument m1, m2
    :return: dictionary of all distances with keys being cluster_id and values being list of distances for segments
    """
    distance_calc = distance_calculation_method_for(distance_measure)
    result = {}
    for pattern_id, cluster_centroid in cluster_centroids.items():
        seg_correlations = labels_df[labels_df[SyntheticDataSegmentCols.pattern_id] == pattern_id][
            SyntheticDataSegmentCols.actual_correlation]
        result[pattern_id] = [distance_calc(seg, cluster_centroid) for seg in seg_correlations]
    return result


def calculate_distances_between_cluster_centroids(cluster_centroids: {}, distance_measure: str):
    """
    Calculates the distance between all cluster centroids
    :param cluster_centroids: dictionary with key being the pattern_id and value the upper triu correlation vector for
    the cluster
    :param distance_measure: name of a distance measure that takes to matrices as argument m1, m2
    :return: dictionary of keys being a tuple of two clusters id being compared and value being the distance between these two
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


def find_all_level_sets(labels_df: pd.DataFrame):
    """
    Calculates all possible level sets and which pattern tuples belong to it. This method only works
    if patterns to model are ideal!
    :param labels_df: a labels dataframe
    :return: dictionary of key level set id and values list of tuples of pattern pairs
    """
    patterns = labels_df[SyntheticDataSegmentCols.pattern_id].unique().tolist()
    # all combinations of patterns
    all_pattern_combinations = list(itertools.combinations_with_replacement(patterns, 2))
    pattern_models = {pid: labels_df[labels_df[SyntheticDataSegmentCols.pattern_id] == pid][
        SyntheticDataSegmentCols.correlation_to_model].iloc[0] for pid in patterns}

    # dictionary of pattern id and pattern model
    level_sets = {i: [] for i in range(6)}

    # cycle through all pattern pairs and put them in the right level set based on number of changes
    for combination in all_pattern_combinations:
        p1 = pattern_models[combination[0]]
        p2 = pattern_models[combination[1]]

        # add up all the changes in the pattern
        n_changes = 0
        for idx, value1 in enumerate(p1):
            value2 = p2[idx]
            # difference between the two values in ideal distance, that is why we round
            n_changes += round(abs(value1 - value2), 0)

        # add pattern to the level set with that number of changes
        level_sets[n_changes].append(combination)

    return level_sets
