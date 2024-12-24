import numpy as np
import sklearn as sk


def calculate_pmb(distances_between_segments_to_overall_data: [], distances_between_segments_to_cluster: [],
                  distances_between_cluster_centroids: [], round_to: int = 3):
    """ Calculates PMB as PMB=(1/k * e_1/e_k *
    d_k)^2
    :param distances_between_segments_to_overall_data: list of distances between all segments to the overall data
    centroid, shape: list of length number of segments
    :param distances_between_segments_to_cluster: list of distances between all segments to their cluster centroid,
    shape: length number of clusters
    :param distances_between_cluster_centroids: list of distance between all cluster centroids, shape:
     list of length number of 2 pairs combination of clusters
    :param round_to: number of decimals places to return
    """
    no_cluster = len(distances_between_segments_to_cluster)
    # the sum of the distances of all the segments the centroid of all observations
    e_1 = sum(distances_between_segments_to_overall_data)
    # the sum of the distances of the segments of each cluster to their centroid
    e_k = sum(distances_between_segments_to_cluster)
    # largest distance between two cluster centroids
    d_k = max(distances_between_cluster_centroids)
    pmb = pow((d_k * e_1) / (no_cluster * e_k), 2)
    return round(pmb, round_to)


def clustering_jaccard_coeff(y_pred_clusters: [], ground_truth_clusters: [], round_to: int = 3) -> float:
    """
    Calculates clustering Jaccard Coefficient where the intersection is all the observation that are in the correct
    cluster and the union is the number of observations in the time series.

    Jaccard is tp/N where N is length of ts and tp are all the observations that are in the same cluster
    in the ground truth partition and another partition

    :param y_pred_clusters: list of cluster assignment for each observation
    :param ground_truth_clusters: list of cluster assignment for each observation for the ground truth
    :param round_to: number of decimals places to return
    """
    assert len(ground_truth_clusters) == len(y_pred_clusters), \
        "Provided ground truth has different number of observations to cluster results"
    compared_j = [gt == actual for gt, actual in zip(ground_truth_clusters, y_pred_clusters)]
    tp_j = sum(compared_j)
    n = len(compared_j)
    clustering_jaccard_coefficient = tp_j / n
    return round(clustering_jaccard_coefficient, round_to)


def silhouette_avg_from_distances(distances: np.array, y_pred: [] = None, round_to: int = 3):
    """
    Return silhouette avg for precomputed distances
    :param y_pred: provide a y_pred if not all segments were used in the distance calculation
    :param distances: 2d ndarray of shape (no_segments, no_segments)
    :param round_to: number of decimals places to return
    """
    if np.any(np.isnan(distances)):
        # cannot calculate silhouette scores from nan distances
        print("Calculating silhouette avg failed due to one or more distance being nan")
        return
    assert distances.shape[0] == distances.shape[1] == len(
        y_pred), "shapes of distance matrix and y_pred don't match"
    if not len(np.unique(y_pred)) < distances.shape[0]:
        print("Cannot use silhouette analysis as each remaining segment has a different cluster")
        return None
    result = sk.metrics.silhouette_score(distances, y_pred, metric='precomputed')
    return round(result, round_to)