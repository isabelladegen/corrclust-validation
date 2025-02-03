from dataclasses import dataclass

import numpy as np
from sklearn import metrics


@dataclass
class ClusteringQualityMeasures:
    pmb: str = "PMB"
    silhouette_score: str = "SCW"
    vrc_index: str = "VRC"
    dbi_index: str = "DBI"
    jaccard_index: str = "Jaccard"


def calculate_vrc(distances_seg_cluster_centroid: {}, distance_cluster_centroids_to_data: {},
                  round_to: int = 3) -> float:
    """
    Calculates the calinkski harabasz index (VRC) - higher is better (lowest is 0)
    :param distances_seg_cluster_centroid: dictionary with key cluster id and values list of distances for each
    segment in the cluster to the centroid. Shape len(k)= number of clusters, len(distances) = n_segment in that cluster
    :param distance_cluster_centroids_to_data: dictionary with key cluster id and value distance between this clusters centroid
    to the overall data
    :param round_to: optional to adjust rounding
    :returns vrc index
    """
    clusters = list(distances_seg_cluster_centroid.keys())
    n_clusters = len(clusters)
    n_segments = sum(len(segments) for segments in distances_seg_cluster_centroid.values())
    # calculate BCV and WCV
    bcvs = []  # between cluster variance sum bits
    wcvs = []  # within cluster variance sum bits
    for cluster in clusters:
        # calculate between cluster variance sum bit
        segments_for_cluster = distances_seg_cluster_centroid[cluster]
        bcvs.append(len(segments_for_cluster) * distance_cluster_centroids_to_data[cluster] ** 2)

        # calculate within cluster variance sum bit
        wcvs.append(sum(seg_dist ** 2 for seg_dist in segments_for_cluster))

    # check we have the right number of elements in the lists
    assert len(bcvs) == n_clusters, "Something went wrong we don't have a number for each cluster"
    assert len(wcvs) == n_clusters, "Something went wrong we don't have a number for each cluster"

    bcv = (1 / (n_clusters - 1)) * sum(bcvs)
    wcv = (1 / (n_segments - n_clusters)) * sum(wcvs)

    return round(bcv / wcv, round_to)


def calculate_dbi(distances_seg_cluster_centroid: {}, distances_cluster_centroids: {}, round_to: int = 3) -> float:
    """
    Calculates the davies bouldin index - lower is better (lowest is 0)
    :param distances_seg_cluster_centroid: dictionary with key cluster id and values list of distances for each
    segment in the cluster to the centroid. Shape len(k)= number of clusters, len(distances) = n_segment in that cluster
    :param distances_cluster_centroids: dictionary with key (cluster1, cluster2) and value distance between these
    two cluster centroids
    :param round_to: optional to adjust rounding
    :returns dbi index
    """
    clusters = list(distances_seg_cluster_centroid.keys())
    n_clusters = len(clusters)
    max_per_cluster = []
    # find max for each cluster
    for cluster in clusters:
        segs_in_cluster = distances_seg_cluster_centroid[cluster]
        sigma_k = sum(segs_in_cluster) / len(segs_in_cluster)
        clusters_excluding_cluster = [c for c in clusters if c != cluster]
        calc_for_y = []
        for y in clusters_excluding_cluster:
            segs_in_y_cluster = distances_seg_cluster_centroid[y]
            sigma_y = sum(segs_in_y_cluster) / len(segs_in_y_cluster)
            # lookup cluster centroid distance between cluster and y
            distance = distances_cluster_centroids.get((cluster, y)) if (cluster,
                                                                         y) in distances_cluster_centroids else distances_cluster_centroids.get(
                (y, cluster))
            calc_for_y.append((sigma_k + sigma_y) / distance)

        # pick max for y to sum up in the end
        max_per_cluster.append(max(calc_for_y))

    sum_of_maxes = sum(max_per_cluster)
    return round(sum_of_maxes / n_clusters, round_to)


def calculate_pmb(distances_between_segments_to_overall_data: [], distances_between_segments_to_cluster: [],
                  distances_between_cluster_centroids: [], round_to: int = 3):
    """
    Calculates PMB as PMB=(1/k * e_1/e_k *d_k)^2 - higher is better, >>100!
    :param distances_between_segments_to_overall_data: e_1 list of distances between all segments to the overall data
    centroid, shape: list of length number of segments
    :param distances_between_segments_to_cluster: e_k = list of distances between all segments to their cluster centroid,
    shape: length number of clusters
    :param distances_between_cluster_centroids: D_k = list of distance between all cluster centroids, shape:
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
    pmb = pow((d_k * e_1) / (no_cluster * e_k), 2)  # pow raises to the power of
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


def silhouette_avg_from_distances(distances: np.array, y_pred: [], round_to: int = 3):
    """
    Return silhouette avg for precomputed distances -> 1 is best, -1 is worst, less than 0.5 is not good
    :param distances: 2d ndarray of shape (no_segments, no_segments)
    :param y_pred: cluster assignment 1d len(no_segments), cluster each segment is assigned to
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
    result = metrics.silhouette_score(distances, y_pred, metric='precomputed')

    return round(result, round_to)
