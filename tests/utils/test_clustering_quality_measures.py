import numpy as np
from hamcrest import *

from src.utils.clustering_quality_measures import silhouette_avg_from_distances, calculate_pmb, \
    clustering_jaccard_coeff, calculate_dbi, calculate_vrc
from src.utils.distance_measures import distance_calculation_method_for, DistanceMeasures

# test scenario with two clusters
# cluster 1
e1 = [0.1, 0.2, 0.8]
e2 = [-0.2, 0.0, 0.99]
e3 = [0.0, 0.3, 0.71]
# centroid
c1 = [0, 0, 0.85]

# cluster 2
e4 = [-1, 0.1, 0]
e5 = [-0.5, 0.5, 0.1]
c2 = [-0.7, 0.2, 0]

# data centroid
d = [-0.3, 0.2, 0.4]

y_pred_gt = [1, 1, 1, 2, 2]  # no mistakes
y_pred_ok = [1, 1, 2, 2, 2]  # one mistake
y_pred_bad = [2, 1, 2, 1, 2]  # 4 mistakes

centroids_gt = [c1 if y == 1 else c2 for y in y_pred_gt]
centroids_ok = [c1 if y == 1 else c2 for y in y_pred_ok]
centroids_bad = [c1 if y == 1 else c2 for y in y_pred_bad]

segments = [e1, e2, e3, e4, e5]

seg_clusters_gt = {1: [e1, e2, e3], 2: [e4, e5]}
seg_clusters_ok = {1: [e1, e2], 2: [e3, e4, e5]}
seg_clusters_bad = {1: [e2, e4], 2: [e1, e3, e5]}


def calculate_distances_between_all_segments(segments: [], distance_measure: str):
    dist_calc = distance_calculation_method_for(distance_measure)
    n = len(segments)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = dist_calc(segments[i], segments[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  # mirror across diagonal

    return distance_matrix


def calculate_distances_between_segments_and_cluster_centroids(seg_clusters: {}, distance_measure: str):
    dist_calc = distance_calculation_method_for(distance_measure)
    results = {}
    for cluster_id, segments in seg_clusters.items():
        distances = []
        centroid = c1 if cluster_id == 1 else c2
        for seg in segments:
            distances.append(dist_calc(seg, centroid))
        results[cluster_id] = distances
    return results


def test_silhouette_score():
    distance_matrix_l1 = calculate_distances_between_all_segments(segments, DistanceMeasures.l1_cor_dist)
    l1_gt = silhouette_avg_from_distances(distances=distance_matrix_l1, y_pred=y_pred_gt)
    l1_gt_ok = silhouette_avg_from_distances(distances=distance_matrix_l1, y_pred=y_pred_ok)
    l1_bad = silhouette_avg_from_distances(distances=distance_matrix_l1, y_pred=y_pred_bad)

    assert_that(l1_gt, greater_than(l1_gt_ok))
    assert_that(l1_gt, greater_than(l1_bad))
    assert_that(l1_gt_ok, greater_than(l1_bad))


def test_pmb_index():
    dist_calc = distance_calculation_method_for(DistanceMeasures.l1_cor_dist)
    # calculated distance between cluster centroids
    dist_cluster_centroids = [dist_calc(c1, c2)]
    # calculate distance for all segments to overall data, and to their cluster centroid
    dist_overall_data = []
    dist_cluster_centroid_gt = []
    dist_cluster_centroid_ok = []
    dist_cluster_centroid_bad = []
    for idx, seg in enumerate(segments):
        dist_overall_data.append(dist_calc(seg, d))
        dist_cluster_centroid_gt.append(dist_calc(seg, centroids_gt[idx]))
        dist_cluster_centroid_ok.append(dist_calc(seg, centroids_ok[idx]))
        dist_cluster_centroid_bad.append(dist_calc(seg, centroids_bad[idx]))

    pmb_gt = calculate_pmb(dist_overall_data, dist_cluster_centroid_gt, dist_cluster_centroids)
    pmb_ok = calculate_pmb(dist_overall_data, dist_cluster_centroid_ok, dist_cluster_centroids)
    pmb_bad = calculate_pmb(dist_overall_data, dist_cluster_centroid_bad, dist_cluster_centroids)

    assert_that(pmb_gt, greater_than(pmb_ok))
    assert_that(pmb_gt, greater_than(pmb_bad))
    assert_that(pmb_ok, greater_than(pmb_bad))


def test_jaccard_index():
    j_gt = clustering_jaccard_coeff(y_pred_gt, y_pred_gt)
    j_ok = clustering_jaccard_coeff(y_pred_ok, y_pred_gt)
    j_bad = clustering_jaccard_coeff(y_pred_bad, y_pred_gt)

    assert_that(j_gt, greater_than(j_ok))
    assert_that(j_gt, is_(1))
    assert_that(j_gt, greater_than(j_bad))
    assert_that(j_ok, greater_than(j_bad))


def test_davies_bouldin_index():
    distances_m = DistanceMeasures.l1_cor_dist
    dist_cluster_centroids_gt = calculate_distances_between_segments_and_cluster_centroids(seg_clusters_gt, distances_m)
    dist_cluster_centroids_ok = calculate_distances_between_segments_and_cluster_centroids(seg_clusters_ok, distances_m)
    dist_cluster_centroids_bad = calculate_distances_between_segments_and_cluster_centroids(seg_clusters_bad,
                                                                                            distances_m)
    dist_calc = distance_calculation_method_for(distances_m)
    distances_cluster_centroids = {(1, 2): dist_calc(c1, c2)}

    dbi_gt = calculate_dbi(dist_cluster_centroids_gt, distances_cluster_centroids)
    dbi_ok = calculate_dbi(dist_cluster_centroids_ok, distances_cluster_centroids)
    dbi_bad = calculate_dbi(dist_cluster_centroids_bad, distances_cluster_centroids)

    assert_that(dbi_gt, less_than(dbi_ok))
    assert_that(dbi_gt, less_than(dbi_bad))
    assert_that(dbi_ok, less_than(dbi_bad))


def test_davies_bouldin_index_with_identical_cluster_centroids():
    # if cluster centroids are identical, we set the cluster centroid a small number to avoid division by zero
    distances_m = DistanceMeasures.l1_cor_dist
    distances_segments_to_cluster_centroid = calculate_distances_between_segments_and_cluster_centroids(seg_clusters_gt,
                                                                                                        distances_m)

    # assume zero distance between the cluster centroids
    distances_cluster_centroids = {(1, 2): 0.0}

    dbi_gt = calculate_dbi(distances_segments_to_cluster_centroid, distances_cluster_centroids)

    assert_that(dbi_gt, is_(876666666666666.6))


def test_calinski_harabasz_index():
    distances_m = DistanceMeasures.l1_cor_dist
    dist_cluster_centroids_gt = calculate_distances_between_segments_and_cluster_centroids(seg_clusters_gt, distances_m)
    dist_cluster_centroids_ok = calculate_distances_between_segments_and_cluster_centroids(seg_clusters_ok, distances_m)
    dist_cluster_centroids_bad = calculate_distances_between_segments_and_cluster_centroids(seg_clusters_bad,
                                                                                            distances_m)
    dist_calc = distance_calculation_method_for(distances_m)
    dist_cluster_centroids_data = {1: dist_calc(c1, d), 2: dist_calc(c2, d)}

    vrc_gt = calculate_vrc(dist_cluster_centroids_gt, dist_cluster_centroids_data)
    vrc_ok = calculate_vrc(dist_cluster_centroids_ok, dist_cluster_centroids_data)
    vrc_bad = calculate_vrc(dist_cluster_centroids_bad, dist_cluster_centroids_data)

    assert_that(vrc_gt, greater_than(vrc_ok))
    assert_that(vrc_gt, greater_than(vrc_bad))
    assert_that(vrc_ok, greater_than(vrc_bad))
