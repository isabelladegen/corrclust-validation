import numpy as np
from hamcrest import *

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import SyntheticDataVariates
from src.utils.distance_measures import DistanceMeasures
from src.utils.labels_utils import calculate_y_pred_from, calculate_overall_data_correlation, \
    calculate_distance_between_segment_and_data_centroid, calculate_cluster_centroids, \
    calculate_distances_between_each_segment_and_its_cluster_centroid, calculate_distances_between_cluster_centroids, \
    calculate_distance_matrix_for, calculate_y_pred_and_updated_gt_y_pred_from
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data_and_labels_for_bad_partitions
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR

test_data_dir = TEST_DATA_DIR

run_name = "misty-forest-56"
data_type = SyntheticDataType.non_normal_correlated
data, gt_label, bad_partitions_labels = load_synthetic_data_and_labels_for_bad_partitions(run_name,
                                                                                          data_type=data_type,
                                                                                          data_dir=test_data_dir,
                                                                                          load_only=3)


def test_returns_y_pred_from_labels_df():
    y_pred_gt = calculate_y_pred_from(gt_label)
    p1_labels = list(bad_partitions_labels.values())[0]
    y_pred_p1 = calculate_y_pred_from(p1_labels)

    # check length of y pred matches data
    assert_that(len(y_pred_gt), is_(data.shape[0]))
    assert_that(len(y_pred_p1), is_(data.shape[0]))

    # check segment end at the right index
    s2_start = gt_label.loc[1, SyntheticDataSegmentCols.start_idx]
    s2_end = gt_label.loc[1, SyntheticDataSegmentCols.end_idx]
    s2_length = gt_label.loc[1, SyntheticDataSegmentCols.length]
    s2_pattern = gt_label.loc[1, SyntheticDataSegmentCols.pattern_id]
    assert_that(y_pred_gt[s2_start - 1], is_(not_(s2_pattern)))  # the previous pattern is different
    assert_that(y_pred_gt[s2_end + 1], is_(not_(s2_pattern)))  # the next pattern is different
    s2 = y_pred_gt[s2_start:s2_end + 1]  # np does not select end index
    assert_that(np.all(s2 == s2_pattern))  # the whole segment has the right pattern
    assert_that(len(s2), is_(s2_length))  # has right length

    s2_p1_start = p1_labels.loc[1, SyntheticDataSegmentCols.start_idx]
    s2_p1_end = p1_labels.loc[1, SyntheticDataSegmentCols.end_idx]
    s2_p1_length = p1_labels.loc[1, SyntheticDataSegmentCols.length]
    s2_p1_pattern = p1_labels.loc[1, SyntheticDataSegmentCols.pattern_id]
    assert_that(y_pred_p1[s2_p1_start - 1], is_(not_(s2_p1_pattern)))  # the previous pattern is different
    assert_that(y_pred_p1[s2_p1_end + 1], is_(not_(s2_p1_pattern)))  # the next pattern is different
    s2_p1 = y_pred_p1[s2_p1_start:s2_p1_end + 1]  # np does not include end index
    assert_that(np.all(s2_p1 == s2_p1_pattern))  # the whole segment has the right pattern
    assert_that(len(s2_p1), is_(s2_p1_length))  # has right length


def test_returns_y_pred_and_update_y_pred_from_labels_df_and_full_gt_y_pred_when_segments_have_been_dropped():
    full_original_y_pred_gt = calculate_y_pred_from(gt_label)
    p1_labels = list(bad_partitions_labels.values())[0]
    # drop some segments from a bad partition
    p1_labels = p1_labels.head(10)

    # calculate y_pred from labels and update y_pred_gt to match p1_labels length
    y_pred_p1, y_pred_gt = calculate_y_pred_and_updated_gt_y_pred_from(p1_labels, full_original_y_pred_gt)

    # check length of y pred matches data
    assert_that(len(full_original_y_pred_gt), is_(data.shape[0]))
    p1_length = p1_labels[SyntheticDataSegmentCols.length].sum()
    assert_that(len(y_pred_p1), is_(p1_length))
    assert_that(len(y_pred_gt), is_(p1_length))


def test_calculates_overall_data_centroid():
    corr = calculate_overall_data_correlation(data[SyntheticDataVariates.columns()].to_numpy())
    assert_that(corr, contains_exactly(0.056, 0.02, -0.005))


def test_calculates_distances_between_each_segment_to_the_overall_data_centroid():
    l1_dist = calculate_distance_between_segment_and_data_centroid(gt_label, [0.4, 0.5, 0.6],
                                                                   DistanceMeasures.l1_cor_dist)
    l1_ref = calculate_distance_between_segment_and_data_centroid(gt_label, [0.4, 0.5, 0.6],
                                                                  DistanceMeasures.l1_with_ref)
    assert_that(len(l1_dist), is_(gt_label.shape[0]))
    assert_that(len(l1_ref), is_(gt_label.shape[0]))
    assert_that(l1_dist[0], less_than(l1_ref[0]))


def test_calculates_all_cluster_centroids():
    centroids = calculate_cluster_centroids(gt_label, data[SyntheticDataVariates.columns()].to_numpy())
    patterns = gt_label[SyntheticDataSegmentCols.pattern_id].unique()
    assert_that(len(centroids), is_(len(patterns)))
    assert_that(centroids[3], contains_exactly(0.0, 1.0, 0.0))
    assert_that(centroids[25], contains_exactly(-1.0, -1.0, 1.0))


def test_calculates_distances_between_each_segment_to_each_cluster_centroid():
    centroids = calculate_cluster_centroids(gt_label, data[SyntheticDataVariates.columns()].to_numpy())
    l1_dist = calculate_distances_between_each_segment_and_its_cluster_centroid(gt_label, centroids,
                                                                                DistanceMeasures.l1_cor_dist)

    assert_that(len(l1_dist), is_(gt_label.shape[0]))


def test_calculates_distances_between_all_cluster_centroids():
    centroids = calculate_cluster_centroids(gt_label, data[SyntheticDataVariates.columns()].to_numpy())
    cluster_dist = calculate_distances_between_cluster_centroids(centroids, DistanceMeasures.l1_cor_dist)
    n = len(gt_label[SyntheticDataSegmentCols.pattern_id].unique())
    expected_len = int(n * (n - 1) / 2)

    assert_that(len(cluster_dist), is_(expected_len))
    assert_that(cluster_dist[(0, 1)], is_(1.007))
    assert_that(cluster_dist[(0, 25)], is_(1.995))


def test_calculates_distance_matrix_from_a_labels_df():
    l1_matrix = calculate_distance_matrix_for(gt_label, DistanceMeasures.l1_cor_dist)
    l2_matrix = calculate_distance_matrix_for(gt_label, DistanceMeasures.l2_cor_dist)

    assert_that(l1_matrix.shape, is_((gt_label.shape[0], gt_label.shape[0])))
    assert_that(l2_matrix.shape, is_((gt_label.shape[0], gt_label.shape[0])))
