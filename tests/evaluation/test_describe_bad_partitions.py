from hamcrest import *

from src.evaluation.distance_metric_assessment import DistanceMeasureCols
from src.evaluation.describe_bad_partitions import DescribeBadPartitions, DescribeBadPartCols
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.labels_utils import calculate_y_pred_from
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR

ds_name = "misty-forest-56"
internal_measures = [DescribeBadPartCols.silhouette_score, DescribeBadPartCols.pmb]
distance_measure = DistanceMeasureCols.l1_cor_dist
test_data_dir = TEST_DATA_DIR
bp = DescribeBadPartitions(ds_name, distance_measure=distance_measure, internal_measures=internal_measures,
                           data_dir=test_data_dir)


def test_describe_bad_partitions():
    # for test runs only 3 bad partitions are read
    assert_that(bp.summary_df.shape[0], is_(4))

    # inserts results for gt at index 0
    gt_row = bp.summary_df.iloc[0]
    assert_that(gt_row[DescribeBadPartCols.name], is_(ds_name))
    assert_that(gt_row[DescribeBadPartCols.n_patterns], is_(23))
    assert_that(gt_row[DescribeBadPartCols.n_segments], is_(100))
    assert_that(gt_row[DescribeBadPartCols.n_observations], is_(1226400))
    assert_that(bp.data.shape[0], is_(1226400))
    assert_that(bp.gt_label[SyntheticDataSegmentCols.length].sum(), is_(1226400))
    assert_that(gt_row[DescribeBadPartCols.errors].round(2), is_(7.9))
    assert_that(gt_row[DescribeBadPartCols.n_obs_shifted], is_(0))
    assert_that(gt_row[DescribeBadPartCols.n_wrong_clusters], is_(0))


def test_describe_bad_partition_returns_list_of_ground_truth_cluster_for_each_observation():
    gt_clusters = calculate_y_pred_from(bp.gt_label)

    # each observation has it's cluster id
    assert_that(len(gt_clusters), is_(1226400))
    # there are in total 23 clusters
    assert_that(len(set(gt_clusters)), is_(23))
    # assert first segment is cluster 0
    assert_that(all(item == 0 for item in gt_clusters[0:899]), is_(True))
    # assert second segment is cluster 1
    assert_that(all(item == 1 for item in gt_clusters[900:2099]), is_(True))
    # assert last segment is cluster 21
    assert_that(all(item == 7 for item in gt_clusters[1183200:1226399]), is_(True))


def test_calculates_jaccard_index_for_each_gt_and_partition():
    assert_that(bp.summary_df.iloc[0][DescribeBadPartCols.jaccard_index], is_(1))
    assert_that(bp.summary_df.iloc[1][DescribeBadPartCols.jaccard_index], is_(0.96))
    assert_that(bp.summary_df.iloc[2][DescribeBadPartCols.jaccard_index], is_(0.777))
    assert_that(bp.summary_df.iloc[3][DescribeBadPartCols.jaccard_index], is_(0.976))


def test_calculates_internal_measures_for_the_given_distance_measure():
    assert_that(bp.summary_df.iloc[0][DescribeBadPartCols.silhouette_score], is_(0.969))
    assert_that(bp.summary_df.iloc[1][DescribeBadPartCols.silhouette_score], is_(0.537))
    assert_that(bp.summary_df.iloc[2][DescribeBadPartCols.silhouette_score], is_(0.363))
    assert_that(bp.summary_df.iloc[3][DescribeBadPartCols.silhouette_score], is_(0.743))

    assert_that(bp.summary_df.iloc[0][DescribeBadPartCols.pmb], is_(225.913))
    assert_that(bp.summary_df.iloc[1][DescribeBadPartCols.pmb], is_(1.218))
    assert_that(bp.summary_df.iloc[2][DescribeBadPartCols.pmb], is_(0.199))
    assert_that(bp.summary_df.iloc[3][DescribeBadPartCols.pmb], is_(3.610))


def test_only_calculates_the_internal_measure_provided():
    bp1 = DescribeBadPartitions(ds_name, distance_measure=distance_measure, internal_measures=[DescribeBadPartCols.pmb])
    columns = list(bp1.summary_df.columns)
    assert_that(DescribeBadPartCols.silhouette_score not in columns, is_(True))


def test_randomly_selects_k_clusters_from_all_partitions():
    n_clusters = 2
    bp1 = DescribeBadPartitions(ds_name, distance_measure=distance_measure,
                                internal_measures=[DescribeBadPartCols.silhouette_score], n_clusters=n_clusters)

    data = bp1.data
    labels = bp1.gt_label

    # check data has been correctly updated
    assert_that(len(labels[SyntheticDataSegmentCols.pattern_id].unique()), is_(n_clusters))
    assert_that(labels.shape[0], is_(8))

    length = sum(labels[SyntheticDataSegmentCols.length])
    assert_that(data.shape[0], is_(length))
    # assert indices for first segment in data
    start_idx = labels[SyntheticDataSegmentCols.start_idx].tolist()[0]
    end_idx = labels[SyntheticDataSegmentCols.end_idx].tolist()[0]
    indices = set(data.index.tolist())
    assert_that(start_idx in indices, is_(True))
    assert_that(end_idx in indices, is_(True))
    assert_that(end_idx + 1 in indices, is_(False))

    df = bp1.summary_df
    assert_that(df.shape[0], is_(4))


def test_randomly_selects_n_clusters_from_all_partitions():
    n_segments = 10
    bp1 = DescribeBadPartitions(ds_name, distance_measure=distance_measure, internal_measures=[DescribeBadPartCols.pmb],
                                n_segments=n_segments)

    data = bp1.data
    labels = bp1.gt_label

    # check data has been correctly updated
    assert_that(len(labels[SyntheticDataSegmentCols.pattern_id].unique()), is_(9))
    assert_that(labels.shape[0], is_(n_segments))

    length = sum(labels[SyntheticDataSegmentCols.length])
    assert_that(data.shape[0], is_(length))
    # assert indices for first segment in data
    start_idx = labels[SyntheticDataSegmentCols.start_idx].tolist()[0]
    end_idx = labels[SyntheticDataSegmentCols.end_idx].tolist()[0]
    indices = set(data.index.tolist())
    assert_that(start_idx in indices, is_(True))
    assert_that(end_idx in indices, is_(True))
    assert_that(end_idx + 1 in indices, is_(False))

    df = bp1.summary_df
    assert_that(df.shape[0], is_(4))  # n test datasets


def test_can_randomly_select_both_clusters_and_segments():
    n_segments = 10
    n_clusters = 3
    bp1 = DescribeBadPartitions(ds_name, distance_measure=distance_measure, internal_measures=[DescribeBadPartCols.pmb],
                                n_clusters=3, n_segments=n_segments)

    data = bp1.data
    labels = bp1.gt_label

    # check data has been correctly updated
    assert_that(len(labels[SyntheticDataSegmentCols.pattern_id].unique()), is_(n_clusters))
    assert_that(labels.shape[0], is_(n_segments))

    length = sum(labels[SyntheticDataSegmentCols.length])
    assert_that(data.shape[0], is_(length))
    # assert indices for first segment in data
    start_idx = labels[SyntheticDataSegmentCols.start_idx].tolist()[0]
    end_idx = labels[SyntheticDataSegmentCols.end_idx].tolist()[0]
    indices = set(data.index.tolist())
    assert_that(start_idx in indices, is_(True))
    assert_that(end_idx in indices, is_(True))
    assert_that(end_idx + 1 in indices, is_(False))

    df = bp1.summary_df
    assert_that(df.shape[0], is_(4))  # n test datasets
