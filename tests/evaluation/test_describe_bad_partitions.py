from hamcrest import *

from src.evaluation.describe_bad_partitions import DescribeBadPartitions, DescribeBadPartCols, select_data_from_df, \
    run_internal_measure_calculation_for_dataset, read_internal_measures_calculation
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.distance_measures import DistanceMeasures
from src.utils.labels_utils import calculate_y_pred_from
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR, TEST_ROOT_RESULTS_DIR, \
    TEST_GENERATED_DATASETS_FILE_PATH

ds_name = "misty-forest-56"
internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                     ClusteringQualityMeasures.dbi, ClusteringQualityMeasures.vrc]
distance_measure = DistanceMeasures.l1_cor_dist
test_data_dir = TEST_DATA_DIR
describe = DescribeBadPartitions(ds_name, distance_measure=distance_measure, internal_measures=internal_measures,
                                 data_dir=test_data_dir)


def test_describe_bad_partitions():
    # for test runs only 3 bad partitions are read
    assert_that(describe.summary_df.shape[0], is_(5))

    # inserts results for gt at index 0
    gt_row = describe.summary_df.iloc[0]
    assert_that(gt_row[DescribeBadPartCols.name], is_(ds_name))
    assert_that(gt_row[DescribeBadPartCols.n_patterns], is_(23))
    assert_that(gt_row[DescribeBadPartCols.n_segments], is_(100))
    assert_that(gt_row[DescribeBadPartCols.n_observations], is_(1226400))
    assert_that(describe.data.shape[0], is_(1226400))
    assert_that(describe.gt_label[SyntheticDataSegmentCols.length].sum(), is_(1226400))
    assert_that(gt_row[DescribeBadPartCols.errors].round(2), is_(0.11))
    assert_that(gt_row[DescribeBadPartCols.n_obs_shifted], is_(0))
    assert_that(gt_row[DescribeBadPartCols.n_wrong_clusters], is_(0))


def test_describe_bad_partition_returns_list_of_ground_truth_cluster_for_each_observation():
    gt_clusters = calculate_y_pred_from(describe.gt_label)

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
    assert_that(describe.summary_df.iloc[0][ClusteringQualityMeasures.jaccard_index], is_(1))
    assert_that(describe.summary_df.iloc[1][ClusteringQualityMeasures.jaccard_index], is_(0.982))
    assert_that(describe.summary_df.iloc[2][ClusteringQualityMeasures.jaccard_index], is_(0.377))
    assert_that(describe.summary_df.iloc[3][ClusteringQualityMeasures.jaccard_index], is_(0.896))
    assert_that(describe.summary_df.iloc[4][ClusteringQualityMeasures.jaccard_index], is_(0.293))

    # mean MAE for comparison
    assert_that(describe.summary_df.iloc[0][DescribeBadPartCols.errors], is_(0.112))
    assert_that(describe.summary_df.iloc[1][DescribeBadPartCols.errors], is_(0.141))
    assert_that(describe.summary_df.iloc[2][DescribeBadPartCols.errors], is_(0.558))
    assert_that(describe.summary_df.iloc[3][DescribeBadPartCols.errors], is_(0.175))
    assert_that(describe.summary_df.iloc[4][DescribeBadPartCols.errors], is_(0.612))


def test_calculates_internal_measures_for_the_given_distance_measure():
    assert_that(describe.summary_df.iloc[0][ClusteringQualityMeasures.silhouette_score], is_(0.97))
    assert_that(describe.summary_df.iloc[1][ClusteringQualityMeasures.silhouette_score], is_(0.829))
    assert_that(describe.summary_df.iloc[2][ClusteringQualityMeasures.silhouette_score], is_(-0.337))
    assert_that(describe.summary_df.iloc[3][ClusteringQualityMeasures.silhouette_score], is_(0.673))
    assert_that(describe.summary_df.iloc[4][ClusteringQualityMeasures.silhouette_score], is_(-0.383))

    assert_that(describe.summary_df.iloc[0][ClusteringQualityMeasures.pmb], is_(12.769))
    assert_that(describe.summary_df.iloc[1][ClusteringQualityMeasures.pmb], is_(0.509))
    assert_that(describe.summary_df.iloc[2][ClusteringQualityMeasures.pmb], is_(0.002))
    assert_that(describe.summary_df.iloc[3][ClusteringQualityMeasures.pmb], is_(0.059))
    assert_that(describe.summary_df.iloc[4][ClusteringQualityMeasures.pmb], is_(0.001))

    assert_that(describe.summary_df.iloc[0][ClusteringQualityMeasures.vrc], is_(11315.355))
    assert_that(describe.summary_df.iloc[1][ClusteringQualityMeasures.vrc], is_(385.374))
    assert_that(describe.summary_df.iloc[2][ClusteringQualityMeasures.vrc], is_(1.831))
    assert_that(describe.summary_df.iloc[3][ClusteringQualityMeasures.vrc], is_(23.604))
    assert_that(describe.summary_df.iloc[4][ClusteringQualityMeasures.vrc], is_(1.403))

    assert_that(describe.summary_df.iloc[0][ClusteringQualityMeasures.dbi], is_(0.044))
    assert_that(describe.summary_df.iloc[1][ClusteringQualityMeasures.dbi], is_(0.199))
    assert_that(describe.summary_df.iloc[2][ClusteringQualityMeasures.dbi], is_(6.124))
    assert_that(describe.summary_df.iloc[3][ClusteringQualityMeasures.dbi], is_(1.393))
    assert_that(describe.summary_df.iloc[4][ClusteringQualityMeasures.dbi], is_(7.193))


def test_only_calculates_the_internal_measure_provided():
    bp1 = DescribeBadPartitions(ds_name, distance_measure=distance_measure,
                                internal_measures=[ClusteringQualityMeasures.pmb],
                                data_dir=test_data_dir)
    columns = list(bp1.summary_df.columns)
    assert_that(ClusteringQualityMeasures.silhouette_score not in columns, is_(True))


def test_randomly_drops_n_clusters_from_all_partitions():
    n_clusters = 20
    bp1 = DescribeBadPartitions(ds_name, distance_measure=distance_measure,
                                internal_measures=[ClusteringQualityMeasures.silhouette_score],
                                drop_n_clusters=n_clusters,
                                data_dir=test_data_dir)

    labels = bp1.gt_label
    data = select_data_from_df(bp1.data, labels)

    # check data has been correctly updated
    assert_that(len(labels[SyntheticDataSegmentCols.pattern_id].unique()), is_(23 - n_clusters))
    assert_that(labels.shape[0], is_(12))  # number of segments left
    n_segments_left = labels.shape[0]

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
    assert_that(df.shape[0], is_(len(bp1.partitions) + 1))  # test data partitions + gt
    # all partitions should have gt_original_n_patterns - n_patterns left over, same for segments
    assert_that((df[DescribeBadPartCols.n_patterns] <= 23 - n_clusters).all())
    assert_that((df[DescribeBadPartCols.n_segments] <= n_segments_left).all())

    # check ground truth row - for the ground truth we know exactly how many patterns and segment as no shifts
    ground_truth = df.iloc[0]
    assert_that(ground_truth[DescribeBadPartCols.n_observations], is_(length))
    assert_that(ground_truth[DescribeBadPartCols.n_patterns], is_(23 - n_clusters))
    assert_that(ground_truth[DescribeBadPartCols.n_segments], is_(n_segments_left))
    assert_that(ground_truth[DescribeBadPartCols.n_wrong_clusters], is_(0))
    assert_that(ground_truth[DescribeBadPartCols.n_obs_shifted], is_(0))

    # check the same for the first partition
    first_partition = df.iloc[1]
    first_partition_length = sum(list(bp1.partitions.values())[0][SyntheticDataSegmentCols.length])
    assert_that(first_partition[DescribeBadPartCols.n_observations], is_(first_partition_length))
    assert_that(first_partition[DescribeBadPartCols.n_patterns], is_(23 - n_clusters))
    assert_that(first_partition[DescribeBadPartCols.n_segments], is_(n_segments_left))
    assert_that(first_partition[DescribeBadPartCols.n_wrong_clusters], is_(0))
    assert_that(first_partition[DescribeBadPartCols.n_obs_shifted], greater_than(0))

    # check the same for the second partition
    second_partition = df.iloc[2]
    second_partition_length = sum(list(bp1.partitions.values())[1][SyntheticDataSegmentCols.length])
    assert_that(second_partition[DescribeBadPartCols.n_observations], is_(second_partition_length))
    assert_that(second_partition[DescribeBadPartCols.n_patterns], is_(23 - n_clusters))
    assert_that(second_partition[DescribeBadPartCols.n_segments], less_than(n_segments_left))
    assert_that(second_partition[DescribeBadPartCols.n_wrong_clusters], greater_than(0))
    assert_that(second_partition[DescribeBadPartCols.n_obs_shifted], greater_than(0))


def test_randomly_drop_n_segments_from_all_partitions():
    n_segments = 90
    bp1 = DescribeBadPartitions(ds_name, distance_measure=distance_measure,
                                internal_measures=[ClusteringQualityMeasures.pmb],
                                drop_n_segments=n_segments, data_dir=test_data_dir)

    labels = bp1.gt_label
    data = select_data_from_df(bp1.data, labels)

    # check data has been correctly updated
    assert_that(labels.shape[0], is_(100 - n_segments))
    assert_that(len(labels[SyntheticDataSegmentCols.pattern_id].unique()), is_(9))

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
    assert_that(df.shape[0], is_(len(bp1.partitions) + 1))  # test data partitions + gt
    # all partitions should have gt_original_n_patterns - n_patterns left over, same for segments
    assert_that((df[DescribeBadPartCols.n_patterns] <= 23).all())
    assert_that((df[DescribeBadPartCols.n_segments] <= 100 - n_segments).all())

    # check ground truth row - for the ground truth we know exactly how many patterns and segment as no shifts
    ground_truth = df.iloc[0]
    assert_that(ground_truth[DescribeBadPartCols.n_observations], is_(length))
    assert_that(ground_truth[DescribeBadPartCols.n_patterns], is_(9))
    assert_that(ground_truth[DescribeBadPartCols.n_segments], is_(100 - n_segments))
    assert_that(ground_truth[DescribeBadPartCols.n_wrong_clusters], is_(0))
    assert_that(ground_truth[DescribeBadPartCols.n_obs_shifted], is_(0))

    # check the same for the first partition
    first_partition = df.iloc[1]
    first_partition_length = sum(list(bp1.partitions.values())[0][SyntheticDataSegmentCols.length])
    assert_that(first_partition[DescribeBadPartCols.n_observations], is_(first_partition_length))
    assert_that(first_partition[DescribeBadPartCols.n_patterns], is_(9))
    assert_that(first_partition[DescribeBadPartCols.n_segments], is_(100 - n_segments))
    assert_that(first_partition[DescribeBadPartCols.n_wrong_clusters], is_(0))
    assert_that(first_partition[DescribeBadPartCols.n_obs_shifted], greater_than(0))

    # check the same for the second partition
    second_partition = df.iloc[2]
    second_partition_length = sum(list(bp1.partitions.values())[1][SyntheticDataSegmentCols.length])
    assert_that(second_partition[DescribeBadPartCols.n_observations], is_(second_partition_length))
    assert_that(second_partition[DescribeBadPartCols.n_patterns], less_than(9))
    assert_that(second_partition[DescribeBadPartCols.n_segments], is_(100 - n_segments))
    assert_that(second_partition[DescribeBadPartCols.n_wrong_clusters], greater_than(0))
    assert_that(second_partition[DescribeBadPartCols.n_obs_shifted], greater_than(0))


def test_can_randomly_drop_both_clusters_and_segments():
    n_segments = 90
    n_clusters = 20
    bp1 = DescribeBadPartitions(ds_name, distance_measure=distance_measure,
                                internal_measures=[ClusteringQualityMeasures.pmb],
                                drop_n_clusters=n_clusters, drop_n_segments=n_segments, data_dir=test_data_dir)

    labels = bp1.gt_label
    # the data is the full data so to get just the observations that have been kept we need to first select them
    data = select_data_from_df(bp1.data, labels)

    # check data has been correctly updated
    assert_that(len(labels[SyntheticDataSegmentCols.pattern_id].unique()), is_(23 - n_clusters))
    assert_that(labels.shape[0], is_(100 - n_segments))

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
    assert_that(df.shape[0], is_(len(bp1.partitions) + 1))  # test data partitions + gt
    # all partitions should have gt_original_n_patterns - n_patterns left over, same for segments
    assert_that((df[DescribeBadPartCols.n_patterns] <= 23 - n_clusters).all())
    assert_that((df[DescribeBadPartCols.n_segments] <= 100 - n_segments).all())

    # check ground truth row - for the ground truth we know exactly how many patterns and segment as no shifts
    ground_truth = df.iloc[0]
    assert_that(ground_truth[DescribeBadPartCols.n_observations], is_(length))
    assert_that(ground_truth[DescribeBadPartCols.n_patterns], is_(23 - n_clusters))
    assert_that(ground_truth[DescribeBadPartCols.n_segments], is_(100 - n_segments))
    assert_that(ground_truth[DescribeBadPartCols.n_wrong_clusters], is_(0))
    assert_that(ground_truth[DescribeBadPartCols.n_obs_shifted], is_(0))

    # check the same for the first partition
    first_partition = df.iloc[1]
    first_partition_length = sum(list(bp1.partitions.values())[0][SyntheticDataSegmentCols.length])
    assert_that(first_partition[DescribeBadPartCols.n_observations], is_(first_partition_length))
    assert_that(first_partition[DescribeBadPartCols.n_patterns], is_(23 - n_clusters))
    assert_that(first_partition[DescribeBadPartCols.n_segments], is_(100 - n_segments))
    assert_that(first_partition[DescribeBadPartCols.n_wrong_clusters], is_(0))
    assert_that(first_partition[DescribeBadPartCols.n_obs_shifted], greater_than(0))

    # check the same for the second partition
    second_partition = df.iloc[2]
    second_partition_length = sum(list(bp1.partitions.values())[1][SyntheticDataSegmentCols.length])
    assert_that(second_partition[DescribeBadPartCols.n_observations], is_(second_partition_length))
    assert_that(second_partition[DescribeBadPartCols.n_patterns], is_(23 - n_clusters))
    assert_that(second_partition[DescribeBadPartCols.n_segments], less_than(100 - n_segments))
    assert_that(second_partition[DescribeBadPartCols.n_wrong_clusters], greater_than(0))
    assert_that(second_partition[DescribeBadPartCols.n_obs_shifted], greater_than(0))


def test_can_run_calculation_for_internal_measures_on_all_datasets():
    # run test_wandb_create_bad_partitions to create bad partitions if they don't exist for your configuration
    overall_ds_name = "test_stuff"
    # distance_measure = DistanceMeasures.l2_cor_dist
    # distance_measure = DistanceMeasures.l1_with_ref
    distance_measure = DistanceMeasures.l1_cor_dist
    data_type = SyntheticDataType.normal_correlated
    test_results_dir = TEST_ROOT_RESULTS_DIR
    runs = TEST_GENERATED_DATASETS_FILE_PATH
    run_internal_measure_calculation_for_dataset(overall_ds_name=overall_ds_name,
                                                 generated_ds_csv=runs,
                                                 distance_measure=distance_measure,
                                                 data_type=data_type,
                                                 data_dir=test_data_dir,
                                                 results_dir=test_results_dir,
                                                 internal_measures=[ClusteringQualityMeasures.silhouette_score,
                                                                    ClusteringQualityMeasures.pmb],
                                                 n_dropped_clusters=[],
                                                 n_dropped_segments=[],
                                                 )

    # read the files from disk
    datasets = read_internal_measures_calculation(overall_ds_name=overall_ds_name, data_type=data_type,
                                                  root_results_dir=test_results_dir, data_dir=test_data_dir,
                                                  distance_measure=distance_measure, generated_ds_csv=runs)

    assert_that(len(datasets), is_(2))
    assert_that(datasets[0][DescribeBadPartCols.name], has_item("misty-forest-56"))
    assert_that(datasets[1][DescribeBadPartCols.name], has_item("splendid-sunset-12"))


def test_can_run_calculation_for_internal_measures_on_all_datasets_when_dropping_clusters():
    # run test_wandb_create_bad_partitions to create bad partitions if they don't exist for your configuration
    overall_ds_name = "test_stuff"
    distance_measure = DistanceMeasures.l1_cor_dist
    data_type = SyntheticDataType.normal_correlated
    test_results_dir = TEST_ROOT_RESULTS_DIR
    runs = TEST_GENERATED_DATASETS_FILE_PATH
    run_internal_measure_calculation_for_dataset(overall_ds_name=overall_ds_name,
                                                 generated_ds_csv=runs,
                                                 distance_measure=distance_measure,
                                                 data_type=data_type,
                                                 data_dir=test_data_dir,
                                                 results_dir=test_results_dir,
                                                 internal_measures=[ClusteringQualityMeasures.silhouette_score,
                                                                    ClusteringQualityMeasures.pmb],
                                                 n_dropped_clusters=[5, 15],
                                                 n_dropped_segments=[],
                                                 )

    # read the files from disk
    datasets_drop5 = read_internal_measures_calculation(overall_ds_name=overall_ds_name, data_type=data_type,
                                                        root_results_dir=test_results_dir, data_dir=test_data_dir,
                                                        distance_measure=distance_measure, n_dropped_clusters=5,
                                                        generated_ds_csv=runs)

    assert_that(len(datasets_drop5), is_(2))
    assert_that(datasets_drop5[0][DescribeBadPartCols.name], has_item("misty-forest-56"))
    assert_that(datasets_drop5[1][DescribeBadPartCols.name], has_item("splendid-sunset-12"))

    datasets_drop15 = read_internal_measures_calculation(overall_ds_name=overall_ds_name, data_type=data_type,
                                                         root_results_dir=test_results_dir, data_dir=test_data_dir,
                                                         distance_measure=distance_measure, n_dropped_clusters=15,
                                                         generated_ds_csv=runs)

    assert_that(len(datasets_drop15), is_(2))
    assert_that(datasets_drop15[0][DescribeBadPartCols.name], has_item("misty-forest-56"))
    assert_that(datasets_drop15[1][DescribeBadPartCols.name], has_item("splendid-sunset-12"))


def test_can_run_calculation_for_internal_measures_on_all_datasets_when_dropping_segments():
    # run test_wandb_create_bad_partitions to create bad partitions if they don't exist for your configuration
    overall_ds_name = "test_stuff"
    distance_measure = DistanceMeasures.l1_cor_dist
    data_type = SyntheticDataType.normal_correlated
    test_results_dir = TEST_ROOT_RESULTS_DIR
    runs = TEST_GENERATED_DATASETS_FILE_PATH
    run_internal_measure_calculation_for_dataset(overall_ds_name=overall_ds_name,
                                                 generated_ds_csv=runs,
                                                 distance_measure=distance_measure,
                                                 data_type=data_type,
                                                 data_dir=test_data_dir,
                                                 results_dir=test_results_dir,
                                                 internal_measures=[ClusteringQualityMeasures.silhouette_score,
                                                                    ClusteringQualityMeasures.pmb],
                                                 n_dropped_clusters=[],
                                                 n_dropped_segments=[50],
                                                 )

    # read the files from disk
    datasets_drop50 = read_internal_measures_calculation(overall_ds_name=overall_ds_name, data_type=data_type,
                                                         root_results_dir=test_results_dir, data_dir=test_data_dir,
                                                         distance_measure=distance_measure, n_dropped_segments=50,
                                                         generated_ds_csv=runs)

    assert_that(len(datasets_drop50), is_(2))
    assert_that(datasets_drop50[0][DescribeBadPartCols.name], has_item("misty-forest-56"))
    assert_that(datasets_drop50[1][DescribeBadPartCols.name], has_item("splendid-sunset-12"))


def test_calculates_n_segment_within_tolerance_stats():
    """ Calculate and return stats df across the partitions for the dataset"""
    n_within = describe.n_segment_within_tolerance_stats()

    assert_that(n_within['mean'], is_(65.6))
    assert_that(n_within['std'], is_(31.919))
    assert_that(n_within['50%'], is_(73.000))
    assert_that(n_within['min'], is_(23))
    assert_that(n_within['max'], is_(98))


def test_calculates_n_segment_outside_tolerance_stats():
    n_outside = describe.n_segment_outside_tolerance_stats()

    assert_that(n_outside['mean'], is_(34.4))
    assert_that(n_outside['std'], is_(31.919))
    assert_that(n_outside['50%'], is_(27))
    assert_that(n_outside['min'], is_(2))
    assert_that(n_outside['max'], is_(77))


def test_calculates_mae_stats():
    mae = describe.mae_stats()

    assert_that(mae['mean'], is_(0.320))
    assert_that(mae['std'], is_(0.244))
    assert_that(mae['50%'], is_(0.175))
    assert_that(mae['min'], is_(0.112))
    assert_that(mae['max'], is_(0.612))


def test_calculates_segment_length_stats():
    mean_seg_length = describe.segment_length_stats()

    assert_that(mean_seg_length['mean'], is_(12264.0))
    assert_that(mean_seg_length['std'], is_(0))
    assert_that(mean_seg_length['50%'], is_(12264.0))
    assert_that(mean_seg_length['min'], is_(12264.0))
    assert_that(mean_seg_length['max'], is_(12264.0))


def test_calculates_jaccard_stats():
    j_stats = describe.jaccard_stats()

    assert_that(j_stats['mean'], is_(0.71))
    assert_that(j_stats['std'], is_(0.345))
    assert_that(j_stats['50%'], is_(0.896))
    assert_that(j_stats['min'], is_(0.293))
    assert_that(j_stats['max'], is_(1.0))
