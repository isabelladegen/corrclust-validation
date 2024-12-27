
import pytest
from hamcrest import *

from src.data_generation.create_bad_partitions import CreateBadSyntheticPartitions
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.evaluation.describe_synthetic_dataset import DescribeSyntheticDataset, DescribeSyntheticCols
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR

ds_name = "misty-forest-56"
test_data_dir = TEST_DATA_DIR
bp = CreateBadSyntheticPartitions(run_name=ds_name)


def test_assigns_n_segments_to_a_wrong_cluster():
    n_segments = [2, 10, 50, 100]
    n_partitions = 4
    labels = bp.randomly_assign_wrong_cluster(n_partitions=n_partitions, n_segments=n_segments)

    assert_that(len(labels), is_(n_partitions))

    original_cluster_ids = bp.labels[SyntheticDataSegmentCols.pattern_id].tolist()
    original_correlations_to_model = bp.labels[SyntheticDataSegmentCols.correlation_to_model].tolist()
    for p in range(n_partitions):
        # assert that the number of differences in cluster ids are as specified
        new_cluster_ids = labels[p][SyntheticDataSegmentCols.pattern_id].tolist()
        n_differences = sum(i != j for i, j in zip(original_cluster_ids, new_cluster_ids))
        assert_that(n_differences, is_(n_segments[p]))

        # assert that the correlations to model have changed accordingly to
        new_correlations_to_model = labels[p][SyntheticDataSegmentCols.correlation_to_model].tolist()
        n_differences = sum(i != j for i, j in zip(original_correlations_to_model, new_correlations_to_model))
        assert_that(n_differences, is_(n_segments[p]))


def test_shifts_segments_end_index_by_n_observations():
    n_partitions = 3
    n_observations = [100, 200, 600]
    labels = bp.shift_segments_end_index(n_partitions=n_partitions, n_observations=n_observations)

    assert_that(len(labels), is_(n_partitions))

    for p in range(n_partitions):
        new_labels = labels[p]
        n_obs = n_observations[p]

        # first start index is 0
        assert_that(new_labels.loc[0, SyntheticDataSegmentCols.start_idx], is_(0))
        # first end index is n_obs higher
        assert_that(new_labels.loc[0, SyntheticDataSegmentCols.end_idx],
                    is_(bp.labels.loc[0, SyntheticDataSegmentCols.end_idx] + n_obs))
        # first segment length is n_obs longer
        assert_that(new_labels.loc[0, SyntheticDataSegmentCols.length],
                    is_(bp.labels.loc[0, SyntheticDataSegmentCols.length] + n_obs))

        # last segment is n_obs shorter
        last_idx = new_labels.index[-1]
        assert_that(new_labels.loc[last_idx, SyntheticDataSegmentCols.length],
                    is_(bp.labels.loc[last_idx, SyntheticDataSegmentCols.length] - n_obs))

        # last segment start idx is n_obs higher
        assert_that(new_labels.loc[last_idx, SyntheticDataSegmentCols.start_idx],
                    is_(bp.labels.loc[last_idx, SyntheticDataSegmentCols.start_idx] + n_obs))
        # last end index is the same
        assert_that(new_labels.loc[last_idx, SyntheticDataSegmentCols.end_idx],
                    is_(bp.labels.loc[last_idx, SyntheticDataSegmentCols.end_idx]))

        # correlations have changed (test first, 66th and last)
        assert_that(new_labels.loc[0, SyntheticDataSegmentCols.actual_correlation][0],
                    is_not(bp.labels.loc[0, SyntheticDataSegmentCols.actual_correlation][0]))
        assert_that(new_labels.loc[66, SyntheticDataSegmentCols.actual_correlation][0],
                    is_not(bp.labels.loc[66, SyntheticDataSegmentCols.actual_correlation][0]))
        assert_that(new_labels.loc[last_idx, SyntheticDataSegmentCols.actual_correlation][0],
                    is_not(bp.labels.loc[last_idx, SyntheticDataSegmentCols.actual_correlation][0]))


def test_creates_bad_partitions_both_shifting_segments_end_idx_and_assigning_random_wrong_cluster():
    n_partitions = 3
    n_segments = [5, 50, 33]
    n_observations = [200, 300, 800]
    labels = bp.shift_segments_end_index_and_assign_wrong_clusters(n_partitions=n_partitions,
                                                                   n_observations=n_observations,
                                                                   n_segments=n_segments)

    assert_that(len(labels), is_(n_partitions))

    original_cluster_ids = bp.labels[SyntheticDataSegmentCols.pattern_id].tolist()
    for p in range(n_partitions):
        n_obs = n_observations[p]
        new_labels = labels[p]

        # check we shifted the indices
        # last segment is n_obs shorter
        a_idx = 44
        # length is the same
        assert_that(new_labels.loc[a_idx, SyntheticDataSegmentCols.length],
                    is_(bp.labels.loc[a_idx, SyntheticDataSegmentCols.length]))
        # start idx is n_obs higher
        assert_that(new_labels.loc[a_idx, SyntheticDataSegmentCols.start_idx],
                    is_(bp.labels.loc[a_idx, SyntheticDataSegmentCols.start_idx] + n_obs))
        # end index is n_obs higher
        assert_that(new_labels.loc[a_idx, SyntheticDataSegmentCols.end_idx],
                    is_(bp.labels.loc[a_idx, SyntheticDataSegmentCols.end_idx] + n_obs))
        # correlations have been updated
        assert_that(new_labels.loc[a_idx, SyntheticDataSegmentCols.actual_correlation][1],
                    is_not(bp.labels.loc[a_idx, SyntheticDataSegmentCols.actual_correlation][1]))

        # check we changed cluster ids
        new_cluster_ids = labels[p][SyntheticDataSegmentCols.pattern_id].tolist()
        n_differences = sum(i != j for i, j in zip(original_cluster_ids, new_cluster_ids))
        assert_that(n_differences, is_(n_segments[p]))


@pytest.mark.skip(reason="use this to create labels for datasets used for the below test")
def test_check_how_bad_partitions_become():
    n_segments = [66, 100]
    n_partitions = 2
    labels = bp.randomly_assign_wrong_cluster(n_partitions=n_partitions, n_segments=n_segments)

    for idx, label_df in enumerate(labels):
        label_df.to_csv("bad_partitions/" + ds_name + "-bad-" + str(idx) + "-labels.csv", index=False)


def test_describe_bad_partitions():
    bad_labels_0 = "misty-forest-56-bad-0"
    bad_labels_1 = "misty-forest-56-bad-1"
    ds_bad_labels_0 = DescribeSyntheticDataset(ds_name, labels_file=bad_labels_0, data_dir=test_data_dir)
    ds_bad_labels_1 = DescribeSyntheticDataset(ds_name, labels_file=bad_labels_1, data_dir=test_data_dir)
    ds_gt_labels = DescribeSyntheticDataset(ds_name, data_dir=test_data_dir)

    sum_errors_bad_0 = ds_bad_labels_0.correlation_patterns_df[DescribeSyntheticCols.sum_mean_abs_error].sum()
    sum_errors_bad_1 = ds_bad_labels_1.correlation_patterns_df[DescribeSyntheticCols.sum_mean_abs_error].sum()
    sum_errors_gt = ds_gt_labels.correlation_patterns_df[DescribeSyntheticCols.sum_mean_abs_error].sum()

    assert_that(sum_errors_bad_0, greater_than(sum_errors_gt))
    assert_that(sum_errors_bad_1, greater_than(sum_errors_gt))
    assert_that(sum_errors_bad_1, greater_than(sum_errors_bad_0))

