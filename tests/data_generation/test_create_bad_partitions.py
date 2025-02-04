import pytest
from hamcrest import *

from src.data_generation.create_bad_partitions import CreateBadSyntheticPartitions
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.evaluation.describe_synthetic_dataset import DescribeSyntheticDataset, DescribeSyntheticCols
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR

ds_name = "misty-forest-56"
test_data_dir = TEST_DATA_DIR
bp = CreateBadSyntheticPartitions(run_name=ds_name, data_type=SyntheticDataType.non_normal_correlated,
                                  data_dir=test_data_dir)


def test_assigns_n_segments_to_a_wrong_cluster():
    n_segments = [2, 10, 50, 100]
    n_partitions = 4
    resulting_labels = bp.randomly_assign_wrong_cluster(n_partitions=n_partitions, n_segments=n_segments)

    assert_that(len(resulting_labels), is_(n_partitions))

    original_cluster_ids = bp.labels[SyntheticDataSegmentCols.pattern_id].tolist()
    original_correlations_to_model = bp.labels[SyntheticDataSegmentCols.correlation_to_model].tolist()
    for p in range(n_partitions):
        # assert that the number of differences in cluster ids are as specified
        new_labels = resulting_labels[p]
        new_cluster_ids = new_labels[SyntheticDataSegmentCols.pattern_id].tolist()
        n_differences = sum(i != j for i, j in zip(original_cluster_ids, new_cluster_ids))
        assert_that(n_differences, is_(n_segments[p]))

        # assert that the correlations to model have changed accordingly to
        new_correlations_to_model = new_labels[SyntheticDataSegmentCols.correlation_to_model].tolist()
        n_differences = sum(i != j for i, j in zip(original_correlations_to_model, new_correlations_to_model))
        assert_that(n_differences, is_(n_segments[p]))

    # ensure labels file has been recalculated
    a_new_label = resulting_labels[3]
    pattern_id = a_new_label.loc[0, SyntheticDataSegmentCols.pattern_id]
    assert_that(pattern_id, is_(10))  # changed first pattern
    # updated pattern to model
    correlations_lookup = ModelCorrelationPatterns().canonical_patterns()
    assert_that(a_new_label.loc[0, SyntheticDataSegmentCols.correlation_to_model],
                contains_exactly(*correlations_lookup[pattern_id]))
    # recalculated tolerance
    assert_that(a_new_label.loc[0, SyntheticDataSegmentCols.actual_within_tolerance],
                contains_exactly(False, True, False))
    # recalculated MAE which is now bigger
    assert_that(a_new_label.loc[0, SyntheticDataSegmentCols.mae],
                greater_than(bp.labels.loc[0, SyntheticDataSegmentCols.mae]))


def test_shifts_segments_end_index_by_n_observations():
    n_partitions = 3
    n_observations = [100, 200, 600]
    resulting_labels = bp.shift_segments_end_index(n_partitions=n_partitions, n_observations=n_observations)

    assert_that(len(resulting_labels), is_(n_partitions))

    for p in range(n_partitions):
        new_label = resulting_labels[p]
        n_obs = n_observations[p]

        # first start index is 0
        assert_that(new_label.loc[0, SyntheticDataSegmentCols.start_idx], is_(0))
        # first end index is n_obs higher
        assert_that(new_label.loc[0, SyntheticDataSegmentCols.end_idx],
                    is_(bp.labels.loc[0, SyntheticDataSegmentCols.end_idx] + n_obs))
        # first segment length is n_obs longer
        assert_that(new_label.loc[0, SyntheticDataSegmentCols.length],
                    is_(bp.labels.loc[0, SyntheticDataSegmentCols.length] + n_obs))

        # second last segment stays the same length
        second_last_idx = new_label.index[-2]
        assert_that(new_label.loc[second_last_idx, SyntheticDataSegmentCols.length],
                    is_(bp.labels.loc[second_last_idx, SyntheticDataSegmentCols.length]))

        # last segment is n_obs shorter
        last_idx = new_label.index[-1]
        assert_that(new_label.loc[last_idx, SyntheticDataSegmentCols.length],
                    is_(bp.labels.loc[last_idx, SyntheticDataSegmentCols.length] - n_obs))

        # last segment start idx is n_obs higher
        assert_that(new_label.loc[second_last_idx, SyntheticDataSegmentCols.start_idx],
                    is_(bp.labels.loc[second_last_idx, SyntheticDataSegmentCols.start_idx] + n_obs))
        # last end index is the same
        assert_that(new_label.loc[last_idx, SyntheticDataSegmentCols.end_idx],
                    is_(bp.labels.loc[last_idx, SyntheticDataSegmentCols.end_idx]))

        # correlations have changed (test first, 66th and second last as last is too long a segment)
        new_cor_0 = new_label.loc[0, SyntheticDataSegmentCols.actual_correlation]
        old_corr_0 = bp.labels.loc[0, SyntheticDataSegmentCols.actual_correlation]
        assert_that(sum(x != y for x, y in zip(new_cor_0, old_corr_0)), greater_than(1))
        new_cor_66 = new_label.loc[66, SyntheticDataSegmentCols.actual_correlation]
        old_corr_66 = bp.labels.loc[66, SyntheticDataSegmentCols.actual_correlation]
        assert_that(sum(x != y for x, y in zip(new_cor_66, old_corr_66)), greater_than(1))
        new_cor_99 = new_label.loc[second_last_idx, SyntheticDataSegmentCols.actual_correlation]
        old_corr_99 = bp.labels.loc[second_last_idx, SyntheticDataSegmentCols.actual_correlation]
        assert_that(sum(x != y for x, y in zip(new_cor_99, old_corr_99)), greater_than(1))

        # mae has been updated
        assert_that(new_label.loc[0, SyntheticDataSegmentCols.mae],
                    is_not(bp.labels.loc[0, SyntheticDataSegmentCols.mae]))
        assert_that(new_label.loc[66, SyntheticDataSegmentCols.mae],
                    is_not(bp.labels.loc[66, SyntheticDataSegmentCols.mae]))
        assert_that(new_label.loc[second_last_idx, SyntheticDataSegmentCols.mae],
                    is_not(bp.labels.loc[second_last_idx, SyntheticDataSegmentCols.mae]))


def test_creates_bad_partitions_both_shifting_segments_end_idx_and_assigning_random_wrong_cluster():
    n_partitions = 3
    n_segments = [5, 50, 33]
    n_observations = [200, 300, 800]
    resulting_labels = bp.shift_segments_end_index_and_assign_wrong_clusters(n_partitions=n_partitions,
                                                                             n_observations=n_observations,
                                                                             n_segments=n_segments)

    assert_that(len(resulting_labels), is_(n_partitions))

    original_cluster_ids = bp.labels[SyntheticDataSegmentCols.pattern_id].tolist()
    for p in range(n_partitions):
        n_obs = n_observations[p]
        new_labels = resulting_labels[p]

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
        new_cor = new_labels.loc[a_idx, SyntheticDataSegmentCols.actual_correlation]
        old_corr = bp.labels.loc[a_idx, SyntheticDataSegmentCols.actual_correlation]
        assert_that(sum(x != y for x, y in zip(new_cor, old_corr)), greater_than(0))

        # check we changed cluster ids
        new_cluster_ids = resulting_labels[p][SyntheticDataSegmentCols.pattern_id].tolist()
        n_differences = sum(i != j for i, j in zip(original_cluster_ids, new_cluster_ids))
        assert_that(n_differences, is_(n_segments[p]))

    a_new_label = resulting_labels[1]
    pattern_id = a_new_label.loc[0, SyntheticDataSegmentCols.pattern_id]
    assert_that(pattern_id, is_(4))  # changed first pattern
    # updated pattern to model
    correlations_lookup = ModelCorrelationPatterns().canonical_patterns()
    assert_that(a_new_label.loc[0, SyntheticDataSegmentCols.correlation_to_model],
                contains_exactly(*correlations_lookup[pattern_id]))
    # recalculated tolerance
    assert_that(a_new_label.loc[0, SyntheticDataSegmentCols.actual_within_tolerance],
                contains_exactly(True, False, False))
    # recalculated MAE which is now bigger
    assert_that(a_new_label.loc[0, SyntheticDataSegmentCols.mae],
                greater_than(bp.labels.loc[0, SyntheticDataSegmentCols.mae]))


def test_generate_bad_partitions_for_resampled_data_deal_with_shifting_more_than_last_segment_length():
    rs_bp = CreateBadSyntheticPartitions(run_name="breezy-leaf-30", data_type=SyntheticDataType.rs_1min,
                                         data_dir=test_data_dir)
    last_segment_length = rs_bp.labels["length"].iloc[-1]
    resulting_labels = rs_bp.shift_segments_end_index(n_partitions=1, n_observations=[3 * last_segment_length])

    assert_that(resulting_labels[0].shape[0], is_(rs_bp.labels.shape[0] - 2))


def test_generate_bad_partitions_so_that_last_segment_is_too_short():
    rs_bp = CreateBadSyntheticPartitions(run_name="breezy-leaf-30", data_type=SyntheticDataType.rs_1min,
                                         data_dir=test_data_dir)
    last_segment_length = rs_bp.labels["length"].iloc[-1]
    resulting_labels = rs_bp.shift_segments_end_index(n_partitions=1, n_observations=[last_segment_length-2])

    assert_that(resulting_labels[0].shape[0], is_(rs_bp.labels.shape[0] - 1))
