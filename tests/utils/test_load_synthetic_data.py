from hamcrest import *
from pandas.core.dtypes.common import is_datetime64_any_dtype

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import GeneralisedCols
from src.utils.load_synthetic_data import load_synthetic_data, SyntheticDataType, \
    load_synthetic_data_and_labels_for_bad_partitions
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR, TEST_IRREGULAR_P90_DATA_DIR, TEST_IRREGULAR_P30_DATA_DIR

test_data_dir = TEST_DATA_DIR


def test_can_load_raw_data_and_labels():
    run_name = "misty-forest-56"
    data_type = SyntheticDataType.raw
    data, labels = load_synthetic_data(run_name, data_type=data_type, data_dir=test_data_dir)

    # check data structure
    assert_that(data.shape, is_((1226400, 5)))  # all data loaded
    assert_that(data.columns, contains_exactly(SyntheticDataSegmentCols.subject_id, GeneralisedCols.datetime, GeneralisedCols.iob, GeneralisedCols.cob,
                                               GeneralisedCols.bg))  # correct columns
    assert_that(is_datetime64_any_dtype(data[GeneralisedCols.datetime].dtype))  # timestamp is date time

    # check data first row
    assert_that(data.loc[0, GeneralisedCols.iob], is_(0.3529612374137545))
    assert_that(data.loc[0, GeneralisedCols.cob], is_(-0.7586487494172459))
    assert_that(data.loc[0, GeneralisedCols.bg], is_(0.1256351542775318))

    # check labels data
    assert_that(labels.shape, is_((100, 11)))  # loaded the 100 segments

    # check first row
    assert_that(labels.loc[0, SyntheticDataSegmentCols.segment_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.start_idx], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.end_idx], is_(899))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.length], is_(900))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.pattern_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.correlation_to_model], contains_exactly(0, 0, 0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_correlation], contains_exactly(0.031, -0.031, 0.009))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_within_tolerance], contains_exactly(True, True, True))


def test_can_load_correlated_normal_data_and_labels():
    run_name = "misty-forest-56"

    data, labels = load_synthetic_data(run_name, data_dir=test_data_dir)

    # check data structure
    assert_that(data.shape, is_((1226400, 5)))  # all data loaded
    assert_that(data.columns, contains_exactly(SyntheticDataSegmentCols.subject_id, GeneralisedCols.datetime, GeneralisedCols.iob, GeneralisedCols.cob,
                                               GeneralisedCols.bg))  # correct columns
    assert_that(is_datetime64_any_dtype(data[GeneralisedCols.datetime].dtype))  # timestamp is date time

    # check data first row
    assert_that(data.loc[0, GeneralisedCols.iob], is_(0.3529612374137545))
    assert_that(data.loc[0, GeneralisedCols.cob], is_(-0.7586487494172459))
    assert_that(data.loc[0, GeneralisedCols.bg], is_(0.1256351542775318))

    # check labels data
    assert_that(labels.shape[0], is_(100))  # loaded the 100 segments

    # check first row
    assert_that(labels.loc[0, SyntheticDataSegmentCols.segment_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.start_idx], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.end_idx], is_(899))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.length], is_(900))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.pattern_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.correlation_to_model], contains_exactly(0, 0, 0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_correlation], contains_exactly(0.031, -0.031, 0.009))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_within_tolerance], contains_exactly(True, True, True))


def test_can_load_correlated_non_normal_data_and_labels():
    run_name = "misty-forest-56"
    data_type = SyntheticDataType.non_normal_correlated
    data, labels = load_synthetic_data(run_name, data_type=data_type, data_dir=test_data_dir)

    # check data structure
    assert_that(data.shape, is_((1226400, 5)))  # all data loaded
    assert_that(data.columns, contains_exactly(SyntheticDataSegmentCols.subject_id,GeneralisedCols.datetime, GeneralisedCols.iob, GeneralisedCols.cob,
                                               GeneralisedCols.bg))  # correct columns
    assert_that(is_datetime64_any_dtype(data[GeneralisedCols.datetime].dtype))  # timestamp is date time

    # check data first row
    assert_that(data.loc[0, GeneralisedCols.iob], is_(1.828845646416472))
    assert_that(data.loc[0, GeneralisedCols.cob], is_(4.0))
    assert_that(data.loc[0, GeneralisedCols.bg], is_(139.33048859742905))

    # check labels data
    assert_that(labels.shape[0], is_(100))  # loaded the 100 segments

    # check first row
    assert_that(labels.loc[0, SyntheticDataSegmentCols.segment_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.start_idx], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.end_idx], is_(899))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.length], is_(900))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.pattern_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.correlation_to_model], contains_exactly(0, 0, 0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_correlation], contains_exactly(0.032, -0.031, 0.009))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_within_tolerance], contains_exactly(True, True, True))


def test_can_load_resampled_data_and_labels():
    run_name = "misty-forest-56"
    data_type = SyntheticDataType.rs_1min
    data, labels = load_synthetic_data(run_name, data_type=data_type, data_dir=test_data_dir)

    # check data structure
    assert_that(data.shape, is_((20440, 5)))  # all data loaded
    assert_that(data.columns, contains_exactly(SyntheticDataSegmentCols.subject_id, GeneralisedCols.datetime, GeneralisedCols.iob, GeneralisedCols.cob,
                                               GeneralisedCols.bg))  # correct columns
    assert_that(is_datetime64_any_dtype(data[GeneralisedCols.datetime].dtype))  # timestamp is date time

    # check data first row
    assert_that(data.loc[0, GeneralisedCols.iob], is_(1.3641881728483534))
    assert_that(data.loc[0, GeneralisedCols.cob], is_(18.083333333333332))
    assert_that(data.loc[0, GeneralisedCols.bg], is_(145.62397466853656))

    # check labels data
    assert_that(labels.shape[0], is_(100))  # loaded the 100 segments

    # check first row
    assert_that(labels.loc[0, SyntheticDataSegmentCols.segment_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.start_idx], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.end_idx], is_(14))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.length], is_(15))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.pattern_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.correlation_to_model], contains_exactly(0, 0, 0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_correlation], contains_exactly(0.31, -0.228, -0.064))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_within_tolerance], contains_exactly(False, False, True))


def test_can_load_irregular_30_data_and_labels():
    run_name = "misty-forest-56"
    data_type = SyntheticDataType.non_normal_correlated
    data, labels = load_synthetic_data(run_name, data_type=data_type, data_dir=TEST_IRREGULAR_P30_DATA_DIR)

    # check data structure
    assert_that(data.shape, is_((858480, 6)))  # all data loaded
    assert_that(data.columns,
                contains_exactly(SyntheticDataSegmentCols.subject_id,SyntheticDataSegmentCols.old_regular_id, GeneralisedCols.datetime, GeneralisedCols.iob, GeneralisedCols.cob,
                                 GeneralisedCols.bg))  # correct columns
    assert_that(is_datetime64_any_dtype(data[GeneralisedCols.datetime].dtype))  # timestamp is date time

    # check data first row
    assert_that(data.loc[0, GeneralisedCols.iob], is_(1.828845646416472))
    assert_that(data.loc[0, GeneralisedCols.cob], is_(4.0))
    assert_that(data.loc[0, GeneralisedCols.bg], is_(139.33048859742905))

    # check labels data
    assert_that(labels.shape[0], is_(100))  # loaded the 100 segments

    # check first row
    assert_that(labels.loc[0, SyntheticDataSegmentCols.segment_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.start_idx], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.end_idx], is_(610))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.length], is_(611))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.pattern_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.correlation_to_model], contains_exactly(0, 0, 0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_correlation], contains_exactly(0.038, -0.04, -0.007))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_within_tolerance], contains_exactly(True, True, True))


def test_can_load_nn_irregular_90_data_and_labels():
    run_name = "misty-forest-56"
    data_type = SyntheticDataType.non_normal_correlated
    data, labels = load_synthetic_data(run_name, data_type=data_type, data_dir=TEST_IRREGULAR_P90_DATA_DIR)

    # check data structure
    assert_that(data.shape, is_((122640, 6)))  # all data loaded
    assert_that(data.columns,
                contains_exactly(SyntheticDataSegmentCols.old_regular_id, SyntheticDataSegmentCols.subject_id,  GeneralisedCols.datetime, GeneralisedCols.iob, GeneralisedCols.cob,
                                 GeneralisedCols.bg))  # correct columns
    assert_that(is_datetime64_any_dtype(data[GeneralisedCols.datetime].dtype))  # timestamp is date time

    # check data first row
    assert_that(data.loc[0, GeneralisedCols.iob], is_(1.828845646416472))
    assert_that(data.loc[0, GeneralisedCols.cob], is_(4.0))
    assert_that(data.loc[0, GeneralisedCols.bg], is_(139.33048859742905))

    # check labels data
    assert_that(labels.shape[0], is_(100))  # loaded the 100 segments

    # check first row
    assert_that(labels.loc[0, SyntheticDataSegmentCols.segment_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.start_idx], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.end_idx], is_(102))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.length], is_(103))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.pattern_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.correlation_to_model], contains_exactly(0, 0, 0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_correlation], contains_exactly(-0.024, -0.11, -0.002))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_within_tolerance], contains_exactly(True, True, True))


def test_can_load_bad_partition_data_and_labels_file():
    # note for all bad partition the data stays the same it is just the labels files that change
    run_name = "misty-forest-56"
    data_type = SyntheticDataType.non_normal_correlated
    orig_data, orig_label = load_synthetic_data(run_name, data_type, data_dir=test_data_dir)
    data, gt_label, bad_partitions_labels = load_synthetic_data_and_labels_for_bad_partitions(run_name,
                                                                                              data_type=data_type,
                                                                                              data_dir=test_data_dir)
    assert_that(data.equals(orig_data))  # the data does not change
    assert_that(gt_label.equals(orig_label))  # the ground truth label are the same
    assert_that(len(bad_partitions_labels), is_(4))  # number of bad partitions

    a_partition = list(bad_partitions_labels.values())[0]
    # last segment has the same index in the bad partition and the ground truth
    assert_that(a_partition.iloc[-1][SyntheticDataSegmentCols.end_idx],
                is_(gt_label.iloc[-1][SyntheticDataSegmentCols.end_idx]))


def test_can_restrict_how_many_partitions_are_loaded():
    # note for all bad partition the data stays the same it is just the labels files that change
    run_name = "misty-forest-56"
    data_type = SyntheticDataType.non_normal_correlated
    data, gt_label, bad_partitions_labels = load_synthetic_data_and_labels_for_bad_partitions(run_name,
                                                                                              data_type=data_type,
                                                                                              data_dir=test_data_dir,
                                                                                              load_only=3)
    assert_that(len(bad_partitions_labels), is_(3))  # number of bad partitions
