from hamcrest import *
from pandas.core.dtypes.common import is_datetime64_any_dtype

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import GeneralisedCols
from src.utils.load_synthetic_data import load_synthetic_data, SyntheticDataType, \
    load_synthetic_data_and_labels_for_bad_partitions
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR

test_data_dir = TEST_DATA_DIR


def test_can_load_raw_data_and_labels():
    run_name = "misty-forest-56"
    data_type = SyntheticDataType.raw
    data, labels = load_synthetic_data(run_name, data_type=data_type, data_dir=test_data_dir)

    # check data structure
    assert_that(data.shape, is_((1226400, 4)))  # all data loaded
    assert_that(data.columns, contains_exactly(GeneralisedCols.datetime, GeneralisedCols.iob, GeneralisedCols.cob,
                                               GeneralisedCols.bg))  # correct columns
    assert_that(is_datetime64_any_dtype(data[GeneralisedCols.datetime].dtype))  # timestamp is date time

    # check data first row
    assert_that(data.loc[0, GeneralisedCols.iob], is_(0.3529612374137545))
    assert_that(data.loc[0, GeneralisedCols.cob], is_(-0.7586487494172459))
    assert_that(data.loc[0, GeneralisedCols.bg], is_(0.1256351542775318))

    # check labels data
    assert_that(labels.shape, is_((100, 9)))  # loaded the 100 segments

    # check first row
    assert_that(labels.loc[0, SyntheticDataSegmentCols.segment_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.start_idx], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.end_idx], is_(899))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.length], is_(900))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.pattern_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.correlation_to_model], contains_exactly(0, 0, 0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_correlation], contains_exactly(0.03, -0.03, 0.01))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_within_tolerance], contains_exactly(True, True, True))


def test_can_load_correlated_normal_data_and_labels():
    run_name = "misty-forest-56"

    data, labels = load_synthetic_data(run_name, data_dir=test_data_dir)

    # check data structure
    assert_that(data.shape, is_((1226400, 4)))  # all data loaded
    assert_that(data.columns, contains_exactly(GeneralisedCols.datetime, GeneralisedCols.iob, GeneralisedCols.cob,
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
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_correlation], contains_exactly(0.03, -0.03, 0.01))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_within_tolerance], contains_exactly(True, True, True))


def test_can_load_correlated_non_normal_data_and_labels():
    run_name = "misty-forest-56"
    data_type = SyntheticDataType.non_normal_correlated
    data, labels = load_synthetic_data(run_name, data_type=data_type, data_dir=test_data_dir)

    # check data structure
    assert_that(data.shape, is_((1226400, 4)))  # all data loaded
    assert_that(data.columns, contains_exactly(GeneralisedCols.datetime, GeneralisedCols.iob, GeneralisedCols.cob,
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
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_correlation], contains_exactly(0.03, -0.03, 0.01))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_within_tolerance], contains_exactly(True, True, True))


def test_can_load_resampled_data_and_labels():
    run_name = "misty-forest-56"
    data_type = SyntheticDataType.rs_1min
    data, labels = load_synthetic_data(run_name, data_type=data_type, data_dir=test_data_dir)

    # check data structure
    assert_that(data.shape, is_((20440, 4)))  # all data loaded
    assert_that(data.columns, contains_exactly(GeneralisedCols.datetime, GeneralisedCols.iob, GeneralisedCols.cob,
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
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_correlation], contains_exactly(0.31, -0.23, -0.06))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_within_tolerance], contains_exactly(False, False, True))


def test_can_load_irregular_30_data_and_labels():
    run_name = "misty-forest-56"
    data_type = SyntheticDataType.irregular_p30_drop
    data, labels = load_synthetic_data(run_name, data_type=data_type, data_dir=test_data_dir)

    # check data structure
    assert_that(data.shape, is_((858480, 5)))  # all data loaded
    assert_that(data.columns,
                contains_exactly("old", GeneralisedCols.datetime, GeneralisedCols.iob, GeneralisedCols.cob,
                                 GeneralisedCols.bg))  # correct columns
    assert_that(is_datetime64_any_dtype(data[GeneralisedCols.datetime].dtype))  # timestamp is date time

    # check data first row
    assert_that(data.loc[0, GeneralisedCols.iob], is_(-1.760493453380778))
    assert_that(data.loc[0, GeneralisedCols.cob], is_(17.0))
    assert_that(data.loc[0, GeneralisedCols.bg], is_(121.94388879686365))

    # check labels data
    assert_that(labels.shape[0], is_(100))  # loaded the 100 segments

    # check first row
    assert_that(labels.loc[0, SyntheticDataSegmentCols.segment_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.start_idx], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.end_idx], is_(640))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.length], is_(641))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.pattern_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.correlation_to_model], contains_exactly(0, 0, 0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_correlation], contains_exactly(0.04, -0.01, -0.03))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_within_tolerance], contains_exactly(True, True, True))


def test_can_load_irregular_90_data_and_labels():
    run_name = "misty-forest-56"
    data_type = SyntheticDataType.irregular_p90_drop
    data, labels = load_synthetic_data(run_name, data_type=data_type, data_dir=test_data_dir)

    # check data structure
    assert_that(data.shape, is_((122640, 5)))  # all data loaded
    assert_that(data.columns,
                contains_exactly("old", GeneralisedCols.datetime, GeneralisedCols.iob, GeneralisedCols.cob,
                                 GeneralisedCols.bg))  # correct columns
    assert_that(is_datetime64_any_dtype(data[GeneralisedCols.datetime].dtype))  # timestamp is date time

    # check data first row
    assert_that(data.loc[0, GeneralisedCols.iob], is_(3.2606729496572813))
    assert_that(data.loc[0, GeneralisedCols.cob], is_(5.0))
    assert_that(data.loc[0, GeneralisedCols.bg], is_(65.77402139933736))

    # check labels data
    assert_that(labels.shape[0], is_(100))  # loaded the 100 segments

    # check first row
    assert_that(labels.loc[0, SyntheticDataSegmentCols.segment_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.start_idx], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.end_idx], is_(104))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.length], is_(105))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.pattern_id], is_(0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.correlation_to_model], contains_exactly(0, 0, 0))
    assert_that(labels.loc[0, SyntheticDataSegmentCols.actual_correlation], contains_exactly(-0.14, 0.09, -0.01))
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
    assert_that(len(bad_partitions_labels), is_(66))  # number of bad partitions

    a_partition = bad_partitions_labels[0]
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

