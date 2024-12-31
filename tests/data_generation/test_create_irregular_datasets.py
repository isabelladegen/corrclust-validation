from hamcrest import *

from src.data_generation.create_irregular_datasets import CreateIrregularDataset
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import SyntheticDataVariates
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR

ds_name = "misty-forest-56"
irds = CreateIrregularDataset(run_name=ds_name, data_type=SyntheticDataType.non_normal_correlated,
                              data_dir=TEST_DATA_DIR, data_cols=SyntheticDataVariates.columns(), seed=1661)


def test_drop_observation_with_a_likelihood_p():
    p = 0.5  # 50%
    irr_data, irr_labels = irds.drop_observation_with_likelihood(p)

    n_samples = irds.orig_data.shape[0]

    lendata = irr_data.shape[0]
    assert_that(lendata, is_(n_samples - (p * n_samples)))
    # indexes have been updated
    assert_that(irr_labels.tail(1)[SyntheticDataSegmentCols.end_idx].values[0], is_(lendata - 1))
    # data df was properly reindex
    assert_that(irr_labels.tail(1)[SyntheticDataSegmentCols.end_idx].values[0], is_(list(irr_data.index)[-1]))

    # compare dfs
    diff_df = irds.orig_labels.compare(irr_labels, result_names=("orig", "new"))

    # cols that needed changing (all the others should not appear in diff df as they have not been updated)
    changed_cols = [SyntheticDataSegmentCols.start_idx, SyntheticDataSegmentCols.end_idx,
                    SyntheticDataSegmentCols.length, SyntheticDataSegmentCols.actual_correlation,
                    SyntheticDataSegmentCols.actual_within_tolerance, SyntheticDataSegmentCols.mae]
    cols_with_differences = list(diff_df.columns.levels[0])
    for cols in changed_cols:
        assert_that(cols in cols_with_differences, is_(True))

    assert_that(len(changed_cols), is_(len(cols_with_differences)))

    # check that the first element in the irregular labels length is +1 the end index
    start_idx = irr_labels.loc[0, SyntheticDataSegmentCols.start_idx]
    end_idx = irr_labels.loc[0, SyntheticDataSegmentCols.end_idx]
    length = irr_labels.loc[0, SyntheticDataSegmentCols.length]
    assert_that(start_idx, is_(0))
    assert_that(length, is_(end_idx + 1))

    # check that the old index of the irregular data for the first end segment is smaller or equal to
    # the old labels end idx, and that the next id belongs to the next old segment
    old_end_idx_of_new_end = irr_data.loc[end_idx, SyntheticDataSegmentCols.old_regular_id]
    orig_end_idx = irds.orig_labels.loc[0, SyntheticDataSegmentCols.end_idx]
    assert_that(old_end_idx_of_new_end, less_than_or_equal_to(orig_end_idx))
    old_2nd_seg_start_id = irr_data.loc[end_idx + 1, SyntheticDataSegmentCols.old_regular_id]
    assert_that(old_2nd_seg_start_id, greater_than(end_idx))
    assert_that(old_2nd_seg_start_id, greater_than(orig_end_idx))

    # check that the new lengths are the same or shorter (given we're dropping samples)
    compare_lengths = diff_df[SyntheticDataSegmentCols.length]["new"] <= diff_df[SyntheticDataSegmentCols.length][
        "orig"]
    assert_that(all(list(compare_lengths)), is_(True))


def test_can_deal_with_segments_that_have_less_than_ncols_samples_left():
    p = 0.9999
    data, labels = irds.drop_observation_with_likelihood(p)
    n_samples = irds.orig_data.shape[0]

    assert_that(len(data), less_than(n_samples))
    assert_that(len(labels), is_(15))

    # all length are bigger than 3
    assert_that((labels[SyntheticDataSegmentCols.length] >= 3).all())

    # all observations in data are in the labels df (if a segment has less than 3 obs left we drop these
    # as we cannot calculate their correlations
    assert_that(labels[SyntheticDataSegmentCols.length].sum(), is_(data.shape[0]))
