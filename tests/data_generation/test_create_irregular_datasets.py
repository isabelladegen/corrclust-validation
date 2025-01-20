import pandas as pd
from hamcrest import *
from pandas._libs.tslibs import to_offset

from src.data_generation.create_irregular_datasets import CreateIrregularDataset
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols, \
    recalculate_labels_df_from_data
from src.utils.configurations import SyntheticDataVariates, GeneralisedCols
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR

ds_name = "misty-forest-56"
data_type = SyntheticDataType.raw
irds = CreateIrregularDataset(run_name=ds_name, data_type=data_type,
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
                    SyntheticDataSegmentCols.mae, SyntheticDataSegmentCols.relaxed_mae]
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


def test_creates_same_irregular_version_for_ds_variation():
    # this needs to be high enough so that resampling faces problems of not having a sample for all timestamps
    # this makes some segment disappear and others have less than 3 samples
    p = 0.998
    raw_irr_data, raw_irr_labels = irds.drop_observation_with_likelihood(p)

    # this method is not to redo the irregular calculation we already did
    try:
        irds.irregular_version_for_data_type(SyntheticDataType.raw, raw_irr_data, raw_irr_labels)
        assert_that(False)
    except Exception:
        assert_that(True)

    # NC IRREGULAR VERSION
    nc_irr_data, nc = irds.irregular_version_for_data_type(SyntheticDataType.normal_correlated, raw_irr_data,
                                                           raw_irr_labels)
    # nc df have same shape
    assert_that(nc_irr_data.shape, is_(raw_irr_data.shape))
    assert_that(nc.shape, is_(raw_irr_labels.shape))

    # labels dataframes only differ in correlation achieved, tolerance and mae
    changed_labels_cols = [SyntheticDataSegmentCols.actual_correlation,
                           SyntheticDataSegmentCols.actual_within_tolerance, SyntheticDataSegmentCols.mae,
                           SyntheticDataSegmentCols.relaxed_mae]
    check_that_the_following_columns_are_different(changed_labels_cols, raw_irr_labels, nc)

    # data dataframes only differ in observations columns
    changed_data_cols = SyntheticDataVariates.columns()
    check_that_the_following_columns_are_different(changed_data_cols, raw_irr_data, nc_irr_data)

    # NN IRREGULAR VERSION
    nn_irr_data, nn_irr_labels = irds.irregular_version_for_data_type(SyntheticDataType.non_normal_correlated,
                                                                      raw_irr_data,
                                                                      raw_irr_labels)
    # nn df have same shape
    assert_that(nn_irr_data.shape, is_(raw_irr_data.shape))
    assert_that(nn_irr_labels.shape, is_(raw_irr_labels.shape))

    # labels dataframes only differ in correlation achieved, tolerance and mae
    check_that_the_following_columns_are_different(changed_labels_cols, raw_irr_labels, nn_irr_labels)

    # data dataframes only differ in observations columns
    check_that_the_following_columns_are_different(changed_data_cols, raw_irr_data, nn_irr_data)

    # RESAMPLED VERSION FROM NN
    rs_irr_data, rs_irr_labels = irds.irregular_version_for_data_type(SyntheticDataType.rs_1min, nn_irr_data,
                                                                      nn_irr_labels)

    # check data is sampled at min, the timestamps are no longer consecutive
    median_diff = pd.Series(rs_irr_data[GeneralisedCols.datetime]).diff().median()
    freq = to_offset(median_diff)
    assert_that(freq.rule_code, is_("min"))
    assert_that(rs_irr_labels.iloc[0][SyntheticDataSegmentCols.start_idx], is_(0))
    assert_that(rs_irr_labels.iloc[-1][SyntheticDataSegmentCols.end_idx], is_(rs_irr_data.shape[0] - 1))
    assert_that(rs_irr_labels.columns,
                contains_exactly(SyntheticDataSegmentCols.segment_id, SyntheticDataSegmentCols.start_idx,
                                 SyntheticDataSegmentCols.end_idx, SyntheticDataSegmentCols.length,
                                 SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.correlation_to_model,
                                 SyntheticDataSegmentCols.actual_correlation,
                                 SyntheticDataSegmentCols.actual_within_tolerance, SyntheticDataSegmentCols.mae,
                                 SyntheticDataSegmentCols.relaxed_mae))

    # resampled mae higher than nn mae
    assert_that(rs_irr_labels.iloc[-1][SyntheticDataSegmentCols.mae],
                greater_than(nn_irr_labels.iloc[-1][SyntheticDataSegmentCols.mae]))


def test_compare_similarity_between_normal_and_non_normal_irregular_data():
    seed = 1661
    p = 0.5
    run_name = "amber-glade-10"
    nc_id = CreateIrregularDataset(run_name=run_name, data_type=SyntheticDataType.normal_correlated,
                                   data_dir=TEST_DATA_DIR, data_cols=SyntheticDataVariates.columns(), seed=seed)
    nc_data, nc_labels = nc_id.drop_observation_with_likelihood(p)

    nn_id = CreateIrregularDataset(run_name=run_name, data_type=SyntheticDataType.non_normal_correlated,
                                   data_dir=TEST_DATA_DIR, data_cols=SyntheticDataVariates.columns(), seed=seed)
    nn_data, nn_labels = nc_id.drop_observation_with_likelihood(p)

    orig_nc_data, orig_nc_labels = load_synthetic_data(run_name, SyntheticDataType.normal_correlated,
                                                       data_dir=TEST_DATA_DIR)
    orig_nn_data, orig_nn_labels = load_synthetic_data(run_name, SyntheticDataType.non_normal_correlated,
                                                       data_dir=TEST_DATA_DIR)

    orig_data_diff = orig_nc_data.compare(orig_nn_data, result_names=("orig", "new"))
    orig_labels_diff = orig_nc_labels.compare(orig_nn_labels, result_names=("orig", "new"))
    data_diff = nc_data.compare(nn_data, result_names=("orig", "new"))
    label_diff = nc_labels.compare(nn_labels, result_names=("orig", "new"))

    new_orig_nc_labels = recalculate_labels_df_from_data(orig_nc_data, orig_nc_labels)
    new_orig_nn_labels = recalculate_labels_df_from_data(orig_nn_data, orig_nn_labels)

    diff_nc_recalculated = orig_nc_labels.compare(new_orig_nc_labels, result_names=("orig", "new"))
    diff_nn_recalculated = orig_nn_labels.compare(new_orig_nn_labels, result_names=("orig", "new"))

    print("Compare labels")


def check_that_the_following_columns_are_different(changed_cols, df_orig, df_new):
    differences_df = df_orig.compare(df_new, result_names=("orig", "new"))
    cols_with_differences = list(differences_df.columns.levels[0])
    for cols in changed_cols:
        assert_that(cols in cols_with_differences, is_(True))
    assert_that(len(changed_cols), is_(len(cols_with_differences)))
    return differences_df
