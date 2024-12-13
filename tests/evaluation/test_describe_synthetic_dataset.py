import itertools

import numpy as np
from hamcrest import *

from src.utils.load_synthetic_data import SyntheticDataSets, SyntheticFileTypes
from src.utils.configurations import GeneralisedCols
from src.utils.plots.matplotlib_helper_functions import Backends
from src.evaluation.describe_synthetic_dataset import DescribeSyntheticDataset, DescribeSyntheticCols
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR

group1_cluster_ids_to_compare = [(0, 1), (0, 2), (0, 3), (0, 6), (0, 9), (1, 4), (0, 18), (1, 7), (1, 10), (1, 19),
                                 (2, 5), (2, 8), (2, 11), (2, 20), (3, 4), (3, 5), (3, 12), (3, 21), (4, 13), (5, 23),
                                 (6, 7), (6, 8), (6, 15), (6, 24), (7, 25), (8, 17), (9, 10), (9, 11), (9, 12), (9, 15),
                                 (10, 13), (11, 17), (12, 13), (15, 17), (18, 19), (18, 20), (18, 21), (18, 24),
                                 (19, 25), (20, 23), (21, 23), (24, 25)]
group2_cluster_ids_to_compare = [(1, 2), (3, 6), (4, 5), (4, 7), (5, 8), (7, 8), (9, 18), (10, 11), (10, 19), (11, 20),
                                 (12, 15), (12, 21), (15, 24), (19, 20), (21, 24), (0, 4), (0, 5), (0, 7), (0, 8),
                                 (0, 10), (0, 11), (0, 12), (0, 15), (0, 19), (0, 20),
                                 (0, 21), (0, 24), (1, 3), (1, 6), (1, 9), (1, 13), (1, 18), (1, 25), (2, 3),
                                 (2, 6), (2, 9), (2, 17), (2, 18), (2, 23), (3, 9), (3, 13), (3, 18), (3, 23),
                                 (4, 10), (4, 12), (4, 19), (4, 21), (5, 11), (5, 12), (5, 20), (5, 21), (6, 9),
                                 (6, 17), (6, 18), (6, 25), (7, 10), (7, 15), (7, 19), (7, 24), (8, 11), (8, 15),
                                 (8, 20), (8, 24), (9, 13), (9, 17), (10, 12), (10, 15), (11, 12), (11, 15), (18, 23),
                                 (18, 25), (19, 21), (19, 24), (20, 21), (20, 24)]
group3_cluster_ids_to_compare = [(0, 13), (0, 17), (0, 23), (0, 25), (1, 5), (1, 8), (1, 11), (1, 12), (1, 15), (1, 20),
                                 (1, 21), (1, 24), (2, 4), (2, 7), (2, 10), (2, 12), (2, 15), (2, 19), (2, 21), (2, 24),
                                 (3, 7), (3, 8), (3, 10), (3, 11), (3, 15), (3, 19), (3, 20), (3, 24), (4, 6), (4, 9),
                                 (4, 18), (4, 23), (4, 25), (5, 6), (5, 9), (5, 13), (5, 17), (5, 18), (6, 10), (6, 11),
                                 (6, 12), (6, 19), (6, 20), (6, 21), (7, 9), (7, 13), (7, 17), (7, 18), (8, 9), (8, 18),
                                 (8, 23), (8, 25), (9, 19), (9, 20), (9, 21), (9, 24), (10, 17), (10, 18), (10, 25),
                                 (11, 13), (11, 18), (11, 23), (12, 17), (12, 18), (12, 23), (13, 15), (13, 19),
                                 (13, 21), (15, 18), (15, 25), (17, 20), (17, 24), (19, 23), (20, 25), (21, 25),
                                 (23, 24)]
group4_cluster_ids_to_compare = [(1, 17), (1, 23), (2, 13), (2, 25), (3, 17), (3, 25), (4, 8), (4, 11), (4, 15),
                                 (4, 20), (4, 24), (5, 7), (5, 10), (5, 15), (5, 19), (5, 24), (6, 13), (6, 23),
                                 (7, 11), (7, 12), (7, 20), (7, 21), (8, 10), (8, 12), (8, 19), (8, 21), (9, 23),
                                 (9, 25), (10, 20), (10, 21), (10, 24), (11, 19), (11, 21), (11, 24), (12, 19),
                                 (12, 20), (12, 24), (13, 17), (13, 18), (13, 23), (13, 25), (15, 19), (15, 20),
                                 (15, 21), (17, 18), (17, 23), (17, 25), (23, 25)]
group5_cluster_ids_to_compare = [(4, 17), (5, 25), (7, 23), (8, 13), (10, 23), (11, 25), (12, 25), (13, 20), (13, 24),
                                 (15, 23), (17, 19), (17, 21)]

a_ds_name = "misty-forest-56"
backend = Backends.none.value
test_data_dir = TEST_DATA_DIR
ds = DescribeSyntheticDataset(a_ds_name, backend=backend, data_dir=TEST_DATA_DIR)


def test_describes_numeric_properties_of_synthetic_dataset():
    run_name = SyntheticDataSets.splendid_sunset
    describe = DescribeSyntheticDataset(run_name, data_dir=test_data_dir)

    assert_that(describe.number_of_variates, is_(3))
    assert_that(describe.number_of_observations, is_(1226400))
    assert_that(describe.number_of_segments, is_(100))
    assert_that(describe.start_date.isoformat(), is_("2017-06-23T00:00:00+00:00"))
    assert_that(describe.end_date.isoformat(), is_("2017-07-07T04:39:59+00:00"))
    assert_that(describe.duration.days, is_(14))
    assert_that(describe.frequency, is_("s"))

    # correlation descriptions
    correlation_patterns = describe.correlation_patterns_df
    assert_that(correlation_patterns.shape[0], is_(23))
    assert_that(correlation_patterns[DescribeSyntheticCols.n_segments].min(), is_(4))
    assert_that(correlation_patterns[DescribeSyntheticCols.n_segments].max(), is_(5))
    row0 = correlation_patterns.iloc[0]
    assert_that(row0[DescribeSyntheticCols.n_segments], is_(5))
    assert_that(row0[SyntheticDataSegmentCols.pattern_id], is_(0))
    assert_that(row0[SyntheticDataSegmentCols.correlation_to_model], contains_exactly(0, 0, 0))
    assert_that(row0[SyntheticDataSegmentCols.length],
                contains_exactly(900, 14400, 10800, 28800, 7200))
    assert_that(row0[DescribeSyntheticCols.achieved_min], contains_exactly(-0.06, -0.0, -0.03))
    assert_that(row0[DescribeSyntheticCols.achieved_max], contains_exactly(0.01, 0.01, 0.02))
    assert_that(row0[DescribeSyntheticCols.achieved_mean], contains_exactly(-0.014, 0.004, -0.004))
    assert_that(row0[DescribeSyntheticCols.achieved_std], contains_exactly(0.027, 0.0055, 0.0182))
    assert_that(row0[DescribeSyntheticCols.error_min], contains_exactly(0.0, 0.0, 0.0))
    assert_that(row0[DescribeSyntheticCols.error_max], contains_exactly(0.06, 0.01, 0.03))
    assert_that(row0[DescribeSyntheticCols.error_mean], contains_exactly(0.018, 0.004, 0.012))
    assert_that(row0[DescribeSyntheticCols.error_std], contains_exactly(0.0239, 0.0055, 0.013))
    assert_that(row0[DescribeSyntheticCols.n_within_tolerance], contains_exactly(5, 5, 5))
    assert_that(row0[DescribeSyntheticCols.n_outside_tolerance], contains_exactly(0, 0, 0))
    assert_that(correlation_patterns.iloc[-1][DescribeSyntheticCols.n_segments], is_(4))
    assert_that(correlation_patterns.iloc[-1][SyntheticDataSegmentCols.pattern_id], is_(25))
    assert_that(correlation_patterns.iloc[-1][SyntheticDataSegmentCols.correlation_to_model],
                contains_exactly(-1, -1, 1))
    assert_that(correlation_patterns.iloc[-1][SyntheticDataSegmentCols.length],
                contains_exactly(10800, 10800, 10800, 3600))

    # mean abs errors across all variates
    sum_mean_abs_error = describe.sum_mean_absolute_error_stats
    assert_that(sum_mean_abs_error['mean'], is_(0.3464))
    assert_that(sum_mean_abs_error['min'], is_(0))
    assert_that(sum_mean_abs_error['max'], is_(0.6475))
    assert_that(sum_mean_abs_error['std'], is_(0.3178))

    assert_that(describe.patterns_with_sum_mean_error_max_df.shape[0], is_(1))
    assert_that(describe.patterns_with_sum_mean_error_min_df.shape[0], is_(4))
    assert_that(describe.patterns_with_sum_mean_error_smaller_equal_mean_df.shape[0], is_(11))
    assert_that(describe.patterns_with_sum_mean_error_bigger_than_mean_df.shape[0], is_(12))

    assert_that(describe.patterns_with_zero_mean_error_df.shape[0], is_(4))
    assert_that(describe.patterns_with_zero_mean_error_df[SyntheticDataSegmentCols.pattern_id],
                contains_exactly(17, 23, 13, 25))

    # segment descriptions
    seg_lengths = describe.segment_length_stats
    assert_that(seg_lengths['mean'], is_(12264.0))
    assert_that(seg_lengths['min'], is_(900.0))
    assert_that(seg_lengths['max'], is_(43200.0))
    assert_that(seg_lengths['std'], is_(11548.891))

    seg_counts = describe.segment_length_counts
    assert_that(len(seg_counts), is_(11))
    assert_that(seg_counts.loc[900], is_(8))
    assert_that(seg_counts.loc[10800], is_(32))
    assert_that(seg_counts.loc[21600], is_(5))

    # values generated
    obs_values = describe.observations_stats
    assert_that(obs_values[GeneralisedCols.iob]['mean'], is_(1.863))
    assert_that(obs_values[GeneralisedCols.cob]['mean'], is_(19.378))
    assert_that(obs_values[GeneralisedCols.bg]['mean'], is_(141.037))
    assert_that(obs_values[GeneralisedCols.iob]['min'], is_(-2.709))
    assert_that(obs_values[GeneralisedCols.cob]['min'], is_(0.0))
    assert_that(obs_values[GeneralisedCols.bg]['min'], is_(5.625))
    assert_that(obs_values[GeneralisedCols.iob]['max'], is_(154.013))
    assert_that(obs_values[GeneralisedCols.cob]['max'], is_(304.0))
    assert_that(obs_values[GeneralisedCols.bg]['max'], is_(536.147))
    assert_that(obs_values[GeneralisedCols.iob]['std'], is_(3.143))
    assert_that(obs_values[GeneralisedCols.cob]['std'], is_(20.428))
    assert_that(obs_values[GeneralisedCols.bg]['std'], is_(50.107))


def test_misty_forest_ds_description():
    correlation_patterns = ds.correlation_patterns_df

    # correlation description works for all pattern
    assert_that(correlation_patterns[DescribeSyntheticCols.n_within_tolerance].isna().sum(), is_(0))

    # correlation df counts number of patterns correctly
    corr_df = ds.correlation_patterns_df.copy()
    corr_df['length'] = corr_df['length'].apply(lambda x: len(x))
    corr_df['result'] = corr_df[DescribeSyntheticCols.n_segments].eq(corr_df['length'])

    row8 = correlation_patterns.iloc[8]  # known problem pattern
    n_segment8 = len(list(row8[SyntheticDataSegmentCols.length]))
    assert_that(n_segment8, is_(4))
    assert_that(row8[DescribeSyntheticCols.n_within_tolerance], contains_exactly(n_segment8, n_segment8, n_segment8))
    assert_that(row8[DescribeSyntheticCols.n_outside_tolerance], contains_exactly(0, 0, 0))

    # check we can get statistics on errors
    seg_out_tol = ds.n_segment_outside_tolerance_df
    assert_that(len(seg_out_tol.columns), is_(5))  # data cols compared plus pattern id and n_segment cols

    # check we get the right number of segments for each pattern
    vc_df = ds.labels[SyntheticDataSegmentCols.pattern_id].value_counts()
    for pattern in ds.patterns:
        count_from_labels = vc_df.loc[pattern]
        count_from_outside_tol_df = ds.n_segment_outside_tolerance_df[
            ds.n_segment_outside_tolerance_df[SyntheticDataSegmentCols.pattern_id] == pattern][
            DescribeSyntheticCols.n_segments].values[0]
        assert_that(count_from_labels, is_(count_from_outside_tol_df))


def test_can_provide_datatype_for_uncorrelated_normal_data():
    run_name = SyntheticDataSets.splendid_sunset
    data_type = SyntheticFileTypes.normal_data  # normal, not correlated data
    describe = DescribeSyntheticDataset(run_name, data_type)

    assert_that(describe.number_of_variates, is_(3))
    assert_that(describe.number_of_observations, is_(1226400))
    assert_that(describe.number_of_segments, is_(100))
    assert_that(describe.duration.days, is_(14))
    assert_that(describe.frequency, is_("s"))

    # check not correlated
    correlation_patterns = describe.correlation_patterns_df
    # just one pattern which is (0,0,0)
    assert_that(correlation_patterns.shape[0], is_(1))
    row0 = correlation_patterns.iloc[0]
    # all segments follow the 0,0,0 correlation
    assert_that(row0[DescribeSyntheticCols.n_within_tolerance], contains_exactly(100, 100, 100))
    assert_that(row0[DescribeSyntheticCols.n_outside_tolerance], contains_exactly(0, 0, 0))

    # no variation all the same none correlated data
    sum_mean_abs_error = describe.sum_mean_absolute_error_stats
    assert_that(sum_mean_abs_error['mean'], is_(0.0351))
    assert_that(sum_mean_abs_error['min'], is_(0.0351))
    assert_that(sum_mean_abs_error['max'], is_(0.0351))
    assert_that(np.isnan(sum_mean_abs_error['std']))

    # values generated follow normal distribution
    obs_values = describe.observations_stats
    assert_that(obs_values[GeneralisedCols.iob]['mean'], is_(-0))
    assert_that(obs_values[GeneralisedCols.cob]['mean'], is_(0))
    assert_that(obs_values[GeneralisedCols.bg]['mean'], is_(0))
    assert_that(obs_values[GeneralisedCols.iob]['min'], is_(-4.850))
    assert_that(obs_values[GeneralisedCols.cob]['min'], is_(-4.686))
    assert_that(obs_values[GeneralisedCols.bg]['min'], is_(-4.648))
    assert_that(obs_values[GeneralisedCols.iob]['max'], is_(4.790))
    assert_that(obs_values[GeneralisedCols.cob]['max'], is_(4.771))
    assert_that(obs_values[GeneralisedCols.bg]['max'], is_(4.742))
    assert_that(obs_values[GeneralisedCols.iob]['std'], is_(1.001))
    assert_that(obs_values[GeneralisedCols.cob]['std'], is_(1.000))
    assert_that(obs_values[GeneralisedCols.bg]['std'], is_(1.001))


def test_can_provide_datatype_for_correlated_normal_data():
    run_name = SyntheticDataSets.splendid_sunset
    data_type = SyntheticFileTypes.normal_correlated_data  # normal, not correlated data
    describe = DescribeSyntheticDataset(run_name, data_type)

    assert_that(describe.number_of_variates, is_(3))
    assert_that(describe.number_of_observations, is_(1226400))
    assert_that(describe.number_of_segments, is_(100))
    assert_that(describe.duration.days, is_(14))
    assert_that(describe.frequency, is_("s"))

    # check correlation same as for distribution shifted data
    correlation_patterns = describe.correlation_patterns_df
    assert_that(correlation_patterns.shape[0], is_(23))
    row0 = correlation_patterns.iloc[0]
    assert_that(row0[DescribeSyntheticCols.n_within_tolerance], contains_exactly(5, 5, 5))
    assert_that(row0[DescribeSyntheticCols.n_outside_tolerance], contains_exactly(0, 0, 0))

    sum_mean_abs_error = describe.sum_mean_absolute_error_stats
    assert_that(sum_mean_abs_error['mean'], is_(0.3462))
    assert_that(sum_mean_abs_error['min'], is_(0))
    assert_that(sum_mean_abs_error['max'], is_(0.6475))
    assert_that(sum_mean_abs_error['std'], is_(0.3176))

    # values generated follow normal distribution but slightly different to non correlation shifted
    obs_values = describe.observations_stats
    assert_that(obs_values[GeneralisedCols.iob]['mean'], is_(0.001))
    assert_that(obs_values[GeneralisedCols.cob]['mean'], is_(-0.002))
    assert_that(obs_values[GeneralisedCols.bg]['mean'], is_(0))
    assert_that(obs_values[GeneralisedCols.iob]['min'], is_(-5.379))
    assert_that(obs_values[GeneralisedCols.cob]['min'], is_(-4.742))
    assert_that(obs_values[GeneralisedCols.bg]['min'], is_(-5.051))
    assert_that(obs_values[GeneralisedCols.iob]['max'], is_(4.849))
    assert_that(obs_values[GeneralisedCols.cob]['max'], is_(5.107))
    assert_that(obs_values[GeneralisedCols.bg]['max'], is_(4.742))
    assert_that(obs_values[GeneralisedCols.iob]['std'], is_(1.034))
    assert_that(obs_values[GeneralisedCols.cob]['std'], is_(1.035))
    assert_that(obs_values[GeneralisedCols.bg]['std'], is_(1.037))


def test_the_different_groups_with_varying_pattern_changes():
    results = ds.groups

    # 6 different groups 0-5
    assert_that(len(results.keys()), is_(6))
    # all 276 patterns combinations have been assigned
    assert_that(sum([len(items) for items in results.values()]), is_(276))

    # assert len in group is correct
    assert_that(len(results[0]), is_(23))
    assert_that(len(results[1]), is_(42))
    assert_that(len(results[2]), is_(75))
    assert_that(len(results[3]), is_(76))
    assert_that(len(results[4]), is_(48))
    assert_that(len(results[5]), is_(12))

    # check items in group are correct
    assert_that(set(results[1]) - set(group1_cluster_ids_to_compare), is_(set()))  # group 1 items correct
    assert_that(set(results[2]) - set(group2_cluster_ids_to_compare), is_(set()))  # group 2 items correct
    assert_that(set(results[3]) - set(group3_cluster_ids_to_compare), is_(set()))  # group 3 items correct
    assert_that(set(results[4]) - set(group4_cluster_ids_to_compare), is_(set()))  # group 4 items correct
    assert_that(set(results[5]) - set(group5_cluster_ids_to_compare), is_(set()))  # group 5 items correct


def test_segment_pairs_for_each_group():
    segments_for_each_pattern = ds.segments_for_each_pattern

    # test that segments for each pattern are correct
    assert_that(len(segments_for_each_pattern), is_(23))
    assert_that(len(segments_for_each_pattern[0]), is_(5))
    assert_that(len(segments_for_each_pattern[25]), is_(4))
    # flatten the list of all the entries in the dict and ensure all 100 segments are in a pattern
    assert_that(len(set(itertools.chain.from_iterable(segments_for_each_pattern.values()))), is_(100))

    groups = ds.segment_pairs_for_group
    assert_that(len(groups), is_(6))
    assert_that(len(groups[0]), is_(170))
    assert_that(len(groups[1]), is_(822))
    assert_that(len(groups[2]), is_(1455))
    assert_that(len(groups[3]), is_(1438))
    assert_that(len(groups[4]), is_(861))
    assert_that(len(groups[5]), is_(204))


def test_plot_actual_correlations_for_each_pattern():
    fig = ds.plot_correlation_matrix_for_each_pattern()
    fig.savefig('images/correlation_for_patterns-misty-forest-56.png')
    assert_that(fig, is_not(None))


def test_plot_description_of_subgroups():
    # find pairs for this pattern
    pattern_id = 15

    # expected order
    order = []  # default
    fig = ds.plot_example_correlation_matrix_for_each_subgroup(pattern_id=pattern_id, order_groups=order)
    fig.savefig('images/subgroups-misty-forest-56.png')
    assert_that(fig, is_not(None))

    # l1 norm order
    l1_order = [0, 1, 2, 3, 5, 4]
    fig = ds.plot_example_correlation_matrix_for_each_subgroup(pattern_id=pattern_id, order_groups=l1_order)
    fig.savefig('images/l1-ordered-subgroups-misty-forest-56.png')
    assert_that(fig, is_not(None))

    # l2 norm order
    l2_order = [0, 1, 3, 2, 5, 4]
    fig = ds.plot_example_correlation_matrix_for_each_subgroup(pattern_id=pattern_id, order_groups=l2_order)
    fig.savefig('images/l2-ordered-subgroups-misty-forest-56.png')
    assert_that(fig, is_not(None))

    # linf norm order
    linf_order = [0, 1, 3, 5, 2, 4]
    fig = ds.plot_example_correlation_matrix_for_each_subgroup(pattern_id=pattern_id, order_groups=linf_order)
    fig.savefig('images/linf-ordered-subgroups-misty-forest-56.png')
    assert_that(fig, is_not(None))


def test_return_modeled_correlation_as_x_matrix_and_label_as_y_vector():
    x, y = ds.x_and_y_of_patterns_modelled()
    n_seg = 100

    assert_that(x.shape[0], is_(n_seg))
    assert_that(x.shape[1], is_(3))
    assert_that(y.shape[0], is_(n_seg))

    assert_that(all(np.equal(x[0, :], np.array([0.03, -0.03, 0.01]))))
    assert_that(y[0], is_(0))

    assert_that(all(np.equal(x[22, :], np.array([-1, -1, 1]))))
    assert_that(y[22], is_(25))

    assert_that(all(np.equal(x[99, :], np.array([-0.09, -0.72, 0.72]))))
    assert_that(y[99], is_(7))


def test_can_min_max_scale_the_data():
    max_v = 10.
    min_v = 0.
    value_range = (min_v, max_v)
    ds_scaled = DescribeSyntheticDataset(a_ds_name, value_range=value_range, backend=backend)

    # same number of columns as unscaled data
    assert_that(len(ds_scaled.data.columns), is_(len(ds.data.columns)))

    scaled_data = ds_scaled.data[ds_scaled.data_cols]
    for col in ds_scaled.data_cols:
        assert_that(scaled_data.min().loc[col], is_(min_v))
        assert_that(scaled_data.max().loc[col], is_(max_v))


def test_can_load_irregular_dataset():
    irregular_30_ds_name = "irregular_p_0_3_" + a_ds_name
    irregular_30 = DescribeSyntheticDataset(run_name=irregular_30_ds_name, backend=backend)
    irregular_30.correlation_patterns_df[DescribeSyntheticCols.sum_mean_abs_error].describe()

    # check that the irregular lengths are the same or shorter (given we're dropping samples)
    compare_lengths = irregular_30.labels[SyntheticDataSegmentCols.length] <= ds.labels[SyntheticDataSegmentCols.length]
    assert_that(all(list(compare_lengths)), is_(True))
