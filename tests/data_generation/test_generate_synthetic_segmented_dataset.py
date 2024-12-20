import itertools

import numpy as np
import pandas as pd
from hamcrest import *
from matplotlib import pyplot as plt
from scipy.stats import genextreme, nbinom

from src.utils.configurations import GeneralisedCols
from src.utils.plots.matplotlib_helper_functions import Backends
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticSegmentedData, SyntheticDataSegmentCols, \
    min_max_scaled_df, random_list_of_patterns_for, random_segment_lengths

backend = Backends.none.value
seed = 66666

variate_names = [GeneralisedCols.iob, GeneralisedCols.cob, GeneralisedCols.bg]
distributions_for_variates = [genextreme, nbinom, genextreme]

# iob distribution parameters
c_iob = -0.22
loc_iob = 0.5
scale_iob = 1.52

args_iob = (c_iob,)
kwargs_iob = {'loc': loc_iob, 'scale': scale_iob}

# cob distribution parameters
n_cob = 1  # number of successes
p_cob = 0.05  # likelihood of success
args_cob = (n_cob, p_cob)
kwargs_cob = {}  # none, loc will be 0

# ig distribution parameters
c_ig = 0.04
loc_ig = 119.27
scale_ig = 39.40

args_ig = (c_ig,)
kwargs_ig = {'loc': loc_ig, 'scale': scale_ig}

distributions_args = [args_iob, args_cob, args_ig]
distributions_kwargs = [kwargs_iob, kwargs_cob, kwargs_ig]

# patterns to model
cholesky_patterns = ModelCorrelationPatterns().patterns_to_model()
loadings_patterns = ModelCorrelationPatterns().ideal_correlations()


def test_generates_two_segments_with_given_correlation():
    number_of_variates = 3
    number_of_segments = 7

    # optional parameters but they make testing possible
    short_segment_durations = [15, 60]  # in minutes
    long_segment_durations = [360, 490]  # in minutes
    all_strong = [0.999, 0.999, 0.999]
    all_weak = [0.1, -0.1, 0.1]
    weak_and_strong = [0.9, 0.3, 0.4]
    negative_weak_and_strong = [-0.99, -0.3, 0.4]
    correlations_to_model = {1: (all_strong, 0.1),
                             2: (all_weak, 0.000001),
                             3: (weak_and_strong, 0.0001),
                             4: (negative_weak_and_strong, 0.3)}  # cycle through these

    generator = SyntheticSegmentedData(number_of_segments, number_of_variates,
                                       distributions_for_variates,
                                       distributions_args, distributions_kwargs, short_segment_durations,
                                       long_segment_durations, correlations_to_model, variate_names)
    generator.generate(seed=seed)
    non_normal_labels_df = generator.non_normal_labels_df
    assert_that(non_normal_labels_df.shape[0], is_(number_of_segments))

    # cycle through patterns until number of segments are created
    assert_that(non_normal_labels_df.iloc[0][SyntheticDataSegmentCols.pattern_id], is_(3))
    assert_that(non_normal_labels_df.iloc[1][SyntheticDataSegmentCols.pattern_id], is_(4))
    assert_that(non_normal_labels_df.iloc[2][SyntheticDataSegmentCols.pattern_id], is_(1))
    assert_that(non_normal_labels_df.iloc[3][SyntheticDataSegmentCols.pattern_id], is_(2))
    assert_that(non_normal_labels_df.iloc[4][SyntheticDataSegmentCols.pattern_id], is_(1))

    assert_that(non_normal_labels_df.iloc[0][SyntheticDataSegmentCols.correlation_to_model], is_(weak_and_strong))
    assert_that(non_normal_labels_df.iloc[2][SyntheticDataSegmentCols.correlation_to_model], is_(all_strong))
    assert_that(non_normal_labels_df.iloc[3][SyntheticDataSegmentCols.correlation_to_model], is_(all_weak))

    assert_that(non_normal_labels_df.iloc[0][SyntheticDataSegmentCols.regularisation], is_(0.0001))
    assert_that(non_normal_labels_df.iloc[1][SyntheticDataSegmentCols.regularisation], is_(0.3))
    assert_that(non_normal_labels_df.iloc[2][SyntheticDataSegmentCols.regularisation], is_(0.1))
    assert_that(non_normal_labels_df.iloc[3][SyntheticDataSegmentCols.regularisation], is_(0.000001))
    assert_that(non_normal_labels_df.iloc[4][SyntheticDataSegmentCols.regularisation], is_(0.1))

    # draw 4 short segments and one long in cyclical order
    assert_that(non_normal_labels_df.iloc[0][SyntheticDataSegmentCols.length], is_(360))
    assert_that(non_normal_labels_df.iloc[1][SyntheticDataSegmentCols.length], is_(60))
    assert_that(non_normal_labels_df.iloc[2][SyntheticDataSegmentCols.length], is_(490))
    assert_that(non_normal_labels_df.iloc[3][SyntheticDataSegmentCols.length], is_(15))
    assert_that(non_normal_labels_df.iloc[4][SyntheticDataSegmentCols.length], is_(60))
    assert_that(non_normal_labels_df.iloc[5][SyntheticDataSegmentCols.length], is_(15))
    assert_that(non_normal_labels_df.iloc[6][SyntheticDataSegmentCols.length], is_(60))

    # check that labels df for raw and correlated normal and non_normal are not the same
    _, raw_labels = generator.raw_generated_data_labels_df()
    _, normal_labels = generator.normal_correlated_generated_data_labels_df()

    mae_nn = non_normal_labels_df.loc[0, SyntheticDataSegmentCols.mae]
    mae_raw = raw_labels.loc[0, SyntheticDataSegmentCols.mae]
    mae_n = normal_labels.loc[0, SyntheticDataSegmentCols.mae]
    assert_that(mae_raw, greater_than(mae_nn))
    assert_that(mae_raw, greater_than(mae_n))

    nn_achieved_correlation = np.array(non_normal_labels_df.loc[0, SyntheticDataSegmentCols.actual_correlation])
    raw_achieved_correlation = np.array(raw_labels.loc[0, SyntheticDataSegmentCols.actual_correlation])
    n_achieved_correlation = np.array(normal_labels.loc[0, SyntheticDataSegmentCols.actual_correlation])
    assert_that(np.array_equal(nn_achieved_correlation, raw_achieved_correlation), is_(False))
    assert_that(np.array_equal(raw_achieved_correlation, n_achieved_correlation), is_(False))

    # visualise data generated
    df = generator.non_normal_data_df
    assert_that(df.shape[1], is_(number_of_variates + 1))
    assert_that(df.columns, contains_exactly(
        *(GeneralisedCols.datetime, GeneralisedCols.iob, GeneralisedCols.cob, GeneralisedCols.bg)))

    generator.plot_distribution_for_segment(0, backend=backend)
    generator.plot_distribution_for_segment(1, backend=backend)
    generator.plot_distribution_for_segment(2, backend=backend)
    generator.plot_distribution_for_segment(3, backend=backend)
    generator.plot_distribution_for_segment(4, backend=backend)
    generator.plot_correlation_matrix_for_segment(0, backend=backend)
    generator.plot_correlation_matrix_for_segment(1, backend=backend)
    generator.plot_correlation_matrix_for_segment(2, backend=backend)
    generator.plot_correlation_matrix_for_segment(3, backend=backend)
    generator.plot_correlation_matrix_for_segment(4, backend=backend)


def test_generates_all_patterns():
    number_of_variates = 3
    number_of_segments = 23

    short_segment_durations = [15 * 60, 60 * 60, 30 * 60, 80 * 60]  # in second
    long_segment_durations = [360 * 60, 490 * 60]  # in seconds

    generator = SyntheticSegmentedData(number_of_segments, number_of_variates,
                                       distributions_for_variates,
                                       distributions_args, distributions_kwargs, short_segment_durations,
                                       long_segment_durations, cholesky_patterns, variate_names)
    generator.generate(seed=seed)

    # assert all patterns have been used
    segment_df = generator.non_normal_labels_df
    random_pattern_list = [20, 2, 15, 5, 12, 0, 18, 13, 3, 9, 11, 19, 24, 25, 8, 23, 17, 6, 7, 10, 4, 1, 21]
    assert_that(segment_df[SyntheticDataSegmentCols.pattern_id], contains_exactly(*random_pattern_list))

    assert_that(generator.non_normal_data_df.shape[0], is_(201000))


def test_downsample_generated_data_to_minutes_and_check_correlation_results():
    number_of_variates = 3
    number_of_segments = 23

    short_segment_durations = [15 * 60, 60 * 60, 30 * 60, 80 * 60]  # in second
    long_segment_durations = [360 * 60, 490 * 60]  # in seconds

    generator = SyntheticSegmentedData(number_of_segments, number_of_variates,
                                       distributions_for_variates,
                                       distributions_args, distributions_kwargs, short_segment_durations,
                                       long_segment_durations, cholesky_patterns, variate_names)
    generator.generate(seed=seed)
    # check original data is second sampled
    assert_that(pd.infer_freq(generator.non_normal_data_df[GeneralisedCols.datetime]), is_("s"))

    generator.resample(rule="1min")

    # check down sampled data's frequency is minutes
    assert_that(pd.infer_freq(generator.resampled_data[GeneralisedCols.datetime]), is_("min"))

    # we're loosing more of the correlations when downsampling
    original_segment_df = generator.non_normal_labels_df
    list_of_lists = original_segment_df[SyntheticDataSegmentCols.actual_within_tolerance].values
    original_flat_results = list(itertools.chain.from_iterable(list_of_lists))
    original_failure = len(original_flat_results) - sum(original_flat_results)

    down_sampled_segment_df = generator.resampled_labels_df
    list_of_lists = down_sampled_segment_df[SyntheticDataSegmentCols.actual_within_tolerance].values
    down_sampled_flat_results = list(itertools.chain.from_iterable(list_of_lists))
    downsampled_failure = len(down_sampled_flat_results) - sum(down_sampled_flat_results)

    assert_that(downsampled_failure, is_(greater_than(original_failure)))

    print("Original failures: " + str(original_failure))
    print("Downsampled failures: " + str(downsampled_failure))


def test_two_generations_with_a_different_seed_are_different():
    number_of_variates = 3
    number_of_segments = 23

    short_segment_durations = [15 * 60, 60 * 60, 30 * 60, 80 * 60]  # in second
    long_segment_durations = [360 * 60, 490 * 60]  # in seconds

    generator = SyntheticSegmentedData(number_of_segments, number_of_variates,
                                       distributions_for_variates,
                                       distributions_args, distributions_kwargs, short_segment_durations,
                                       long_segment_durations, cholesky_patterns, variate_names)

    data1 = generator.generate(seed=10)
    labels1 = generator.non_normal_labels_df
    data2 = generator.generate(seed=10)
    labels2 = generator.non_normal_labels_df
    data3 = generator.generate(seed=666)
    labels3 = generator.non_normal_labels_df

    # labels 1 and 2 are the same but labels 3 is different
    # pattern orders
    labels1_patterns = labels1[SyntheticDataSegmentCols.pattern_id].tolist()
    labels2_patterns = labels2[SyntheticDataSegmentCols.pattern_id].tolist()
    labels3_patterns = labels3[SyntheticDataSegmentCols.pattern_id].tolist()
    assert_that(labels1_patterns, contains_exactly(*labels2_patterns))
    assert_that(labels1_patterns[0], is_not(labels3_patterns[0]))
    assert_that(labels1_patterns[2], is_not(labels3_patterns[2]))
    assert_that(labels1_patterns[20], is_not(labels3_patterns[20]))

    # segment lengths
    labels1_length = labels1[SyntheticDataSegmentCols.length].tolist()
    labels2_length = labels2[SyntheticDataSegmentCols.length].tolist()
    labels3_length = labels3[SyntheticDataSegmentCols.length].tolist()
    assert_that(labels1_length, contains_exactly(*labels2_length))
    assert_that(labels1_length[0], is_not(labels3_length[0]))
    assert_that(labels1_length[3], is_not(labels3_length[3]))
    assert_that(labels1_length[20], is_not(labels3_length[20]))

    # observations
    assert_that(data1.equals(data2))
    assert_that(data1.equals(data3), is_(False))


def test_returns_scaled_version_of_dataset():
    number_of_variates = 3
    number_of_segments = 23

    short_segment_durations = [15 * 60, 60 * 60, 30 * 60, 80 * 60]  # in second
    long_segment_durations = [360 * 60, 490 * 60]  # in seconds

    generator = SyntheticSegmentedData(number_of_segments, number_of_variates,
                                       distributions_for_variates,
                                       distributions_args, distributions_kwargs, short_segment_durations,
                                       long_segment_durations, cholesky_patterns, variate_names)
    generator.generate(seed=seed)

    scale = (-100, -50)  # move data totally out of current range
    min_max_scaled_data = min_max_scaled_df(generator.non_normal_data_df, scale_range=scale, columns=variate_names)

    # was complete nonsense so added visual check
    generator.non_normal_data_df.plot(x=GeneralisedCols.datetime, y=GeneralisedCols.iob, title="original")
    if backend == Backends.visible_tests.value:
        min_max_scaled_data.plot(x=GeneralisedCols.datetime, y=GeneralisedCols.iob, title="scaled")
        plt.show()

    for variate in variate_names:
        # unscaled values are outside the scale range
        assert_that(generator.non_normal_data_df[variate].min(), is_(greater_than(scale[0])))  # min > -100
        assert_that(generator.non_normal_data_df[variate].max(), is_(greater_than(scale[1])))  # max > -50

        # scaled values are within the scale range
        assert_that(round(min_max_scaled_data[variate].min(), 0), is_(scale[0]))
        assert_that(round(min_max_scaled_data[variate].max(), 0), is_(scale[1]))


def test_ensure_segment_creation_stays_within_correlation_strength_given():
    # recreate data if correlation strength specified is not achieved
    number_of_variates = 3
    number_of_segments = 23

    # segments that are less than 1000 have very little chance to stay within the correlation specified
    shortest = 50
    longest = 3600
    short_segment_durations = [shortest, 100]
    long_segment_durations = [longest]

    generator = SyntheticSegmentedData(number_of_segments, number_of_variates,
                                       distributions_for_variates,
                                       distributions_args, distributions_kwargs, short_segment_durations,
                                       long_segment_durations, loadings_patterns, variate_names)
    generator.generate(seed=seed)

    segment_df = generator.non_normal_labels_df

    # check correlations stayed within tolerance
    list_of_lists = segment_df[SyntheticDataSegmentCols.actual_within_tolerance].values
    flat_results = list(itertools.chain.from_iterable(list_of_lists))
    assert_that(any(flat_results))  # some go to true


def test_generate_a_randomly_ordered_list_of_pattern_ids_to_use_for_each_segment():
    pattern_ids = [1, 2, 3, 4]
    l1_length = 4
    l2_length = 6
    l1 = random_list_of_patterns_for(pattern_ids, l1_length, seed=1)
    l2 = random_list_of_patterns_for(pattern_ids, l2_length, seed=1)
    l3 = random_list_of_patterns_for(pattern_ids, l2_length, seed=2)

    assert_that(len(l1), is_(l1_length))
    assert_that(l1, contains_exactly(2, 1, 3, 4))  # checks that seed sets the order
    assert_that(len(l2), is_(l2_length))
    assert_that(l2, contains_exactly(3, 1, 4, 3, 4, 2))  # checks that seed sets the order
    assert_that(l3, contains_exactly(1, 4, 3, 4, 2, 3))  # same length but different seed


def test_generate_random_pattern_order_throws_exception_if_not_possible_to_not_have_repetitions():
    pattern_ids = [1]
    try:
        random_list_of_patterns_for(pattern_ids, 3, seed=1)
        assert True is False, "Should have thrown an exception"
    except ValueError as err:
        assert_that(str(err), is_("No valid pattern placement found that does not cause repetition"))


def test_generate_a_random_list_of_segment_lengths_for_each_segment():
    short_segments = [100, 300, 400, 600]
    long_segments = [8000, 16000]
    length1 = 6
    length2 = 2
    length3 = 10
    l1 = random_segment_lengths(short_segments, long_segments, length1, seed=1)
    l2 = random_segment_lengths(short_segments, long_segments, length2, seed=1)
    l3 = random_segment_lengths(short_segments, long_segments, length3, seed=1)

    assert_that(len(l1), is_(length1))
    assert_that(l1, contains_exactly(400, 300, 8000, 100, 600, 16000))  # each segment length used once

    assert_that(len(l2), is_(length2))
    assert_that(l2, contains_exactly(600, 400))  # only short ones used

    assert_that(len(l3), is_(length3))
    # all short and long and 4 duplicates
    assert_that(l3, contains_exactly(400, 16000, 400, 100, 100, 600, 300, 600, 8000, 300))
