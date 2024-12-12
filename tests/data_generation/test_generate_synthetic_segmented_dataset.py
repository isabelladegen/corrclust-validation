import itertools

import pandas as pd
from hamcrest import *
from matplotlib import pyplot as plt
from scipy.stats import genextreme, nbinom

from src.utils.configurations import GeneralisedCols
from src.utils.plots.matplotlib_helper_functions import Backends
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticSegmentedData, SyntheticDataSegmentCols, \
    min_max_scaled_df

backend = Backends.none.value

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
    generator.generate()
    segment_df = generator.generated_segment_df
    assert_that(segment_df.shape[0], is_(number_of_segments))

    # cycle through patterns until number of segments are created
    assert_that(segment_df.iloc[0][SyntheticDataSegmentCols.pattern_id], is_(1))
    assert_that(segment_df.iloc[1][SyntheticDataSegmentCols.pattern_id], is_(2))
    assert_that(segment_df.iloc[2][SyntheticDataSegmentCols.pattern_id], is_(3))
    assert_that(segment_df.iloc[3][SyntheticDataSegmentCols.pattern_id], is_(4))
    assert_that(segment_df.iloc[4][SyntheticDataSegmentCols.pattern_id], is_(1))

    assert_that(segment_df.iloc[0][SyntheticDataSegmentCols.correlation_to_model], is_(all_strong))
    assert_that(segment_df.iloc[2][SyntheticDataSegmentCols.correlation_to_model], is_(weak_and_strong))
    assert_that(segment_df.iloc[3][SyntheticDataSegmentCols.correlation_to_model], is_(negative_weak_and_strong))

    assert_that(segment_df.iloc[0][SyntheticDataSegmentCols.regularisation], is_(0.1))
    assert_that(segment_df.iloc[1][SyntheticDataSegmentCols.regularisation], is_(0.000001))
    assert_that(segment_df.iloc[2][SyntheticDataSegmentCols.regularisation], is_(0.0001))
    assert_that(segment_df.iloc[3][SyntheticDataSegmentCols.regularisation], is_(0.3))
    assert_that(segment_df.iloc[4][SyntheticDataSegmentCols.regularisation], is_(0.1))

    # draw 4 short segments and one long in cyclical order
    assert_that(segment_df.iloc[0][SyntheticDataSegmentCols.length], is_(15))
    assert_that(segment_df.iloc[1][SyntheticDataSegmentCols.length], is_(60))
    assert_that(segment_df.iloc[2][SyntheticDataSegmentCols.length], is_(15))
    assert_that(segment_df.iloc[3][SyntheticDataSegmentCols.length], is_(60))
    assert_that(segment_df.iloc[4][SyntheticDataSegmentCols.length], is_(360))
    assert_that(segment_df.iloc[5][SyntheticDataSegmentCols.length], is_(15))
    assert_that(segment_df.iloc[6][SyntheticDataSegmentCols.length], is_(60))

    # visualise data generated
    df = generator.generated_df
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
                                       long_segment_durations, cholesky_patterns, variate_names, max_repetitions=100)
    generator.generate()

    # assert all patterns have been used
    segment_df = generator.generated_segment_df
    assert_that(segment_df[SyntheticDataSegmentCols.pattern_id].unique(),
                contains_exactly(*list(cholesky_patterns.keys())))

    length_of_four_short = sum(short_segment_durations)
    first_three_shorts = sum(short_segment_durations[:3])
    # 4 times all the short, 2 times the first long, 2 times the second long and then the first three shorts
    length = 4 * length_of_four_short + 2 * long_segment_durations[0] + 2 * long_segment_durations[
        1] + first_three_shorts
    assert_that(generator.generated_df.shape[0], is_(length))

    list_of_lists = segment_df[SyntheticDataSegmentCols.actual_within_tolerance].values
    flat_results = list(itertools.chain.from_iterable(list_of_lists))
    print("Number of failures:")
    print(len(flat_results) - sum(flat_results))
    print("Row with max repetitions:")
    print(segment_df.iloc[segment_df[SyntheticDataSegmentCols.repeats].idxmax()])


def test_downsample_generated_data_to_minutes_and_check_correlation_results():
    number_of_variates = 3
    number_of_segments = 23

    short_segment_durations = [15 * 60, 60 * 60, 30 * 60, 80 * 60]  # in second
    long_segment_durations = [360 * 60, 490 * 60]  # in seconds

    generator = SyntheticSegmentedData(number_of_segments, number_of_variates,
                                       distributions_for_variates,
                                       distributions_args, distributions_kwargs, short_segment_durations,
                                       long_segment_durations, cholesky_patterns, variate_names)
    generator.generate()
    # check original data is second sampled
    assert_that(pd.infer_freq(generator.generated_df[GeneralisedCols.datetime]), is_("s"))

    generator.resample(rule="1min")

    # check down sampled data's frequency is minutes
    assert_that(pd.infer_freq(generator.resampled_data.index), is_("min"))

    # we're loosing more of the correlations when downsampling
    original_segment_df = generator.generated_segment_df
    list_of_lists = original_segment_df[SyntheticDataSegmentCols.actual_within_tolerance].values
    original_flat_results = list(itertools.chain.from_iterable(list_of_lists))
    original_failure = len(original_flat_results) - sum(original_flat_results)

    down_sampled_segment_df = generator.resampled_segment_df
    list_of_lists = down_sampled_segment_df[SyntheticDataSegmentCols.actual_within_tolerance].values
    down_sampled_flat_results = list(itertools.chain.from_iterable(list_of_lists))
    downsampled_failure = len(down_sampled_flat_results) - sum(down_sampled_flat_results)

    assert_that(downsampled_failure, is_(greater_than(original_failure)))

    print("Original failures: " + str(original_failure))
    print("Downsampled failures: " + str(downsampled_failure))


def test_returns_scaled_version_of_dataset():
    number_of_variates = 3
    number_of_segments = 23

    short_segment_durations = [15 * 60, 60 * 60, 30 * 60, 80 * 60]  # in second
    long_segment_durations = [360 * 60, 490 * 60]  # in seconds

    generator = SyntheticSegmentedData(number_of_segments, number_of_variates,
                                       distributions_for_variates,
                                       distributions_args, distributions_kwargs, short_segment_durations,
                                       long_segment_durations, cholesky_patterns, variate_names)
    generator.generate()

    scale = (-100, -50)  # move data totally out of current range
    min_max_scaled_data = min_max_scaled_df(generator.generated_df, scale_range=scale, columns=variate_names)

    # was complete nonsense so added visual check
    generator.generated_df.plot(x=GeneralisedCols.datetime, y=GeneralisedCols.iob, title="original")
    min_max_scaled_data.plot(x=GeneralisedCols.datetime, y=GeneralisedCols.iob, title="scaled")
    plt.show()

    for variate in variate_names:
        # unscaled values are outside the scale range
        assert_that(generator.generated_df[variate].min(), is_(greater_than(scale[0])))  # min > -100
        assert_that(generator.generated_df[variate].max(), is_(greater_than(scale[1])))  # max > -50

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
                                       long_segment_durations, loadings_patterns, variate_names, max_repetitions=300)
    generator.generate()

    segment_df = generator.generated_segment_df
    # ensure data generation for a segment is repeated to attempt to get the correlation specified
    shortest_results = segment_df[segment_df[SyntheticDataSegmentCols.length] == shortest]
    longest_results = segment_df[segment_df[SyntheticDataSegmentCols.length] == longest]
    # shorter segments are harder to get within repetition
    assert_that(sum(shortest_results[SyntheticDataSegmentCols.repeats]),
                is_(greater_than(sum(longest_results[SyntheticDataSegmentCols.repeats]))))

    # check correlations stayed within tolerance
    list_of_lists = segment_df[SyntheticDataSegmentCols.actual_within_tolerance].values
    flat_results = list(itertools.chain.from_iterable(list_of_lists))
    assert_that(any(flat_results))  # some go to true

    print("Number of failures:")
    print(len(flat_results) - sum(flat_results))
    print("Row with max repetitions:")
    print(segment_df.iloc[segment_df[SyntheticDataSegmentCols.repeats].idxmax()])
