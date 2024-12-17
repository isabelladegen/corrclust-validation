from hamcrest import *

from src.data_generation.create_synthetic_data_wandb import SyntheticDataConfig, one_synthetic_creation_run

from src.utils.wandb_utils import set_test_configurations


def test_wandb_synthetic_data_creation_with_loadings_correlation_method():
    config = SyntheticDataConfig()
    set_test_configurations(config)
    config.max_repetitions = 1  # reduces time #todo remove as this makes no sense with seed
    config.number_of_segments = 10  # reduces time

    # evaluation is None if the run fails
    raw_describe, normal_correlated_describe, non_normal_describe, downsampled_1min_describe = one_synthetic_creation_run(
        config, seed=10)

    assert_that(raw_describe.n_patterns, is_(10))  # each segment has a different pattern
    assert_that(raw_describe.number_of_segments, is_(config.number_of_segments))

    assert_that(normal_correlated_describe.n_patterns, is_(10))
    assert_that(normal_correlated_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(normal_correlated_describe.number_of_observations, is_(raw_describe.number_of_observations))

    assert_that(non_normal_describe.n_patterns, is_(10))
    assert_that(non_normal_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(non_normal_describe.number_of_observations, is_(raw_describe.number_of_observations))

    assert_that(downsampled_1min_describe.n_patterns, is_(10))
    assert_that(downsampled_1min_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(downsampled_1min_describe.number_of_observations, is_(raw_describe.number_of_observations / 60))


def test_wandb_synthetic_data_creation_works_for_cholesky_method_too():
    config = SyntheticDataConfig()
    set_test_configurations(config)
    config.max_repetitions = 1  # reduces time
    config.number_of_segments = 5  # reduces time
    config.correlation_model = "cholesky"
    config.do_distribution_fit = True
    config.max_repetitions = 1  # reduce time

    # returns none
    raw_describe, normal_correlated_describe, non_normal_describe, downsampled_1min_describe = one_synthetic_creation_run(
        config, seed=10)
    assert_that(raw_describe.n_patterns, is_(config.number_of_segments))  # each segment has a different pattern
    assert_that(raw_describe.number_of_segments, is_(config.number_of_segments))

    assert_that(normal_correlated_describe.n_patterns, is_(5))
    assert_that(normal_correlated_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(normal_correlated_describe.number_of_observations, is_(raw_describe.number_of_observations))

    assert_that(non_normal_describe.n_patterns, is_(5))
    assert_that(non_normal_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(non_normal_describe.number_of_observations, is_(raw_describe.number_of_observations))

    assert_that(downsampled_1min_describe.n_patterns, is_(5))
    assert_that(downsampled_1min_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(downsampled_1min_describe.number_of_observations, is_(raw_describe.number_of_observations / 60))
