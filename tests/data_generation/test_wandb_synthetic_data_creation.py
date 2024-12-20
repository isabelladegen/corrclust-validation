from hamcrest import *

from src.data_generation.create_synthetic_data_wandb import SyntheticDataConfig, one_synthetic_creation_run

from src.utils.wandb_utils import set_test_configurations


def test_wandb_synthetic_data_creation_with_loadings_correlation_method():
    config = SyntheticDataConfig()
    set_test_configurations(config)
    config.number_of_segments = 10  # reduces time

    # evaluation is None if the run fails
    results = one_synthetic_creation_run(config, seed=10)
    raw_describe = results['raw']
    normal_correlated_describe = results['nc']
    non_normal_describe = results['nn']
    rs_1min_describe = results['rs']

    n_observations = 115500
    assert_that(raw_describe.n_patterns, is_(10))  # each segment has a different pattern
    assert_that(raw_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(raw_describe.number_of_observations, is_(n_observations))

    assert_that(normal_correlated_describe.n_patterns, is_(10))
    assert_that(normal_correlated_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(normal_correlated_describe.number_of_observations, is_(n_observations))

    assert_that(non_normal_describe.n_patterns, is_(10))
    assert_that(non_normal_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(non_normal_describe.number_of_observations, is_(n_observations))

    assert_that(rs_1min_describe.n_patterns, is_(10))
    assert_that(rs_1min_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(rs_1min_describe.number_of_observations, is_(n_observations / 60))


def test_wandb_synthetic_data_creation_works_for_cholesky_method_too():
    config = SyntheticDataConfig()
    set_test_configurations(config)
    config.number_of_segments = 5  # reduces time
    config.correlation_model = "cholesky"
    config.do_distribution_fit = True

    # returns none
    results = one_synthetic_creation_run(config, seed=10)
    raw_describe = results['raw']
    normal_correlated_describe = results['nc']
    non_normal_describe = results['nn']
    rs_1min_describe = results['rs']

    assert_that(raw_describe.n_patterns, is_(config.number_of_segments))  # each segment has a different pattern
    assert_that(raw_describe.number_of_segments, is_(config.number_of_segments))

    assert_that(normal_correlated_describe.n_patterns, is_(5))
    assert_that(normal_correlated_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(normal_correlated_describe.number_of_observations, is_(raw_describe.number_of_observations))

    assert_that(non_normal_describe.n_patterns, is_(5))
    assert_that(non_normal_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(non_normal_describe.number_of_observations, is_(raw_describe.number_of_observations))

    assert_that(rs_1min_describe.n_patterns, is_(5))
    assert_that(rs_1min_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(rs_1min_describe.number_of_observations, is_(raw_describe.number_of_observations / 60))
