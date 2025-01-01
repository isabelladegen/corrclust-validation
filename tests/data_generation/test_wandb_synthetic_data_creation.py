from hamcrest import *

from src.data_generation.wandb_create_synthetic_data import SyntheticDataConfig, one_synthetic_creation_run

from src.utils.wandb_utils import set_test_configurations



def test_wandb_synthetic_data_creation_with_loadings_correlation_method():
    config = SyntheticDataConfig()
    set_test_configurations(config)
    config.number_of_segments = 10  # reduces time

    # evaluation is None if the run fails
    results, summary = one_synthetic_creation_run(config, seed=10)
    raw_describe = results['raw']
    normal_correlated_describe = results['nc']
    non_normal_describe = results['nn']
    rs_1min_describe = results['rs']

    assert_that(summary["dataset seed"], is_(10))

    n_observations = 115500
    assert_that(raw_describe.n_patterns, is_(config.number_of_segments))  # each segment has a different pattern
    assert_that(summary["mean pattern frequency RAW"], is_(1))  # only 10 segment so each pattern is used once
    assert_that(raw_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(raw_describe.number_of_observations, is_(n_observations))
    assert_that(summary["n observations RAW"], is_(n_observations))
    assert_that(raw_describe.frequency, is_("s"))
    assert_that(summary["frequency RAW"], is_("s"))
    assert_that(summary["max MAE RAW"], is_(1.0))  # not correlated so highest error

    assert_that(normal_correlated_describe.n_patterns, is_(10))
    assert_that(normal_correlated_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(normal_correlated_describe.number_of_observations, is_(n_observations))
    assert_that(summary["n observations NC"], is_(n_observations))
    assert_that(normal_correlated_describe.frequency, is_("s"))
    assert_that(summary["frequency NC"], is_("s"))
    assert_that(summary["max MAE NC"], is_(0.217))

    assert_that(non_normal_describe.n_patterns, is_(10))
    assert_that(non_normal_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(non_normal_describe.number_of_observations, is_(n_observations))
    assert_that(summary["n observations NC"], is_(n_observations))
    assert_that(non_normal_describe.frequency, is_("s"))
    assert_that(summary["frequency NN"], is_("s"))
    assert_that(summary["max MAE NN"], is_(0.065))  # interesting that this improves

    assert_that(rs_1min_describe.n_patterns, is_(10))
    assert_that(rs_1min_describe.number_of_segments, is_(config.number_of_segments))
    assert_that(rs_1min_describe.number_of_observations, is_(n_observations / 60))
    assert_that(summary["n observations RS"], is_(n_observations / 60))
    assert_that(rs_1min_describe.frequency, is_("min"))  # minutes
    assert_that(summary["frequency RS"], is_("min"))
    assert_that(summary["max MAE RS"], is_(0.29))  # higher than any of the correlated version

def test_wandb_synthetic_data_creation_works_for_cholesky_method_too():
    config = SyntheticDataConfig()
    set_test_configurations(config)
    config.number_of_segments = 5  # reduces time
    config.correlation_model = "cholesky"
    config.do_distribution_fit = True

    # returns none
    results, summary = one_synthetic_creation_run(config, seed=10)
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
