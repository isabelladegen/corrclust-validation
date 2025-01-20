from datetime import datetime, timezone, timedelta
from io import StringIO
from statistics import mean

import numpy as np
import pandas as pd
import pytest
from hamcrest import *
from scipy.stats import genextreme, spearmanr, norm, lognorm, nbinom
from statsmodels.stats.moment_helpers import corr2cov

from src.utils.configurations import GeneralisedCols
from src.utils.plots.matplotlib_helper_functions import Backends
from src.evaluation.distribution_fit import DistributionFit, distribution_col, args_col, loc_col, scale_col
from src.data_generation.generate_synthetic_correlated_data import GenerateData, calculate_spearman_correlation, \
    calculate_correlation_error, is_pos_def, generate_observations, generate_correlation_matrix

backend = Backends.none.value
seed = 666


def print_correlations_for(data):
    spear_cor = calculate_spearman_correlation(data=data)
    print("Spearman's correlation:")
    print(spear_cor)
    return spear_cor


def mean_absolute_error_from_actual_and_specified_coefficients(actual_correlations, specified_correlations,
                                                               round_to: int = 3):
    """
    Calculate the mean absolute error between actual correlation coefficients and specified ones
    """
    error = np.round(np.sum(abs(np.array(specified_correlations) - np.array(actual_correlations)), axis=1) / (
        len(actual_correlations)), round_to)
    return error


# TODO move into a class that generates the timeseries, probably better overall than per segment
def to_timeseries_df(data: np.array, delta: timedelta, columns):
    n_obs = data.shape[0]
    assert data.shape[1] == len(columns), "Please provide a column name for each of the features"

    # create timestamps adding time delta to a given start time n_obs times
    # datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0)
    start_time = datetime(2017, 6, 23, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    times = [start_time]
    for x in range(n_obs - 1):
        times.append(times[x] + delta)

    # create df
    df = pd.DataFrame(data, columns=columns)
    df.insert(loc=0, column=GeneralisedCols.datetime, value=times)
    df[GeneralisedCols.datetime] = pd.to_datetime(df[GeneralisedCols.datetime])
    return df


def test_different_cholesky_correlation_patterns():
    # test different correlations - cov matrix needs to be positive definite
    # correlations = [-0.8, -0.8, 0.3]  # work
    # correlations = [-0.99, -0.3, 0.4]  # work
    # correlations = [0.9, 0.3, 0.4]  # work
    # correlations = [0.6, 0.3, 0.5]  # from example
    # correlations = [0.9, 0.8, 0.6]  # strong and weak
    # correlations = [0.8, 0.8, 0.8]  # all strong
    # correlations = [-0.95, 0.6, 0.9]  # not positive definite cov matrix
    # correlations = [0.8, 0.9, 0.99]  # not positive definite cov matrix
    # correlations = [0.1, 0.3, 0.1]  # all weak
    # correlations = [0.1, -0.1, 0.1]  # all weak
    # correlations = [0.8, 0.9, 0.7]  # works
    # correlations = [0.999, 0.999, 0.999]  # works
    # 0, 1, 1	(0.172, 0.541, 0.541)
    # correlations = [0.172, 0.541, 0.541] # works
    correlations = [0.8, -0.8, -0.1]  # works #reg 0.1
    cov_reg = 0.1
    variates = 3
    n = 10000

    # generate correlated genextreme data
    c = -0.04
    loc = 119.28
    scale = 39.40
    args = [(c,)]
    kwargs = [{'loc': loc, 'scale': scale}]
    distributions = [genextreme]

    generator = GenerateData(n, variates, correlations, distributions, args=args, kwargs=kwargs, method="Cholesky",
                             regularisation=cov_reg)
    generator.generate(seed=seed)

    # test resulting correlation
    gen_cor_errors = generator.calculate_correlation_error()
    for err in gen_cor_errors:
        assert_that(err, is_(less_than(0.2)))

    # plot correlation graph
    generator.plot_correlation_matrix(backend=backend)
    generator.plot_pdf_and_histogram(backend=backend)


def test_different_loadings_correlation_patterns():
    # test different correlations - cov matrix needs to be positive definite
    # loadings will find a positive definite one no matter the numbers given, it might be considerable different
    different_correlations = [
        [-0.8, -0.8, 0.3],
        [-0.99, -0.3, 0.4],
        [0.9, 0.3, 0.4],
        [0.6, 0.3, 0.5],
        [0.9, 0.8, 0.6],
        [0.8, 0.8, 0.8],
        [0.1, 0.3, 0.1],
        [0.1, -0.1, 0.1],
        [0.8, 0.9, 0.7],
        [1, 1, 1],
        [0.172, 0.541, 0.541],
        [0.8, -0.8, -0.1],
    ]
    # cov_reg = 0.1
    variates = 3
    n = 10000

    # generate correlated genextreme data
    c = -0.04
    loc = 119.28
    scale = 39.40
    args = [(c,)]
    kwargs = [{'loc': loc, 'scale': scale}]
    distributions = [genextreme]

    for correlations in different_correlations:
        generator = GenerateData(n, variates, correlations, distributions, args=args, kwargs=kwargs, method="loadings")
        generator.generate(seed=seed)

        # test resulting correlation
        gen_cor_errors = generator.calculate_correlation_error()
        for err in gen_cor_errors:
            assert_that(err, is_(less_than(0.1)))

        # plot correlation graph
        # generator.plot_correlation_matrix(backend=backend)
        # generator.plot_pdf_and_histogram(backend=backend)


def test_generate_synthetic_genextreme_correlated_data():
    # test different correlations - cov matrix needs to be positive definite
    correlations = [0.8, 0.9, 0.7]  # works

    variates = 3
    n = 10000

    # generate correlated genextreme data
    c = -0.04
    loc = 119.28
    scale = 39.40
    args = [(c,)]
    kwargs = [{'loc': loc, 'scale': scale}]
    distributions = [genextreme]

    generator = GenerateData(n, variates, correlations, distributions, args=args, kwargs=kwargs, method="loadings")
    synthetic_data = generator.generate(seed=seed)

    # test resulting correlation
    gen_cor_errors = generator.calculate_correlation_error()
    for err in gen_cor_errors:
        assert_that(err, is_(less_than(0.2)))

    # plot results
    generator.plot_pdf_and_histogram(title="PDF vs histogram of specifically distributed correlated data",
                                     backend=backend)

    # test distribution fit
    # n number of random vectors to draw, p is between 0 and 1 and does not need to be specified
    genextreme_bounds = {'c': (-1, 1), 'loc': (90, 150), 'scale': (20, 40)}
    bounds = [genextreme_bounds]
    # variate 1
    dist_fit_v1 = DistributionFit(synthetic_data[:, 0], distributions, bounds)
    dist_fit_v1.fit()
    best_v1_dist = dist_fit_v1.best_distribution()
    assert_that(best_v1_dist[distribution_col], is_(genextreme.name))
    assert_that(abs(best_v1_dist[args_col][0] - c), less_than(0.2))
    assert_that(abs(best_v1_dist[loc_col] - loc), less_than(1))
    assert_that(round(abs(best_v1_dist[scale_col] - scale), 2), less_than(1))

    # variate 2
    dist_fit_v2 = DistributionFit(synthetic_data[:, 1], distributions, bounds)
    dist_fit_v2.fit()
    best_v2_dist = dist_fit_v2.best_distribution()
    assert_that(best_v2_dist[distribution_col], is_(genextreme.name))
    assert_that(abs(best_v2_dist[args_col][0] - c), less_than(0.2))
    assert_that(abs(best_v2_dist[loc_col] - loc), less_than(1))
    assert_that(round(abs(best_v2_dist[scale_col] - scale), 2), less_than(1))

    # variate 3
    dist_fit_v3 = DistributionFit(synthetic_data[:, 1], distributions, bounds)
    dist_fit_v3.fit()
    best_v3_dist = dist_fit_v3.best_distribution()
    assert_that(best_v3_dist[distribution_col], is_(genextreme.name))
    assert_that(abs(best_v3_dist[args_col][0] - c), less_than(0.2))
    assert_that(abs(best_v3_dist[loc_col] - loc), less_than(1))
    assert_that(round(abs(best_v3_dist[scale_col] - scale), 2), less_than(1))

    # plot fit
    dist_fit_v1.plot_results_for(genextreme, backend=backend)
    dist_fit_v2.plot_results_for(genextreme, backend=backend)
    dist_fit_v3.plot_results_for(genextreme, backend=backend)

    # plot correlation graph
    generator.plot_correlation_matrix()


def test_can_generate_synthetic_correlated_data_for_different_distributions():
    correlations = [0.9]  # works
    variates = 2
    n = 10000

    # genextrem params
    c = -0.04
    loc = 119.28
    scale = 39.40
    gen_ex_kwargs = {"loc": loc, "scale": scale}

    # lognormal params
    s = 0.8
    lnloc = -5
    lnscale = 34
    lognorm_kwargs = {"loc": lnloc, "scale": lnscale}

    ge_and_logn_args = [(c,), (s,)]
    ge_and_logn_kwargs = [gen_ex_kwargs, lognorm_kwargs]
    distributions = [genextreme, lognorm]

    generator = GenerateData(n, variates, correlations, distributions, args=ge_and_logn_args,
                             kwargs=ge_and_logn_kwargs)
    generator.generate(seed=seed)

    errors = generator.calculate_correlation_error()

    # test resulting correlation
    for err in errors:
        assert_that(err, is_(less_than(0.2)))

    generator.plot_pdf_and_histogram(backend=backend)


def test_can_generate_synthetic_correlated_data_using_different_distributions_including_discrete_distributions():
    correlations = [0.9]
    variates = 2
    n_obs = 10000

    c = -0.04
    loc = 119.28
    scale = 39.40
    n = 5
    p = 0.2
    gen_ex_kwargs = {"loc": loc, "scale": scale}
    nbinom_kwargs = {}
    args = [(c,), (n, p)]
    kwargs = [gen_ex_kwargs, nbinom_kwargs]
    distributions = [genextreme, nbinom]

    generator = GenerateData(n_obs, variates, correlations, distributions, args=args, kwargs=kwargs)
    synthetic_data = generator.generate(seed=seed)

    errors = generator.calculate_correlation_error()
    print("Spearman's errors between specified and achieved cor")
    print(errors)

    # correlation error smaller than 0.2
    error = 0.2
    for err in errors:
        assert_that(err, is_(less_than(error)))

    # plot distributions
    generator.plot_pdf_and_histogram(backend=backend)

    # test distribution fit
    # n number of random vectors to draw, p is between 0 and 1 and does not need to be specified
    nbinom_bounds = {'n': (0, 10)}
    genextreme_bounds = {'c': (-1, 1), 'loc': (90, 150), 'scale': (20, 40)}
    bounds = [genextreme_bounds, nbinom_bounds]
    dist_fit_v1 = DistributionFit(synthetic_data[:, 0], distributions, bounds)
    dist_fit_v1.fit()
    best_v1_dist = dist_fit_v1.best_distribution()

    dist_fit_v2 = DistributionFit(synthetic_data[:, 1], distributions, bounds)
    dist_fit_v2.fit()
    best_v2_dist = dist_fit_v2.best_distribution()

    # assert distribution name and parameter
    assert_that(best_v1_dist[distribution_col], is_(genextreme.name))
    assert_that(abs(best_v1_dist[args_col][0] - c), less_than(0.2))
    assert_that(abs(best_v1_dist[loc_col] - loc), less_than(1))
    assert_that(abs(best_v1_dist[scale_col] - scale), less_than(1))

    assert_that(best_v2_dist[distribution_col], is_(nbinom.name))
    assert_that(best_v2_dist[args_col][0], is_(n))
    assert_that(abs(best_v2_dist[args_col][1] - p), less_than(0.1))
    assert_that(best_v2_dist[loc_col], is_(0.0))

    dist_fit_v1.plot_results_for(genextreme, backend=backend)
    dist_fit_v2.plot_results_for(nbinom, backend=backend)


def test_can_generate_different_shorter_segments_of_a_given_correlation_reliably():
    # played around with number of samples required, 15 would be one every minute
    # the smaller the number of samples the bigger the resulting correlation error < 0.05 -> better to create
    # high frequency data and get better correlation
    n = 900
    correlations = [0.8, 0.9, 0.7]
    variates = 3
    c = -0.04
    loc = 119.28
    scale = 39.40
    tolerated_cor_error = 0.05

    errors_per_run = []
    errors_smaller_than_05 = []

    args = [(c,)]
    kwargs = [{'loc': loc, 'scale': scale}]
    distributions = [genextreme]

    generator = GenerateData(n, variates, correlations, distributions, args=args, kwargs=kwargs)

    # run 100 times
    for attempt in range(100):
        generator.generate(seed=seed)
        errors = generator.calculate_correlation_error()

        # test resulting correlation
        errors_per_run.append(errors)
        error_small_enough = [x < tolerated_cor_error for x in errors]
        errors_smaller_than_05.append(all(error_small_enough))

    # check 95% of runs errors are smaller than 0.05
    print("Number of correct correlations:")
    print(sum(errors_smaller_than_05))
    assert_that(sum(errors_smaller_than_05), is_(greater_than(9)))


def test_correlation_stays_within_strength_band_specified():
    # don't change these otherwise asserts will fail
    correlation1 = [0.1, 0.8, 0.8]  # not, strong, strong

    # distributions
    variates = 3
    c = -0.04
    loc = 119.28
    scale = 39.40

    args = [(c,)]
    kwargs = [{'loc': loc, 'scale': scale}]
    distributions = [genextreme]

    # run 1: generate 5 observations (almost never stay within error)
    generator1 = GenerateData(15, variates, correlation1, distributions, args=args, kwargs=kwargs,
                              regularisation=0.1)
    generator1.generate(seed=seed)
    within_strength1 = generator1.check_if_achieved_correlation_is_within_original_strengths()
    achieved_cors = generator1.achieved_correlations()
    expected_result = [abs(achieved_cors[0]) < 0.2, abs(achieved_cors[1]) >= 0.7, abs(achieved_cors[2] >= 0.7)]
    print("number of failures 1:")
    print(len(achieved_cors) - sum(within_strength1))
    assert_that(within_strength1, contains_exactly(*expected_result))

    # run 2: generate 1000 observations
    generator2 = GenerateData(1000, variates, correlation1, distributions, args=args, kwargs=kwargs,
                              regularisation=0.1)
    generator2.generate(seed=seed)
    within_strength2 = generator2.check_if_achieved_correlation_is_within_original_strengths()
    achieved_cors2 = generator2.achieved_correlations()
    expected_result2 = [abs(achieved_cors2[0]) < 0.2, abs(achieved_cors2[1]) >= 0.7, abs(achieved_cors2[2] >= 0.7)]
    print("number of failures 2:")
    print(len(achieved_cors2) - sum(within_strength2))
    assert_that(within_strength2, contains_exactly(*expected_result2))


def test_generated_data_has_regularly_sampled_time_stamps_for_a_given_frequency():
    n = 900
    correlations = [0.8, 0.9, 0.7]
    variates = 3
    columns = ["col 1", "col 2", "col 3"]
    c = -0.04
    loc = 119.28
    scale = 39.40
    args = [(c,)]
    kwargs = [{'loc': loc, 'scale': scale}]
    distributions = [genextreme]

    generator = GenerateData(n, variates, correlations, distributions, args=args, kwargs=kwargs)
    data = generator.generate(seed=seed)

    # sampled at 1 second frequency
    delta = timedelta(seconds=1)
    synthetic_df = to_timeseries_df(data=data, delta=delta, columns=columns)

    assert_that(synthetic_df.shape, is_((n, variates + 1)))

    # check cols
    assert_that(synthetic_df.columns[0], is_(GeneralisedCols.datetime))
    assert_that(synthetic_df.columns[1], is_(columns[0]))
    assert_that(synthetic_df.columns[2], is_(columns[1]))
    assert_that(synthetic_df.columns[3], is_(columns[2]))

    # check times type and frequency
    assert_that(all(synthetic_df[GeneralisedCols.datetime].apply(lambda v: isinstance(v, datetime))))
    df_dt = synthetic_df.set_index([GeneralisedCols.datetime])
    assert_that(pd.infer_freq(df_dt.index), is_("s"))
    assert_that(synthetic_df.iloc[0][GeneralisedCols.datetime],
                is_(datetime(2017, 6, 23, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)))


def test_difference_to_original_correlation_after_correlating_data_sampled_at_seconds():
    n = 900
    correlations = [0.8, 0.9, 0.7]
    variates = 3
    columns = ["col 1", "col 2", "col 3"]
    c = -0.04
    loc = 119.28
    scale = 39.40

    min_cor_errors = []
    five_min_cor_errors = []

    args = [(c,)]
    kwargs = [{'loc': loc, 'scale': scale}]
    distributions = [genextreme]

    generator = GenerateData(n, variates, correlations, distributions, args=args, kwargs=kwargs)

    # check mean correlation error when down sampling to minutes and five minutes
    for run in range(100):
        data = generator.generate(seed=seed + run)
        # sampled at 1 second frequency
        delta = timedelta(seconds=1)
        df = to_timeseries_df(data=data, delta=delta, columns=columns)

        # downsample to minutes
        minute_sampled_df = df.resample('1min', on=GeneralisedCols.datetime).mean()
        assert_that(minute_sampled_df.shape[0], is_(15))
        min_sampled_data = minute_sampled_df[columns].to_numpy()
        min_errors = calculate_correlation_error(correlations, min_sampled_data)
        min_cor_errors.append(round(mean(min_errors), 3))

        # downsample to 5 minutes
        five_min_sampled_df = df.resample('5min', on=GeneralisedCols.datetime).mean()
        assert_that(five_min_sampled_df.shape[0], is_(3))
        five_min_sampled_data = five_min_sampled_df[columns].to_numpy()
        five_min_errors = calculate_correlation_error(correlations, five_min_sampled_data)
        five_min_cor_errors.append(round(mean(five_min_errors), 3))

    # calculate min, max, mean error over 100 runs
    mean_min_error = round(mean(min_cor_errors), 3)
    min_min_error = min(min_cor_errors)
    max_min_error = max(min_cor_errors)

    mean_5min_error = round(mean(five_min_cor_errors), 3)
    min_5min_error = min(five_min_cor_errors)
    max_5min_error = max(five_min_cor_errors)

    assert_that(mean_min_error, is_(0.118))
    assert_that(min_min_error, is_(0.019))
    assert_that(max_min_error, is_(0.521))

    assert_that(mean_5min_error, is_(0.363))
    assert_that(min_5min_error, is_(0.2))
    assert_that(max_5min_error, is_(1.267))


@pytest.mark.skip(reason="takes a long time")
def test_difference_to_original_correlation_after_correlating_data_using_10000_seconds():
    n = 900000  # sampled at milliseconds
    correlations = [0.8, 0.9, 0.7]
    variates = 3
    columns = ["col 1", "col 2", "col 3"]
    c = -0.04
    loc = 119.28
    scale = 39.40

    min_cor_errors = []
    five_min_cor_errors = []

    args = [(c,)]
    kwargs = [{'loc': loc, 'scale': scale}]
    distributions = [genextreme]

    generator = GenerateData(n, variates, correlations, distributions, args=args, kwargs=kwargs)

    # check mean correlation error when down sampling to minutes and five minutes
    for run in range(100):
        data = generator.generate(seed=seed + run)
        # sampled at 1 second frequency
        delta = timedelta(milliseconds=1)
        df = to_timeseries_df(data=data, delta=delta, columns=columns)
        # downsample to minutes
        minute_sampled_df = df.resample('min', on=GeneralisedCols.datetime).mean()
        assert_that(minute_sampled_df.shape[0], is_(15))

        min_sampled_data = minute_sampled_df[columns].to_numpy()
        min_errors = calculate_correlation_error(correlations, min_sampled_data)
        min_cor_errors.append(mean(min_errors))

        # downsample to 5 minutes
        five_min_sampled_df = df.resample('5min', on=GeneralisedCols.datetime).mean()
        assert_that(five_min_sampled_df.shape[0], is_(3))
        five_min_sampled_data = five_min_sampled_df[columns].to_numpy()
        five_min_errors = calculate_correlation_error(correlations, five_min_sampled_data)
        five_min_cor_errors.append(mean(five_min_errors))

    # calculate min, max, mean error over 100 runs
    mean_min_error = round(mean(min_cor_errors), 3)
    min_min_error = min(min_cor_errors)
    max_min_error = max(min_cor_errors)

    mean_5min_error = round(mean(five_min_cor_errors))
    min_5min_error = min(five_min_cor_errors)
    max_5min_error = max(five_min_cor_errors)

    assert_that(mean_min_error, is_(0.108))
    assert_that(min_min_error, is_(0.012))
    assert_that(max_min_error, is_(0.356))

    assert_that(mean_5min_error, is_(0))
    assert_that(min_5min_error, is_(0.2))
    assert_that(max_5min_error, is_(1.3))


def test_correlation_between_three_times_the_same_ts():
    normal_data = norm.rvs(size=(1000, 1), loc=0, scale=1)
    same_three_ts = np.concatenate([normal_data, normal_data, normal_data], axis=1)

    # calculating cor and cov from data
    correlations = spearmanr(same_three_ts).correlation[np.triu_indices(3, 1)]
    assert_that(all(correlations), is_(True))  # they are all exactly 1
    cov = np.cov(same_three_ts, rowvar=False)
    # -> false -> I believe this is due to numerical rounding issues
    # assert not is_pos_def(cov) sometimes this is true but mostly not just depends on actual observations

    reg_term = np.diag([0.0001, 0.0001, 0.0001])  # regularisation term
    reg_cov = cov + reg_term  # regulated covariance matrix
    # -> true
    assert is_pos_def(reg_cov)

    # going the other way round and calculate cov from correlation
    cor_matrix = [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]
    std = np.std(same_three_ts, axis=0)
    cov_matrix = corr2cov(cor_matrix, std)

    # -> false - not always
    # assert not is_pos_def(cov_matrix)

    reg_cov_matrix = cov_matrix + reg_term
    # -> true
    assert is_pos_def(reg_cov_matrix)

    # going the other way round with 1,1,-1 correlation
    cor2_matrix = [[1., 1., 1.], [1., 1., -1.], [1., -1., 1.]]
    std = np.std(same_three_ts, axis=0)
    cov2_matrix = corr2cov(cor2_matrix, std)

    # -> false
    assert not is_pos_def(cov2_matrix)

    reg_cov2_matrix = cov2_matrix + reg_term
    # -> false not a possible correlation combination
    assert not is_pos_def(reg_cov2_matrix)


def test_correlation_calculation():
    # generate 2d data
    data = generate_observations(seed=seed, distribution=norm, size=(10000000, 1), loc=0, scale=1)
    same_obs = np.column_stack((data, data))
    correlation = calculate_spearman_correlation(same_obs, 0)
    assert_that(correlation, is_([1]))

    # generate 3d data
    same_obs_3d = np.column_stack((data, data, data))
    correlation_3d = calculate_spearman_correlation(same_obs_3d, 0)
    assert_that(correlation_3d, is_([1, 1, 1]))


def test_correlations_get_clipped_not_rounded():
    data = get_data_that_creates_almost_perfect_correlation()
    corr_2 = calculate_spearman_correlation(data, 2)
    corr_3 = calculate_spearman_correlation(data, 3)
    assert_that(corr_2[0], is_(-0.1))
    assert_that(corr_3[0], is_(-0.108))
    assert_that(corr_2[1], is_(-0.1))
    assert_that(corr_3[1], is_(-0.101))
    assert_that(corr_2[2], is_(0.99))
    assert_that(corr_3[2], is_(0.999))


def test_generate_correlation_matrix_from_upper_half_array():
    # n=3
    correlations = [1, 0.7, 0.3]
    m = generate_correlation_matrix(correlations)
    assert_that(m.shape, is_((3, 3)))
    assert_that(m[1, 2], is_(correlations[2]))
    assert_that(m[0, 2], is_(correlations[1]))
    assert_that(m[0, 1], is_(correlations[0]))

    # n = 4
    corr_n4 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    m_n4 = generate_correlation_matrix(corr_n4)
    assert_that(m_n4[0, 1], is_(corr_n4[0]))
    assert_that(m_n4[0, 2], is_(corr_n4[1]))
    assert_that(m_n4[0, 3], is_(corr_n4[2]))
    assert_that(m_n4[1, 2], is_(corr_n4[3]))
    assert_that(m_n4[1, 3], is_(corr_n4[4]))
    assert_that(m_n4[2, 3], is_(corr_n4[5]))


def get_data_that_creates_almost_perfect_correlation():
    # this string is from the misty-forest-56 irregular p90 data, second segment that produces if rounded
    # a correlation of 1
    segment_data = """ ,old id,datetime,iob,cob,ig
    103,913,2017-06-23 00:15:13+00:00,0.3332366395313703,16.0,142.80991423419823
    104,926,2017-06-23 00:15:26+00:00,0.1127219211354838,29.0,173.65067227224006
    105,968,2017-06-23 00:16:08+00:00,-1.6198441508353114,33.0,182.11634072875484
    106,970,2017-06-23 00:16:10+00:00,-0.6198646313203509,7.0,112.86961549608073
    113,1045,2017-06-23 00:17:25+00:00,-0.4452857865036661,1.0,79.02923366601365
    114,1053,2017-06-23 00:17:33+00:00,4.462903933558053,49.0,213.37927606753397
    115,1072,2017-06-23 00:17:52+00:00,-0.3438326685214177,13.0,134.90306317458462
    116,1082,2017-06-23 00:18:02+00:00,5.9916672759393474,22.0,156.908620193476
    117,1089,2017-06-23 00:18:09+00:00,0.4565345518080042,29.0,172.62898109044602
    118,1102,2017-06-23 00:18:22+00:00,1.010470851408681,14.0,135.05694984232164
    119,1107,2017-06-23 00:18:27+00:00,1.01650571500467,13.0,132.71223187689796
    120,1116,2017-06-23 00:18:36+00:00,5.508692903707404,47.0,208.15701246633364
    121,1131,2017-06-23 00:18:51+00:00,0.6545690929377757,15.0,140.43172012664414
    122,1132,2017-06-23 00:18:52+00:00,-0.6814737468648702,10.0,123.33904724000732
    123,1139,2017-06-23 00:18:59+00:00,-0.8516694894789838,4.0,103.1382151036333
    124,1162,2017-06-23 00:19:22+00:00,0.7297774728484316,11.0,126.18553028088552
    125,1202,2017-06-23 00:20:02+00:00,-0.108249396524037,12.0,129.9246482174557
    126,1207,2017-06-23 00:20:07+00:00,3.9240197163045822,12.0,130.96039987899786
    127,1216,2017-06-23 00:20:16+00:00,-0.9456590305114074,39.0,193.64714373710572
    128,1234,2017-06-23 00:20:34+00:00,-0.1134658042636976,2.0,86.20653553894044
    129,1259,2017-06-23 00:20:59+00:00,2.3026727632953747,16.0,141.32384657603075
    130,1265,2017-06-23 00:21:05+00:00,17.55458494430853,7.0,113.52689065744276
    131,1279,2017-06-23 00:21:19+00:00,1.8888140905328537,5.0,104.03713084643988
    132,1280,2017-06-23 00:21:20+00:00,3.1158526286938124,10.0,125.64553338471543
    133,1281,2017-06-23 00:21:21+00:00,-0.7511914391984209,45.0,205.5988232499348
    134,1295,2017-06-23 00:21:35+00:00,1.6084735419266378,5.0,106.0595747633577
    135,1299,2017-06-23 00:21:39+00:00,3.6325710053477542,20.0,151.9637074631624
    136,1301,2017-06-23 00:21:41+00:00,0.5694822517945354,11.0,127.77202987480824
    137,1317,2017-06-23 00:21:57+00:00,1.4108916398297588,2.0,87.43600096590438
    138,1344,2017-06-23 00:22:24+00:00,9.317829461099036,4.0,99.85461323661686
    139,1358,2017-06-23 00:22:38+00:00,3.3522151948411523,0.0,66.4932616197205
    140,1372,2017-06-23 00:22:52+00:00,7.108723516259212,1.0,79.90442920995386
    141,1389,2017-06-23 00:23:09+00:00,-0.735732104229448,45.0,206.03636763834945
    142,1399,2017-06-23 00:23:19+00:00,-1.1356218639872144,20.0,152.32381261995846
    143,1407,2017-06-23 00:23:27+00:00,1.572323087032001,11.0,128.6837751628173
    144,1408,2017-06-23 00:23:28+00:00,0.8411286912107956,18.0,147.19340827731315
    145,1411,2017-06-23 00:23:31+00:00,14.860319530254108,57.0,227.7690998035361
    146,1420,2017-06-23 00:23:40+00:00,-0.725186988055841,3.0,93.55848681491456
    147,1441,2017-06-23 00:24:01+00:00,1.8574598744419524,24.0,161.13774879379557
    148,1460,2017-06-23 00:24:20+00:00,2.397748437458243,5.0,107.00741740350628
    149,1476,2017-06-23 00:24:36+00:00,8.164585411236787,3.0,93.81535364994228
    150,1477,2017-06-23 00:24:37+00:00,2.1865901927674525,32.0,178.04648930267857
    151,1502,2017-06-23 00:25:02+00:00,2.233933026070926,2.0,86.72425551375554
    152,1503,2017-06-23 00:25:03+00:00,4.567431207534226,0.0,64.68830638743225
    153,1513,2017-06-23 00:25:13+00:00,0.4536911903051929,13.0,134.20068893576536
    154,1521,2017-06-23 00:25:21+00:00,-1.1794234943909017,31.0,175.84472352464127
    155,1525,2017-06-23 00:25:25+00:00,1.8899721474382167,26.0,167.0228577758376
    156,1527,2017-06-23 00:25:27+00:00,-1.166822037831822,1.0,78.44055965413179
    157,1532,2017-06-23 00:25:32+00:00,1.2397050721909624,15.0,138.41377413414963
    158,1534,2017-06-23 00:25:34+00:00,0.1347491041724801,1.0,77.10968491170061
    159,1548,2017-06-23 00:25:48+00:00,1.4615222750363157,7.0,113.3690829887038
    160,1557,2017-06-23 00:25:57+00:00,4.158112405128565,0.0,73.05311578600293
    161,1562,2017-06-23 00:26:02+00:00,2.8983970472903446,6.0,108.39848735275116
    162,1563,2017-06-23 00:26:03+00:00,-0.466355936354539,16.0,140.80166460618238
    163,1571,2017-06-23 00:26:11+00:00,2.290632902392456,61.0,234.69805776075847
    164,1572,2017-06-23 00:26:12+00:00,2.874346785926635,26.0,166.4671005347005
    165,1580,2017-06-23 00:26:20+00:00,5.9035083559776975,8.0,116.16445120128876
    166,1582,2017-06-23 00:26:22+00:00,2.607950473096113,13.0,133.00600245404382
    167,1590,2017-06-23 00:26:30+00:00,-0.2656160535365098,12.0,129.9509141637496
    168,1606,2017-06-23 00:26:46+00:00,1.267551613725462,0.0,70.65773835973033
    169,1629,2017-06-23 00:27:09+00:00,1.878730508127842,2.0,87.04986971152493
    170,1648,2017-06-23 00:27:28+00:00,0.0474640348306035,45.0,205.6614313171237
    171,1656,2017-06-23 00:27:36+00:00,1.1106257806244957,0.0,55.17555168723456
    172,1673,2017-06-23 00:27:53+00:00,1.5646713898116265,8.0,116.00652855859838
    173,1688,2017-06-23 00:28:08+00:00,0.4780518050290203,2.0,91.7265429213654
    174,1691,2017-06-23 00:28:11+00:00,2.2810058717948016,2.0,87.9864974604358
    175,1694,2017-06-23 00:28:14+00:00,3.9142778573298607,5.0,106.62635381527912
    176,1696,2017-06-23 00:28:16+00:00,0.6649932999203434,9.0,119.77654364180202
    177,1714,2017-06-23 00:28:34+00:00,0.54941202370194,9.0,119.69991241539007
    178,1718,2017-06-23 00:28:38+00:00,-1.114929754475277,10.0,125.37590411583098
    179,1725,2017-06-23 00:28:45+00:00,1.6636550120582825,12.0,130.7190057586062
    180,1728,2017-06-23 00:28:48+00:00,1.9050623441528665,20.0,151.77689233782087
    181,1733,2017-06-23 00:28:53+00:00,1.6521797277950343,13.0,133.13794702949787
    182,1742,2017-06-23 00:29:02+00:00,0.458155180135441,70.0,251.5610576599644
    183,1752,2017-06-23 00:29:12+00:00,-1.1763380497290308,56.0,226.6947493241181
    184,1771,2017-06-23 00:29:31+00:00,0.3127041829553071,25.0,163.79561250888287
    185,1781,2017-06-23 00:29:41+00:00,2.7424277186517494,5.0,107.66282507595858
    186,1790,2017-06-23 00:29:50+00:00,0.1889770327085604,5.0,105.97618082078782
    187,1805,2017-06-23 00:30:05+00:00,-0.3281114894113878,2.0,91.18605051662152
    188,1808,2017-06-23 00:30:08+00:00,1.6175044548731383,11.0,128.00380348266845
    189,1809,2017-06-23 00:30:09+00:00,2.215427487089644,53.0,220.14546932446933
    190,1816,2017-06-23 00:30:16+00:00,7.882201618283044,35.0,185.31000777317607
    191,1865,2017-06-23 00:31:05+00:00,-0.2944411505979069,16.0,142.08975626939124
    192,1876,2017-06-23 00:31:16+00:00,5.525185282546812,12.0,129.97961041216135
    193,1881,2017-06-23 00:31:21+00:00,0.9351407671475594,29.0,172.76712985483638
    194,1897,2017-06-23 00:31:37+00:00,0.5280255588885661,18.0,147.96676680531937
    195,1899,2017-06-23 00:31:39+00:00,0.7201602555180177,61.0,235.2805833947168
    196,1901,2017-06-23 00:31:41+00:00,0.1576386824303262,36.0,187.4924063870528
    197,1907,2017-06-23 00:31:47+00:00,1.123037080670696,7.0,113.37461263235424
    198,1912,2017-06-23 00:31:52+00:00,2.7529077883689284,0.0,53.308437549118366
    199,1927,2017-06-23 00:32:07+00:00,2.202446466731689,13.0,132.26754372018635
    200,1928,2017-06-23 00:32:08+00:00,2.0600270482893457,31.0,176.48888507095884
    201,1934,2017-06-23 00:32:14+00:00,-0.5947459467545657,1.0,79.70000892681671
    202,1942,2017-06-23 00:32:22+00:00,1.214701102598286,11.0,127.21232865243188
    203,1949,2017-06-23 00:32:29+00:00,0.1403824779934047,7.0,113.01509339050844
    """
    return pd.read_csv(StringIO(segment_data))[['iob', 'cob', 'ig']]
