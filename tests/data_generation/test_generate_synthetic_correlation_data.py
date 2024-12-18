from datetime import datetime, timezone, timedelta
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
        data = generator.generate(seed=seed+run)
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
    assert_that(max_min_error, is_(0.52))

    assert_that(mean_5min_error, is_(0.361))
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
        data = generator.generate(seed=seed+run)
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
