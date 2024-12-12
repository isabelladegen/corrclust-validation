from hamcrest import *
import numpy as np
from scipy.stats import nbinom, genextreme, poisson, lognorm

from src.utils.plots.matplotlib_helper_functions import Backends
from src.evaluation.distribution_fit import DistributionFit, distribution_col

# create some nbinom data
rng = np.random.default_rng()
size = 10000
nbinom_data = nbinom.rvs(*(5, 0.5), size=size, random_state=rng)

# create some genextreme data
genextreme_data = genextreme.rvs(*(0.04,), loc=39, scale=119, size=size, random_state=rng)

# create some lognormal data
lognorm_data = lognorm.rvs(*(0.36,), loc=67, scale=31, size=size, random_state=rng)

distributions = [nbinom, poisson, genextreme]
nbinom_bounds = {'n': (0, 40)}
poisson_bounds = {'mu': (0, 40)}
genextreme_bounds = {'c': (-1, 1), 'loc': (0, 30), 'scale': (20, 150)}
lognorm_bounds = {'s': (0, 5), 'loc': (0, 70), 'scale': (0, 40)}
bounds = [nbinom_bounds, poisson_bounds, genextreme_bounds]

backend = Backends.none.value


def test_calculate_summary_for_discrete_and_continuous_distributions_discrete_data():
    # fit this data
    dist_fit = DistributionFit(nbinom_data, distributions, bounds)
    dist_fit.fit()
    df = dist_fit.summary_df

    assert_that(df.shape, is_((3, 8)))
    assert_that(df.iloc[0][distribution_col], is_('nbinom'))  # surprising

    assert_that(dist_fit.best_distribution()[distribution_col], is_('nbinom'))


def test_calculate_summary_for_discrete_and_continuous_distributions_continuous_data():
    # fit this data
    dist_fit = DistributionFit(genextreme_data, distributions, bounds)
    dist_fit.fit()
    df = dist_fit.summary_df

    assert_that(df.shape, is_((3, 8)))
    assert_that(df.iloc[0][distribution_col], is_('genextreme'))  # surprising

    assert_that(dist_fit.best_distribution()[distribution_col], is_('genextreme'))


def test_show_pmf_pdf_and_qq_plot_for_distribution_nbinom_data():
    dist_fit = DistributionFit(nbinom_data, distributions, bounds)
    dist_fit.fit()

    dist_fit.plot_results_for(distributions[0], backend=backend)
    dist_fit.plot_results_for(distributions[1], backend=backend)
    dist_fit.plot_results_for(distributions[2], backend=backend)


def test_show_pmf_pdf_and_qq_plot_for_distribution_genextreme_data():
    dist_fit = DistributionFit(genextreme_data, distributions, bounds)
    dist_fit.fit()

    dist_fit.plot_results_for(distributions[0], backend=backend)
    dist_fit.plot_results_for(distributions[1], backend=backend)
    dist_fit.plot_results_for(distributions[2], backend=backend)


def test_show_pdf_for_lognormal_data():
    dists_ = [lognorm]
    dist_fit = DistributionFit(lognorm_data, dists_, [lognorm_bounds])
    dist_fit.fit()

    dist_fit.plot_results_for(dists_[0], backend=Backends.visible_tests.value)



