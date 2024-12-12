import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import fit, _discrete_distns, _continuous_distns
from statsmodels.graphics.gofplots import qqplot

from src.utils.plots.matplotlib_helper_functions import fontsize, reset_matplotlib, Backends, display_legend

distribution_col = 'Distribution'
is_discrete_col = 'Is discrete?'
fit_succeeded_col = 'Fit succeeded'
args_col = 'Args'
loc_col = 'loc'
scale_col = 'scale'
nllf_col = 'Negative log likelihood'  # negative log likelihood of fitted params
square_error_col = 'Sum squared error'


class DistributionFit:
    def __init__(self, data: np.array, distributions: [], bounds):
        """
        :param data: 1d numpy array
        :param distributions: list of scipy distribution functions e.g stats.nbionom
        :param bounds: list of bounds for distributions, must be in the same order as the distribution list
        """
        self.data = data
        self.distributions = distributions
        self.bounds = bounds
        self.__results = []
        self.summary_df = None

    def fit(self):
        """ Fits data to given distribution and creates summary_df"""
        self.__results = []
        for idx, distribution in enumerate(self.distributions):
            res = fit(distribution, self.data, self.bounds[idx])
            self.__results.append(res)

        self.summary_df = self.__calculate_results_df()

    def __calculate_results_df(self):
        distribution_names = []
        is_discrete = []
        fit_succeeded = []
        args = []
        locs = []
        scales = []
        nllfs = []  # negative log likelihood of fitted params
        square_errors = []

        # for square error calculation
        # y = frequency, x=bin edges=quantiles
        y, x = np.histogram(self.data, bins=100, density=True)
        x = x[0:-1]  # remove last edge to make same length as frequency

        # calculate results summary for each distribution
        for idx, distribution in enumerate(self.distributions):
            result = self.__results[idx]
            dist_name = distribution.name
            distribution_names.append(dist_name)

            fit_succeeded.append(result.success)

            # get args
            arg_names = [arguments.strip() for arguments in distribution.shapes.split(',')]
            args_fitted = tuple([result.params[name] for name in range(len(arg_names))])
            args.append(args_fitted)

            # discrete and continuous distribution, get kwargs and fitted pmf/pdf
            if dist_name in _discrete_distns._distn_names:
                is_discrete.append(True)
                loc = result.params.loc
                locs.append(loc)
                scales.append(None)
                kwargs = {'loc': loc}
                fitted_pmf_pdf = distribution.pmf(x, *args_fitted, **kwargs)
            elif dist_name in _continuous_distns._distn_names:
                is_discrete.append(False)
                loc = result.params.loc
                locs.append(loc)
                scale = result.params.scale
                scales.append(scale)
                kwargs = {'loc': loc, 'scale': scale}
                fitted_pmf_pdf = distribution.pdf(x, *args_fitted, **kwargs)
            else:
                assert False, dist_name + " not in scipy continuous or discrete distributions"

            nllfs.append(result.nllf())

            # calculate square error
            sum_of_square_error = sum((fitted_pmf_pdf - y) ** 2)
            square_errors.append(sum_of_square_error)

        # assemble dataframe
        results_dictionary = {
            distribution_col: distribution_names,
            is_discrete_col: is_discrete,
            fit_succeeded_col: fit_succeeded,
            args_col: args,
            loc_col: locs,
            scale_col: scales,
            nllf_col: nllfs,
            square_error_col: square_errors
        }
        df = pd.DataFrame(results_dictionary)
        return df.sort_values(by=[fit_succeeded_col, nllf_col, square_error_col], ascending=[False, True, True])

    def best_distribution(self):
        """By smallest sum of square errors"""
        best_dist = self.summary_df.iloc[0]
        if best_dist[fit_succeeded_col]:
            return best_dist
        else:
            return None

    def plot_results_for(self, distribution, backend: str = Backends.visible_tests.value):
        """Give actual scipy distribution, e.g. nbinom (not as string)"""
        # get result for distribution
        idx = self.distributions.index(distribution)
        name = distribution.name
        result = self.summary_df[self.summary_df[distribution_col] == name]
        fit_result = self.__results[idx]
        decimal = 3
        args = [round(value, decimal) for value in result[args_col].values[0]]
        loc = round(result[loc_col].values[0], decimal)
        scale = round(result[scale_col].values[0], decimal)
        nllf = round(result[nllf_col].values[0], decimal)
        square_error = round(result[square_error_col].values[0], decimal)

        # discrete or continuous
        if name in _discrete_distns._distn_names:
            kwargs = {'loc': loc}
            first_title = "PMF"
        elif name in _continuous_distns._distn_names:
            kwargs = {'loc': loc, 'scale': scale}
            first_title = "PDF"

        theoretical_dist = distribution(*args, **kwargs)

        reset_matplotlib(backend)
        fig_size = (18, 10)
        fig, axs = plt.subplots(nrows=1,
                                ncols=2,
                                sharey=False,
                                sharex=False,
                                figsize=fig_size, squeeze=0)

        args_str = str(args).replace('[', '').replace(']', '').replace('\'', '').strip()
        kwargs_str = str(kwargs).replace('{', '').replace('}', '').replace('\'', '').strip()
        nll_str = str(nllf)
        ssqe_str = str(square_error)
        fig.suptitle(name + ", args=" + args_str + " " + kwargs_str + "; nll=" + nll_str + ", ssqe=" + ssqe_str,
                     fontsize=fontsize)

        # PMF/PDF vs histogram left image
        ax = axs[0, 0]
        fit_result.plot(ax=ax)
        ax.set_title("Fitted " + first_title + " and Histogram", fontsize=fontsize)
        ax.set_ylabel(first_title, fontsize=fontsize)
        ax.set_xlabel('x', fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

        # qq plot right image
        ax = axs[0, 1]
        ax.set_title("Data vs Theoretical QQ", fontsize=fontsize)
        qqplot(self.data, dist=theoretical_dist, line="45", ax=ax)
        ax.set_ylabel('Data Quantiles', fontsize=fontsize)
        ax.set_xlabel('Theoretical Quantiles', fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

        fig.set_tight_layout(True)
        plt.show()
        return fig
