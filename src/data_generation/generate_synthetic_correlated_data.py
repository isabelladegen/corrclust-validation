import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import cholesky
from scipy.stats import spearmanr, pearsonr, norm, _discrete_distns, _continuous_distns
from sequana.viz import corrplot
from statsmodels.stats.moment_helpers import corr2cov

from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, fontsize


class GenerateData:
    def __init__(self, nobservation: int, nvariates: int, correlations: [float], distributions: [], args: [],
                 kwargs: [], method="loadings", regularisation=0.0001):
        """"
        Generate synthetic data for given distributions and parameters
        :param nobservation: Number of observations to generate
        :param nvariates: Number of variates
        :param correlations: List of correlation in order (variate1/variate2, variate1/variate3, variate2/variate3)...
        :param distributions: List of distributions in order of the variates for which the distribution is, if only one
         distribution is provided it will be used for all variates, e.g [genextreme]
        :param args: List of tuples of args for each of the distributions given, e.g [(0.5,)]
        :param kwargs: List of dictionaries of kwargs for each of the distributions given, e.g [{'loc': 0, 'scale': 1}]
        :param method: 'cholesky' decomposition or 'loadings' to correlate data, covariance regularisation is ignored
        when method is loadings
        :param regularisation: small number to be added to diagonal of covariance/correlation matrix to ensure matrix is
        full rank and all eigenvalues are >0 and not too small to give numerical issues. Only needed for 'cholesky'
        method
        """
        self.nobservation = nobservation
        self.nvariates = nvariates
        self.correlations = correlations
        self.distributions = distributions
        self.method = method.lower()
        self.covariance_regulation = regularisation
        self.args = args
        self.kwargs = kwargs
        self.generated_data = None  # call generate to get new data #np.array, this is the non normal correlated data
        self.normal_data = None  # normal data before anything was applied (the raw data)
        # normal correlated data before distribution was shifted
        self.normal_correlated_data = None

    def generate(self, seed: int, round_to:int=3):
        """
        Generate synthetic data for given distributions and parameters
        Note the credit for the loadings methods to correlate data goes to Henry Reeves.
        :param seed: random seed
        """
        self.generated_data = None
        normal_data = generate_observations(seed, norm, size=(self.nobservation, self.nvariates), loc=0, scale=1)
        if self.method == "cholesky":
            cor_data = cholesky_correlate_data(data=normal_data, correlations=self.correlations,
                                               cov_reg=self.covariance_regulation)
        elif self.method == "loadings":
            # treated as approximate correlation
            cor_data = loading_correlate_data(data=normal_data, correlations=self.correlations)
        else:
            assert False, "Unknown method '{}'".format(self.method)
        cor_data = np.round(cor_data, decimals=round_to)
        dist_data = move_to_distributions(cor_data, self.distributions, self.args, self.kwargs)
        dist_data = np.round(dist_data, decimals=round_to)
        self.normal_data = normal_data
        self.normal_correlated_data = cor_data
        self.generated_data = dist_data
        return self.generated_data

    def calculate_correlation_error(self, round_to: int = 3):
        """Calculate abs difference between specified correlation and achieved correlation in generated data"""
        return calculate_correlation_error(self.correlations, self.generated_data, round_to)

    def plot_pdf_and_histogram(self, title="PDF and histogram of data", backend=Backends.none.value):
        """ Plots pdf and histogram
        :param title: (optional) overall title of the plot
        :param backend: (optional) matplotlib backend given via Backends
        """
        # if only one distribution given it will be used for all variates, otherwise all distributions need to be given
        distributions = self.distributions
        args = self.args
        kwargs = self.kwargs

        if len(distributions) == 1:
            distributions = self.nvariates * distributions
            args = self.nvariates * args
            kwargs = self.nvariates * kwargs

        # setup figure
        reset_matplotlib(backend)
        fig_size = (18, 8)
        fig, axs = plt.subplots(nrows=1,
                                ncols=self.nvariates,
                                sharey=False,
                                sharex=False,
                                figsize=fig_size, squeeze=0)

        fig.suptitle(title, fontsize=fontsize)

        # plot each of the variates
        for col_idx in range(self.nvariates):
            ax = axs[0, col_idx]
            x = self.generated_data[:, col_idx]

            # get distribution settings for variate
            distribution = distributions[col_idx]
            name = distribution.name
            arg = args[col_idx]
            kwarg = kwargs[col_idx]

            # discrete and continuous distribution, get kwargs and fitted pmf/pdf
            if name in _discrete_distns._distn_names:
                pmf_x = np.arange(x.min(), x.max(), 2)  # plot every other vertical line of pmf
                y = distribution.pmf(pmf_x, *arg, **kwarg)
                ax.vlines(pmf_x, 0, y, colors='r', alpha=0.5, lw=1, label='PMF')
            elif name in _continuous_distns._distn_names:
                pdf_x = np.linspace(distribution.ppf(0.0001, *arg, **kwarg),
                                    distribution.ppf(0.9999, *arg, **kwarg), 200)
                y = distribution.pdf(pdf_x, *arg, **kwarg)
                ax.plot(pdf_x, y, 'r-', lw=2, alpha=0.5, label='PDF')

            # plot data
            ax.hist(x, density=True, bins='auto', histtype='stepfilled', alpha=0.2)

            # subplot configuration
            ax.set_title(distribution.name, fontsize=fontsize)
            # show parameters instead
            ax.legend(loc='best', frameon=False, fontsize=fontsize)

            ax.tick_params(axis='x', labelsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)

        plt.show()
        return fig

    def plot_correlation_matrix(self, title="Spearman Correlation Matrix", backend=Backends.none.value):
        """Attention the order of the variates sadly changes and cannot be fixed"""
        reset_matplotlib(backend)
        fig = plt.figure(figsize=(8, 5))
        correlations = calculate_spearman_correlation(self.generated_data, round_to=3)
        correlation_matrix = generate_correlation_matrix(correlations)
        c = corrplot.Corrplot(correlation_matrix, compute_correlation=False)
        c.plot(fig=fig, fontsize=fontsize, lower="number", upper="ellipse", rotation=0)
        fig.suptitle(title, fontsize=fontsize)
        fig.tight_layout()
        plt.show()
        return fig

    def achieved_correlations(self):
        return calculate_spearman_correlation(self.generated_data)

    def check_if_achieved_correlation_is_within_original_strengths(self, strong_cor: float = 0.7, not_cor: float = 0.2):
        """Checks if the correlation achieved stays within the original "strengths" specified
        Returns false for weak correlations not_cor < achieved cor < strong_cor!
        """
        # calculate achieved correlation
        achieved_correlations = self.achieved_correlations()
        return check_correlations_are_within_original_strength(self.correlations, achieved_correlations, strong_cor,
                                                               not_cor)


def calculate_correlation_error(correlations, data, round_to: int = 3):
    actual_correlation = calculate_spearman_correlation(data, round_to=round_to)
    return [abs(round(cor - actual_correlation[idx], round_to)) for idx, cor in enumerate(correlations)]


def generate_observations(seed: int, distribution, *args, **kwargs):
    """ Generates random observations
    :param seed: random seed
    :param distribution: name of the scipy.stats distribution function, usually norm
    :param args: args for that distribution rvs function
    :param kwargs: kwargs for that distribution rvs function

    Example: generate_observations(genextreme, c, size=(10000, 3), loc=loc, scale=scale)
    """
    np.random.seed(seed=seed)
    # generate variates with same distributions
    data = distribution.rvs(*args, **kwargs)
    return np.round(data, decimals=3)


def is_pos_def(x):
    # note that numpy (and scipy and sympy) have limited precision for float, for numpy it is 1e-15, if an
    # eigenvalue is smaller than that it will be treated as zero
    return np.all(np.linalg.eigvals(x) > 1e-14)


def loading_correlate_data(data: np.array, correlations: []):
    cor_matrix = generate_correlation_matrix(correlations)
    # eigendecomposition with negatives removed
    eig_vals, eig_vects = np.linalg.eigh(cor_matrix)
    eig_vals[eig_vals < 0] = 0
    # correlate data with loadings
    cor_data = data @ (np.sqrt(eig_vals) * eig_vects).T
    return cor_data


def cholesky_correlate_data(data: np.array, correlations: [], cov_reg=0.0001):
    # note the bigger the cov_reg the more likely the correlation will succeed as specified but the less
    # like specified the resulting data is distributed
    # generate correlation matrix
    cor_matrix = generate_correlation_matrix(correlations)
    # calculate covariance matrix of correlation matrix
    std = np.std(data, axis=0)
    cov_matrix = corr2cov(cor_matrix, std)
    is_pos_dev = is_pos_def(cov_matrix)

    # regulate cov matrix by adding small number to diagonal
    if not is_pos_dev:
        # regulate cov matrix
        reg_term = np.diag([cov_reg, cov_reg, cov_reg])  # regularisation term
        cov_matrix = cov_matrix + reg_term

    # only regulate once this might still not be positive definite at which point an exception is thrown by cholesky
    # returns Cholesky decomposition, L * L.H, of the square matrix a, where L is lower-triangular
    # dimension is n_variate x n_variate
    L = cholesky(cov_matrix)

    # matrix product - note order! it has to be L @ data not the other way round
    cor_data = L @ data.T
    return cor_data.T


def calculate_spearman_correlation(data: np.array, round_to: int = 2):
    """ Calculate the correlation for the data. Assumes that columns are variates and rows are observations.
    :return: np.array of correlations ordered by np.triu_indices of the upper half of the correlation matrix

    """
    return_result = []
    n = data.shape[1]
    result = spearmanr(data)
    if n == 2:
        return_result = [result.correlation]
    else:
        return_result = result.correlation[np.triu_indices(n, 1)]
    return [round(x, round_to) for x in return_result]


def check_correlations_are_within_original_strength(original_cor, actual_cor, strong_cor: float = 0.7,
                                                    not_cor: float = 0.2):
    """Checks if the correlation of actual_cor stays within the original "strengths" of original_cor
    Returns false for weak correlations not_cor < achieved cor < strong_cor!
    """
    # default all to failed
    result = len(actual_cor) * [False]

    for idx, cor in enumerate(actual_cor):
        if abs(original_cor[idx]) >= strong_cor:
            result[idx] = abs(cor) >= strong_cor
        if abs(original_cor[idx]) <= not_cor:
            result[idx] = abs(cor) <= not_cor

    return result


def generate_correlation_matrix(correlations: np.array):
    """
    Generate a correlation matrix given the tuple for the upper half of the matrix
    :param correlations: np.array of correlations ordered by the upper half indices: np.triu_indices(n, 1)
    :return: np.array for correlation matrix of shape n x n
    """
    # create identity
    elements = len(correlations)
    # n(n-1) = number of elements in triangular matrix of size nxn, solved for n
    n = int(np.sqrt(2 * elements + 1 / 4) + 1 / 2)
    m = np.identity(n)
    # indices for lower and upper, offset of one leaves diagonal alone
    i_upper = np.triu_indices(n, 1)
    i_lower = np.tril_indices(n, -1)
    # set upper and lower with values
    m[i_upper] = correlations
    m[i_lower] = m.T[i_lower]  # important to transpose as order will be wrong otherwise
    return m


def calculate_personr(data):
    n = data.shape[1]
    indices = np.triu_indices(n, 1)
    correlations = []
    for i in range(n):
        x = data[:, indices[0][i]]
        y = data[:, indices[1][i]]
        result = pearsonr(x, y)
        correlations.append(result.correlation)
    return correlations


def move_to_distributions(normal_data, distributions, args, kwargs):
    """ Moves normal distributed data to given distribution keeping correlation
        :param normal_data: 2d nd array of normally distributed data with particular correlations must have mean=0, std=1
        :param distributions: list of names of the scipy.stats distribution function to move to in order of the variates in the data
        :param args: list of args tuples for each distribution for ppf function
        :param kwargs: list of kwargs dictionaries for each distribution for ppf function

        Example: move_to_distributions(normaldata, [genextreme], [(c,)], [{'loc':loc, 'scale':scale}])
    """
    # move to uniform
    uniform_data = norm.cdf(normal_data)
    # move form uniform to given distribution via inverse cdf (=ppf) function
    if len(distributions) == 1:
        result = distributions[0].ppf(uniform_data, *args[0], **kwargs[0])
    else:
        resulting_1d_arrays = []
        for variate_idx in range(normal_data.shape[1]):
            shifted_dist = distributions[variate_idx].ppf(uniform_data[:, variate_idx], *args[variate_idx],
                                                          **kwargs[variate_idx])
            resulting_1d_arrays.append(shifted_dist)
        result = np.column_stack(tuple(resulting_1d_arrays))
    return result
