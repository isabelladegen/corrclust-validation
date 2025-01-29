from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from statsmodels.stats.power import TTestPower

from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, display_legend, fontsize

ci_lower = "lower"
ci_higher = "higher"
ci_overlap = "overlap"
ci_width_iob = "ci96_width | iob mean"
ci_width_cob = "ci96_width | cob mean"
ci_width_ig = "ci96_width | ig mean"


@dataclass
class ConfidenceIntervalCols:
    ci_96hi = "ci96_hi"
    ci_96lo = "ci96_lo"
    width = "ci width"
    standard_error = "ci standard error"


@dataclass
class StatsCols:
    variant: str = 'data variant'
    is_significant: str = 'significant'
    p_value: str = 'p-value'
    effect_size: str = 'effect size'
    alpha: str = 'adjusted alpha'
    achieved_power: str = 'achieved_power'
    n_for_power_80: str = 'required n for 80%'
    none_zero_pairs: str = 'none zero pairs'
    statistic: str = 'statistic'
    n_pairs: str = 'n pairs'


class WilcoxResult:
    def __init__(self, statistic: float, p_value: float, n_pairs: int, none_zero: int, round_to: int = 3):
        self.statistic = statistic
        self.p_value = p_value
        self.n_pairs = n_pairs
        self.non_zero = none_zero
        self.__round_to = round_to

    def is_significant(self, alpha: float = 0.05, bonferroni_adjust: int = 1) -> bool:
        """
        Returns if p_value is significant
        :param bonferroni_adjust: divides alpha by bonferroni adjustment for significance testing
        """
        adjusted_alpha = self.adjusted_alpha(alpha=alpha, bonferroni_adjust=bonferroni_adjust)
        is_significant = self.p_value < adjusted_alpha
        return is_significant

    def effect_size(self) -> float:
        """
        Returns effect size r = z_scores/sqrt(N_non_zero_pairs). Assumes two sided test
        """
        z_score = norm.ppf(self.p_value / 2)  # divide by 2 for two-tailed test
        r = abs(z_score) / np.sqrt(self.non_zero)
        return round(r, self.__round_to)

    @staticmethod
    def adjusted_alpha(alpha: float = 0.05, bonferroni_adjust: int = 1) -> float:
        return round(alpha / bonferroni_adjust, 4)

    def achieved_power(self, alpha: float = 0.05, bonferroni_adjust: int = 1) -> float:
        """
        Returns the power achieved as float, so 0.7 = 70%
        :param alpha: optional, alpha that should be used
        :param bonferroni_adjust: optional, int to divide alpha by for multiple tests
        :return: achieved power
        """
        adjusted_alpha = self.adjusted_alpha(alpha=alpha, bonferroni_adjust=bonferroni_adjust)
        power_analysis = TTestPower()
        achieved_power = power_analysis.power(
            effect_size=self.effect_size(),
            nobs=self.non_zero,
            alpha=adjusted_alpha,
            alternative='two-sided'
        )
        return round(achieved_power, self.__round_to)

    def sample_size_for_power(self, target_power: float = 0.8, alpha: float = 0.05, bonferroni_adjust: int = 1) -> int:
        adjusted_alpha = self.adjusted_alpha(alpha=alpha, bonferroni_adjust=bonferroni_adjust)
        power_analysis = TTestPower()
        required_n = power_analysis.solve_power(
            effect_size=self.effect_size(),
            power=target_power,
            alpha=adjusted_alpha,
            alternative='two-sided',
            nobs=None
        )
        return np.ceil(required_n)

    def as_series(self, variant_name: str, target_power: float = 0.8, alpha: float = 0.05, bonferroni_adjust: int = 1):
        return pd.Series({
            StatsCols.variant: variant_name,
            StatsCols.is_significant: self.is_significant(alpha, bonferroni_adjust),
            StatsCols.p_value: round(self.p_value, 4),
            StatsCols.effect_size: self.effect_size(),
            StatsCols.alpha: self.adjusted_alpha(alpha=alpha, bonferroni_adjust=bonferroni_adjust),
            StatsCols.achieved_power: self.achieved_power(alpha, bonferroni_adjust),
            StatsCols.n_for_power_80: self.sample_size_for_power(target_power, alpha, bonferroni_adjust),
            StatsCols.none_zero_pairs: self.non_zero,
            StatsCols.statistic: self.statistic,
            StatsCols.n_pairs: self.n_pairs,
        })


def calculate_hi_ci(mean_series, std_df, count_df):
    column = mean_series.name
    return mean_series + 1.96 * std_df[column] / np.sqrt(count_df[column])


def calculate_lo_ci(mean_series, std_df, count_df):
    column = mean_series.name
    return mean_series - 1.96 * std_df[column] / np.sqrt(count_df[column])


def calculate_hi_lo_difference_ci(n1, n2, s1, s2, m1, m2, z=1.96):
    """Returns ci for e.g difference of mean values (m1, m2) but could be difference of other values
    :param n1: count for first group
    :param n2: count for second group
    :param s1: std for first group
    :param s2: std for second group
    :param m1: mean for first group
    :param m2: mean for second group
    :param z: critical value defaults to 1.96 for 95% confidence intervals
    :returns pandas series of low and high confidence intervals calculation below and standard_error:
    (m1-m2) +/- z sqrt(s1^2/n1 +s2^2/n2)
    If the CI values are positive the mean in the first population is larger than the second, otherwise
    the mean in the first population is smaller than the second.
    """
    diff_mean = m1 - m2
    standard_error = standard_error_for_big_samples(n1, n2, s1, s2)
    lo_ci = diff_mean - z * standard_error
    hi_ci = diff_mean + z * standard_error
    return lo_ci, hi_ci, standard_error


def standard_error_for_big_samples(n1, n2, s1, s2):
    """Big samples n > 30"""
    return np.sqrt((s1 ** 2 / n1) + (s2 ** 2 / n2))


def standardized_effect_size_of_mean_difference(n1, n2, s1, s2, m1, m2):
    """Cohens'd effect size for mean differences when number of samples >30 in each group"""
    diff_mean = m1 - m2
    return diff_mean / standard_error_for_big_samples(n1, n2, s1, s2)


def n_for_power(effect_size, z_alpha=1.96, z_beta=0.84):
    """"Default to 95% CI and Power to 80%"""
    return 2 * ((z_alpha + z_beta) / effect_size) ** 2


def gaussian_critical_z_value_for(a, two_tailed=True):
    """
    Calculates the gaussian critical z value for a (e.g 0.05 for alpha or 0.2 for beta).
    If two_tailed is True than we use 1-a/2 to find the probability otherwise we use 1-a to find the
    percentage
    """
    denom = 2 if two_tailed else 1
    probability = 1 - (a / denom)
    z_value = norm.ppf(probability)
    return z_value


def gaussian_probability_from_z_value(z_value):
    """
    Calculate probability from a z_value
    """
    return norm.cdf(z_value)


def ci_power(d, n, alpha=0.05, two_tailed=True):
    """Calculates the CI power based on the effect size and number of samples provided
    :param d: effect size
    :param n: number of samples
    :param alpha: probability of Type 1 error
    :param two_tailed: decides whether to use z_alpha/2 (two tailed) or z_alpha (one tailed)
    """
    z_alpha_2 = gaussian_critical_z_value_for(alpha, two_tailed=two_tailed)
    z_beta = d * np.sqrt(n / 2) - z_alpha_2
    probability_power = gaussian_probability_from_z_value(z_beta)
    return probability_power


# P value from CI
def p_value_from_ci_se(se, difference_in_mean):
    """Source: https://www.bmj.com/content/343/bmj.d2304"""
    z = difference_in_mean / se
    x = (z * -0.717) - (0.416 * (z ** 2))
    p = np.exp(x)
    return p


def plot_power_for_effect_size_and_samples_for_ci(effect_sizes, samples, alpha=0.05, two_tailed=True,
                                                  power_line=0.8,
                                                  backend=Backends.none.value):
    """
    Plots power for effect sizes and samples formula used is n=2*(z_alpha/2 - z_beta/d)^2
    :param effect_sizes: the effect sizes d to calculate power for
    :param samples: n the number of samples to calculate power for
    :param alpha: alpha value to use
    :param two_tailed: weather to use z_alpha/2 (two tailed) or z_alpha
    :param power_line: plot horizontal red line for a specific power
    :param backend: Backends (if figure will be shown or not)
    :return: powers dictionary (effect_size, sample)=power, fig
    """
    powers = {}
    for effect_size in effect_sizes:
        for sample in samples:
            powers[(effect_size, sample)] = ci_power(d=effect_size, n=sample, alpha=alpha,
                                                     two_tailed=two_tailed)

    reset_matplotlib(backend=backend)
    fig_size = (15, 12)
    fig, axs = plt.subplots(nrows=2,
                            ncols=1,
                            sharey=False,
                            sharex=False,
                            figsize=fig_size, squeeze=0)
    colormap = plt.cm.Dark2  # pylint: disable-msg=E1101

    # y=power, x=number of observations, labels = effect-size
    ax1 = axs[0, 0]
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(effect_sizes))]
    for i, es in enumerate(effect_sizes):
        keys = [(es, sample) for sample in samples]
        power = [powers[k] for k in keys]
        ax1.plot(samples, power, marker='.', lw=2, alpha=1, color=colors[i], label='es=%4.2F' % es)
    ax1.axhline(y=power_line, color='r', linestyle='dashed')
    ax1.set_xlabel('Number of Observations')
    ax1.set_title('Power of Test', fontsize=fontsize)
    display_legend(ax1, fig)

    ax2 = axs[1, 0]
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(samples))]
    for i, n in enumerate(samples):
        keys = [(es, n) for es in effect_sizes]
        power = [powers[k] for k in keys]
        ax2.plot(effect_sizes, power, marker='.', lw=2, alpha=1, color=colors[i], label='N=%4.2F' % n)
    ax2.axhline(y=power_line, color='r', linestyle='dashed')
    ax2.set_xlabel('Effect Size')
    display_legend(ax2, fig)
    return powers, fig


def number_of_unique_two_combinations(n):
    return (n * (n - 1)) / 2


def number_of_samples_for_correlations(alpha: float, beta: float, rho1: float, rho0: float, b: float = 3,
                                       c2: float = 1,
                                       two_tailed: bool = True):
    """Calculates the number of samples required for the given parameters
    Source: Sample Size Charts for Spearman and Kendall Coefficients, Justine O et all

    :param alpha: type 1 error control (significance level)
    :param beta: type 2 error control (power), for 80% enter 0.8!
    :param rho1: correlation coefficient for alternative hypothesis, -1<=rho<=1
    :param rho0: correlation coefficient for null hypothesis, -1<=rho<=1
    :param b: 3 for Pearson and Spearman and 4 for Kendall Coefficient
    :param c2:  (already squared value), 1 for Pearsons, planning value or 1.06 for Spearman, 0.437 for Kendall Coefficient
    """
    z_rho1 = np.arctanh(rho1)  # fisher z transform
    z_rho0 = np.arctanh(rho0)  # fisher z transform
    z_alpha = gaussian_critical_z_value_for(alpha, two_tailed)
    # we calculate 1-beta and the one tailed critical value (as common), for beta =0.8 this results in 0.84
    z_1mbeta = gaussian_critical_z_value_for(1 - beta, two_tailed=False)
    n = b + c2 * (((z_alpha + z_1mbeta) / (z_rho1 - z_rho0)) ** 2)
    return n


def alpha_for_correlations(n: int, beta: float, rho1: float, rho0: float, b: float = 3, c2: float = 1):
    """Calculates alpha achieved for a given correlation
    Source: Sample Size Charts for Spearman and Kendall Coefficients, Justine O et all

    :param n: number of samples
    :param beta: type 2 error control (power), for 80% enter 0.8!
    :param rho1: correlation coefficient achieved, -1<=rho<=1
    :param rho0: correlation coefficient for null hypothesis, -1<=rho<=1
    :param b: 3 for Pearson and Spearman and 4 for Kendall Coefficient
    :param c2:  (already squared value), 1 for Pearson's, planning value or 1.06 for Spearman, 0.437 for Kendall Coefficient
    """
    z_rho1 = np.arctanh(rho1)  # fisher z transform
    z_rho0 = np.arctanh(rho0)  # fisher z transform
    # we calculate 1-beta and the one tailed critical value (as common), for beta =0.8 this results in 0.84
    z_1mbeta = gaussian_critical_z_value_for(1 - beta, two_tailed=False)
    z_alpha = (np.sqrt((n - b) / c2) * (z_rho1 - z_rho0)) - z_1mbeta
    # calculate alpha from critical value
    probability = gaussian_probability_from_z_value(z_alpha)
    alpha = 1 - probability
    return alpha


def compare_ci_for_differences(lo_ci_value, hi_ci_value):
    if lo_ci_value <= 0 <= hi_ci_value:
        return ci_overlap
    if hi_ci_value < 0:
        return ci_lower
    if lo_ci_value > 0:
        return ci_higher


def corr_significant_results(correlation_result, no=0.1, slight=0.3, strong=0.7):
    """
    :returns all significant correlation, and only kendall tau split into significant no correlation, slight, medium, strong
    """
    all_sig = correlation_result[(correlation_result['Spearmanr sig']) | (
        correlation_result['Kendalltau sig']) | (correlation_result['Personsr sig'])]
    k_sig = all_sig[all_sig['Kendalltau sig']]
    no_sig = k_sig[k_sig['Kendalltau'].abs().lt(no)]
    slight_sig = k_sig[(k_sig['Kendalltau'].abs().ge(no)) & (k_sig['Kendalltau'].abs().lt(slight))]
    medium_sig = k_sig[(k_sig['Kendalltau'].abs().ge(slight)) & (k_sig['Kendalltau'].abs().lt(strong))]
    strong_sig = k_sig[k_sig['Kendalltau'].abs().ge(strong)]
    return all_sig, k_sig, no_sig, slight_sig, medium_sig, strong_sig


def find_multiplicity_adjusted_z_alpha(alpha, n_tests):
    adjusted_alpha = alpha / n_tests
    adjusted_z_alpha = norm.ppf(1 - adjusted_alpha / 2)
    return adjusted_z_alpha, adjusted_alpha


def filter_mean_diffs(mean_diff_df, variate, ci_relationship, old_mean_ci_dict):
    """
    Finds the mean diff CI for ci_relationship and variate
    :param mean_diff_df:
    :param variate:
    :param ci_relationship:
    :param old_mean_ci_dict:
    :return: filtered_df
    """
    filtered_df = mean_diff_df[variate][ci_relationship]
    old_findings = old_mean_ci_dict[ci_relationship][variate]
    print(set(old_findings).issubset(filtered_df["id"]))
    print(len(old_findings))
    return filtered_df


def cohens_d(m1, m2, s1, s2):
    """Cohen's d for assumption n1=n2, simple pooled standard deviation"""
    return (m1 - m2) / np.sqrt((s1 ** 2 + s2 ** 2) / 2)
