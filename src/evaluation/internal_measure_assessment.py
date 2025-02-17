import itertools
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel

from src.evaluation.describe_bad_partitions import default_internal_measures, default_external_measures, \
    DescribeBadPartCols
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import internal_measure_evaluation_dir_for
from src.utils.stats import standardized_effect_size_of_mean_difference, calculate_hi_lo_difference_ci, \
    ConfidenceIntervalCols, StatsCols, calculate_power, cohens_d, cohens_d_paired


@dataclass
class IAResultsCSV:
    correlation_summary: str = "correlation_summary.csv"
    effect_size_difference_worst_best: str = "effect_size_difference_of_worst_to_best_partition.csv"
    descriptive_statistics_measure_summary: str = "descriptive_statistics_internal_measures_correlation.csv"
    ci_of_differences_between_measures: str = "ci_differences_between_internal_measure_correlation.csv"
    paired_t_test: str = "paired_t_test_between_internal_measure_correlation.csv"
    mean_correlation_data_variant: str = "mean_correlation_data_variants.csv"
    paired_t_test_data_variant: str = "paired_t_test_data_variants.csv"
    gt_worst_measure_data_variants: str = "gt_worst_measure_data_variants.csv"


def get_name_paired_t_test_between_distance_measures(internal_measure: str) -> str:
    return internal_measure + "_paired_t_test_distance_measures.csv"


def get_full_filename_for_results_csv(full_results_dir: str, csv_filename: str):
    """ Returns the full filename to read or save a csv result
    :param full_results_dir: the full path where to save the file or read the file from
    :param csv_filename: the full filename to save or read, see InternalMeasureAssessmentCSV for options
    :returns the full path to the csv file
    """
    return os.path.join(full_results_dir, csv_filename)


@dataclass
class InternalMeasureCols:
    name: str = "name"
    partitions: str = "N_pi+1"
    persons_r: str = 'r'
    p_value: str = 'P'
    effect_size: str = 'd'


def read_internal_assessment_result_for(result_type, overall_dataset_name: str, results_dir: str, data_type: str,
                                        data_dir: str, distance_measure: str):
    store_results_in = internal_measure_evaluation_dir_for(overall_dataset_name=overall_dataset_name,
                                                           results_dir=results_dir, data_type=data_type,
                                                           data_dir=data_dir,
                                                           distance_measure=distance_measure)
    file_name = get_full_filename_for_results_csv(store_results_in, result_type)
    df = pd.read_csv(file_name, index_col=0)
    return df


def column_name_correlation_coefficient_for(internal_index: str,
                                            external_index: str = ClusteringQualityMeasures.jaccard_index) -> str:
    measure_pair = internal_index + ', ' + external_index
    return InternalMeasureCols.persons_r + ' ' + measure_pair


class InternalMeasureAssessment:
    def __init__(self, distance_measure: str, dataset_results: [pd.DataFrame],
                 internal_measures: [str] = default_internal_measures,
                 external_measures: [str] = default_external_measures, round_to: int = 3):
        self.distance_measure = distance_measure
        self.dataset_results = dataset_results
        self.__internal_measures = internal_measures
        self.__external_measures = external_measures
        self.measures_combinations = list(itertools.product(self.__internal_measures, self.__external_measures))
        self.measures_combinations_col_names = [pair[0] + ', ' + pair[1] for pair in self.measures_combinations]
        self.measures_corr_col_names = [InternalMeasureCols.persons_r + ' ' + item for item in
                                        self.measures_combinations_col_names]
        self.measures_p_col_names = [InternalMeasureCols.p_value + ' ' + item for item in
                                     self.measures_combinations_col_names]

        self.__comparing_internal_measures = list(itertools.combinations(self.measures_corr_col_names, 2))
        self.compare_internal_measures_cols = [item[0] + ' vs ' + item[1] for item in
                                               self.__comparing_internal_measures]
        self.__round_to = round_to

        # calculate correlations between all combinations of internal and external measures
        names = []
        partitions = []
        correlations = {col_names: [] for col_names in self.measures_corr_col_names}
        p_values = {col_names: [] for col_names in self.measures_p_col_names}

        # calculate correlations for each dataset and each measure pair
        for ds in self.dataset_results:
            name = ds.iloc[0][DescribeBadPartCols.name]
            n_partition = ds.shape[0]  # partitions including ground truth

            # calculate correlations for internal index with jaccard
            for p_idx, pair in enumerate(self.measures_combinations):
                m1_values = ds[pair[0]].to_numpy()
                m2_values = ds[pair[1]].to_numpy()
                # handle constants
                if (m1_values == m1_values[0]).all() or (m2_values == m2_values[0]).all():
                    # correlation is not defined as the variance of one or more is zero
                    if (m1_values == m1_values[0]).all() and (m2_values == m2_values[0]).all():
                        # both are constants (should not happen for Jaccard as not calculated on data)
                        cor = 1  # choice to call two constants perfectly correlated
                        p = 1  # we have no evidence for this though as we couldn't examine any variation
                    else:
                        # only one is a constant
                        cor = 0  # choice to call a constants unrelated to a variable that varies
                        p = 1  # we have no evidence for this as we couldn't examine any variation
                elif np.any(~np.isfinite(m1_values)) or np.any(~np.isfinite(m2_values)):
                    print("oh common")
                    cor = np.nan
                    p = 1
                else:
                    # actually calculating the correlation
                    stat_result = pearsonr(m1_values, m2_values)
                    cor = stat_result.statistic
                    p = stat_result.pvalue
                # update result
                correlations[self.measures_corr_col_names[p_idx]].append(round(cor, round_to))
                p_values[self.measures_p_col_names[p_idx]].append(round(p, round_to))

            # update results
            names.append(name)
            partitions.append(n_partition)

        # add p_values dict to correlations
        correlations.update(p_values)
        corr_df = pd.DataFrame(correlations)
        corr_df.insert(loc=0, column=InternalMeasureCols.name, value=names, allow_duplicates=True)
        corr_df.insert(loc=1, column=InternalMeasureCols.partitions, value=partitions, allow_duplicates=True)

        self.correlation_summary = corr_df

    def effect_size_and_ci_of_difference_of_means_gt_and_worst_partition(self, internal_measure: str,
                                                                         worst_ranked_by: str, z=1.96):
        """Calculates Cohen's d effect size and CI (with z=1.96 this is the 95% CI) of the differences in mean of the
        ground truth and worst partition. The mean is calculated across the N_D datasets for the provided
        internal_measure. The worst partition is judged by the lowest number for worst_ranked_by measure provided
        returns: effect_size, lo_ci, hi_ci, standard_error
        """
        n1 = n2 = len(self.dataset_results)
        gts = []
        worsts = []

        # get gt and worst value
        for ds in self.dataset_results:
            # value for ground truth
            gts.append(ds.iloc[0][internal_measure])
            ds_sorted = ds.sort_values(by=worst_ranked_by, ascending=True)  # lowest value first
            # worst value for internal measure
            worsts.append(ds_sorted.iloc[0][internal_measure])

        gts = np.array(gts)
        worsts = np.array(worsts)

        m1 = gts.mean()
        m2 = worsts.mean()
        s1 = gts.std()
        s2 = worsts.std()
        effect_size = standardized_effect_size_of_mean_difference(n1, n2, s1, s2, m1, m2)
        lo_ci, hi_ci, standard_error = calculate_hi_lo_difference_ci(n1, n2, s1, s2, m1, m2, z)
        return effect_size, lo_ci, hi_ci, standard_error

    def descriptive_statistics_for_internal_measures_correlation(self):
        """ Calculates the descriptive stats for each internal measure and it's correlation to the external measures
        and returns a pd.Dataframe with the internal measure v external measure as column names and the descriptive
        statistics as rows.
        """
        return self.correlation_summary[self.measures_corr_col_names].describe().round(2)

    def paired_samples_t_test_on_fisher_transformed_correlation_coefficients(self, alpha=0.05,
                                                                             alternative: str = 'two-sided'):
        """
        Calculates a paired samples t-test for fisher transformed correlation coefficients.
        This test takes into consideration that the correlations are with a dependent variable (Jaccard) - hence the
        fisher transformation as well as that there are multiple subject's correlation on 67 partitions available - hence
        the paired t test.
        :param alpha: what alpha to use for the power calculation
        :param alternative: what alternative hypothesis to use for the power calculation
        :return: df where the rows are indexed by StatCols p-value, statistics, the columns are the different internal
         measures combinations
        """
        error_msg = "Calculate at least two internal indices to be able to compare them"
        assert len(self.__comparing_internal_measures) > 0, error_msg

        # calculate fisher z score of each of the correlation coefficient
        # use the absolute of the values as we don't care about the direction of the correlation just the strength
        df = self.correlation_summary[self.measures_corr_col_names]
        dbi_cols = [col for col in df.columns if ClusteringQualityMeasures.dbi in col]
        # turn copy warning off given that we use the values to create a new df
        with pd.option_context('mode.chained_assignment', None):
            # for DBI where lower values are better we need to invert the correlation coefficients for a fair comparison
            df[dbi_cols] = df[dbi_cols].multiply(-1)
        df = pd.DataFrame(np.arctanh(df.values), index=df.index, columns=df.columns)

        # measures that we need to compared
        compare = self.__comparing_internal_measures

        # results
        names = []
        p_values = []
        statistics = []
        effect_sizes = []
        powers = []

        # perform paired t-test on the transformed scores
        for idx, measure_pair in enumerate(compare):
            m1_coefficients = df[measure_pair[0]]
            m2_coefficients = df[measure_pair[1]]

            # calculate statistic
            t_stat, p_value = ttest_rel(m1_coefficients, m2_coefficients, alternative=alternative)

            # calculate effect size (Cohen's d for paired samples)
            d = cohens_d_paired(m1_coefficients, m2_coefficients)

            power = calculate_power(effect_size=d, n_samples=len(m1_coefficients), alpha=alpha, alternative=alternative)

            names.append(self.compare_internal_measures_cols[idx])
            p_values.append(p_value)
            statistics.append(t_stat)
            effect_sizes.append(d)
            powers.append(power)

        result = pd.DataFrame({
            InternalMeasureCols.name: names,
            StatsCols.p_value: p_values,
            StatsCols.statistic: statistics,
            StatsCols.effect_size: effect_sizes,
            StatsCols.achieved_power: powers,
        })
        result = result.set_index(keys=InternalMeasureCols.name).T.round(self.__round_to)
        return result

    def ci_of_differences_between_internal_measure_correlations(self, z=1.96):
        """ Calculates the CI of mean difference between each of the internal measures correlation.
        the rows are indexed by lo, hi ci and standard error, the columns are the different internal measures combinations
        """
        assert len(
            self.__comparing_internal_measures) > 0, "Calculate at least two internal indices to be able to compare them"
        df = self.descriptive_statistics_for_internal_measures_correlation()

        mean = df.loc['mean']
        count = df.loc['count']
        std = df.loc['std']

        # measures that we need to compared
        compare = self.__comparing_internal_measures

        names = []
        lo_cis = []
        hi_cis = []
        standard_errors = []
        effect_sizes = []

        for idx, measure_pair in enumerate(compare):
            m1 = abs(mean[measure_pair[0]])
            m2 = abs(mean[measure_pair[1]])
            n1 = count[measure_pair[0]]
            n2 = count[measure_pair[1]]
            s1 = std[measure_pair[0]]
            s2 = std[measure_pair[1]]

            lo_ci, hi_ci, standard_error = calculate_hi_lo_difference_ci(n1, n2, s1, s2, m1, m2, z)
            effect_size = cohens_d(m1, m2, s1, s2)

            names.append(self.compare_internal_measures_cols[idx])
            lo_cis.append(lo_ci)
            hi_cis.append(hi_ci)
            standard_errors.append(standard_error)
            effect_sizes.append(effect_size)

        result = pd.DataFrame({
            InternalMeasureCols.name: names,
            ConfidenceIntervalCols.ci_96lo: lo_cis,
            ConfidenceIntervalCols.ci_96hi: hi_cis,
            ConfidenceIntervalCols.standard_error: standard_errors,
            StatsCols.effect_size: effect_sizes,
        })
        result = result.set_index(keys=InternalMeasureCols.name).T.round(self.__round_to)
        return result

    def differences_between_worst_and_best_partition(self):
        """Calculates effect sizes and ci of the difference in correlation between the worst and best partition for
        each internal measure.
        """
        names = []
        effect_sizes = []
        lo_cis = []
        hi_cis = []
        standard_errors = []
        for internal_measure in self.__internal_measures:
            effect_size, lo_ci, hi_ci, standard_error = self.effect_size_and_ci_of_difference_of_means_gt_and_worst_partition(
                internal_measure=internal_measure,
                worst_ranked_by=ClusteringQualityMeasures.jaccard_index)
            names.append(internal_measure)
            effect_sizes.append(effect_size)
            lo_cis.append(lo_ci)
            hi_cis.append(hi_ci)
            standard_errors.append(standard_error)

        result = pd.DataFrame({InternalMeasureCols.name: names, InternalMeasureCols.effect_size: effect_sizes,
                               ConfidenceIntervalCols.ci_96lo: lo_cis, ConfidenceIntervalCols.ci_96hi: hi_cis,
                               ConfidenceIntervalCols.standard_error: standard_errors})
        return result.round(self.__round_to)
