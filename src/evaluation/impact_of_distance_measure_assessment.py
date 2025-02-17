import itertools

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from src.evaluation.internal_measure_assessment import read_internal_assessment_result_for, IAResultsCSV, \
    InternalMeasureCols
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.distance_measures import short_distance_measure_names
from src.utils.stats import StatsCols, calculate_power, cohens_d_paired


def get_col_name_distance_internal_corr(internal_measure: str, distance_measure: str) -> str:
    return short_distance_measure_names[distance_measure] + '_' + internal_measure


def get_col_distances_compared(distance_measure1: str, distance_measure2: str) -> str:
    return short_distance_measure_names[distance_measure1] + '_' + short_distance_measure_names[distance_measure2]


class ImpactDistanceMeasureAssessment:
    """
    This class can be used to assess the differences between different distance measure on the outcome
    of an internal measure for a data variant
    """

    def __init__(self, overall_ds_name: str, root_result_dir: str, data_type: str, data_dir: str, internal_measure: str,
                 distance_measures: [str], round_to: int = 3):
        self.distance_measures = distance_measures
        self.overall_ds_name = overall_ds_name
        self.root_result_dir = root_result_dir
        self.data_type = data_type
        self.data_dir = data_dir

        self.__internal_measure = internal_measure
        self.__round_to = round_to
        self.__comparing_distances = list(itertools.combinations(self.distance_measures, 2))
        self.corr_col_name = InternalMeasureCols.persons_r + ' ' + self.__internal_measure + ', ' + ClusteringQualityMeasures.jaccard_index

        # load correlation results for each distance measure and keep only the correlations for the internal index
        # rename the columns to include distance measures
        # files loaded are the correlations for bad partitions 2010 per data variant per distance measures
        correlations_tmp = []
        for distance_measure in self.distance_measures:
            corr_sum_df = read_internal_assessment_result_for(result_type=IAResultsCSV.correlation_summary,
                                                              overall_dataset_name=self.overall_ds_name,
                                                              results_dir=self.root_result_dir,
                                                              data_type=self.data_type,
                                                              data_dir=self.data_dir, distance_measure=distance_measure)
            corr_sum_df = corr_sum_df[[InternalMeasureCols.name, self.corr_col_name]].copy()
            corr_sum_df = corr_sum_df.rename(columns={
                self.corr_col_name: get_col_name_distance_internal_corr(self.__internal_measure, distance_measure)})
            # set name as index for concat
            corr_sum_df = corr_sum_df.set_index(InternalMeasureCols.name)
            correlations_tmp.append(corr_sum_df)
        # combine all correlations for the distance measures into one df, columns now e.g. L1_SCW etc.,
        # name column is the run name (subjects)
        self.correlations_df = pd.concat(correlations_tmp, axis=1).reset_index()

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
        error_msg = "Calculate at least two distance measures' correlation to be able to compare them"
        assert len(self.__comparing_distances) > 0, error_msg

        # calculate fisher z score of each of the correlation coefficients for each distance measure
        corr_cols = [col for col in self.correlations_df.columns if col != InternalMeasureCols.name]
        # use the absolute of the values as we don't care about the direction of the correlation just the strength
        df = self.correlations_df[corr_cols]
        values = df.values
        # for DBI where lower values are better we need to invert the correlation coefficients for a fair comparison
        if self.__internal_measure == ClusteringQualityMeasures.dbi:
            values = -1 * values
        df = pd.DataFrame(np.arctanh(values), index=df.index, columns=df.columns)

        # results
        names = []  # distance measures compared
        p_values = []
        statistics = []
        effect_sizes = []
        powers = []

        # perform paired t-test on the transformed scores
        for measure_pair in self.__comparing_distances:
            measure1 = measure_pair[0]
            measure2 = measure_pair[1]
            distance_m1_col = get_col_name_distance_internal_corr(self.__internal_measure, measure1)
            distance_m2_col = get_col_name_distance_internal_corr(self.__internal_measure, measure2)
            m1_coefficients = df[distance_m1_col]
            m2_coefficients = df[distance_m2_col]

            # calculate statistic
            t_stat, p_value = ttest_rel(m1_coefficients.to_numpy(), m2_coefficients.to_numpy(), alternative=alternative)

            # calculate effect size (Cohen's d for paired samples)
            d = cohens_d_paired(m1_coefficients, m2_coefficients)

            power = calculate_power(effect_size=d, n_samples=len(m1_coefficients), alpha=alpha, alternative=alternative)

            name = get_col_distances_compared(measure1, measure2)
            names.append(name)
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
