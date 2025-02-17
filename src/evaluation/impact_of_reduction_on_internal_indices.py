import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from src.evaluation.internal_measure_assessment import read_internal_assessment_result_for, IAResultsCSV, \
    InternalMeasureCols
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import get_root_folder_for_reduced_segments, get_data_dir, \
    get_root_folder_for_reduced_cluster
from src.utils.stats import cohens_d_paired, calculate_power, StatsCols


@dataclass
class ReductionType:
    clusters: str = 'clusters'
    segments: str = 'segments'


def get_col_name_reduction_internal_corr(internal_measure: str, n_dropped: int, reduction_type: str) -> str:
    return str(n_dropped) + '_dropped_' + reduction_type + '_' + internal_measure


def get_col_reductions_compared(reduction1: int, reduction2: int, reduction_type: str) -> str:
    return "drop_" + reduction_type + '_' + str(reduction1) + '_vs_' + str(reduction2)


class ImpactReductionOnInternalIndices:
    """
    This class can be used to assess the differences between different cluster/segment reduction on the correlation
    outcome of an internal measure for a data variant
    """

    def __init__(self, overall_ds_name: str, reduced_root_result_dir: str, unreduced_root_result_dir: str,
                 data_type: str, root_reduced_data_dir: str, unreduced_data_dir: str, data_completeness: str,
                 n_dropped: [int], reduction_type: str, internal_measure: str, distance_measure: str,
                 round_to: int = 3):
        """

        :param overall_ds_name: overall ds name
        :param reduced_root_result_dir: result
        :param data_type: normal or non-normal
        :param root_reduced_data_dir: root of the reduced data (does not include n_dropped, data_type or data completeness)
        :param unreduced_data_dir: data dir for unreduced ata
        :param data_completeness: see DataCompleteness for options
        :param n_dropped: list of segments or clusters dropped
        :param reduction_type: whether we're reducing clusters or segments see ReductionType for options
        :param internal_measure: which internal measure to assess see ClusteringQualityMeasures for options
        :param distance_measure: which distance measure to use see DistanceMeasures for options
        :param round_to: optional what numbers to round to, default 3
        """
        self.overall_ds_name = overall_ds_name
        self.reduced_root_result_dir = reduced_root_result_dir
        self.unreduced_root_result_dir = unreduced_root_result_dir
        self.unreduced_data_dir = unreduced_data_dir
        self.data_type = data_type
        self.root_reduced_data_dir = root_reduced_data_dir
        self.data_completeness = data_completeness
        self.n_dropped = n_dropped
        self.reduction_type = reduction_type
        self.__internal_measure = internal_measure
        self.distance_measure = distance_measure
        self.__round_to = round_to

        reductions = self.n_dropped.copy()
        reductions.insert(0, 0)  # insert no reduction
        self.comparing_reductions = list(itertools.combinations(reductions, 2))

        self.corr_col_name = InternalMeasureCols.persons_r + ' ' + self.__internal_measure + ', ' + ClusteringQualityMeasures.jaccard_index

        # load correlation results for each reduction and keep only the correlations for the given internal index
        # rename the columns to include reduction
        # files loaded are the correlations for bad partitions 2010 per data variant per distance measures
        correlations_tmp = []
        # load unreduced result
        corr_sum_df = read_internal_assessment_result_for(result_type=IAResultsCSV.correlation_summary,
                                                          overall_dataset_name=self.overall_ds_name,
                                                          results_dir=self.unreduced_root_result_dir,
                                                          data_type=self.data_type,
                                                          data_dir=self.unreduced_data_dir,
                                                          distance_measure=self.distance_measure)
        corr_sum_df = corr_sum_df[[InternalMeasureCols.name, self.corr_col_name]].copy()
        corr_sum_df = corr_sum_df.rename(columns={
            self.corr_col_name: get_col_name_reduction_internal_corr(self.__internal_measure, 0, self.reduction_type)})
        # set name as index for concat
        corr_sum_df = corr_sum_df.set_index(InternalMeasureCols.name)
        correlations_tmp.append(corr_sum_df)

        # load reduced results
        for n in self.n_dropped:
            results_dir = ''
            root_data_dir = ''
            if self.reduction_type == ReductionType.segments:
                root_data_dir = get_root_folder_for_reduced_segments(self.root_reduced_data_dir, n)
                results_dir = get_root_folder_for_reduced_segments(self.reduced_root_result_dir, n)
            elif self.reduction_type == ReductionType.clusters:
                root_data_dir = get_root_folder_for_reduced_cluster(self.root_reduced_data_dir, n)
                results_dir = get_root_folder_for_reduced_cluster(self.reduced_root_result_dir, n)

            data_dir = get_data_dir(root_data_dir, self.data_completeness)
            corr_sum_df = read_internal_assessment_result_for(result_type=IAResultsCSV.correlation_summary,
                                                              overall_dataset_name=self.overall_ds_name,
                                                              results_dir=results_dir,
                                                              data_type=self.data_type,
                                                              data_dir=data_dir,
                                                              distance_measure=distance_measure)
            corr_sum_df = corr_sum_df[[InternalMeasureCols.name, self.corr_col_name]].copy()
            corr_sum_df = corr_sum_df.rename(columns={
                self.corr_col_name: get_col_name_reduction_internal_corr(self.__internal_measure, n,
                                                                         self.reduction_type)})
            # set name as index for concat
            corr_sum_df = corr_sum_df.set_index(InternalMeasureCols.name)
            correlations_tmp.append(corr_sum_df)
        # combine all correlations for the reductions into one df, columns now e.g. 0_dropped_clusters_SCW etc.,
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
        :return: df where the rows are indexed by StatCols p-value, statistics, the columns are the different reduction
            levels
        """
        error_msg = "Compare at least two different reductions' correlations with Jaccard"
        assert len(self.comparing_reductions) > 0, error_msg

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
        for reduction_pair in self.comparing_reductions:
            reduction1 = reduction_pair[0]
            reduction2 = reduction_pair[1]
            reduction_m1_col = get_col_name_reduction_internal_corr(self.__internal_measure, reduction1,
                                                                    self.reduction_type)
            reduction_m2_col = get_col_name_reduction_internal_corr(self.__internal_measure, reduction2,
                                                                    self.reduction_type)
            m1_coefficients = df[reduction_m1_col]
            m2_coefficients = df[reduction_m2_col]

            # calculate statistic
            t_stat, p_value = ttest_rel(m1_coefficients.to_numpy(), m2_coefficients.to_numpy(), alternative=alternative)

            # calculate effect size (Cohen's d for paired samples)
            d = cohens_d_paired(m1_coefficients, m2_coefficients)

            power = calculate_power(effect_size=d, n_samples=len(m1_coefficients), alpha=alpha, alternative=alternative)

            name = get_col_reductions_compared(reduction1, reduction2, self.reduction_type)
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
