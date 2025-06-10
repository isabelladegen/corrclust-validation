from dataclasses import dataclass
from itertools import combinations, product

import pandas as pd

from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.experiments.run_calculate_internal_measures_for_ground_truth import \
    read_ground_truth_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import get_data_completeness_from
from src.utils.stats import calculate_wilcox_signed_rank, StatsCols
from src.visualisation.run_average_rank_visualisations import data_variant_description

# value for ascending, if true lowest value will get rank 1, if falls highest value will get rank 1
internal_measure_lower_values_best = {
    ClusteringQualityMeasures.silhouette_score: False,  # higher is better
    ClusteringQualityMeasures.dbi: True,  # lower is better
    ClusteringQualityMeasures.vrc: False,  # higher is better
    ClusteringQualityMeasures.pmb: False,  # higher is better
}


def hypotheses_string(m1: str, m2: str, effect_size: float, alternative: str):
    if alternative == 'less':
        return "<".join([m1, m2])
    elif alternative == 'greater':
        return ">".join([m1, m2])
    elif alternative == 'two-sided':
        if effect_size < 0:
            return "<".join([m1, m2])
        else:
            return ">".join([m1, m2])
    return "?"


@dataclass
class GroupAssessmentCols:
    alpha: str = "alpha"
    statistic: str = "statistic"
    p_value: str = "p value"
    effect_size: str = "effect size"
    achieved_power: str = "achieved power"
    non_zero_pairs: str = "non zero pairs"
    is_significat: str = "is significant"
    group: str = "group"
    compared_counts: str = "compared_counts"
    distance_measures_in_group: str = "distance measures in group"
    compared_distance_measures: str = "compared distance measures"


class InternalMeasureGroundTruthAssessment:
    """Calculates per data variant and per internal measure which distance measure works best and worst"""

    def __init__(self, overall_ds_name: str, internal_measures: [str], distance_measures: [str], data_type: str,
                 data_dir: str, root_results_dir: str, round_to: int = 3):
        self.overall_ds_name = overall_ds_name
        self.distance_measures = distance_measures
        self.internal_measures = internal_measures
        self.data_dir = data_dir
        self.data_type = data_type
        self.root_results_dir = root_results_dir
        self.round_to = round_to
        # key=distance measure, value df of ground truth calculation
        self.ground_truth_calculation_dfs = {}

        for distance_measure in distance_measures:
            # read all internal measures for a distance measure
            per_distance_measure_df = read_ground_truth_clustering_quality_measures(
                overall_ds_name=self.overall_ds_name,
                data_type=self.data_type,
                root_results_dir=self.root_results_dir,
                data_dir=self.data_dir, distance_measure=distance_measure)
            self.ground_truth_calculation_dfs[distance_measure] = per_distance_measure_df

    def rank_distance_measures_for_each_internal_measure(self):
        """
        Ranks each distance measure for each internal measure and returns dictionary of ranks keyed
        by internal measure name
        :return dictionary{key=internal measure name: values= df with rows=run-names, columns= distance measures,
        cells=rank for that distance measure for that run
        """
        ranked_dict = {}
        raw_scores = self.raw_scores_for_each_internal_measure()

        for internal_measure, df in raw_scores.items():
            # Rank across columns (axis=1) for each row
            ranked_df = df.rank(axis=1, method='dense', ascending=internal_measure_lower_values_best[internal_measure])
            ranked_dict[internal_measure] = ranked_df
        return ranked_dict

    def raw_scores_for_each_internal_measure(self):
        """
        Reshapes data into dictionary of raw score per internal measure
        :return dictionary{key=internal measure name: values= df with rows=run-names, columns=distance measures, values=
        internal measure score for that distance measure and run}
        """
        result_dict = {}
        run_names = self.ground_truth_calculation_dfs[self.distance_measures[0]][DescribeBadPartCols.name].to_list()

        for measure in self.internal_measures:
            # Create empty dataframe with runs as index
            measure_df = pd.DataFrame(index=run_names)
            # Fill in values for each distance measure
            for distance_measure, df in self.ground_truth_calculation_dfs.items():
                if measure in df.columns:
                    # Set the run name as index for easier merging
                    temp_df = df.set_index(DescribeBadPartCols.name)
                    measure_df[distance_measure] = temp_df[measure]

            result_dict[measure] = measure_df

        return result_dict

    def stats_for_ranks_across_all_runs(self):
        stats_results = {}
        ranks = self.rank_distance_measures_for_each_internal_measure()
        for measure in self.internal_measures:
            stats_results[measure] = ranks[measure].describe().round(self.round_to)
        return stats_results

    def stats_for_raw_values_across_all_runs(self):
        stats_results = {}
        raw_values = self.raw_scores_for_each_internal_measure()
        for measure in self.internal_measures:
            stats_results[measure] = raw_values[measure].describe().round(self.round_to)
        return stats_results

    def grouping_for_each_internal_measure(self, stats_value: str):
        """
        Calculates the grouping of distance measures per internal measure. Lower groups
        are better performing distance measures, higher worse.
        :param stats_value: which rank stats value to use, e.g. 50% = median ranks across the n=30 pairs
        :return:
        """
        rank_stats = self.stats_for_ranks_across_all_runs()
        groupings = {}
        for measure in self.internal_measures:
            ranks = rank_stats[measure].loc[stats_value]
            unique_ranks = sorted(ranks.unique())

            # Create a mapping from actual rank to group numbers (1, 2, 3...)
            rank_to_group = {rank: i + 1 for i, rank in enumerate(unique_ranks)}

            # Group indices (distance measure) by group numbers
            result = {}
            for distance_measure, rank in ranks.items():
                group_num = rank_to_group[rank]
                if group_num not in result:
                    result[group_num] = []
                result[group_num].append(distance_measure)

            # Sort the dictionary by keys
            result = dict(sorted(result.items()))
            groupings[measure] = result
        return groupings

    def wilcoxons_signed_rank_step_down(self, alpha: float = 0.05,
                                        bonferroni_adjust: int = 1, alternative: str = 'two-sided',
                                        non_zero: float = 0.0001):
        """
        Calculates for each internal measure the wilcoxon's signed rank test between different distance measures in a
        step-down fashion until it finds a significant difference between two measures
        :param alpha: significance level
        :param bonferroni_adjust: divide alpha by this to adjust for multiplicity
        :param alternative: which alternative to test
        :param non_zero: what differences to consider zero
        :return: df with rows for each internal measure and columns wilcoxon results
        """
        # dm are tested in order of rank (lowest is best)
        # dictionary{key=internal measure name: values= df with rows=describe stats, columns= distance measures,
        # cells=rank stats for that distance measure across all subjects
        ranked_dm = self.stats_for_ranks_across_all_runs()

        # raw values
        raw_values = self.raw_scores_for_each_internal_measure()

        # results to build dataframe
        internal_measures = []
        is_significances = []
        p_values = []
        effect_sizes = []
        achieved_powers = []
        nz_pairs = []
        statistics = []
        alphas_used = []
        best_dms = []
        compared_dms = []
        non_sig_dms = []
        not_tested_dms = []

        # run tests for each internal index
        for internal_index in self.internal_measures:
            # df of columns are distance measures, values are the scores for the runs
            values = raw_values[internal_index]

            # order distance values
            ranks = ranked_dm[internal_index]
            ordered_dm = ranks.loc['mean'].sort_values().index.tolist()
            dm_1 = ordered_dm[0]
            measure1_raw = values[dm_1]

            best_dms.append(dm_1)
            dms_tested = []
            wilc_result = None

            # cycle through all dm until significant difference is found
            for dm_2 in ordered_dm[1:]:
                dms_tested.append(dm_2)
                measure2_raw = values[dm_2]
                wilc_result = calculate_wilcox_signed_rank(measure1_raw, measure2_raw, non_zero,
                                                           alternative=alternative)
                # stop if significant
                if wilc_result.is_significant(alpha=alpha, bonferroni_adjust=bonferroni_adjust):
                    break

            # save result for last dm tested
            compared_dms.append(dm_2)
            not_tested_dms.append(set(ordered_dm[1:]) - set(dms_tested))
            internal_measures.append(internal_index)
            p_values.append(wilc_result.p_value)
            statistics.append(wilc_result.statistic)
            effect_sizes.append(wilc_result.effect_size(alternative=alternative))
            achieved_powers.append(wilc_result.achieved_power(alpha=alpha, bonferroni_adjust=bonferroni_adjust,
                                                              alternative=alternative))
            alphas_used.append(alpha)
            nz_pairs.append(wilc_result.non_zero)
            is_significances.append(True)
            non_sig_dms.append(dms_tested)

        # create dataframe
        results_dict = {
            "Internal Measure": internal_measures,
            "Best ranked dm": best_dms,
            "compared to dm": compared_dms,
            GroupAssessmentCols.is_significat: is_significances,
            GroupAssessmentCols.p_value: p_values,
            GroupAssessmentCols.effect_size: effect_sizes,
            GroupAssessmentCols.achieved_power: achieved_powers,
            GroupAssessmentCols.non_zero_pairs: nz_pairs,
            "Non sig dm": non_sig_dms,
            "Not tested dms": not_tested_dms,
            GroupAssessmentCols.alpha: alphas_used,
            GroupAssessmentCols.statistic: statistics,
        }
        df = pd.DataFrame(results_dict)

        return df

    def wicoxons_between(self, dm1: str, dm2: str, alpha: float = 0.05,
                         target_power: float = 0.8, bonferroni_adjust: int = 1, alternative: str = 'two-sided',
                         non_zero: float = 0.0001):
        """
        Calculates  the wilcoxon's signed rank between dm1 and dm2.
        :param dm1: distance measure 1
        :param dm2: distance measure 2
        :param alpha: significance level
        :param target_power: to calculate n required to achieve target power, e.g 0.8 for 80%
        :param bonferroni_adjust: divide alpha by this to adjust for multiplicity
        :param alternative: which alternative to test
        :param non_zero: what differences to consider zero
        :return: df of wilxocon's signed rank result
        """
        # raw values
        raw_values = self.raw_scores_for_each_internal_measure()

        # results to build dataframe
        data_variants = []
        internal_measures = []
        hypotheses = []
        is_significances = []
        effect_sizes = []
        p_values = []
        nz_pairs = []
        achieved_powers = []
        statistics = []
        alphas_used = []
        n_target_powers = []
        data_variant = data_variant_description[(get_data_completeness_from(self.data_dir), self.data_type)]

        # run tests for each internal index
        for internal_index in self.internal_measures:
            # df of columns are distance measures, values are the scores for the runs
            values = raw_values[internal_index]

            wilc_result = calculate_wilcox_signed_rank(values[dm1], values[dm2], non_zero,
                                                       alternative=alternative)
            es = wilc_result.effect_size(alternative=alternative)
            data_variants.append(data_variant)
            internal_measures.append(internal_index)
            hypotheses.append(hypotheses_string(dm1, dm2, effect_size=es, alternative=alternative))
            p_values.append(wilc_result.p_value)
            nz_pairs.append(wilc_result.non_zero)
            statistics.append(wilc_result.statistic)
            effect_sizes.append(es)
            achieved_powers.append(wilc_result.achieved_power(alpha=alpha, bonferroni_adjust=bonferroni_adjust,
                                                              alternative=alternative))
            alphas_used.append(alpha)
            n_target_powers.append(wilc_result.sample_size_for_power(target_power=target_power, alternative=alternative, alpha=alpha, bonferroni_adjust=bonferroni_adjust))
            is_significances.append(wilc_result.is_significant(alpha=alpha, bonferroni_adjust=bonferroni_adjust))

        results_dict = {
            "Data Variant": data_variants,
            "Internal Measure": internal_measures,
            "H": hypotheses,
            GroupAssessmentCols.p_value: p_values,
            GroupAssessmentCols.effect_size: effect_sizes,
            GroupAssessmentCols.non_zero_pairs: nz_pairs,
            GroupAssessmentCols.achieved_power: achieved_powers,
            GroupAssessmentCols.is_significat: is_significances,
            GroupAssessmentCols.alpha: alphas_used,
            GroupAssessmentCols.statistic: statistics,
            StatsCols.n_for_power_80: n_target_powers,
        }

        # create dataframe
        df = pd.DataFrame(results_dict)

        return df

    def wilcoxons_signed_rank_until_all_significant(self, stats_value: str = '50%', alpha: float = 0.05,
                                                    bonferroni_adjust: int = 1, alternative: str = 'two-sided',
                                                    non_zero: float = 0.00000001):
        """ Calculates for each internal measure the wilcoxon's signed rank test until all measures of
        top group are significantly better than the next, or next next group

        """
        wilxoxons_signed_ranks = {}
        # get distance measure groupings
        groupings = self.grouping_for_each_internal_measure(stats_value=stats_value)
        raw_values = self.raw_scores_for_each_internal_measure()
        for internal_index in self.internal_measures:
            # results to build dataframe
            statistics = []
            p_values = []
            effect_sizes = []
            achieved_powers = []
            n_pairs = []
            is_significances = []
            results_for_group = []
            distance_measures_in_group = []
            compares = []
            alphas_used = []

            groups = groupings[internal_index]
            # df of columns are distance measures, values are the scores for the runs
            values = raw_values[internal_index]

            # test group 1
            group = 1
            top_group = groups[group]
            # compare significance within top group
            # if top group has just one distance measure than compare is empty and this will not be ran
            compare = list(combinations(top_group, 2))
            for m1, m2 in compare:
                compares.append((m1, m2))
                results_for_group.append(group)
                distance_measures_in_group.append(groups[group])
                # swap values 1 and values 2 for DBI where lower values are best, so we can interpret the statistical
                # outcomes sign consistently across internal measures
                values1 = values[m2] if internal_measure_lower_values_best[internal_index] else values[m1]
                values2 = values[m1] if internal_measure_lower_values_best[internal_index] else values[m2]
                wilcox_result = calculate_wilcox_signed_rank(values1=values1, values2=values2, non_zero=non_zero,
                                                             alternative=alternative)
                statistics.append(wilcox_result.statistic)
                p_values.append(round(wilcox_result.p_value, 5))
                n_pairs.append(wilcox_result.non_zero)
                is_significances.append(wilcox_result.is_significant(alpha=alpha, bonferroni_adjust=bonferroni_adjust))
                effect_sizes.append(wilcox_result.effect_size(alternative=alternative))
                achieved_powers.append(
                    wilcox_result.achieved_power(alpha=alpha, bonferroni_adjust=bonferroni_adjust,
                                                 alternative=alternative))
                alphas_used.append(wilcox_result.adjusted_alpha(alpha=alpha, bonferroni_adjust=bonferroni_adjust))

            # test top group with next group(s), stop if significant
            max_group = max(groups.keys())
            for group in range(2, max_group + 1):
                next_group = groups[group]
                measure_comb = list(product(top_group, next_group))
                this_group_significances = []
                for m1, m2 in measure_comb:
                    compares.append((m1, m2))
                    results_for_group.append((1, group))
                    distance_measures_in_group.append(groups[group])
                    # swap values 1 and values 2 for DBI where lower values are best, so we can interpret the
                    # statistical outcomes sign consistently across internal measures
                    values1 = values[m2] if internal_measure_lower_values_best[internal_index] else values[m1]
                    values2 = values[m1] if internal_measure_lower_values_best[internal_index] else values[m2]
                    wilcox_result = calculate_wilcox_signed_rank(values1=values1, values2=values2,
                                                                 non_zero=non_zero,
                                                                 alternative=alternative)
                    statistics.append(wilcox_result.statistic)
                    p_values.append(round(wilcox_result.p_value, 5))
                    n_pairs.append(wilcox_result.non_zero)
                    this_group_significances.append(
                        wilcox_result.is_significant(alpha=alpha, bonferroni_adjust=bonferroni_adjust))
                    is_significances.append(
                        wilcox_result.is_significant(alpha=alpha, bonferroni_adjust=bonferroni_adjust))
                    effect_sizes.append(wilcox_result.effect_size(alternative=alternative))
                    achieved_powers.append(
                        wilcox_result.achieved_power(alpha=alpha, bonferroni_adjust=bonferroni_adjust,
                                                     alternative=alternative))
                    alphas_used.append(wilcox_result.adjusted_alpha(alpha=alpha, bonferroni_adjust=bonferroni_adjust))
                if all(this_group_significances):
                    # we found the top group, all other distance measures score significantly worse
                    break

            # build dataframe
            results_dict = {
                GroupAssessmentCols.group: results_for_group,
                GroupAssessmentCols.compared_distance_measures: compares,
                GroupAssessmentCols.p_value: p_values,
                GroupAssessmentCols.effect_size: effect_sizes,
                GroupAssessmentCols.achieved_power: achieved_powers,
                GroupAssessmentCols.non_zero_pairs: n_pairs,
                GroupAssessmentCols.alpha: alphas_used,
                GroupAssessmentCols.is_significat: is_significances,
                GroupAssessmentCols.statistic: statistics,
                GroupAssessmentCols.distance_measures_in_group: distance_measures_in_group,
            }
            df = pd.DataFrame(results_dict)
            wilxoxons_signed_ranks[internal_index] = df

        return wilxoxons_signed_ranks
