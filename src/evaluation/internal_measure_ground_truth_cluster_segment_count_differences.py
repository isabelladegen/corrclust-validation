import itertools
import os

import pandas as pd

from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.evaluation.internal_measure_ground_truth_assessment import GroupAssessmentCols, \
    internal_measure_lower_values_best
from src.experiments.run_calculate_internal_measures_for_ground_truth import \
    read_ground_truth_clustering_quality_measures
from src.utils.configurations import get_irregular_folder_name_from
from src.utils.stats import calculate_wilcox_signed_rank


class InternalMeasureGroundTruthClusterSegmentCount:
    """Calculates per data variant and per internal measure which count (segment or cluster works best)"""

    def __init__(self, overall_ds_name: str, internal_measures: [str], distance_measure: str, data_type: str,
                 data_dirs: [str], reduced_root_result: str, original_root_result: str, round_to: int = 3):
        self.overall_ds_name = overall_ds_name
        self.distance_measure = distance_measure
        self.internal_measures = internal_measures
        self.data_dirs = data_dirs
        self.data_type = data_type
        self.round_to = round_to
        self.reduced_root_result = reduced_root_result
        self.original_root_result = original_root_result
        # key=distance measure, value df of ground truth calculation
        self.ground_truth_calculation_dfs = {}

        for data_dir in data_dirs:
            # read all internal measures for given cluster or segment count
            # get count key
            last_folder_name = os.path.basename(os.path.normpath(data_dir))
            if 'irregular' in last_folder_name:
                data_dir_without_completeness = data_dir.replace(last_folder_name, '')
            else:
                data_dir_without_completeness = data_dir

            count_name = os.path.basename(os.path.normpath(data_dir_without_completeness))
            if "reduced" in data_dir:
                result_dir = str(os.path.join(self.reduced_root_result, count_name))
            else:
                result_dir = self.original_root_result
            per_count_internal_measure_df = read_ground_truth_clustering_quality_measures(
                overall_ds_name=self.overall_ds_name,
                data_type=self.data_type,
                root_results_dir=result_dir,
                data_dir=data_dir, distance_measure=distance_measure)
            self.ground_truth_calculation_dfs[count_name] = per_count_internal_measure_df
        self.count_names = list(self.ground_truth_calculation_dfs.keys())

    def raw_scores_for_each_internal_measure(self):
        """
        Reshapes data into dictionary of raw score per internal measure
        :return dictionary{key=internal measure name: values= df with rows=run-names, columns=counts, values=
        internal measure score for that distance measure and run}
        """
        result_dict = {}
        run_names = self.ground_truth_calculation_dfs[self.count_names[0]][DescribeBadPartCols.name].to_list()

        for internal_index in self.internal_measures:
            # Create empty dataframe with runs as index
            measure_df = pd.DataFrame(index=run_names)
            # Fill in values for each distance measure
            for count_name, df in self.ground_truth_calculation_dfs.items():
                if internal_index in df.columns:
                    # Set the run name as index for easier merging
                    temp_df = df.set_index(DescribeBadPartCols.name)
                    measure_df[count_name] = temp_df[internal_index]

            result_dict[internal_index] = measure_df

        return result_dict

    def stats_for_raw_values_across_all_runs(self):
        stats_results = {}
        raw_values = self.raw_scores_for_each_internal_measure()
        for measure in self.internal_measures:
            stats_results[measure] = raw_values[measure].describe().round(self.round_to)
        return stats_results

    def wilcoxons_signed_rank_between_all_counts(self, stats_value: str = '50%', alpha: float = 0.05,
                                                 bonferroni_adjust: int = 1, alternative: str = 'two-sided',
                                                 non_zero: float = 0.00000001):
        """
        Calculates for each internal measure the wilcoxon's signed rank test of all count combinations
        """
        wilxoxons_signed_ranks = {}
        count_name_combinations = list(itertools.combinations(self.count_names, 2))

        raw_values = self.raw_scores_for_each_internal_measure()
        for internal_index in self.internal_measures:
            # results to build dataframe
            statistics = []
            p_values = []
            effect_sizes = []
            achieved_powers = []
            n_pairs = []
            is_significances = []
            results_for_count_compares = []
            compares = []
            alphas_used = []

            # df of columns are distance measures, values are the scores for the runs
            values = raw_values[internal_index]

            # compare significance between count_names
            for v1, v2 in count_name_combinations:
                compares.append((v1, v2))
                results_for_count_compares.append((v1, v2))
                # swap values 1 and values 2 for DBI where lower values are best, so we can interpret the statistical
                # outcomes sign consistently across internal measures
                values1 = values[v2] if internal_measure_lower_values_best[internal_index] else values[v1]
                values2 = values[v1] if internal_measure_lower_values_best[internal_index] else values[v2]
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

            # build dataframe
            results_dict = {
                GroupAssessmentCols.compared_counts: results_for_count_compares,
                GroupAssessmentCols.p_value: p_values,
                GroupAssessmentCols.effect_size: effect_sizes,
                GroupAssessmentCols.achieved_power: achieved_powers,
                GroupAssessmentCols.non_zero_pairs: n_pairs,
                GroupAssessmentCols.alpha: alphas_used,
                GroupAssessmentCols.is_significat: is_significances,
                GroupAssessmentCols.statistic: statistics,
            }
            df = pd.DataFrame(results_dict)
            wilxoxons_signed_ranks[internal_index] = df

        return wilxoxons_signed_ranks
