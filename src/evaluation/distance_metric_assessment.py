import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.utils.configurations import Aggregators

from src.utils.distance_measures import DistanceMeasures, distance_calculation_method_for
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, fontsize, \
    use_latex_labels
from src.utils.stats import ConfidenceIntervalCols, calculate_hi_lo_difference_ci, gaussian_critical_z_value_for, \
    cohens_d, ci_overlap, compare_ci_for_differences
from src.evaluation.describe_synthetic_dataset import DescribeSyntheticDataset, plot_corr_ellipses
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols

default_order = [0, 1, 2, 3, 4, 5]


@dataclass
class DistanceMeasureCols:
    criterion: str = "Criterion"
    pair1: str = "Pattern pair 1"
    pair2: str = "Pattern pair 2"
    alpha: str = "alpha"
    level_set: str = "level set"
    pairs: str = "segment pairs"
    pattern_pairs: str = "pattern pairs"
    type: str = "distance measure"
    compared: str = "compared level sets"
    effect_size: str = "Cohen's d"
    stat_diff: str = "Stat diff"
    mean_diff: str = "Mean diff"
    avg_rate: str = "avg rate of increase"
    monotonic: str = "monotonic"
    cv: str = "Coefficient of Variation (CV)"
    rc: str = "Relative Contrast (RC)"


@dataclass
class EvaluationCriteria:
    inter_i: str = "Interpretability: L_0 close to zero"
    inter_ii: str = "Interpretability: proper level sets ordering"
    inter_iii: str = "Interpretability: average level sets RC"
    inter_iv: str = "Interpretability: average level sets CV"
    inter_v: str = "Interpretability: rate of increase between level sets"
    disc_i: str = "Discriminative Power: overall RC"
    disc_ii: str = "Discriminative Power: overall CV"
    disc_iii: str = "Discriminative Power: macro F1 score"
    stab_i: str = "Stability: completed"
    stab_ii: str = "Stability: count of nan and inf distances"


minkowsky_distances = [DistanceMeasures.l1_cor_dist, DistanceMeasures.l2_cor_dist,
                       DistanceMeasures.l5_cor_dist, DistanceMeasures.l10_cor_dist,
                       DistanceMeasures.l50_cor_dist, DistanceMeasures.l100_cor_dist,
                       DistanceMeasures.linf_cor_dist]

minkowski_with_ref_distances = [DistanceMeasures.l1_with_ref, DistanceMeasures.l2_with_ref,
                                DistanceMeasures.l5_with_ref, DistanceMeasures.l10_with_ref,
                                DistanceMeasures.l50_with_ref, DistanceMeasures.l100_with_ref,
                                DistanceMeasures.linf_with_ref]


def get_p_from_distance(measure):
    p_str = measure.split(' ')[0].replace('L', '')
    if p_str == 'inf':
        p = np.inf
    else:
        p = int(p_str)
    return p


class DistanceMetricAssessment:
    def __init__(self, ds: DescribeSyntheticDataset, measures: [], backend: str = Backends.none.value):
        self.__ds = ds
        self.backend = backend
        self.__measures = measures
        self.segment_pair_distance_df = self.__calculate_distances_between_segment_pairs_per_level_set()
        self.level_sets = self.segment_pair_distance_df[DistanceMeasureCols.level_set].unique().tolist()
        self.per_level_set_distance_statistics_df = self.__calculate_per_level_sets_distance_statistics()
        self.effect_sizes_between_level_sets_df = self.__calculate_effect_size_for_difference_between_level_sets()
        self.ci_for_mean_differences, self.alpha_for_ci = self.__calculate_ci_for_mean_differences_between_level_sets()

        # cached variables
        # dictionary of key (alpha, bonferroni, two-sided) and value being a df of the ci of mean differences
        self.__ci_mean_differences_between_pattern_pairs_per_level_sets = {}

        self.__colors = {'overlap': 'teal', 'lower': 'deeppink', 'higher': 'dodgerblue'}

    def distances_statistics_for_each_pattern(self):
        """Calculates the distances statistics between each pattern combination"""
        dist_df = self.segment_pair_distance_df[self.segment_pair_distance_df[DistanceMeasureCols.level_set] == 0]
        segments_in_pattern = self.__ds.segments_for_each_pattern
        stats_dfs = []
        for pattern in self.__ds.patterns:
            p_segment_pairs = list(itertools.combinations(segments_in_pattern[pattern], 2))
            for distance_measure in self.__measures:
                p_stats = dist_df[dist_df[DistanceMeasureCols.pairs].isin(p_segment_pairs)][
                    distance_measure].describe().to_frame().T
                p_stats.insert(0, SyntheticDataSegmentCols.pattern_id, pattern)
                stats_dfs.append(p_stats)
        result = pd.concat(stats_dfs).reset_index(names=DistanceMeasureCols.type)
        return result

    def ci_of_differences_between_patterns(self, stats: pd.DataFrame, alpha: float = 0.05, bonferroni: bool = True,
                                           two_tailed: bool = True):
        """Calculates df of ci for the stats provided
        :param stats: df of "pattern_id, mean, std, etc for distances between segments in each pattern"""
        patterns_to_compare = list(itertools.combinations(stats[SyntheticDataSegmentCols.pattern_id].unique(), 2))
        a, ci_mean_diff_df = self.calculate_ci_of_mean_differences_between_level_sets(patterns_to_compare, stats,
                                                                                      SyntheticDataSegmentCols.pattern_id,
                                                                                      alpha, bonferroni, two_tailed)
        return ci_mean_diff_df, a

    def __calculate_distances_between_segment_pairs_per_level_set(self):
        segment_pairs_dict = self.__ds.segment_pairs_for_level_sets
        segment_correlations = self.__ds.segment_correlations_df
        results_levelset_names = []
        results_segpairs = []
        results_patternpairs = []
        results_measures = {measure: [] for measure in self.__measures}
        for level_set_name, segment_pairs in segment_pairs_dict.items():
            for pair in segment_pairs:
                # get  correlations
                seg1 = pair[0]
                seg2 = pair[1]
                corr1 = segment_correlations.loc[seg1][SyntheticDataSegmentCols.actual_correlation]
                corr2 = segment_correlations.loc[seg2][SyntheticDataSegmentCols.actual_correlation]

                for distance_measure in self.__measures:
                    calc_distance = distance_calculation_method_for(distance_measure)
                    dist = calc_distance(corr1, corr2)
                    results_measures[distance_measure].append(dist)

                # find pattern pair for segments
                p1 = self.__ds.labels[self.__ds.labels[SyntheticDataSegmentCols.segment_id] == seg1][
                    SyntheticDataSegmentCols.pattern_id].values[0]
                p2 = self.__ds.labels[self.__ds.labels[SyntheticDataSegmentCols.segment_id] == seg2][
                    SyntheticDataSegmentCols.pattern_id].values[0]
                results_patternpairs.append((p1, p2))

                # update other columns
                results_levelset_names.append(level_set_name)
                results_segpairs.append(pair)

        result_dic = {DistanceMeasureCols.level_set: results_levelset_names,
                      DistanceMeasureCols.pairs: results_segpairs,
                      DistanceMeasureCols.pattern_pairs: results_patternpairs
                      }
        for measure in self.__measures:
            result_dic[measure] = results_measures[measure]
        segment_pair_distances_df = pd.DataFrame(result_dic)
        return segment_pair_distances_df

    def __calculate_per_level_sets_distance_statistics(self):
        dist_df = self.segment_pair_distance_df

        stats_dfs = []
        for distance_measure in self.__measures:
            for level_set in self.level_sets:
                stats_df = dist_df[dist_df[DistanceMeasureCols.level_set] == level_set][
                    distance_measure].describe().to_frame().T
                stats_df.insert(0, DistanceMeasureCols.level_set, level_set)
                stats_dfs.append(stats_df)
        result = pd.concat(stats_dfs).reset_index(names=DistanceMeasureCols.type)
        return result

    def __calculate_effect_size_for_difference_between_level_sets(self):
        stats = self.per_level_set_distance_statistics_df
        level_set_combinations = itertools.combinations(self.level_sets, 2)
        compared_level_sets = []
        dist_measures = []
        effect_sizes = []
        for level_set1, level_set2 in level_set_combinations:
            g1 = stats[stats[DistanceMeasureCols.level_set] == level_set1]
            g2 = stats[stats[DistanceMeasureCols.level_set] == level_set2]
            d = cohens_d(g1[Aggregators.mean].to_numpy(), g2[Aggregators.mean].to_numpy(),
                         g1[Aggregators.std].to_numpy(), g2[Aggregators.std].to_numpy())
            measures = g1[DistanceMeasureCols.type].tolist()
            n_measures = len(measures)
            compared_level_sets.append([(level_set1, level_set2)] * n_measures)
            effect_sizes.append(d)
            dist_measures.append(measures)

        result = pd.DataFrame({
            DistanceMeasureCols.compared: list(itertools.chain.from_iterable(compared_level_sets)),
            DistanceMeasureCols.effect_size: list(itertools.chain.from_iterable(effect_sizes)),
            DistanceMeasureCols.type: list(itertools.chain.from_iterable(dist_measures))
        })
        return result

    def ordered_level_sets_and_mean_distances_by_smallest_first(self, distance_measure: str):
        df = self.per_level_set_distance_statistics_df[
            self.per_level_set_distance_statistics_df[DistanceMeasureCols.type] == distance_measure]
        sorted_df = df.sort_values(by=Aggregators.mean)
        return sorted_df[DistanceMeasureCols.level_set].tolist(), sorted_df[Aggregators.mean].tolist()

    def __calculate_ci_for_mean_differences_between_level_sets(self, alpha: float = 0.05, bonferroni: bool = True,
                                                               two_tailed: bool = True):
        stats = self.per_level_set_distance_statistics_df
        level_set_combinations = list(itertools.combinations(self.level_sets, 2))
        a, ci_mean_diff_df = self.calculate_ci_of_mean_differences_between_level_sets(level_set_combinations, stats,
                                                                                      DistanceMeasureCols.level_set,
                                                                                      alpha,
                                                                                      bonferroni, two_tailed)
        return ci_mean_diff_df, a

    def calculate_ci_of_mean_differences_between_level_sets(self, combinations, stats: pd.DataFrame,
                                                            level_set_selector: str,
                                                            alpha: float = 0.05,
                                                            bonferroni: bool = True,
                                                            two_tailed: bool = True):
        """ Calculates the ci of mean differences between various combinations of level sets
        :param combinations: list of tuples that need to be compared [(g1, g2)]
        :param stats: dataframe with the stats, must have columns: level-set_selector, Aggregators.count, Aggregators.mean
        Aggregators.std
        :param level_set_selector: name of the column in the stats dataframe for that level set
        """
        compared_level_sets = []
        dist_measures = []
        mean_diffs = []
        lo_diffs = []
        hi_diffs = []
        stat_diffs = []
        ci_widths = []
        standard_errors = []
        a = alpha
        if bonferroni:
            a = alpha / len(combinations)
        z_alpha = gaussian_critical_z_value_for(a, two_tailed=two_tailed)
        for level_set1, level_set2 in combinations:
            g1 = stats[stats[level_set_selector] == level_set1]
            g2 = stats[stats[level_set_selector] == level_set2]
            n1 = g1[Aggregators.count].reset_index(drop=True)
            n2 = g2[Aggregators.count].reset_index(drop=True)
            m1 = g1[Aggregators.mean].reset_index(drop=True)
            m2 = g2[Aggregators.mean].reset_index(drop=True)
            s1 = g1[Aggregators.std].reset_index(drop=True)
            s2 = g2[Aggregators.std].reset_index(drop=True)
            diff_ms = m1 - m2
            lo_ci, hi_ci, standard_error = calculate_hi_lo_difference_ci(n1, n2, s1, s2, m1, m2, z_alpha)
            ci_width = hi_ci - lo_ci
            stat_diff = []
            for idx in range(len(lo_ci)):
                stat_diff.append(compare_ci_for_differences(lo_ci[idx], hi_ci[idx]))

            compared_level_sets.append([(level_set1, level_set2)] * len(lo_ci))
            dist_measures.append(g1[DistanceMeasureCols.type])
            mean_diffs.append(diff_ms)
            lo_diffs.append(lo_ci)
            hi_diffs.append(hi_ci)
            stat_diffs.append(stat_diff)
            ci_widths.append(ci_width)
            standard_errors.append(standard_error)
        ci_mean_diff_df = pd.DataFrame(
            {DistanceMeasureCols.compared: list(itertools.chain.from_iterable(compared_level_sets)),
             DistanceMeasureCols.type: list(itertools.chain.from_iterable(dist_measures)),
             DistanceMeasureCols.stat_diff: list(itertools.chain.from_iterable(stat_diffs)),
             DistanceMeasureCols.mean_diff: list(itertools.chain.from_iterable(mean_diffs)),
             ConfidenceIntervalCols.ci_96lo: list(itertools.chain.from_iterable(lo_diffs)),
             ConfidenceIntervalCols.ci_96hi: list(itertools.chain.from_iterable(hi_diffs)),
             ConfidenceIntervalCols.width: list(itertools.chain.from_iterable(ci_widths)),
             ConfidenceIntervalCols.standard_error: list(
                 itertools.chain.from_iterable(standard_errors)),
             })
        return a, ci_mean_diff_df

    def plot_ci_of_differences_between_level_sets_for_measures(self, measures: [], show_title=True):
        title = r'95\% CI diff in means for each level set $d_i$ compared to $d_j$; $i=0,\ldots,4$, $j=i+1,\ldots,5$, $\alpha=' + str(
            round(self.alpha_for_ci, 3)) + '$'

        columns = [0, 1, 2, 3, 4]  # comparison between this level set and the other ones
        # setup figure
        reset_matplotlib(self.backend)
        use_latex_labels()
        fig_size = (15, len(measures) * 5)
        fig, axs = plt.subplots(nrows=len(measures),
                                ncols=len(columns),
                                sharey=False,
                                sharex=False,
                                figsize=fig_size, squeeze=0)

        if show_title:
            fig.suptitle(title, fontsize=fontsize)

        color = 'teal'
        gap = 0.2

        # data
        df = self.ci_for_mean_differences[self.ci_for_mean_differences[DistanceMeasureCols.type].isin(measures)]

        # y labels, use the y's for 0 level set that is compared with all others
        compared_level_sets = df[DistanceMeasureCols.compared].unique().tolist()
        y_level_sets = [item for item in compared_level_sets if item[0] == 0]
        y_ticks = range(1, len(y_level_sets) + 1)
        # create y ticks a gap apart for each measure

        # columns are each level set compared to its supposedly higher level sets
        for cdx, column in enumerate(columns):
            col_y_level_sets = [item for item in compared_level_sets if item[0] == column]
            # data for level_sets
            df_for_level_set = df[df[DistanceMeasureCols.compared].isin(col_y_level_sets)]

            # plot measures on different row to highlight differences better
            for rdx, measure in enumerate(measures):
                ax = axs[rdx][cdx]
                # for each new measure add a gap
                y = y_ticks[:len(col_y_level_sets)]
                df_measure = df_for_level_set[df_for_level_set[DistanceMeasureCols.type] == measure]
                # plot mean and annotate value
                mean_values = df_measure[DistanceMeasureCols.mean_diff]
                ax.scatter(mean_values, y, c=color, marker='P')
                for idx, mean in enumerate(mean_values):
                    ax.annotate(xy=(mean, y[idx]), text=round(mean, 1), xytext=(0, gap), textcoords='offset points',
                                va='bottom', color=color, fontsize=fontsize, ha='center')

                # plot horizontal lines
                ci_hi = df_measure[ConfidenceIntervalCols.ci_96hi]
                ci_lo = df_measure[ConfidenceIntervalCols.ci_96lo]
                ax.hlines(y=y, xmin=ci_lo, xmax=ci_hi, colors=color, ls='solid', lw=4)

                ax.set_yticks(y_ticks[:len(col_y_level_sets)], col_y_level_sets)
                min_y = 1 - gap
                max_y = y_ticks[-1] + 2 * gap
                ax.set_ylim(bottom=min_y, top=max_y)
                space = 10 if abs(mean_values.mean()) > 10 else 0.5
                ax.set_xlim(left=min(0, ci_lo.min()) - space, right=max(0, ci_hi.max()) + space)
                # show 0 - if the CI overlaps the difference is not significant
                ax.vlines(x=0, ymin=min_y, ymax=max_y, color='lightgray', linestyle='--')
                # ax.grid(which='both', axis='x', linestyle='--', color='lightgray')

        # overall row labels
        for ax, measure in zip(axs[:, 0], measures):
            ax.set_ylabel(measure, rotation=90, size=fontsize)

        # overall column labels
        for level_set_id in columns:
            ax = axs[len(measures) - 1, level_set_id]
            label = "$d_" + str(level_set_id) + "$"
            ax.set_xlabel(label, rotation=0, size=fontsize)

        plt.tight_layout()
        plt.show()
        return fig

    def plot_ci_per_level_set_for_pattern_pairs_where_distances_do_not_agree(self, measures: [] = [],
                                                                             show_title: bool = True):
        """
        Plots the ci for within pattern differences for each of the measures (rows) and each of the level sets (level sets)
        but only when the ci don't agree between the measures (e.g. one overlap, another higher or lower).
        Only the first 10 pattern pairs are plotted. If a level set has no differences it is not plotted
        """
        title = r'95\% CI of difference in means for each pattern pair in $d_i$ where the distance measure disagree in significance of difference'

        # if not provided plot all
        if len(measures) is 0:
            measures = self.__measures

        pairs_with_differences = self.find_within_level_set_differences_where_the_distances_dont_agree(
            measures=measures)
        level_sets = [key for key, value in pairs_with_differences.items() if len(value) > 0]

        # setup figure
        reset_matplotlib(self.backend)
        use_latex_labels()
        fig_size = (15, 10)
        fig, axs = plt.subplots(nrows=len(measures),
                                ncols=len(level_sets),
                                sharey=False,
                                sharex=False,
                                figsize=fig_size, squeeze=0)

        if show_title:
            fig.suptitle(title, fontsize=fontsize)

        gap = 0.2

        # data
        data = self.calculate_ci_mean_differences_between_pattern_pairs_for_each_level_set()
        # only keep the measures we want
        data = data[data[DistanceMeasureCols.type].isin(measures)]
        # only keep first 10 pairs for each level set
        n_plot = 10
        shortened_pairs = [pairs[:n_plot] for pairs in pairs_with_differences.values()]
        # flatten list
        shortened_pairs = (itertools.chain.from_iterable(shortened_pairs))

        # filter data
        data["compared"] = list(zip(data[DistanceMeasureCols.pair1], data[DistanceMeasureCols.pair2]))
        filtered_data = data[data["compared"].isin(shortened_pairs)]

        # level sets are each level set, y-axis for each column is the pairs, x-axis the CI
        for cdx, column in enumerate(level_sets):
            data_level_set = filtered_data[filtered_data[DistanceMeasureCols.level_set] == column]

            # y labels for level sets is the pattern pairs within that level set, these are: g1 (pattern1, pattern2) and g2
            # (pattern 1, pattern 2). These will be the same for all measures
            data_level_set_measure = data_level_set[data_level_set[DistanceMeasureCols.type] == measures[0]]
            level_set_pair1 = list(data_level_set_measure[DistanceMeasureCols.pair1])
            level_set_pair2 = list(data_level_set_measure[DistanceMeasureCols.pair2])
            y_level_sets = [
                str(p1).replace('(', '').replace(')', '').replace(" ", '') + "/"
                + str(p2).replace('(', '').replace(')', '').replace(" ", '')
                for p1, p2 in zip(level_set_pair1, level_set_pair2)]
            y_ticks = range(1, len(y_level_sets) + 1)
            # create y ticks a gap apart for each measure

            # plot measures on different row to highlight differences better
            for rdx, measure in enumerate(measures):
                data_for_measure = data_level_set[data_level_set[DistanceMeasureCols.type] == measure]
                ax = axs[rdx][cdx]

                # plot mean and annotate value
                color = [self.__colors[diff] for diff in list(data_for_measure[DistanceMeasureCols.stat_diff])]
                mean_values = data_for_measure[DistanceMeasureCols.mean_diff]
                # skip if now values
                if len(mean_values) > 0:
                    ax.scatter(mean_values, y_ticks, c=color, marker='P')

                    # plot horizontal lines
                    ci_hi = data_for_measure[ConfidenceIntervalCols.ci_96hi]
                    ci_lo = data_for_measure[ConfidenceIntervalCols.ci_96lo]
                    ax.hlines(y=y_ticks, xmin=ci_lo, xmax=ci_hi, colors=color, ls='solid', lw=4)

                    ax.set_yticks(y_ticks, y_level_sets)
                    min_y = 1 - gap
                    max_y = y_ticks[-1] + 2 * gap
                    ax.set_ylim(bottom=min_y, top=max_y)
                    space = 10 if abs(mean_values.mean()) > 10 else 0.5
                    ax.set_xlim(left=min(0, ci_lo.min()) - space, right=max(0, ci_hi.max()) + space)
                    # show 0 - if the CI overlaps the difference is not significant
                    ax.vlines(x=0, ymin=min_y, ymax=max_y, color='lightgray', linestyle='--')
                    # ax.grid(which='both', axis='x', linestyle='--', color='lightgray')

        # overall row labels
        for ax, row in zip(axs[:, 0], measures):
            ax.set_ylabel(row, rotation=90, size=fontsize)

        # overall column labels
        for idx, level_set_id in enumerate(level_sets):
            ax = axs[len(measures) - 1, idx]
            label = r'$d_' + str(level_set_id) + '$'
            ax.set_xlabel(label, rotation=0, size=fontsize)

        plt.tight_layout()
        plt.show()
        return fig

    def plot_ci_for_ordered_level_sets_for_measures(self, measures: [], show_title=True):
        """
        Plots the confidence intervals for the provided measures but only the comparison between
        the level sets in order of their mean difference. One plot per distance measure as columns.
        """
        # setup figure
        reset_matplotlib(self.backend)
        use_latex_labels()
        fig_size = (15, 5)
        fig, axs = plt.subplots(nrows=1,
                                ncols=len(measures),
                                sharey=False,
                                sharex=False,
                                figsize=fig_size, squeeze=0)

        if show_title:
            title = r'95\% CI diff in means between the level sets ordered from smallest to biggest distance, $\alpha=' + str(
                round(self.alpha_for_ci, 3)) + '$'
            fig.suptitle(title, fontsize=fontsize)

        colors = {measures[0]: 'teal', measures[1]: 'dodgerblue'}
        gap = 0.2

        # data
        df = self.ci_for_mean_differences[self.ci_for_mean_differences[DistanceMeasureCols.type].isin(measures)]
        distance_df = self.per_level_set_distance_statistics_df[
            self.per_level_set_distance_statistics_df[DistanceMeasureCols.type].isin(measures)]
        # y ticks: each measure has 5 comparisons
        y_ci = [1.5, 2.5, 3.5, 4.5, 5.5]
        y_distances = [1, 2, 3, 4, 5, 6]

        # columns are the different measures
        for cdx, measure in enumerate(measures):
            ax = axs[0][cdx]

            # ordered distances for measure
            level_sets_ordered = self.ordered_level_sets_and_mean_distances_by_smallest_first(measure)[0]
            # level sets compared in order
            y_tuples = []
            for i in range(len(level_sets_ordered) - 1):
                y_tuples.append((level_sets_ordered[i], level_sets_ordered[i + 1]))
            y_labels = [(min(g1, g2), max(g1, g2)) for g1, g2 in y_tuples]

            # data
            df_measure = df[df[DistanceMeasureCols.type] == measure]
            data = df_measure[df_measure[DistanceMeasureCols.compared].isin(y_labels)]
            # sort the data in order of the smallest level set first and the biggest last
            data[DistanceMeasureCols.compared] = data[DistanceMeasureCols.compared].astype("category")
            data[DistanceMeasureCols.compared] = data[DistanceMeasureCols.compared].cat.set_categories(y_labels)
            sorted_data = data.sort_values(DistanceMeasureCols.compared)

            distance_data = distance_df[distance_df[DistanceMeasureCols.type] == measure]
            # sort the data by the level set
            distance_data[DistanceMeasureCols.level_set] = distance_data[DistanceMeasureCols.level_set].astype(
                "category")
            distance_data[DistanceMeasureCols.level_set] = distance_data[
                DistanceMeasureCols.level_set].cat.set_categories(
                level_sets_ordered)
            sorted_distance = distance_data.sort_values(DistanceMeasureCols.level_set)

            # plot mean and annotate value
            mean_values = sorted_data[DistanceMeasureCols.mean_diff]
            ax.scatter(mean_values, y_ci, c=colors[measure], marker='P')
            for idx, mean in enumerate(mean_values):
                ax.annotate(xy=(mean, y_ci[idx]), text=round(mean, 1), xytext=(0, gap), textcoords='offset points',
                            va='bottom', color=colors[measure], fontsize=fontsize, ha='center')

            # plot distance for level set
            distances = sorted_distance[Aggregators.mean]
            ax.scatter(distances, y_distances, c='dimgrey', marker='*')
            for idx, dist in enumerate(distances):
                ax.annotate(xy=(dist, y_distances[idx]), text=round(dist, 1), xytext=(0, gap),
                            textcoords='offset points',
                            va='bottom', color='dimgrey', fontsize=fontsize, ha='center')

            # plot CI lines
            ci_hi = sorted_data[ConfidenceIntervalCols.ci_96hi]
            ci_lo = sorted_data[ConfidenceIntervalCols.ci_96lo]
            ax.hlines(y=y_ci, xmin=ci_lo, xmax=ci_hi, colors=colors[measure], ls='solid', lw=4)

            # plot labels
            # combine both y ticks and labels into one list
            ys = list(sum(zip(y_distances, y_ci + [0]), ())[:-1])
            distance_labels = ['d' + str(level_set) for level_set in level_sets_ordered]
            ys_labels = list(sum(zip(distance_labels, y_labels + [0]), ())[:-1])
            ax.set_yticks(ys, ys_labels)
            min_y = 1 - gap
            max_y = y_distances[-1] + 3 * gap
            ax.set_ylim(bottom=min_y, top=max_y)
            space = 5 if distances.mean() > 10 else 0.5
            x_min = min(0, min(list(ci_lo) + list(distances))) - space
            x_max = max(0, max(list(ci_hi) + list(distances))) + space
            ax.set_xlim(left=x_min, right=x_max)
            # show 0 - if the CI overlaps the difference is not significant
            ax.vlines(x=0, ymin=min_y, ymax=max_y, color='lightgray', linestyle='--')
            ax.hlines(y=y_distances, xmin=x_min, xmax=x_max, color='lightgray', linestyle='dotted')

        for ax, measure in zip(axs[0, :], measures):
            ax.set_xlabel(measure, rotation=0, size=fontsize)

        plt.tight_layout()
        plt.show()
        return fig

    def plot_correlation_matrices_of_biggest_distances_for_level_sets(self, g1: int, g2: int, plot_diagonal=False,
                                                                      what: str = "biggest",
                                                                      measure=DistanceMeasures.l2_cor_dist,
                                                                      show_title: bool = True):
        """Plots correlation matrices of patterns compared in the two level sets given
        :param g1: index of level sets plotted on row 1, 0-5
        :param g2: index of level sets plotted on row 2, 0-5
        :param plot_diagonal: whether to plot the diagonal 1 correlation or not
        :param what: "biggest" or "smallest" selects the pattern with either the 2 biggest or smallest distances
        between segments
        :param measure: which distance metric to get results for
        :param show_title: whether to show title
        """
        # setup figure
        reset_matplotlib(self.backend)
        use_latex_labels()
        fig_size = (15, 15)
        no_rows = 2
        no_cols = 2
        fig, axs = plt.subplots(nrows=no_rows,
                                ncols=no_cols,
                                sharey=False,
                                sharex=False,
                                figsize=fig_size)
        cmap = "bwr"
        normed_colour = mpl.colors.Normalize(-1, 1)

        title = "Visualisation of the patterns of the segment pair with the " + what + " distances for each level set"
        if show_title:
            fig.suptitle(title, fontsize=fontsize)

        # data
        level_set1 = self.segment_pair_distance_df[self.segment_pair_distance_df[DistanceMeasureCols.level_set] == g1]
        level_set2 = self.segment_pair_distance_df[self.segment_pair_distance_df[DistanceMeasureCols.level_set] == g2]

        # get segment pairs for each of the two level sets
        if what == "biggest":
            seg1pair = level_set1.loc[level_set1[measure].idxmax()][DistanceMeasureCols.pairs]
            seg2pair = level_set2.loc[level_set2[measure].idxmax()][DistanceMeasureCols.pairs]
        elif what == "smallest":
            seg1pair = level_set1.loc[level_set1[measure].idxmin()][DistanceMeasureCols.pairs]
            seg2pair = level_set2.loc[level_set2[measure].idxmin()][DistanceMeasureCols.pairs]
        else:
            assert False, "unknown what parameter"

        # get pattern of each of these segments pairs
        pattern11 = self.__ds.labels.iloc[seg1pair[0]][SyntheticDataSegmentCols.pattern_id]
        pattern12 = self.__ds.labels.iloc[seg1pair[1]][SyntheticDataSegmentCols.pattern_id]
        pattern21 = self.__ds.labels.iloc[seg2pair[0]][SyntheticDataSegmentCols.pattern_id]
        pattern22 = self.__ds.labels.iloc[seg2pair[1]][SyntheticDataSegmentCols.pattern_id]

        # plot each of them: level set 1 on row 1, level set 2 on row 2
        ax00 = self.__plot_matrix_for_pattern(pattern11, axs, 0, 0, cmap, normed_colour, plot_diagonal=plot_diagonal)
        ax01 = self.__plot_matrix_for_pattern(pattern12, axs, 0, 1, cmap, normed_colour, plot_diagonal=plot_diagonal)
        ax10 = self.__plot_matrix_for_pattern(pattern21, axs, 1, 0, cmap, normed_colour, plot_diagonal=plot_diagonal,
                                              add_to_x_label='Segment A')
        ax11 = self.__plot_matrix_for_pattern(pattern22, axs, 1, 1, cmap, normed_colour, plot_diagonal=plot_diagonal,
                                              add_to_x_label='Segment B')

        # label rows
        ax00.set_ylabel(r'level set: ' + '$d_' + str(g1) + '$', size=fontsize)
        ax10.set_ylabel(r'level set: ' + '$d_' + str(g2) + '$', size=fontsize)

        plt.tight_layout()
        plt.show()
        return fig

    def __plot_matrix_for_pattern(self, pattern_id, axs, rdx, cdx, cmap, normed_colour, plot_diagonal=False,
                                  add_to_x_label: str = ''):
        ax = axs[rdx][cdx]
        # get all segment data for this pattern
        df = self.__ds.data_by_pattern_id[pattern_id]
        correlation = df.corr()
        plot_corr_ellipses(correlation, ax=ax, plot_diagonal=plot_diagonal, cmap=cmap, norm=normed_colour)
        ax.margins(0.1)

        # label x with pattern
        label = str(pattern_id) + ": " + str(
            self.__ds.labels[self.__ds.labels[SyntheticDataSegmentCols.pattern_id] == pattern_id].iloc[0][
                SyntheticDataSegmentCols.correlation_to_model])
        if len(add_to_x_label) > 0:
            label = label + '\n' + add_to_x_label

        ax.set_xlabel(label, rotation=0, size=fontsize)
        return ax

    def find_min_or_max_distances_for_each_level_set_for_a_measure(self, what: str = "min",
                                                                   measure: str = DistanceMeasures.l2_cor_dist):
        """ Return  df of min or max distance for the given distance measure.
        Result df has columns: level_sets, segment pairs, distance, pattern_id pairs
        """
        dist_df = self.segment_pair_distance_df

        indices = []
        for level_set in self.level_sets:
            if what == "min":
                idx = dist_df[dist_df[DistanceMeasureCols.level_set] == level_set][
                    measure].idxmin()
            elif what == "max":
                idx = dist_df[dist_df[DistanceMeasureCols.level_set] == level_set][
                    measure].idxmax()
            indices.append(idx)
        result = dist_df.iloc[indices].reset_index(drop=True)[
            [DistanceMeasureCols.level_set, DistanceMeasureCols.pairs, DistanceMeasureCols.pattern_pairs, measure]]
        return result

    def plot_box_diagrams_of_distances_as_function_of_p(self, level_sets: []):
        """This assumes that the distance measures are the lp ones. Plots box diagrams of the distances
        with the different p as x and the distance as y axis. If level set is 'all' the mean is taken. If level set is a number
        only the distance data for that level set will be plotted
        """
        # build dictionary of p's for x axis. Key is p, value is measure name
        measures = {}
        if DistanceMeasures.l1_with_ref in self.__measures:
            measures[1] = DistanceMeasures.l1_with_ref
        if DistanceMeasures.l2_with_ref in self.__measures:
            measures[2] = DistanceMeasures.l2_with_ref
        if DistanceMeasures.l5_with_ref in self.__measures:
            measures[5] = DistanceMeasures.l5_with_ref
        if DistanceMeasures.l10_with_ref in self.__measures:
            measures[10] = DistanceMeasures.l10_with_ref
        if DistanceMeasures.l50_with_ref in self.__measures:
            measures[50] = DistanceMeasures.l50_with_ref
        if DistanceMeasures.l100_with_ref in self.__measures:
            measures[100] = DistanceMeasures.l100_with_ref
        if DistanceMeasures.linf_with_ref in self.__measures:
            measures[np.inf] = DistanceMeasures.linf_with_ref
        if DistanceMeasures.l1_cor_dist in self.__measures:
            measures[1] = DistanceMeasures.l1_cor_dist
        if DistanceMeasures.l2_cor_dist in self.__measures:
            measures[2] = DistanceMeasures.l2_cor_dist
        if DistanceMeasures.l5_cor_dist in self.__measures:
            measures[5] = DistanceMeasures.l5_cor_dist
        if DistanceMeasures.l10_cor_dist in self.__measures:
            measures[10] = DistanceMeasures.l10_cor_dist
        if DistanceMeasures.l50_cor_dist in self.__measures:
            measures[50] = DistanceMeasures.l50_cor_dist
        if DistanceMeasures.l100_cor_dist in self.__measures:
            measures[100] = DistanceMeasures.l100_cor_dist
        if DistanceMeasures.linf_cor_dist in self.__measures:
            measures[np.inf] = DistanceMeasures.linf_cor_dist
        ps = list(measures.keys())
        measure_names = list(measures.values())

        reset_matplotlib(self.backend)
        fig_size = (15, 10)
        fig, axs = plt.subplots(nrows=1,
                                ncols=1,
                                sharey=False,
                                sharex=False,
                                figsize=fig_size)
        sns.set(style="whitegrid")

        data = pd.melt(self.segment_pair_distance_df, id_vars=[DistanceMeasureCols.level_set], value_vars=measure_names)
        if level_sets[0] in self.level_sets:
            ax = sns.boxplot(data=data, x="variable", y="value", hue=DistanceMeasureCols.level_set, ax=axs,
                             palette="rainbow")
        else:
            ax = sns.boxplot(data=data, x="variable", y="value", ax=axs,
                             palette="rainbow")

        ax.set_xlabel('Minkowski Distance', fontsize=fontsize)
        ax.set_ylabel('Distance', fontsize=fontsize)

        ax.tick_params(axis='x', labelsize=fontsize)
        ax.set_xticklabels(["L" + str(p) for p in ps], fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

        # no legend needed if level set is all
        if level_sets[0] in self.level_sets:
            ax.legend(title="Pattern Level Set", title_fontsize=fontsize, fontsize=fontsize)
            for i, level_set in enumerate(level_sets):
                ax.legend_.texts[i].set_text(level_set)

        plt.show()
        return fig

    def plot_box_diagrams_of_distances_for_all_level_sets(self, measures: [] = [], order: [] = default_order):
        if len(measures) == 0:
            measures = self.__measures

        if len(order) == 0:
            order = self.level_sets

        reset_matplotlib(self.backend)
        fig_size = (15, 10)
        fig, axs = plt.subplots(nrows=1,
                                ncols=1,
                                sharey=False,
                                sharex=False,
                                figsize=fig_size)
        sns.set(style="whitegrid")

        data = pd.melt(self.segment_pair_distance_df, id_vars=[DistanceMeasureCols.level_set], value_vars=measures)
        ax = sns.boxplot(data=data, x=DistanceMeasureCols.level_set, y="value", hue="variable", ax=axs,
                         palette="rainbow",
                         order=order)

        ax.set_xlabel('Level Set d', fontsize=fontsize)
        ax.set_ylabel('Distance', fontsize=fontsize)

        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

        ax.legend(title="Distance Measure", title_fontsize=fontsize, fontsize=fontsize)
        for i, measure in enumerate(measures):
            ax.legend_.texts[i].set_text(measure)

        plt.show()
        return fig

    def calculate_statistics_for_pattern_pairs_for_measure(self, alpha=0.05):
        """
        Calculates count, mean, std, min, max and CI of mean difference between each row and the previous row.
        Mean are ordered by smallest first, therefor z-value is for one-sided test means between row either overlap or
        are previous row is significantly lower.
        We do not Bonferroni adjust here (maybe we should?)
        """
        dist_df = self.segment_pair_distance_df
        pattern_pairs = list(dist_df[DistanceMeasureCols.pattern_pairs].unique())
        stats_dfs = []
        for pair in pattern_pairs:
            dist_for_pair = dist_df[dist_df[DistanceMeasureCols.pattern_pairs] == pair]
            level_sets = list(dist_for_pair[DistanceMeasureCols.level_set].unique())
            assert len(level_sets) == 1, "a pattern pair appears in more than one level set"
            level_set = level_sets[0]
            for distance_measure in self.__measures:
                p_stats = dist_for_pair[distance_measure].describe().to_frame().T
                p_stats.insert(0, DistanceMeasureCols.level_set, level_set)
                p_stats.insert(1, DistanceMeasureCols.pattern_pairs, [pair])
                stats_dfs.append(p_stats)
        all_stats = pd.concat(stats_dfs).reset_index(names=DistanceMeasureCols.type)

        # CI interpretation when considering CI of distance for each pattern pair
        stats_for_measure = []
        z_value = gaussian_critical_z_value_for(alpha, two_tailed=False)
        for distance_measure in self.__measures:
            stats_per_d = all_stats[all_stats[DistanceMeasureCols.type] == distance_measure]
            # sort by means
            stats_per_d.sort_values(by=["mean"], inplace=True, ignore_index=True)
            stats_diff = [ci_overlap]  # first element has no comparison
            lo_cis = [0]
            hi_cis = [0]
            for idx in range(1, stats_per_d.shape[0]):
                # calculate CI of differences pairwise for ordered patterns
                row_data1 = stats_per_d.iloc[idx - 1]
                row_data2 = stats_per_d.iloc[idx]
                lo_ci, hi_ci, standard_error = calculate_hi_lo_difference_ci(row_data1["count"], row_data2["count"],
                                                                             row_data1["std"], row_data2["std"],
                                                                             row_data1["mean"], row_data2["mean"],
                                                                             z_value)
                stats_diff.append(compare_ci_for_differences(lo_ci, hi_ci))
                hi_cis.append(hi_ci)
                lo_cis.append(hi_ci)
            stats_per_d[DistanceMeasureCols.stat_diff] = stats_diff
            stats_per_d[ConfidenceIntervalCols.ci_96hi] = hi_cis
            stats_per_d[ConfidenceIntervalCols.ci_96lo] = lo_cis
            stats_for_measure.append(stats_per_d)

        result = pd.concat(stats_for_measure)

        return result

    def calculate_df_of_pattern_pair_level_sets(self):
        """
            Returns df with columns distance measure, level set (list of level set id), count (of patterns in that row),
            pattern pairs (list of patterns).
            Each row contains pattern pairs between which there is no significant differences
        """
        all_df = self.calculate_statistics_for_pattern_pairs_for_measure()

        # dtype object is important to be able to insert lists into cell
        result = pd.DataFrame({DistanceMeasureCols.type: [], DistanceMeasureCols.level_set: [], "count": [],
                               DistanceMeasureCols.pattern_pairs: []}, dtype=object)
        for measure in self.__measures:
            data = all_df[all_df[DistanceMeasureCols.type] == measure]

            # indices for next level set to start
            indices = list(data[data[DistanceMeasureCols.stat_diff] == "lower"].index)
            indices.append(data.shape[0])  # include last level set

            start_idx = 0
            for end_idx in indices:
                sub_frame = data[start_idx:end_idx]
                pairs = list(sub_frame[DistanceMeasureCols.pattern_pairs])
                level_sets = list(sub_frame[DistanceMeasureCols.level_set].unique())
                df_index = len(result.index)
                result.at[df_index, DistanceMeasureCols.type] = measure
                result.at[df_index, DistanceMeasureCols.pattern_pairs] = pairs
                result.at[df_index, DistanceMeasureCols.level_set] = level_sets
                result.at[df_index, "count"] = len(pairs)

                start_idx = end_idx

        return result

    def calculate_ci_mean_differences_between_pattern_pairs_for_each_level_set(self, alpha: float = 0.05,
                                                                               bonferroni: bool = True,
                                                                               two_tailed: bool = True):
        """Calculates ci of mean difference between each pattern pair in each level set"""
        if self.__ci_mean_differences_between_pattern_pairs_per_level_sets.get((alpha, bonferroni, two_tailed)) is None:
            # calculate it - otherwise return already calculated version
            stats_df = self.calculate_statistics_for_pattern_pairs_for_measure()

            pattern_pair1 = []
            pattern_pair2 = []
            level_sets = []
            bonferroni_alpha = []
            dist_measures = []
            mean_diffs = []
            lo_diffs = []
            hi_diffs = []
            stat_diffs = []
            ci_widths = []
            standard_errors = []

            for level_set in stats_df[DistanceMeasureCols.level_set].unique():
                stats_for_level_set = stats_df[stats_df[DistanceMeasureCols.level_set] == level_set]
                pattern_pairs = list(stats_for_level_set[DistanceMeasureCols.pattern_pairs].unique())
                combinations = list(itertools.combinations(pattern_pairs, 2))

                # bonferroni correction for the level set
                a = alpha
                if bonferroni:
                    a = alpha / len(combinations)
                z_alpha = gaussian_critical_z_value_for(a, two_tailed=two_tailed)

                for pair1, pair2 in combinations:
                    g1 = stats_for_level_set[stats_for_level_set[DistanceMeasureCols.pattern_pairs] == pair1]
                    g2 = stats_for_level_set[stats_for_level_set[DistanceMeasureCols.pattern_pairs] == pair2]
                    n1 = g1[Aggregators.count].reset_index(drop=True)
                    n2 = g2[Aggregators.count].reset_index(drop=True)
                    m1 = g1[Aggregators.mean].reset_index(drop=True)
                    m2 = g2[Aggregators.mean].reset_index(drop=True)
                    s1 = g1[Aggregators.std].reset_index(drop=True)
                    s2 = g2[Aggregators.std].reset_index(drop=True)
                    diff_ms = m1 - m2
                    lo_ci, hi_ci, standard_error = calculate_hi_lo_difference_ci(n1, n2, s1, s2, m1, m2, z_alpha)
                    ci_width = hi_ci - lo_ci
                    stat_diff = []
                    for idx in range(len(lo_ci)):
                        stat_diff.append(compare_ci_for_differences(lo_ci[idx], hi_ci[idx]))

                    pattern_pair1.append([pair1] * len(lo_ci))  # len is for different distance measures
                    pattern_pair2.append([pair2] * len(lo_ci))
                    level_sets.append([level_set] * len(lo_ci))
                    bonferroni_alpha.append([a] * len(lo_ci))
                    dist_measures.append(g1[DistanceMeasureCols.type])
                    mean_diffs.append(diff_ms)
                    lo_diffs.append(lo_ci)
                    hi_diffs.append(hi_ci)
                    stat_diffs.append(stat_diff)
                    ci_widths.append(ci_width)
                    standard_errors.append(standard_error)
            ci_mean_diff_df = pd.DataFrame({
                DistanceMeasureCols.pair1: list(itertools.chain.from_iterable(pattern_pair1)),
                DistanceMeasureCols.pair2: list(itertools.chain.from_iterable(pattern_pair2)),
                DistanceMeasureCols.level_set: list(itertools.chain.from_iterable(level_sets)),
                DistanceMeasureCols.type: list(itertools.chain.from_iterable(dist_measures)),
                DistanceMeasureCols.stat_diff: list(itertools.chain.from_iterable(stat_diffs)),
                DistanceMeasureCols.alpha: list(itertools.chain.from_iterable(bonferroni_alpha)),
                DistanceMeasureCols.mean_diff: list(itertools.chain.from_iterable(mean_diffs)),
                ConfidenceIntervalCols.ci_96lo: list(itertools.chain.from_iterable(lo_diffs)),
                ConfidenceIntervalCols.ci_96hi: list(itertools.chain.from_iterable(hi_diffs)),
                ConfidenceIntervalCols.width: list(itertools.chain.from_iterable(ci_widths)),
                ConfidenceIntervalCols.standard_error: list(
                    itertools.chain.from_iterable(standard_errors)),
            })
            self.__ci_mean_differences_between_pattern_pairs_per_level_sets[
                (alpha, bonferroni, two_tailed)] = ci_mean_diff_df
        return self.__ci_mean_differences_between_pattern_pairs_per_level_sets[(alpha, bonferroni, two_tailed)]

    def find_within_level_set_differences_where_the_distances_dont_agree(self, measures: [] = []):
        """
            Returns a dictionary with the keys being the level set name and the values being a list of tuples of pattern
            pairs ((p1, p2), (p2, p3)) for which the differences in means between segments from (p1, p2) and segments
            from (p2, p3) are not consistent across the different distance measures. I.e. one measure might judge the
            CI for differences in means as 'overlap', one as 'higher', and one as 'lower'
        """
        # use all distance measures if none given
        if len(measures) is 0:
            measures = self.__measures

        ci_mean_diff = self.calculate_ci_mean_differences_between_pattern_pairs_for_each_level_set()
        result = {}

        for level_set in self.level_sets:
            # ci for that gourp
            level_set_data = ci_mean_diff[ci_mean_diff[DistanceMeasureCols.level_set] == level_set]
            # create a new column of tuples of pattern pair 1, pair 2
            level_set_data.insert(0, "compared",
                                  list(zip(level_set_data[DistanceMeasureCols.pair1],
                                           level_set_data[DistanceMeasureCols.pair2])))
            # level set data by compared -> now each should unique pattern tuple has an entry per distance measure
            # then count the values for stat_diff (higher, lower, overlap), this number will be the same
            # as the number of distance measures if all measures have the same stat_diff
            stat_diff_counts = level_set_data.groupby(["compared"])[
                DistanceMeasureCols.stat_diff].value_counts().reset_index(name='count')
            # select the ones where the count is not equal to number of distance measures
            patterns = stat_diff_counts[stat_diff_counts["count"] < len(self.__measures)]["compared"].unique()
            result[level_set] = patterns
        return result

    def plot_ci_pattern_pair_heat_map(self, level_set: int, measure: str,
                                      show_stats_for: [] = ['overlap', 'lower', 'higher'], show_title: bool = True):
        """Plots a heatmap for n/a (pair not in level set), similar, overlap, higher, lower for all pattern
        pair combinations
        """
        # get ci data for level set
        stats_df = self.calculate_ci_mean_differences_between_pattern_pairs_for_each_level_set()
        ci_mean_diff = stats_df[stats_df[DistanceMeasureCols.level_set] == level_set]
        ci_mean_diff = ci_mean_diff[ci_mean_diff[DistanceMeasureCols.type] == measure][
            [DistanceMeasureCols.pair1, DistanceMeasureCols.pair2, DistanceMeasureCols.stat_diff]]

        title_add = ''
        show_cbar = True
        # colors = tuple([self.__colors[item] for item in show_stats_for])
        colors = tuple(self.__colors.values())
        cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))

        # filter stat
        if len(show_stats_for) < 3:
            ci_mean_diff = ci_mean_diff[ci_mean_diff[DistanceMeasureCols.stat_diff].isin(show_stats_for)]
            title_add = '. Only show ' + ','.join(show_stats_for) + '.'
            show_cbar = False

        # set values to 0 (overlap), 1 (lower), 2 (higher)
        stat_diff = [self.__translate_stat_diff(item) for item in (list(ci_mean_diff[DistanceMeasureCols.stat_diff]))]

        # only squares with data
        pair1 = list(ci_mean_diff[DistanceMeasureCols.pair1])
        pair2 = list(ci_mean_diff[DistanceMeasureCols.pair2])
        nx = len(pair1)
        ny = len(pair2)
        data = np.ones([nx, ny]) * -1
        data[np.ix_(range(nx), range(ny))] = stat_diff
        df = pd.DataFrame(data, index=pair1, columns=pair2)

        reset_matplotlib(self.backend)
        fig_size = (15, 10)
        fig, ax = plt.subplots(nrows=1,
                               ncols=1,
                               sharey=False,
                               sharex=False,
                               figsize=fig_size)
        sns.set(style="whitegrid")

        title = 'Heatmap of distance similarity for level set ' + str(level_set) + " and measure " + measure + title_add

        if show_title:
            fig.suptitle(title, fontsize=fontsize)

        snax = sns.heatmap(df, ax=ax, cmap=cmap, cbar=show_cbar)
        snax.tick_params(axis='x', labelrotation=90)
        snax.xaxis.tick_top()

        ax.set_xlabel('Pattern pair 1', size=fontsize)
        ax.set_ylabel('Pattern pair 2', size=fontsize)

        if show_cbar:
            colorbar = snax.collections[0].colorbar
            colorbar.set_ticks([0.35, 1, 1.66])
            colorbar.set_ticklabels(['Similar', 'Lower', 'Higher'], fontsize=fontsize)

        fig.tight_layout()
        plt.show()
        return fig

    @staticmethod
    def __translate_stat_diff(value: str) -> int:
        if value == "overlap":
            return 0
        elif value == "lower":
            return 1
        elif value == "higher":
            return 2
        else:
            assert False, "unkown value"

    def discriminative_power_criteria(self):
        """
            Calculates the following discriminative information about each measure:
            -average rate of increase in mean distance from smaller to bigger changes (bigger is better, check it is
             monotonic first as it is not valid otherwise)
        """
        # get mean difference of distance between each level sets 0-1, 0-2, 0-3, ... , 1-2, 1-3 etc.
        # check this is monotonic per level set 0-4
        # add all up and divide by total

        df = self.ci_for_mean_differences
        measures = []
        averages = []
        monotonicities = []
        cvies = []
        rcies = []

        # don't include comparing correlations from level set 0 (same pattern)
        exclude_g0 = self.segment_pair_distance_df[self.segment_pair_distance_df[DistanceMeasureCols.level_set] != 0]

        mins = exclude_g0[self.__measures].min(axis=0)
        maxs = exclude_g0[self.__measures].max(axis=0)
        stds = exclude_g0[self.__measures].std(axis=0)
        means = exclude_g0[self.__measures].mean(axis=0)

        for measure in self.__measures:
            df_m = df[df[DistanceMeasureCols.type] == measure]
            total_number = df_m.shape[0]
            measures.append(measure)
            avg = df_m[DistanceMeasureCols.mean_diff].abs().sum() / total_number
            averages.append(avg.round(2))

            min_d = mins.loc[measure]
            max_d = maxs.loc[measure]
            std_d = stds.loc[measure]
            mean_d = means.loc[measure]

            cvies.append((std_d / mean_d).round(2))
            rcies.append(((max_d - min_d) / min_d).round(2))

            # check monotonic
            is_monotonic = True
            for g1 in range(5):
                level_sets_list = [(g1, g2) for g2 in range(g1, 6)]
                monotonic = df_m[df_m[DistanceMeasureCols.compared].isin(level_sets_list)][
                    DistanceMeasureCols.mean_diff].abs().is_monotonic_increasing
                if not monotonic:
                    is_monotonic = False
                    break
            monotonicities.append(is_monotonic)

        result = pd.DataFrame({
            DistanceMeasureCols.type: measures,
            DistanceMeasureCols.avg_rate: averages,
            DistanceMeasureCols.monotonic: monotonicities,
            DistanceMeasureCols.cv: cvies,
            DistanceMeasureCols.rc: rcies
        })
        result.set_index(DistanceMeasureCols.type, inplace=True)
        return result

    def raw_results_for_each_criteria(self):
        """ Calculates the raw results for each distance measure and each criterion as described in the paper.
        Rows are the criteria (see EvaluationCriteria), columns are criteria followed by the distance measures.
        :returns pd.Dataframe
        """
        # setup columns, indices and empty row arrays for dataframe
        columns = self.__measures.copy()
        indices = [
            EvaluationCriteria.inter_i,
            EvaluationCriteria.inter_ii,
            EvaluationCriteria.inter_iii,
            EvaluationCriteria.inter_iv,
            EvaluationCriteria.inter_v,
            EvaluationCriteria.disc_i,
            EvaluationCriteria.disc_ii,
            EvaluationCriteria.disc_iii,
            EvaluationCriteria.stab_i,
            EvaluationCriteria.stab_ii
        ]

        inter_i = []
        inter_ii = []
        inter_iii = []
        inter_iv = []
        inter_v = []
        disc_i = []
        disc_ii = []
        disc_iii = []
        stab_i = []
        stab_ii = []

        # mean distances for level set 0 indexed by distance measure
        distances_for_level_set0 = self.per_level_set_distance_statistics_df.loc[
            self.per_level_set_distance_statistics_df[DistanceMeasureCols.level_set] == 0][
            [DistanceMeasureCols.type, Aggregators.mean]].round(3).set_index(DistanceMeasureCols.type, drop=True)

        # statistical diff between adjacent level sets
        ls_pairs = [(self.level_sets[i], self.level_sets[i + 1]) for i in range(len(self.level_sets) - 1)]
        ci_adjacent = self.ci_for_mean_differences.loc[
            (self.ci_for_mean_differences[DistanceMeasureCols.compared]).isin(ls_pairs)]

        # for each distance measure calculate all criteria
        for measure in self.__measures:
            inter_i.append(distances_for_level_set0.loc[measure, Aggregators.mean])
            inter_ii.append(ci_adjacent.loc[(ci_adjacent[DistanceMeasureCols.type] == measure)].eq('lower').all())
            inter_iii.append(0)
            inter_iv.append(0)
            inter_v.append(0)
            disc_i.append(0)
            disc_ii.append(0)
            disc_iii.append(0)
            stab_i.append(0)
            stab_ii.append(0)

        data = [
            inter_i,
            inter_ii,
            inter_iii,
            inter_iv,
            inter_v,
            disc_i,
            disc_ii,
            disc_iii,
            stab_i,
            stab_ii,
        ]
        return pd.DataFrame(data=data, columns=columns, index=indices)
