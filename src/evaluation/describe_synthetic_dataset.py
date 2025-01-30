import itertools
from dataclasses import dataclass
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.collections import EllipseCollection
from pandas._libs.tslibs import to_offset

from src.utils.labels_utils import calculate_n_segments_within_tolerance_for, \
    calculate_n_segments_outside_tolerance_for, find_all_level_sets
from src.utils.load_synthetic_data import load_synthetic_data, SyntheticDataType, load_labels_file_for
from src.utils.configurations import GeneralisedCols, SyntheticDataVariates, SYNTHETIC_DATA_DIR, \
    bad_partition_dir_for_data_type
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, fontsize, Backends, use_latex_labels
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols, min_max_scaled_df


@dataclass
class DescribeSyntheticCols:
    achieved_min: str = "achieved min"
    achieved_max: str = "achieved max"
    achieved_mean: str = "achieved mean"
    achieved_std: str = "achieved std"
    error_min: str = "error min"
    error_max: str = "error max"
    error_mean: str = "error mean"
    error_std: str = "error std"
    n_within_tolerance: str = "n within tolerance"
    n_outside_tolerance: str = "n outside tolerance"
    sum_mean_abs_error: str = "sum mean absolute error"
    correlation_matrix_col = "correlation matrix"
    n_segments = "n segments"


def to_correlation_matrix(corr_pairs: []):
    """ Creates the correlation matrix from a list or np.array of the np.triu without diagonal values"""
    N = int((1 + np.sqrt(1 + 8 * len(corr_pairs))) / 2)  # number of elements without diagonal N(N−1)/2
    corr_m = np.identity(N)  # diagonal 1 for correlation
    corr_m[np.triu_indices(N, k=1)] = corr_pairs  # assign array of upper correlations to the upper half
    corr_m[np.tril_indices(3, k=-1)] = corr_pairs  # assign array of upper correlations to the lower half
    return corr_m


class DescribeSyntheticDataset:
    def __init__(self, run_name: str, data_type: str = SyntheticDataType.non_normal_correlated,
                 data_cols: [str] = SyntheticDataVariates.columns(), value_range: (float, float) = None,
                 data_dir: str = SYNTHETIC_DATA_DIR, backend: str = Backends.none.value, bad_partition_name: str = ""):
        """
        :param run_name: the name of the wandb run that generated the dataset
        :param data_type: which data variation to load, by default most AID like SyntheticDataType.non_normal_correlated
        read e.g bad labels file than you can provide the full name without the '_labels.csv' for that file
        :param data_cols: which columns in the data df are the values excluding time
        :param value_range: if not None data will be min max scaled to the range provided before description
        :param data_dir: the root directory from which to read data, by default SYNTHETIC_DATA_DIR
        :param backend: backend to use
        :param bad_partition_name: file name for bad partition, will load data for run name and labels for bad partition
        """
        self.bad_partition_name = bad_partition_name
        self.backend = backend
        self.run_name = run_name
        self.data_type = data_type
        self.data_cols = data_cols
        self.data_dir = data_dir
        self.value_range = value_range
        self.data, self.labels = load_synthetic_data(self.run_name, self.data_type, data_dir=data_dir)
        if bad_partition_name is not "":
            # load labels for bad partition and drops the previous one!!
            bad_partitions_dir = bad_partition_dir_for_data_type(self.data_type, self.data_dir)
            bad_file_full_path = path.join(bad_partitions_dir, self.bad_partition_name)
            self.labels = load_labels_file_for(Path(bad_file_full_path))

        if self.value_range is not None:  # needs to scale data first
            self.data = min_max_scaled_df(self.data, scale_range=self.value_range, columns=self.data_cols)

        self.number_of_variates = len(self.data_cols)
        self.number_of_observations = self.data.shape[0]
        self.number_of_segments = self.labels.shape[0]
        self.start_date = self.data.iloc[0][GeneralisedCols.datetime]
        self.end_date = self.data.iloc[-1][GeneralisedCols.datetime]
        self.duration = self.end_date - self.start_date
        # for irregular data the frequency is not so clear so we're using the median frequency
        median_diff = pd.Series(self.data[GeneralisedCols.datetime]).diff().median()
        freq = to_offset(median_diff)
        self.frequency = freq.rule_code
        self.patterns = self.labels[SyntheticDataSegmentCols.pattern_id].unique().tolist()
        self.n_patterns = len(self.patterns)
        self.mae_stats = self.labels[SyntheticDataSegmentCols.mae].describe().round(3)
        self.patterns_stats = self.labels[SyntheticDataSegmentCols.pattern_id].value_counts().describe().round(3)

        self.n_segment_within_tolerance = calculate_n_segments_within_tolerance_for(self.labels)
        self.n_segment_outside_tolerance = calculate_n_segments_outside_tolerance_for(self.labels)

        # absolute errors
        correlation_to_model_as_numpy = np.array(self.labels[SyntheticDataSegmentCols.correlation_to_model].to_list())
        achieved_correlation_as_numpy = np.array(self.labels[SyntheticDataSegmentCols.actual_correlation].to_list())
        absolute_errors = np.absolute(np.array(correlation_to_model_as_numpy) - np.array(achieved_correlation_as_numpy))

        # dfs with pattern id col and achieved correlations for each variate with indices as columns
        correlations_df = pd.DataFrame(achieved_correlation_as_numpy)
        correlations_df[SyntheticDataSegmentCols.pattern_id] = self.labels[SyntheticDataSegmentCols.pattern_id]
        abs_errors_df = pd.DataFrame(absolute_errors)
        abs_errors_df[SyntheticDataSegmentCols.pattern_id] = self.labels[SyntheticDataSegmentCols.pattern_id]

        # Within tolerance with id and variate indices as columns
        within_tolerance_as_numpy = np.array(self.labels[SyntheticDataSegmentCols.actual_within_tolerance].to_list())
        within_tolerance_df = pd.DataFrame(within_tolerance_as_numpy)
        within_tolerance_df[SyntheticDataSegmentCols.pattern_id] = self.labels[SyntheticDataSegmentCols.pattern_id]

        # creates df from value counts with columns: SyntheticDataSegmentCols.pattern_id,
        # SyntheticDataSegmentCols.correlation_to_model, counts, 'length' (list of segment lengths for pattern),
        # min, max, mean, std achieved correlations, min, max, mean, std absolute error in correlation
        as_df = self.labels[[SyntheticDataSegmentCols.pattern_id]].value_counts().to_frame().reset_index()
        as_df.rename(columns={'count': DescribeSyntheticCols.n_segments}, inplace=True)

        # add list of segment length, achieved corr cols and error corr cols
        correlation_to_model = []
        seg_lengths_by_pattern = []
        achieved_min_results = []
        achieved_max_results = []
        achieved_mean_results = []
        achieved_std_results = []
        error_min_results = []
        error_max_results = []
        error_mean_results = []
        error_std_results = []
        n_within_tol_results = []
        n_outside_tol_results = []
        number_of_unique_pairs = len(set(itertools.combinations(list(range(self.number_of_variates)), 2)))
        var_columns = list(range(number_of_unique_pairs))

        for idx, row in as_df.iterrows():
            pattern_id = row[SyntheticDataSegmentCols.pattern_id]
            labels_for_id = self.labels[self.labels[SyntheticDataSegmentCols.pattern_id] == pattern_id]
            lengths = list(labels_for_id[SyntheticDataSegmentCols.length])
            seg_lengths_by_pattern.append(lengths)
            corr_to_model = labels_for_id.iloc[0][SyntheticDataSegmentCols.correlation_to_model]
            correlation_to_model.append(corr_to_model)
            corr_stats = correlations_df[correlations_df[SyntheticDataSegmentCols.pattern_id] == pattern_id][
                var_columns].describe().round(4)
            achieved_min_results.append(list(corr_stats.loc['min']))
            achieved_max_results.append(list(corr_stats.loc['max']))
            achieved_mean_results.append(list(corr_stats.loc['mean']))
            achieved_std_results.append(list(corr_stats.loc['std']))

            error_stats = abs_errors_df[abs_errors_df[SyntheticDataSegmentCols.pattern_id] == pattern_id][
                var_columns].describe().round(4)
            error_min_results.append(list(error_stats.loc['min']))
            error_max_results.append(list(error_stats.loc['max']))
            error_mean_results.append(list(error_stats.loc['mean']))
            error_std_results.append(list(error_stats.loc['std']))

            count_tol_df = within_tolerance_df[within_tolerance_df[SyntheticDataSegmentCols.pattern_id] == pattern_id][
                var_columns]
            total = number_of_unique_pairs * [count_tol_df.shape[0]]
            count_trues = list(count_tol_df.sum())
            count_falses = [abs(xi - yi) for xi, yi in zip(total, count_trues)]
            n_within_tol_results.append(count_trues)
            n_outside_tol_results.append(count_falses)

        as_df[SyntheticDataSegmentCols.correlation_to_model] = correlation_to_model
        as_df[SyntheticDataSegmentCols.length] = seg_lengths_by_pattern
        as_df[DescribeSyntheticCols.achieved_min] = achieved_min_results
        as_df[DescribeSyntheticCols.achieved_max] = achieved_max_results
        as_df[DescribeSyntheticCols.achieved_mean] = achieved_mean_results
        as_df[DescribeSyntheticCols.achieved_std] = achieved_std_results
        as_df[DescribeSyntheticCols.error_min] = error_min_results
        as_df[DescribeSyntheticCols.error_max] = error_max_results
        as_df[DescribeSyntheticCols.error_mean] = error_mean_results
        as_df[DescribeSyntheticCols.error_std] = error_std_results
        as_df[DescribeSyntheticCols.n_within_tolerance] = n_within_tol_results
        as_df[DescribeSyntheticCols.n_outside_tolerance] = n_outside_tol_results
        as_df[DescribeSyntheticCols.sum_mean_abs_error] = [sum(e) for e in error_mean_results]

        # correlations using spearman correlations! Less impacted by distribution
        self.correlation_patterns_df = as_df

        # create segments correlation df
        # index is segment_id, actual correlation, correlation_matrix
        np_correlations = list(achieved_correlation_as_numpy)
        corr_matrices = [to_correlation_matrix(corr) for corr in np_correlations]
        self.segment_correlations_df = pd.DataFrame({DescribeSyntheticCols.correlation_matrix_col: corr_matrices,
                                                     SyntheticDataSegmentCols.actual_correlation: np_correlations
                                                     })
        self.segment_correlations_df[SyntheticDataSegmentCols.pattern_id] = self.labels[
            SyntheticDataSegmentCols.pattern_id]

        self.patterns_with_zero_mean_error_df = self.correlation_patterns_df[
            self.correlation_patterns_df[DescribeSyntheticCols.sum_mean_abs_error] == 0][[
            SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.correlation_to_model,
            DescribeSyntheticCols.achieved_mean, SyntheticDataSegmentCols.length, DescribeSyntheticCols.n_segments]]

        sum_mean_absolute_error = self.correlation_patterns_df[DescribeSyntheticCols.sum_mean_abs_error]
        self.sum_mean_absolute_error_stats = sum_mean_absolute_error.describe().round(4)
        self.patterns_with_sum_mean_error_max_df = self.correlation_patterns_df[
            self.correlation_patterns_df[DescribeSyntheticCols.sum_mean_abs_error] >= sum_mean_absolute_error.max()]
        self.patterns_with_sum_mean_error_min_df = self.correlation_patterns_df[
            self.correlation_patterns_df[DescribeSyntheticCols.sum_mean_abs_error] <= sum_mean_absolute_error.min()]
        self.patterns_with_sum_mean_error_smaller_equal_mean_df = self.correlation_patterns_df[
            self.correlation_patterns_df[DescribeSyntheticCols.sum_mean_abs_error] <= sum_mean_absolute_error.mean()]
        self.patterns_with_sum_mean_error_bigger_than_mean_df = self.correlation_patterns_df[
            self.correlation_patterns_df[DescribeSyntheticCols.sum_mean_abs_error] > sum_mean_absolute_error.mean()]

        self.segment_length_stats = self.labels[SyntheticDataSegmentCols.length].describe().round(3)
        self.segment_length_counts = self.labels[SyntheticDataSegmentCols.length].value_counts()
        self.observations_stats = self.data[SyntheticDataVariates.columns()].describe().round(3)

        out_tol = np.array(list(self.correlation_patterns_df[DescribeSyntheticCols.n_outside_tolerance]))
        cols_comb = itertools.combinations(self.data_cols, 2)
        # tolerance frame is <=0.2 for 0 and >= 0.7
        self.n_segment_outside_tolerance_df = pd.DataFrame(out_tol, columns=cols_comb)
        pattern_id_col = SyntheticDataSegmentCols.pattern_id
        self.n_segment_outside_tolerance_df.insert(loc=0, column=pattern_id_col,
                                                   value=self.correlation_patterns_df[pattern_id_col])
        self.n_segment_outside_tolerance_df.insert(loc=1, column=DescribeSyntheticCols.n_segments,
                                                   value=self.correlation_patterns_df[DescribeSyntheticCols.n_segments])
        self.n_segment_outside_tolerance_df = self.n_segment_outside_tolerance_df.sort_values(
            by=pattern_id_col).reset_index(drop=True)

        # dictionary of key level set id and values list of tuples of pattern pairs
        self.level_sets = find_all_level_sets(self.labels)
        # list index is level set id, value is a list of all pattern pairs as tuple in that level set
        self.patterns_by_level_set = list(self.level_sets.values())

        # dictionary with key being pattern id and values being all the segment ids in that pattern
        segments_in_pattern = {}
        for pattern in self.patterns:
            # get segment ids for this pattern
            segment_id = self.segment_correlations_df[
                self.segment_correlations_df[SyntheticDataSegmentCols.pattern_id] == pattern].index.tolist()
            segments_in_pattern[pattern] = segment_id
        self.segments_for_each_pattern = segments_in_pattern

        # dictionary with keys being level sets 0-5 and values being tuples of segment pairs in this level set
        self.segment_pairs_for_level_sets = self.__create_segment_pairs_for_each_level_set()

        # calculate data by pattern id
        self.data_by_pattern_id = self.__create_dictionary_of_data_by_pattern_id()

    def __create_segment_pairs_for_each_level_set(self):
        """
        Creates a dictionary with the key being the level set id 0-5 and the value being a list of segment id tuples
        to compare for this level set
        """
        result = {}  # key is level set id, value is list of segment tuples to calculate distance inbetween
        # get all the segment ids for each pattern
        segments_in_pattern = self.segments_for_each_pattern

        # all possible pairs of segments in the same cluster
        for group_id, pattern_pairs in self.level_sets.items():
            if group_id == 0:  # don't want to double the segments x,y and y,x for level set 0 where the pattern is the same
                result[group_id] = list(itertools.chain.from_iterable(
                    [list(itertools.combinations(segments, 2)) for p, segments in segments_in_pattern.items()]))
            else:
                result[group_id] = list(itertools.chain.from_iterable(
                    [list(itertools.product(segments_in_pattern[p1], segments_in_pattern[p2])) for p1, p2 in
                     pattern_pairs]))

        return result

    def plot_correlation_matrix_for_each_pattern(self, method: str = "spearman", plot_diagonal=False, show_title=True):
        """ Plot's the actual correlation matrix for each of the 23 patterns
        Method: pandas corr method ‘pearson’, ‘kendall’, ‘spearman’ or callable
        """
        # setup figure
        reset_matplotlib(self.backend)
        fig_size = (20, 20)
        no_rows = 5
        fig, axs = plt.subplots(nrows=no_rows,
                                ncols=no_rows,
                                sharey=True,
                                sharex=True,
                                figsize=fig_size)
        cmap = "bwr"
        normed_colour = mpl.colors.Normalize(-1, 1)
        patterns = self.patterns.copy()
        patterns.sort()

        pattern_idx = 0
        for rdx in range(no_rows):
            for cdx in range(no_rows):
                ax = axs[rdx][cdx]

                # turn grid off
                ax.grid(False, which='major')
                ax.grid(False, which='minor')
                ax.tick_params(length=0)

                # get all segment data for this pattern
                pattern = patterns[pattern_idx]
                df = self.data_by_pattern_id[pattern]
                correlation = df.corr(method=method)
                plot_corr_ellipses(correlation, plot_diagonal=plot_diagonal, show_top_labels=(rdx == 0), ax=ax,
                                   cmap=cmap, norm=normed_colour)
                ax.margins(0.1)

                # pattern as x-label
                label = str(pattern) + ": " + str(
                    self.labels[self.labels[SyntheticDataSegmentCols.pattern_id] == pattern].iloc[0][
                        SyntheticDataSegmentCols.correlation_to_model])
                ax.set_xlabel(label, rotation=0, size=fontsize)

                # all patterns have been plotted - need to break both row and column loop
                if pattern_idx == len(patterns) - 1:
                    break
                pattern_idx += 1

            # all patterns have been plotted - need to break both row and column loop
            if pattern_idx == len(patterns) - 1:
                break

        axs[4, 4].axis('off')
        axs[4, 3].axis('off')

        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        pos = axs[4, 4].get_position()
        cbar_ax = fig.add_axes([pos.x0,
                                pos.y0 - 0.08,
                                pos.width,
                                pos.height])
        cbar_ax.axis('off')
        fig.colorbar(mpl.cm.ScalarMappable(norm=normed_colour, cmap=cmap),
                     ax=cbar_ax, orientation='vertical', label='Correlation coefficient')

        plt.tight_layout()
        plt.show()
        return fig

    def __create_dictionary_of_data_by_pattern_id(self):
        # dictionary with key being pattern_id end value being df  of data for that pattern id
        data_by_pattern_id = {}
        data_df = self.data[self.data_cols]
        # iterate through all segments and store the data for that data in a dataframe
        for idx, row in self.labels.iterrows():
            start_idx = row[SyntheticDataSegmentCols.start_idx]
            end_idx = row[SyntheticDataSegmentCols.end_idx]
            length = row[SyntheticDataSegmentCols.length]
            pattern_id = row[SyntheticDataSegmentCols.pattern_id]

            # select data
            segment_df = data_df.iloc[start_idx:end_idx + 1]
            segment = segment_df.to_numpy()
            assert segment.shape[0] == length, "Mistake with indexing dataframe"

            # store data by pattern id
            if pattern_id in data_by_pattern_id.keys():
                current_df = data_by_pattern_id[pattern_id]
                data_by_pattern_id[pattern_id] = pd.concat([current_df, segment_df])
            else:
                data_by_pattern_id[pattern_id] = segment_df

        return data_by_pattern_id

    def plot_example_correlation_matrix_for_each_subgroup(self, pattern_id: int = 15, order_groups: [] = [],
                                                          plot_diagonal: bool = False):
        """
        Plot example seg1 and seg2 correlation matrix on rows for the 5 subgroups (columns)
        """
        # setup figure
        reset_matplotlib(self.backend)
        use_latex_labels()
        widths = [2, 2, 2, 2, 2, 2, 0.5]
        heights = [1, 1]
        fig_size = (15, 5)
        no_rows = 2
        no_cols = 7

        fig = plt.figure(constrained_layout=True, figsize=fig_size)
        grid = fig.add_gridspec(ncols=no_cols, nrows=no_rows, width_ratios=widths, height_ratios=heights, wspace=0.3)

        cmap = "bwr"
        normed_colour = mpl.colors.Normalize(-1, 1)

        # index is level set index, tuple 0 is pattern for seg 1, tuple 1 is pattern for seg 2
        # find pattern for each level set for given pattern_id
        patterns_default_order = []
        for patterns in self.patterns_by_level_set:
            patterns_with_pattern_id = list(filter(lambda x: pattern_id in x, patterns))
            # take the first
            patterns_default_order.append(patterns_with_pattern_id[0])

        if len(order_groups) == 0:
            patterns_to_plot = patterns_default_order
        else:  # order patterns
            patterns_to_plot = [patterns_default_order[index] for index in order_groups]

        # columns are the level sets, level_set_id is also column id
        for level_set_id, patterns_in_level_set in enumerate(patterns_to_plot):
            # rows are segments (think the different patterns in tuple)
            # row 1 is the pattern_id, row 2 is the other pattern, all pattern pairs have smaller pattern first
            pattern_1 = pattern_id
            pattern_2 = patterns_in_level_set[0] if patterns_in_level_set[0] != pattern_id else patterns_in_level_set[1]

            # get data for patterns
            df_1 = self.data_by_pattern_id[pattern_1]
            df_2 = self.data_by_pattern_id[pattern_2]
            corr1 = df_1.corr()
            corr2 = df_2.corr()

            # create axes for plots
            ax1 = fig.add_subplot(grid[0, level_set_id])
            ax2 = fig.add_subplot(grid[1:, level_set_id])

            plot_corr_ellipses(corr1, plot_diagonal=plot_diagonal, ax=ax1, cmap=cmap, norm=normed_colour)
            plot_corr_ellipses(corr2, plot_diagonal=plot_diagonal, ax=ax2, cmap=cmap, norm=normed_colour)

            # turn x-tick labels off in row 2
            ax2.get_xaxis().set_ticklabels([])

            # turn y-tick labels off for col > 0
            if level_set_id > 0:
                ax1.get_yaxis().set_visible(False)
                ax2.get_yaxis().set_visible(False)
            else:
                # add y labels
                ax1.set_ylabel("Segment A", fontsize=fontsize)
                ax2.set_ylabel("Segment B", fontsize=fontsize)

            # pattern as x-label
            label_1 = str(pattern_1) + ": " + str(
                self.labels[self.labels[SyntheticDataSegmentCols.pattern_id] == pattern_1].iloc[0][
                    SyntheticDataSegmentCols.correlation_to_model])
            ax1.set_xlabel(label_1, rotation=0, size=fontsize)
            label_2 = r'' + str(pattern_2) + ': ' + str(
                self.labels[self.labels[SyntheticDataSegmentCols.pattern_id] == pattern_2].iloc[0][
                    SyntheticDataSegmentCols.correlation_to_model]) + '\n\n' + '$d_' + str(level_set_id) + '$'
            ax2.set_xlabel(label_2, rotation=0, size=fontsize)

        # plot colorbar in end column over both rows
        cbar_ax = fig.add_subplot(grid[:, -1])
        fig.colorbar(mpl.cm.ScalarMappable(norm=normed_colour, cmap=cmap), cax=cbar_ax, orientation='vertical')

        plt.show()
        return fig

    def x_and_y_of_patterns_modelled(self):
        """ Returns feature matrix X and class label vector y. X has each correlation pair as column and returns each
         segments actual correlation and y has the class label for each row in X that has been modeled. (y_true)
        """
        y = self.labels[SyntheticDataSegmentCols.pattern_id].to_numpy()
        x = np.array(self.labels[SyntheticDataSegmentCols.actual_correlation].to_list())
        return x, y


def plot_corr_ellipses(corr_df: pd.DataFrame, ax, plot_diagonal=False, show_top_labels=True, **kwargs):
    gap = 0.1  # space between ellipses on the grid

    cor = corr_df.to_numpy()
    n = cor.shape[0]
    # if diagonal is included or not
    k = 0 if plot_diagonal else 1

    # list of xy locations for each ellipse center
    upper_indices = np.triu_indices(n, k=k)
    # invert row index due to matplotlib grid starting at lower left corner unlike upper left like matrix indices
    inverted_row_idx = [abs(i - upper_indices[0][-1]) for i in upper_indices[0]]
    column_idx = upper_indices[1]
    # xy location of where to paint the ellipses, they are painted in matrix order top left corner first
    # if no diagonal is plotted it is the index-1 for y
    diag_correction = 0 if plot_diagonal else 1
    xy = [[(j + (j - diag_correction) * gap) - diag_correction, (i + i * gap)] for i, j in
          zip(inverted_row_idx, column_idx)]

    # set the relative sizes of the major/minor axes according to the strength of
    # the positive/negative correlation
    w = np.ones(len(xy))
    h = 1 - np.abs(cor[upper_indices])
    a = 45 * np.sign(cor[upper_indices])

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           transOffset=ax.transData, array=cor[upper_indices], edgecolor='black', **kwargs)
    ax.add_collection(ec)

    # Modify the tick params based on show_top_labels first row only
    if show_top_labels:
        ax.xaxis.set_tick_params(labeltop=True)
        ax.xaxis.set_tick_params(labelbottom=False)
    else:
        ax.xaxis.set_tick_params(labeltop=False)
        ax.xaxis.set_tick_params(labelbottom=False)

    n_ticks = n if plot_diagonal else n - k
    ticks = [tick + tick * gap for tick in np.arange(n_ticks)]
    ax.set_xticks(ticks)
    x_labels = corr_df.columns if plot_diagonal else corr_df.columns[1:]
    ax.set_xticklabels(x_labels)
    ax.set_yticks(ticks)
    reversed_cols = list(reversed(corr_df.index))
    y_labels = reversed_cols if plot_diagonal else reversed_cols[1:]
    ax.set_yticklabels(y_labels)

    rounded_cor = list(cor[upper_indices].round(2))
    # add correlation label
    for cidx, text in enumerate(rounded_cor):
        y_idx = inverted_row_idx[cidx]  # for n=3 this is 1, 1, 0
        x_idx = column_idx[cidx] - 1  # for n=3 this is 0, 1, 1
        ax.text(ticks[x_idx] - 2 * gap, ticks[y_idx], text, fontsize=fontsize, color='silver', zorder=10)

    # each tick is 1 wide
    half_w = 0.5
    ax.set_xlim(-half_w - gap, n_ticks - half_w + (n_ticks * gap))
    ax.set_ylim(-half_w - gap, n_ticks - half_w + (n_ticks * gap))
    ax.autoscale_view()
    return ec
