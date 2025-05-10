import re
from os import path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl

from src.data_generation.generate_synthetic_correlated_data import generate_correlation_matrix
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols, CorrType
from src.evaluation.describe_subjects_for_data_variant import DescribeSubjectsForDataVariant
from src.evaluation.describe_synthetic_dataset import plot_corr_ellipses
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import get_irregular_folder_name_from, base_dataset_result_folder_for_type, ResultsType, \
    OVERALL_SEGMENT_LENGTH_IMAGE, OVERALL_MAE_IMAGE
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, fontsize


def custom_number_text_formatter(x, _):
    abs_x = abs(x)
    if abs_x > 10000:
        return f'{x:.0e}'  # Scientific notation for large numbers
    elif abs_x >= 10:  # Regular numbers no decimal
        return f'{x:.0f}'
    elif abs_x >= 1:  # Regular numbers show one decimal but not .0
        return f'{x:.1f}'.rstrip('0').rstrip('.')
    elif abs_x >= 0.1:  # Decimal numbers
        return f'{x:.1f}'
    elif abs_x >= 0.01:  # Decimal numbers
        return f'{x:.2f}'
    elif abs_x >= 0.001:  # Decimal numbers
        return f'{x:.3f}'
    elif abs_x > 0:  # Very small numbers
        return f'{x:.0e}'  # Scientific notation for tiny numbers no d
    elif abs_x == 0:
        return '0'
    else:
        return f'{x:.1f}'  # Regular format for other numbers


def add_stats(data: np.ndarray, ax: plt.Axes, y_pos: float, fix_x_axis:bool=False) -> None:
    """Add min, max and mean annotations to the plot."""
    mean = np.mean(data)
    min_val = np.min(data)
    max_val = np.max(data)

    # Add mean annotation centered above the box
    x_center = np.mean(ax.get_xlim())
    ax.text(x_center, y_pos + 0.2, f'μ={custom_number_text_formatter(mean, "-")}',
            verticalalignment='center', horizontalalignment='center',
            fontsize=fontsize, fontweight='bold')

    # Add min/max annotations with adjusted positions
    ax.text(min_val, y_pos - 0.3, f'min={custom_number_text_formatter(min_val, "-")}',
            verticalalignment='center', horizontalalignment='left',
            fontsize=fontsize)
    max_pos = max_val
    if fix_x_axis:
        max_pos = min(max_val, 0.5)
    ax.text(max_pos, y_pos - 0.15, f'max={custom_number_text_formatter(max_val, "-")}',
            verticalalignment='center', horizontalalignment='left',
            fontsize=fontsize)


def get_row_name_from(folder):
    irr_folder_name = get_irregular_folder_name_from(folder)
    if irr_folder_name == "":
        return "Complete 100%"
    # change _p30 to partial 30%
    match_no = re.search(r'p(\d+)$', irr_folder_name)
    if match_no:
        number = int(match_no.group(1)) / 100
        if number == 0.3:
            return "Partial 70%"
        elif number == 0.9:
            return "Sparse 10%"
    assert False, "Unknown folder extension in: " + folder


def create_violin_grid_log_scale_x_axis(data_dict: {}, figsize: tuple = (18, 15),
                                        backend: str = Backends.none.value) -> plt.Figure:
    """
    Create a grid of horizontal violin plots with statistics using dictionary keys as labels.
    Source - mostly claude with some modifications from me

    :param data_dict : {str, {str, np.ndarray}} Nested dictionary containing data for each plot square
        Format: {'row_name': {'column_name': data_array}}
    :param figsize : tuple, optional Figure size in inches (width, height)
    :returns: matplotlib.figure.Figure
    Save like plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    """

    # Extract row and column names from dictionary structure
    row_names = list(data_dict.keys())
    # Get unique column names from all nested dictionaries
    column_names = list(list(data_dict.values())[0].keys())

    # Set the style for publication-quality figures
    reset_matplotlib(backend)
    # sns.set_context("paper", font_scale=1.2)

    # Create figure with custom size
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(row_names), len(column_names), figure=fig)
    fig.subplots_adjust(hspace=0.1, wspace=0.4, left=0.15)

    # Set up colors
    colors = sns.color_palette("husl", n_colors=len(column_names))

    # Find global and column min and max for shared x-axis
    global_min = float('inf')
    global_max = float('-inf')
    for row_data in data_dict.values():
        for col_data in row_data.values():
            global_min = min(global_min, np.min(col_data))
            global_max = max(global_max, np.max(col_data))

    # Add padding to global limits (multiplicative for log scale)
    global_min = global_min * 0.9  # 10% padding on log scale
    global_max = global_max * 1.1

    # Plot each subplot
    axes = []
    for i, row in enumerate(row_names):
        row_axes = []
        for j, col in enumerate(column_names):
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)

            try:
                data = data_dict[row][col]
            except KeyError:
                print(f"Warning: No data found for condition: row='{row}', column='{col}'")
                continue

            # Set x-axis to logarithmic scale
            ax.set_xscale('log')

            # Create violin plot with cut=0 to prevent extending beyond data range
            sns.violinplot(data=data, orient='h', color=colors[j], alpha=0.6, ax=ax, cut=0)

            # Add box plot inside violin
            sns.boxplot(data=data, orient='h', width=0.2, color='white',
                        showfliers=False, ax=ax)

            # Customize appearance
            if i == 0:  # Add column titles only to the top row
                ax.set_title(col, fontsize=fontsize, fontweight='bold')

            # Set x-axis limits to be the same for all plots
            ax.set_xlim(global_min, global_max)

            # Format x-axis labels to be more readable
            ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_number_text_formatter))

            # Remove y-ticks and labels
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row, fontsize=fontsize, fontweight='bold')

            # Only show x-axis elements for bottom row
            if i == len(row_names) - 1:
                plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])  # Hide x-tick labels
                ax.tick_params(axis='x', which='both', length=0)  # Hide tick marks but keep grid
                # ax.set_xticks([])  # Remove x-axis ticks completely

            # Add prominent vertical grid lines
            ax.grid(True, axis='x', linestyle='-', alpha=0.3, which='both')
            ax.grid(True, axis='x', linestyle='-', alpha=0.6, which='major')

            # Customize spines
            sns.despine(ax=ax, left=True)

            # Add statistics
            add_stats(data, ax, 0)

        axes.append(row_axes)

    fig.supxlabel("x-axes log scale", fontsize=fontsize)

    # Adjust layout
    plt.tight_layout()
    return fig


def create_violin_grid(data_dict: {}, figsize: tuple = (12, 12), fix_x_axis: bool = False, x_lim: float = 0.9,
                       backend: str = Backends.none.value) -> plt.Figure:
    """
    Create a grid of horizontal violin plots with statistics using dictionary keys as labels.
    Source: developed with the help of claude and a lot of input from me

    :param data_dict : {str, {str, np.ndarray}} Nested dictionary containing data for each plot square
        Format: {'row_name': {'column_name': data_array}}
    :param figsize : tuple, optional Figure size in inches (width, height)
    :returns: matplotlib.figure.Figure
    Save like plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    """

    # Extract row and column names from dictionary structure
    row_names = list(data_dict.keys())
    # Get unique column names from all nested dictionaries
    column_names = list(list(data_dict.values())[0].keys())

    # Set the style for publication-quality figures
    reset_matplotlib(backend)

    # Create figure with custom size
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(row_names), len(column_names), figure=fig)
    fig.subplots_adjust(hspace=0.5, wspace=0.6, left=0.15)

    # Set up colors
    colors = sns.color_palette("husl", n_colors=len(column_names))

    # Find global min and max for shared x-axis
    global_min = float('inf')
    global_max = float('-inf')
    for row_data in data_dict.values():
        for col_data in row_data.values():
            global_min = min(global_min, np.min(col_data))
            global_max = max(global_max, np.max(col_data))

    # Add padding to global limits
    global_min = global_min - 0.2
    global_max = global_max + 0.2

    # Plot each subplot
    axes = []
    for i, row in enumerate(row_names):
        row_axes = []
        for j, col in enumerate(column_names):
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)

            try:
                data = data_dict[row][col]
            except KeyError:
                print(f"Warning: No data found for condition: row='{row}', column='{col}'")
                continue

            # Create violin plot with cut=0 to prevent extending beyond data range
            sns.violinplot(data=data, orient='h', color=colors[j], alpha=0.6, ax=ax, cut=0)

            # Add box plot inside violin
            sns.boxplot(data=data, orient='h', width=0.2, color='white',
                        showfliers=False, ax=ax)

            # Customize appearance
            if i == 0:  # Add column titles only to the top row
                ax.set_title(col, fontsize=fontsize, fontweight='bold')

            # Set x-axis limits to be the same for all plots
            if fix_x_axis:
                ax.set_xlim(0, x_lim)
            else:
                ax.set_xlim(global_min, global_max)

            # Format x-axis labels to be more readable
            ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_number_text_formatter))

            # Remove y-ticks and labels
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row, fontsize=fontsize, fontweight='bold')

            # Only show x-axis elements for bottom row
            if i == len(row_names) - 1:
                plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])  # Hide x-tick labels
                ax.tick_params(axis='x', which='both', length=0)  # Hide tick marks but keep grid

            # Add prominent vertical grid lines
            ax.grid(True, axis='x', linestyle='-', alpha=0.5, which='major', color='gray')  # major grid
            ax.grid(True, axis='x', linestyle='--', alpha=0.3, which='minor', color='gray')  # minor grid
            ax.xaxis.set_minor_locator(AutoMinorLocator())  # minor grid spacing

            # Customize spines
            sns.despine(ax=ax, left=True)

            # Add statistics
            add_stats(data, ax, 0, fix_x_axis)

        axes.append(row_axes)

    # Adjust layout
    plt.tight_layout()
    return fig


def create_correlation_grid(data_dict: {}, figsize: tuple = (12, 12), backend: str = Backends.none.value) -> plt.Figure:
    """
    Create a grid of correlation matrix visualisations using dictionary keys as labels.

    :param data_dict : {str, {str, np.ndarray}} Nested dictionary containing correlation data for each data variant
        Format: {'row_name': {'column_name': data_array}} - rows are complete, partial, sparse
    :param figsize : tuple, optional Figure size in inches (width, height)
    :returns: matplotlib.figure.Figure
    """

    # Extract row and column names from dictionary structure
    row_names = list(data_dict.keys())
    # Get unique column names from all nested dictionaries
    column_names = list(list(data_dict.values())[0].keys())

    # Set the style for publication-quality figures
    reset_matplotlib(backend)

    # Create figure with custom size
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(row_names), len(column_names), figure=fig)
    fig.subplots_adjust(hspace=0.5, wspace=0.3, left=0.15)

    # Set up colors
    cmap = "bwr"
    normed_colour = mpl.colors.Normalize(-1, 1)

    # Plot each subplot
    axes = []
    for i, row in enumerate(row_names):
        row_axes = []
        for j, col in enumerate(column_names):
            ax = fig.add_subplot(gs[i, j])
            ax.set_aspect('equal')
            row_axes.append(ax)

            # turn grid off
            ax.grid(False, which='major')
            ax.grid(False, which='minor')
            ax.tick_params(length=0)
            # Light square around each subplot
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)  # Make thinner
                spine.set_color('#cccccc')  # Light gray color

            try:
                correlations = data_dict[row][col]
            except KeyError:
                print(f"Warning: No correlations found for condition: row='{row}', column='{col}'")
                continue

            # create correlation plot
            plot_corr_ellipses(correlations, plot_diagonal=False, show_top_labels=False, ax=ax,
                               cmap=cmap, norm=normed_colour)
            ax.margins(0.1)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.yaxis.set_tick_params(labelright=False)

            # Customize appearance
            if i == 0:  # Add column titles only to the top row
                ax.set_title(col, fontsize=fontsize, fontweight='bold')

            if j == 0:  # Add row titles only for the first column
                ax.set_ylabel(row, fontsize=fontsize, fontweight='bold')

        axes.append(row_axes)

    # Adjust layout
    plt.tight_layout()
    return fig


def create_scatter_grid(data_dict: {}, measure_cols: list, figsize: tuple = (12, 12),
                        backend: str = Backends.none.value) -> plt.Figure:
    """
    Create a grid of scatter plots using dictionary keys as labels.

    :param data_dict : {str, {str, pd.DataFrame}} Nested dictionary containing data for each plot square
        Format: {'row_name': {'column_name': list_of_dataframes}}
        Where each DataFrame has the measures as columns and columns refer to the data generation stages and
        rows refer to the data completeness
    :param measure_cols : list of str, columns to plot from the DataFrames
    :param figsize : tuple, optional Figure size in inches (width, height)
    :returns: matplotlib.figure.Figure
    """

    # Extract row and column names from dictionary structure
    row_names = list(data_dict.keys())
    column_names = list(list(data_dict.values())[0].keys())

    # Set the style for publication-quality figures
    reset_matplotlib(backend)

    # Create figure with standard size (no extra width needed now)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(row_names), len(column_names))
    fig.subplots_adjust(hspace=0.5, wspace=0.3, right=0.85)

    # Set up colors for different measures
    measure_colors = sns.color_palette("husl", n_colors=len(measure_cols))
    markers = ['o', 'x', 's', '^']  # Marker styles

    # Plot each subplot
    axes = []
    legend_handles = []  # Will store handles for the legend
    for i, completeness in enumerate(row_names):
        row_axes = []
        for j, generation_stage in enumerate(column_names):
            ax = fig.add_subplot(gs[i, j])
            ax2 = ax.twinx()  # Create secondary axis

            row_axes.append(ax)

            try:
                dfs = data_dict[completeness][generation_stage]
            except KeyError:
                print(f"Warning: No data found for condition: row='{completeness}', column='{generation_stage}'")
                continue

            # find min_value across all subjects
            min_subject_val = np.inf
            max_subject_val = -np.inf
            for df in dfs:
                min_val = df[measure_cols[1]].min()
                max_val = df[measure_cols[1]].max()
                if min_val < min_subject_val:
                    min_subject_val = min_val
                if max_val > max_subject_val:
                    max_subject_val = max_val

            # Plot each measure for each DataFrame
            for df in dfs:
                # min_val = df[measure_cols[1]].min()
                # print(", ".join([completeness, generation_stage, str(min_val)]))
                ax.scatter(df.index, df[measure_cols[0]],
                           alpha=0.6,
                           color=measure_colors[0],
                           s=20,
                           marker=markers[0])

                ax2.scatter(df.index, df[measure_cols[1]],
                            alpha=0.6,
                            color=measure_colors[1],
                            s=20,
                            marker=markers[1])
                # For all but silhouette we must make sure the axis doesn't look 0 when it isn't
                # and that we do not hide outliers which come from the centroid problems
                if measure_cols[1] != ClusteringQualityMeasures.silhouette_score:
                    ax2.set_ylim(min_subject_val, max_subject_val)
                    ax2.yaxis.set_major_locator(plt.LinearLocator(5))
                    ticks = ax2.get_yticks()
                    ticks[0] = min_subject_val
                    # if next tick too close to new min tick, remove it
                    if abs(ticks[1] - min_subject_val) < 3:
                        ticks = np.delete(ticks, 1)
                    ax2.set_yticks(ticks)

            # Customize appearance
            if i == 0:  # Add column titles only to the top row
                ax.set_title(generation_stage, fontsize=fontsize, fontweight='bold')

            # Set axis limits for silhouette to range of the measure
            if measure_cols[1] == ClusteringQualityMeasures.silhouette_score:
                # the other measures have outliers and so we cannot use this
                ax2.set_ylim(-1, 1)  # Set secondary axis limits
            ax.set_xlim(-2, 69)  # slight padding around 0-67

            # Format axis labels
            ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_number_text_formatter))
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(custom_number_text_formatter))

            ax2.grid(False)  # Turn off secondary grid to avoid double grid

            if j == 0:
                ax.set_ylabel(completeness, fontsize=fontsize, fontweight='bold')

            # Only show x-axis elements for bottom row
            if i == len(row_names) - 1:
                plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', length=0)

            # Add grid
            ax.grid(True, linestyle='-', alpha=0.3, which='major')
            ax.grid(True, linestyle='--', alpha=0.2, which='minor')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Customize spines
            sns.despine(ax=ax)

        axes.append(row_axes)

    # Add legend directly in the last subplot
    for k, measure in enumerate(measure_cols):
        measure_name = ClusteringQualityMeasures.get_display_name_for_measure(measure)
        legend_handles.append(plt.Line2D([0], [0],
                                         marker=markers[k],
                                         color=measure_colors[k],
                                         label=measure_name,
                                         markersize=8,
                                         linestyle='None'))
    last_ax = axes[-1][-1]
    last_ax.legend(handles=legend_handles,
                   loc='upper left',
                   fontsize=fontsize - 2,
                   frameon=True,
                   edgecolor='black',
                   facecolor='white')

    # Adjust layout
    plt.tight_layout()
    return fig


def create_scatter_row(data_dict: {}, reference_measure: str, measure_cols: [str],
                       data_type: str, completeness: str, figsize: tuple = (18, 4),
                       backend: str = Backends.none.value) -> plt.Figure:
    """
    Create a row of scatter plots using dictionary keys as labels.

    :param data_dict : {str, {str, pd.DataFrame}} Nested dictionary containing data for each plot square
        Format: {'row_name': {'column_name': list_of_dataframes}}
        Where each DataFrame has the measures as columns and columns refer to the data generation stages and
        rows refer to the data completeness
    :param measure_cols : list of str, columns to plot from the DataFrames
    :param figsize : tuple, optional Figure size in inches (width, height)
    :returns: matplotlib.figure.Figure
    """
    # Set the style for publication-quality figures
    reset_matplotlib(backend)

    # Create figure with standard size (no extra width needed now)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, len(measure_cols))
    fig.subplots_adjust(hspace=0.5, wspace=0.3, right=0.85)

    # Set up colors for different measures
    measure_colors = sns.color_palette("husl", n_colors=2)
    markers = ['o', 'x']  # Marker styles

    for i, measure in enumerate(measure_cols):
        ax = fig.add_subplot(gs[0, i])
        ax2 = ax.twinx()  # Create secondary axis

        try:
            row_name = get_row_name_from(completeness)
            col_name = SyntheticDataType.get_display_name_for_data_type(data_type)
            dfs = data_dict[row_name][col_name]
        except KeyError:
            print(f"Warning: No data found for condition: row='{completeness}', column='{data_type}'")
            continue

        # find min_value across all subjects
        min_subject_val = np.inf
        max_subject_val = -np.inf
        for df in dfs:
            min_val = df[measure].min()
            max_val = df[measure].max()
            if min_val < min_subject_val:
                min_subject_val = min_val
            if max_val > max_subject_val:
                max_subject_val = max_val

        # Plot each measure for each DataFrame
        for df in dfs:
            ax.scatter(df.index, df[reference_measure],
                       alpha=0.6,
                       color=measure_colors[0],
                       s=20,
                       marker=markers[0])

            ax2.scatter(df.index, df[measure],
                        alpha=0.6,
                        color=measure_colors[1],
                        s=20,
                        marker=markers[1])
            # For all but silhouette we must make sure the axis doesn't look 0 when it isn't
            # and that we do not hide outliers which come from the centroid problems
            if measure_cols[1] != ClusteringQualityMeasures.silhouette_score:
                ax2.set_ylim(min_subject_val, max_subject_val)
                ax2.yaxis.set_major_locator(plt.LinearLocator(5))
                ticks = ax2.get_yticks()
                ticks[0] = min_subject_val
                # if next tick too close to new min tick, remove it
                if abs(ticks[1] - min_subject_val) < 3:
                    ticks = np.delete(ticks, 1)
                ax2.set_yticks(ticks)

        # Set axis limits for silhouette to range of the measure
        if measure == ClusteringQualityMeasures.silhouette_score:
            # the other measures have outliers and so we cannot use this
            ax2.set_ylim(-1, 1)  # Set secondary axis limits

        ax.set_ylim(bottom=0)
        ax.set_xlim(-2, 69)  # slight padding around 0-67

        # Format axis labels
        ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_number_text_formatter))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(custom_number_text_formatter))

        ax2.grid(False)  # Turn off secondary grid to avoid double grid

        # set measure as title
        measure_name = ClusteringQualityMeasures.get_display_name_for_measure(measure)
        ax.set_title(measure_name, fontsize=fontsize, fontweight='bold', color=measure_colors[1])

        # Add grid
        ax.grid(True, linestyle='-', alpha=0.3, which='major')
        ax.grid(True, linestyle='--', alpha=0.2, which='minor')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Customize spines
        sns.despine(ax=ax)

    # Adjust layout
    plt.tight_layout()
    return fig


class VisualiseMultipleDataVariants:
    """Use this class for visualising data properties from multiple subjects for multiple data variants"""

    def __init__(self, run_file: str, overall_ds_name: str, dataset_types: [str], data_dirs: [str],
                 additional_cor: [str] = [], backend: str = Backends.none.value):
        self.run_file = run_file
        self.overall_ds_name = overall_ds_name
        self.dataset_types = dataset_types
        self.data_dirs = data_dirs
        self.additional_cor = additional_cor
        self.backend = backend
        self.row_names = []
        self.all_data = {}
        for folder in data_dirs:
            row_name = get_row_name_from(folder)
            self.row_names.append(row_name)
            column_results = {}
            for ds_type in dataset_types:
                column_name = SyntheticDataType.get_display_name_for_data_type(ds_type)
                ds = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                                    data_type=ds_type, data_dir=folder,
                                                    additional_corr=self.additional_cor, load_data=True)
                column_results[column_name] = ds
            self.all_data[row_name] = column_results
        self.col_names = [SyntheticDataType.get_display_name_for_data_type(ds_type) for ds_type in dataset_types]

    def violin_plots_of_overall_segment_lengths(self, save_fig: bool, root_result_dir: str):
        """
        Creates a plot of overall segment lengths, rows will be standard, irregular p 0.3, irregular p 0.9 and columns
        will be "RAW/NC/NN" and "RS"
        :param save_fig: whether to save the figure
        :param root_result_dir: root result dir to save the figure this will be put in the dataset-description
        :return: fig
        """
        # create data dict, for this RAW, NC and NN are all the same so we're just using NN
        column_keys = {
            SyntheticDataType.get_display_name_for_data_type(
                SyntheticDataType.non_normal_correlated): "Raw/Correlated/Non-normal",
            SyntheticDataType.get_display_name_for_data_type(
                SyntheticDataType.rs_1min): SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.rs_1min)}
        data_dict = {}
        for row, row_data in self.all_data.items():
            row_dict = {}
            for col, col_data in row_data.items():
                if col in column_keys.keys():
                    row_dict[column_keys[col]] = col_data.all_segment_lengths_values()
            data_dict[row] = row_dict

        fig = create_violin_grid_log_scale_x_axis(data_dict=data_dict, backend=self.backend, figsize=(18, 15))
        plt.show()

        if save_fig:
            folder = base_dataset_result_folder_for_type(root_result_dir, ResultsType.dataset_description)
            fig.savefig(str(path.join(folder, OVERALL_SEGMENT_LENGTH_IMAGE)), dpi=300, bbox_inches='tight')
        return fig

    def violin_plots_of_overall_mae(self, root_result_dir: str, cor_types: [str] = [CorrType.spearman], save_fig=False):
        """
            Creates a plot of overall segment lengths, rows will be standard, irregular p 0.3, irregular p 0.9 and columns
            will be "RAW/NC/NN" and "RS"
            :param root_result_dir: root result dir to save the figure this will be put in the dataset-description
            :param cor_types: list of cor types to calculate for, default just spearman
            :param save_fig: whether to save the figure
            :return: fig
        """
        figs = {}
        for cor in cor_types:
            # create data dict for mae, for this all ds are different so we plot all
            data_dict = {}
            for row, row_data in self.all_data.items():
                row_dict = {}
                for col, col_data in row_data.items():
                    row_dict[col] = col_data.all_mae_values(SyntheticDataSegmentCols.relaxed_mae, corr_type=cor)
                data_dict[row] = row_dict

            fig = create_violin_grid(data_dict=data_dict, backend=self.backend, fix_x_axis=True, x_lim=0.9,
                                     figsize=(24, 10))
            plt.show()

            if save_fig:
                folder = base_dataset_result_folder_for_type(root_result_dir, ResultsType.dataset_description)
                fig.savefig(str(path.join(folder, cor + "_" + OVERALL_MAE_IMAGE)), dpi=300, bbox_inches='tight')
            figs[cor] = fig

        return figs

    def correlation_pattern_for_pattern(self, root_result_dir: str, pattern_id: int = 19, stats: str = "50%",
                                        save_fig=False):
        """
            Creates a IOB, COB, and IG Correlations, rows are completed, partial, sparse and columns
            are be raw, correlates, non-normal and downsampled
            :param root_result_dir: root result dir to save the figure this will be put in the dataset-description
            :param pattern_id: which pattern to show the overall correlation for
            :param stats: which stats to plot pattern for, default median (50%), see other pandas describe stats
            :param save_fig: whether to save the figure
            :return: fig
        """
        # create data dict for mae, for this all ds are different so we plot all
        data_dict = {}
        for row, row_data in self.all_data.items():
            row_dict = {}
            for col, col_data in row_data.items():
                actual_cor = col_data.achieved_correlation_stats_for_pattern(pattern_id).loc[stats]
                # gosh, don't ask but the method wants a correlation matrix as df
                row_dict[col] = pd.DataFrame(generate_correlation_matrix(actual_cor.to_numpy()))
            data_dict[row] = row_dict

        fig = create_correlation_grid(data_dict=data_dict, backend=self.backend, figsize=(15, 10))
        plt.show()

        if save_fig:
            folder = base_dataset_result_folder_for_type(root_result_dir, ResultsType.dataset_description)
            fig.savefig(
                str(path.join(folder, stats + "_pattern_id_" + str(
                    pattern_id) + "_correlation_matrix_across_data_variants.png")),
                dpi=300, bbox_inches='tight')

        return fig
