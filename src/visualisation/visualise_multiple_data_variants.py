import re
from os import path

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator, LogLocator

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.evaluation.describe_subjects_for_data_variant import DescribeSubjectsForDataVariant
from src.utils.configurations import get_irregular_folder_name_from, base_dataset_result_folder_for_type, ResultsType, \
    OVERALL_SEGMENT_LENGTH_IMAGE, OVERALL_MAE_IMAGE
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, fontsize


def custom_number_text_formatter(x, _):
    if abs(x) > 1000:
        return f'{x:.0e}'  # Scientific notation for large numbers
    else:
        return f'{x:.1f}'  # Regular format for other numbers


def add_stats(data: np.ndarray, ax: plt.Axes, y_pos: float) -> None:
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
    ax.text(max_val, y_pos - 0.15, f'max={custom_number_text_formatter(max_val, "-")}',
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


def create_violin_grid(data_dict: {}, figsize: tuple = (12, 12), backend: str = Backends.none.value) -> plt.Figure:
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
    fig.subplots_adjust(hspace=0.5, wspace=0.3, left=0.15)

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
            add_stats(data, ax, 0)

        axes.append(row_axes)

    # Adjust layout
    plt.tight_layout()
    return fig


class VisualiseMultipleDataVariants:
    """Use this class for visualising data properties from multiple subjects for multiple data variants"""

    def __init__(self, run_file: str, overall_ds_name: str, dataset_types: [str], data_dirs: [str],
                 backend: str = Backends.none.value):
        self.run_file = run_file
        self.overall_ds_name = overall_ds_name
        self.dataset_types = dataset_types
        self.data_dirs = data_dirs
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
                                                    data_type=ds_type, data_dir=folder)
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

    def violin_plots_of_overall_mae(self, root_result_dir: str, save_fig=False):
        """
            Creates a plot of overall segment lengths, rows will be standard, irregular p 0.3, irregular p 0.9 and columns
            will be "RAW/NC/NN" and "RS"
            :param root_result_dir: root result dir to save the figure this will be put in the dataset-description
            :param save_fig: whether to save the figure
            :return: fig
        """
        # create data dict for mae, for this all ds are different so we plot all
        data_dict = {}
        for row, row_data in self.all_data.items():
            row_dict = {}
            for col, col_data in row_data.items():
                row_dict[col] = col_data.all_mae_values(SyntheticDataSegmentCols.relaxed_mae)
            data_dict[row] = row_dict

        fig = create_violin_grid(data_dict=data_dict, backend=self.backend, figsize=(15, 10))
        plt.show()

        if save_fig:
            folder = base_dataset_result_folder_for_type(root_result_dir, ResultsType.dataset_description)
            fig.savefig(str(path.join(folder, OVERALL_MAE_IMAGE)), dpi=300, bbox_inches='tight')
        return fig
