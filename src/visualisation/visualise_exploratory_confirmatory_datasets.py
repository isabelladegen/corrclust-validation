from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import get_data_dir
from src.utils.load_synthetic_data import load_labels, SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends, reset_matplotlib, fontsize
from src.visualisation.visualise_multiple_data_variants import get_row_name_from


def create_paired_scatter_grid(exloratory_data_dict: {}, confirmatory_data_dict: {}, data_col_name: str,
                               figsize: tuple = (12, 12), y_lim: (float, float) = (0, 1),
                               backend: str = Backends.none.value) -> plt.Figure:
    """
    Create a grid of scatter plots using dictionary keys as labels.

    :param exploratory_data_dict : {str, {str, pd.DataFrame}} Nested dictionary containing labels_df for each plot
    square
        Format: {'row_name': {'column_name': list_of_dataframes}}
        Where each DataFrame has the measures as columns and columns refer to the data generation stages and
        rows refer to the data completeness
    :param confirmatory_data_dict : {str, {str, pd.DataFrame}} Nested dictionary containing labels_df data for
    each plot square
        Format: {'row_name': {'column_name': list_of_dataframes}}
        Where each DataFrame has the measures as columns and columns refer to the data generation stages and
        rows refer to the data completeness
    :param data_col_name : name of column to plot from labels df
    :param figsize : tuple, optional Figure size in inches (width, height)
    :param y_lim: tuple, optional y axis limits, defaults to (0,1)
    :returns: matplotlib.figure.Figure
    """

    # Extract row and column names from dictionary structure
    completeness_levels = list(exloratory_data_dict.keys())
    data_generation_stages = list(list(exloratory_data_dict.values())[0].keys())

    # Set the style for publication-quality figures
    reset_matplotlib(backend)

    # Create figure with standard size (no extra width needed now)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(completeness_levels), len(data_generation_stages))
    fig.subplots_adjust(hspace=0.5, wspace=0.3, right=0.85)

    # Set up colors for different measures
    dataset_colours = sns.color_palette("husl", n_colors=2)
    markers = ['o', 'x']  # Marker styles

    # Plot each subplot
    axes = []
    legend_handles = []  # Will store handles for the legend
    for i, completeness in enumerate(completeness_levels):
        row_axes = []
        for j, generation_stage in enumerate(data_generation_stages):
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)

            try:
                exp_dfs = exloratory_data_dict[completeness][generation_stage]
                con_dfs = confirmatory_data_dict[completeness][generation_stage]
            except KeyError:
                print(f"Warning: No data found for condition: row='{completeness}', column='{generation_stage}'")
                continue

            # Plot each measure for each DataFrame
            for subject_idx, exp_df in enumerate(exp_dfs):
                ax.scatter(exp_df.index, exp_df[data_col_name],
                           alpha=0.6,
                           color=dataset_colours[0],
                           s=20,
                           marker=markers[0])

                conf_df = con_dfs[subject_idx]
                ax.scatter(conf_df.index, conf_df[data_col_name],
                           alpha=0.6,
                           color=dataset_colours[1],
                           s=20,
                           marker=markers[1])

            # Customize appearance
            if i == 0:  # Add column titles only to the top row
                ax.set_title(SyntheticDataType.get_display_name_for_data_type(generation_stage), fontsize=fontsize,
                             fontweight='bold')

            ax.set_ylim(y_lim[0], y_lim[1])  # Set y lim

            if j == 0:
                ax.set_ylabel(get_row_name_from(completeness), fontsize=fontsize, fontweight='bold')

            # Only show x-axis elements for bottom row
            if i == len(completeness_levels) - 1:
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

    # Add legend directly in the last column's first subplot
    for k, generation_stage in enumerate(['Exploratory', 'Confirmatory']):
        legend_handles.append(plt.Line2D([0], [0],
                                         marker=markers[k],
                                         color=dataset_colours[k],
                                         label=generation_stage,
                                         markersize=8,
                                         linestyle='None'))
    last_ax = axes[0][-1]
    last_ax.legend(handles=legend_handles,
                   loc='upper left',
                   fontsize=fontsize - 2,
                   frameon=True,
                   edgecolor='black',
                   facecolor='white')

    # Adjust layout
    plt.tight_layout()
    return fig


class VisualiseExploratoryConfirmatoryDatasets:
    def __init__(self, run_namex_exploratory: str, run_name_confirmatory: str, overall_ds_name: str,
                 dataset_types: [str], completeness_levels: [str], exploratory_data_dir: str,
                 confirmatory_data_dir: str, backend: str = Backends.none.value):
        self.run_names_exploratory = run_namex_exploratory
        self.run_names_confirmatory = run_name_confirmatory
        self.overall_ds_name = overall_ds_name
        self.dataset_types = dataset_types
        self.completeness_levels = completeness_levels
        self.exploratory_data_dir = exploratory_data_dir
        self.confirmatory_data_dir = confirmatory_data_dir
        self.backend = backend

        # double dictionaries of {completeness: {data_type: labels}}
        self.exploratory_labels = {completeness: {} for completeness in completeness_levels}
        self.confirmatory_labels = {completeness: {} for completeness in completeness_levels}

        for completeness in self.completeness_levels:
            for data_type in dataset_types:
                exploratory_labels = []
                data_dir = get_data_dir(root_data_dir=self.exploratory_data_dir, extension_type=completeness)
                for run_name in self.run_names_exploratory:
                    label = load_labels(run_name, data_type, data_dir=data_dir)
                    exploratory_labels.append(label)
                self.exploratory_labels[completeness][data_type] = exploratory_labels

                confirmatory_labels = []
                data_dir = get_data_dir(root_data_dir=self.confirmatory_data_dir, extension_type=completeness)
                for run_name in self.run_names_confirmatory:
                    label = load_labels(run_name, data_type, data_dir=data_dir)
                    confirmatory_labels.append(label)
                self.confirmatory_labels[completeness][data_type] = confirmatory_labels

    def plot_relaxed_mae_per_subject_scatter_plot(self, figsize: (float, float) = (16, 10)) -> plt.Figure:
        fig = create_paired_scatter_grid(exloratory_data_dict=self.exploratory_labels,
                                         confirmatory_data_dict=self.confirmatory_labels,
                                         data_col_name=SyntheticDataSegmentCols.relaxed_mae, backend=self.backend,
                                         figsize=figsize)

        plt.show()
        return fig

    def plot_pattern_id_for_each_segment(self, figsize: (float, float) = (16, 10)) -> plt.Figure:
        fig = create_paired_scatter_grid(exloratory_data_dict=self.exploratory_labels,
                                         confirmatory_data_dict=self.confirmatory_labels,
                                         data_col_name=SyntheticDataSegmentCols.pattern_id, backend=self.backend,
                                         y_lim=(0, 26),
                                         figsize=figsize)

        plt.show()
        return fig
