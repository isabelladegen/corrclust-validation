import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

from src.evaluation.describe_clustering_quality_for_data_variant import DescribeClusteringQualityForDataVariant
from src.evaluation.internal_measure_ground_truth_assessment import InternalMeasureGroundTruthAssessment
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import get_image_results_path, \
    get_clustering_quality_multiple_data_variants_result_folder, ResultsType, OVERALL_CLUSTERING_QUALITY_DISTRIBUTION, \
    OVERALL_CORRELATION_COEFFICIENT_DISTRIBUTION, OVERALL_CLUSTERING_SCATTER_PLOT, GROUND_TRUTH_CI_PLOT, \
    MULTI_MEASURES_SCATTER_PLOT
from src.utils.distance_measures import short_distance_measure_names
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends, reset_matplotlib, fontsize
from src.visualisation.visualise_multiple_data_variants import get_row_name_from, create_violin_grid, \
    create_violin_grid_log_scale_x_axis, create_scatter_grid, create_scatter_row


def plot_internal_index(df, ax, color, x_pos, alpha: float = 0.05):
    # Calculate mean and confidence intervals
    means = df.mean().values
    n = len(df)
    sem = df.sem().values
    ci = stats.t.ppf(1 - alpha / 2, n - 1) * sem
    # Plot confidence intervals as error bars
    ax.errorbar(x_pos, means, yerr=ci, fmt='o', color=color, capsize=5, markersize=8, elinewidth=2,
                capthick=2)
    # Fill the area between the confidence interval bounds
    ax.fill_between(x_pos, means - ci, means + ci, color=color, alpha=0.2)


def create_ci_grid(data_dict: {}, internal_indices: [str], distance_measures: [str], figsize: (float, float) = (12, 12),
                   backend: str = Backends.none.value) -> plt.Figure:
    """
    Create a grid of ci plots using dictionary keys as labels.

    :param data_dict : {str, {str, pd.DataFrame}} Nested dictionary containing data for each plot square
        Format: {'row_name': {'column_name': list_of_dataframes}}
        Where each DataFrame has the distance measures as columns and columns refer to the data generation stages and
        rows refer to the data completeness
    :param internal_indices : two indices to plot ci for
    :param distance_measures : list of str, columns to plot from the DataFrames
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
    colours = ['#008000', '#800080']

    # X positions for each distance measure
    x_pos = np.arange(len(distance_measures))

    # Plot each subplot
    axes = []
    legend_handles = []  # Will store handles for the legend
    for i, completeness in enumerate(row_names):
        row_axes = []
        for j, generation_stage in enumerate(column_names):
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)

            try:
                data = data_dict[completeness][generation_stage]
            except KeyError:
                print(f"Warning: No data found for condition: row='{completeness}', column='{generation_stage}'")
                continue

            # plot the data for the two internal indices
            for idx, internal_index in enumerate(internal_indices):
                plot_internal_index(data[internal_index], ax, colours[idx], x_pos)

            # Set x-axis labels and ticks
            ax.set_xticks(x_pos)

            # Add a grid for easier reading
            ax.grid(True, linestyle='--', alpha=0.7)

            # Customize appearance
            if i == 0:  # Add column titles only to the top row
                ax.set_title(generation_stage, fontsize=fontsize, fontweight='bold')

            if j == 0:
                ax.set_ylabel(completeness, fontsize=fontsize, fontweight='bold')

            # Only show x-axis elements for bottom row
            if i == len(row_names) - 1:
                display_labels = [short_distance_measure_names[dist] for dist in distance_measures]
                ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=fontsize)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', length=0)

        axes.append(row_axes)

    # Add legend directly in the last subplot
    for k, ii in enumerate(internal_indices):
        legend_handles.append(plt.Line2D([0], [0],
                                         marker='s',
                                         color=colours[k],
                                         label=ClusteringQualityMeasures.get_display_name_for_measure(ii),
                                         markersize=8,
                                         linestyle='None'))
    last_ax = axes[-1][-1]
    last_ax.legend(handles=legend_handles,
                   loc='center left',
                   fontsize=fontsize - 2,
                   frameon=True,
                   edgecolor='black',
                   facecolor='white')

    # Adjust layout
    plt.tight_layout()
    return fig


class VisualiseGroundTruthClusteringQualityMeasuresForDataVariants:
    def __init__(self, overall_ds_name: str, dataset_types: [str], data_dirs: [str],
                 result_root_dir: str, internal_measures: [str], distance_measures: [str],
                 backend: str = Backends.none.value):
        self.overall_ds_name = overall_ds_name
        self.dataset_types = dataset_types
        self.data_dirs = data_dirs
        self.result_root_dir = result_root_dir
        self.internal_measures = internal_measures
        self.distance_measures = distance_measures
        self.backend = backend
        self.row_names = []
        self.all_variants_ground_truth = {}
        for data_dir in data_dirs:
            row_name = get_row_name_from(data_dir)  # data completeness
            self.row_names.append(row_name)
            generation_stages = {}
            for data_type in dataset_types:
                generation_stage = SyntheticDataType.get_display_name_for_data_type(data_type)
                gt = InternalMeasureGroundTruthAssessment(overall_ds_name=self.overall_ds_name,
                                                          internal_measures=self.internal_measures,
                                                          distance_measures=self.distance_measures, data_dir=data_dir,
                                                          data_type=data_type,
                                                          root_results_dir=self.result_root_dir)
                # for the value: dictionary of key=internal_index, value=df of gt results with distance measures as
                # columns
                generation_stages[generation_stage] = gt.raw_scores_for_each_internal_measure()
            self.all_variants_ground_truth[row_name] = generation_stages
        self.col_names = [SyntheticDataType.get_display_name_for_data_type(ds_type) for ds_type in dataset_types]

    def ci_mean_ground_truth_for_quality_measures(self, alpha=0.05, save_fig: bool = False,
                                                  figsize: (float, float) = (15, 8)):
        """
        Plots grid of ci for ground truth for the given quality measures
        """
        # create data dict to plot which matches the row, column structure of the plot
        data_dict = {}
        for row, row_data in self.all_variants_ground_truth.items():
            row_dict = {}
            for col, col_data in row_data.items():
                # df of ground truth for internal measures for each subject, has columns for the distance  measures
                row_dict[col] = col_data
            data_dict[row] = row_dict

        fig = create_ci_grid(data_dict=data_dict, internal_indices=self.internal_measures,
                             distance_measures=self.distance_measures, backend=self.backend,
                             figsize=figsize)

        plt.show()

        if save_fig:
            folder = get_clustering_quality_multiple_data_variants_result_folder(
                results_type=ResultsType.internal_measure_evaluation,
                overall_dataset_name=self.overall_ds_name,
                results_dir=self.result_root_dir,
                distance_measure='')
            # add an image results folder
            internal_measure_names = '_'.join(self.internal_measures)
            file_name = get_image_results_path(folder, internal_measure_names + '_' + GROUND_TRUTH_CI_PLOT)
            fig.savefig(file_name, dpi=300, bbox_inches='tight')
        return fig


class VisualiseClusteringQualityMeasuresForDataVariants:
    def __init__(self, run_file: str, overall_ds_name: str, dataset_types: [str], data_dirs: [str],
                 result_root_dir: str, distance_measure: str, backend: str = Backends.none.value):
        self.run_file = run_file
        self.overall_ds_name = overall_ds_name
        self.dataset_types = dataset_types
        self.data_dirs = data_dirs
        self.result_root_dir = result_root_dir
        self.distance_measure = distance_measure
        self.backend = backend
        self.row_names = []
        self.all_variants_describe = {}
        for data_dir in data_dirs:
            row_name = get_row_name_from(data_dir)  # data completeness
            self.row_names.append(row_name)
            generation_stages = {}
            for data_type in dataset_types:
                generation_stage = SyntheticDataType.get_display_name_for_data_type(data_type)
                describe = DescribeClusteringQualityForDataVariant(wandb_run_file=self.run_file,
                                                                   overall_ds_name=self.overall_ds_name,
                                                                   data_type=data_type, data_dir=data_dir,
                                                                   results_root_dir=self.result_root_dir,
                                                                   distance_measure=self.distance_measure)
                generation_stages[generation_stage] = describe
            self.all_variants_describe[row_name] = generation_stages
        self.col_names = [SyntheticDataType.get_display_name_for_data_type(ds_type) for ds_type in dataset_types]

    def violin_plots_for_quality_measure(self, quality_measure: str, save_fig: bool = False,
                                         figsize: (float, float) = (15, 10)):
        """
        Plots grid of data variants with the data generation stages as columns and the data completeness as row.
        Each subplot shows the distribution of the given quality measure for that data variant
        :param quality_measure: see ClusteringQualityMeasures for options
        :param save_fig: whether to save the figure or not
        :param figsize: size of figure default for 4 columns and 3 rows
        :return:
        """
        # create data dict to plot which matches the row, column structure of the plot
        data_dict = {}
        for row, row_data in self.all_variants_describe.items():
            row_dict = {}
            for col, col_data in row_data.items():
                row_dict[col] = col_data.all_values_for_clustering_quality_measure(quality_measure)
            data_dict[row] = row_dict

        # find mean of data to decide if we're using log or linear x-axis scale
        max_mean = 0
        for row_data in data_dict.values():
            for col_data in row_data.values():
                max_mean = max(max_mean, np.mean(col_data))

        if max_mean > 10000:
            # use log scale axes
            fig = create_violin_grid_log_scale_x_axis(data_dict=data_dict, backend=self.backend, figsize=figsize)
        else:
            # use linear axes
            fig = create_violin_grid(data_dict=data_dict, backend=self.backend, figsize=figsize)

        plt.show()

        if save_fig:
            folder = get_clustering_quality_multiple_data_variants_result_folder(
                results_type=ResultsType.internal_measures_calculation,
                overall_dataset_name=self.overall_ds_name,
                results_dir=self.result_root_dir,
                distance_measure=self.distance_measure)
            # add an image results folder
            file_name = get_image_results_path(folder, quality_measure + OVERALL_CLUSTERING_QUALITY_DISTRIBUTION)
            fig.savefig(file_name, dpi=300, bbox_inches='tight')
        return fig

    def violin_plots_for_correlation_coefficients(self, quality_measure: str, save_fig: bool = False,
                                                  figsize: (float, float) = (15, 10)):
        """
        Plots grid of data variants with the data generation stages as columns and the data completeness as row.
        Each subplot shows the distribution of the given quality measure's correlation coefficients
         with the jaccard index
        :param quality_measure: see ClusteringQualityMeasures for options (cannot be jaccard index)
        :param save_fig: whether to save the figure or not
        :param figsize: figure size default for 4 columns 3 rows
        :return:
        """
        msg = "Provide an other quality measure that is compared to the Jaccard Index"
        assert quality_measure != ClusteringQualityMeasures.jaccard_index, msg
        # create data dict to plot which matches the row, column structure of the plot
        data_dict = {}
        for row, row_data in self.all_variants_describe.items():
            row_dict = {}
            for col, col_data in row_data.items():
                row_dict[col] = col_data.all_values_for_correlations_with_jaccard_index_for_quality_measure(
                    quality_measure)
            data_dict[row] = row_dict

        # use linear axes
        fig = create_violin_grid(data_dict=data_dict, backend=self.backend, figsize=figsize)

        plt.show()

        if save_fig:
            folder = get_clustering_quality_multiple_data_variants_result_folder(
                results_type=ResultsType.internal_measure_evaluation,
                overall_dataset_name=self.overall_ds_name,
                results_dir=self.result_root_dir,
                distance_measure=self.distance_measure)
            # add an image results folder
            file_name = get_image_results_path(folder, quality_measure + OVERALL_CORRELATION_COEFFICIENT_DISTRIBUTION)
            fig.savefig(file_name, dpi=300, bbox_inches='tight')
        return fig

    def scatter_plots_for_quality_measures(self, quality_measures: [str], save_fig: bool = False,
                                           figsize: (float, float) = (18, 10)):
        """
        Plots grid of data variants with the data generation stages as columns and the data completeness as row.
        Each subplot shows a scatter plots of the given quality measures for that data variant. y is the value
        of the measure and x the different segmented clusterings
        :param quality_measures: see ClusteringQualityMeasures for options, will use a different color for each
        :param save_fig: whether to save the figure or not
        :param figsize: figure size by default good for 4 cols and 3 rows
        :return:
        """
        # create data dict to plot which matches the row, column structure of the plot
        data_dict = {}
        for row, row_data in self.all_variants_describe.items():
            row_dict = {}
            for col, col_data in row_data.items():
                # df of quality measure calculation summary for each subject, has columns for the measures, rows are
                # partitions (segmented clusterings)
                row_dict[col] = list(col_data.quality_measures_results.values())
            data_dict[row] = row_dict

        fig = create_scatter_grid(data_dict=data_dict, measure_cols=quality_measures, backend=self.backend,
                                  figsize=figsize)

        plt.show()

        if save_fig:
            folder = get_clustering_quality_multiple_data_variants_result_folder(
                results_type=ResultsType.internal_measures_calculation,
                overall_dataset_name=self.overall_ds_name,
                results_dir=self.result_root_dir,
                distance_measure=self.distance_measure)
            # add an image results folder
            file_name = get_image_results_path(folder, quality_measures[0] + '_with_' + quality_measures[
                1] + OVERALL_CLUSTERING_SCATTER_PLOT)
            fig.savefig(file_name, dpi=300, bbox_inches='tight')
        return fig

    def scatter_plots_for_multiple_quality_measures(self, reference_measure: str, quality_measures: [str],
                                                    data_type: str, completeness: str,
                                                    save_fig: bool = False,
                                                    figsize: (float, float) = (18, 3.5)):
        # create data dict to plot which matches the row, column structure of the plot
        data_dict = {}
        for row, row_data in self.all_variants_describe.items():
            row_dict = {}
            for col, col_data in row_data.items():
                # df of quality measure calculation summary for each subject, has columns for the measures, rows are
                # partitions (segmented clusterings)
                row_dict[col] = list(col_data.quality_measures_results.values())
            data_dict[row] = row_dict

        fig = create_scatter_row(data_dict=data_dict, reference_measure=reference_measure,
                                 measure_cols=quality_measures, data_type=data_type, completeness=completeness,
                                 backend=self.backend,
                                 figsize=figsize)

        plt.show()

        if save_fig:
            folder = get_clustering_quality_multiple_data_variants_result_folder(
                results_type=ResultsType.internal_measures_calculation,
                overall_dataset_name=self.overall_ds_name,
                results_dir=self.result_root_dir,
                distance_measure=self.distance_measure)
            # add an image results folder
            file_name = "_".join([data_type, completeness, MULTI_MEASURES_SCATTER_PLOT])
            file_name = get_image_results_path(folder, file_name)
            fig.savefig(file_name, dpi=300, bbox_inches='tight')
        return fig
