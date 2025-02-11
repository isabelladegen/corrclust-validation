import numpy as np
from matplotlib import pyplot as plt

from src.evaluation.describe_clustering_quality_for_data_variant import DescribeClusteringQualityForDataVariant
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import get_image_results_path, \
    get_clustering_quality_multiple_data_variants_result_folder, ResultsType, OVERALL_CLUSTERING_QUALITY_DISTRIBUTION, \
    OVERALL_CORRELATION_COEFFICIENT_DISTRIBUTION, OVERALL_CLUSTERING_SCATTER_PLOT
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_multiple_data_variants import get_row_name_from, create_violin_grid, \
    create_violin_grid_log_scale_x_axis, create_scatter_grid


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

    def violin_plots_for_quality_measure(self, quality_measure: str, save_fig: bool = False):
        """
        Plots grid of data variants with the data generation stages as columns and the data completeness as row.
        Each subplot shows the distribution of the given quality measure for that data variant
        :param quality_measure: see ClusteringQualityMeasures for options
        :param save_fig: whether to save the figure or not
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
            fig = create_violin_grid_log_scale_x_axis(data_dict=data_dict, backend=self.backend, figsize=(15, 10))
        else:
            # use linear axes
            fig = create_violin_grid(data_dict=data_dict, backend=self.backend, figsize=(15, 10))

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

    def violin_plots_for_correlation_coefficients(self, quality_measure: str, save_fig: bool = False):
        """
        Plots grid of data variants with the data generation stages as columns and the data completeness as row.
        Each subplot shows the distribution of the given quality measure's correlation coefficients
         with the jaccard index
        :param quality_measure: see ClusteringQualityMeasures for options (cannot be jaccard index)
        :param save_fig: whether to save the figure or not
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
        fig = create_violin_grid(data_dict=data_dict, backend=self.backend, figsize=(15, 10))

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

    def scatter_plots_for_quality_measures(self, quality_measures: [str], save_fig: bool = False):
        """
        Plots grid of data variants with the data generation stages as columns and the data completeness as row.
        Each subplot shows a scatter plots of the given quality measures for that data variant. y is the value
        of the measure and x the different segmented clusterings
        :param quality_measures: see ClusteringQualityMeasures for options, will use a different color for each
        :param save_fig: whether to save the figure or not
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
                                  figsize=(18, 10))

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
