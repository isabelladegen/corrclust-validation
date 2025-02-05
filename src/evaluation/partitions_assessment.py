import pandas as pd
from matplotlib import pyplot as plt

from src.evaluation.run_cluster_quality_measures_calculation import read_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, fontsize, set_axis_label_font_size
from src.evaluation.describe_bad_partitions import DescribeBadPartCols


class PartitionAssessment:
    def __init__(self, overall_dataset_name: str, data_type: str, root_results_dir: str, data_dir: str,
                 distance_measure: str, run_names: [str], n_clusters: int = 0,
                 n_segments: int = 0):

        self.overall_dataset_name = overall_dataset_name
        self.data_type = data_type
        self.root_results_dir = root_results_dir
        self.data_dir = data_dir
        self.distance_measure = distance_measure
        self.n_clusters = n_clusters
        self.n_segments = n_segments
        self.run_names = run_names
        # list of dfs of outcomes for each partition in each of the datasets
        self.partition_outcomes = read_clustering_quality_measures(overall_ds_name=self.overall_dataset_name,
                                                                   data_type=self.data_type,
                                                                   root_results_dir=self.root_results_dir,
                                                                   data_dir=self.data_dir,
                                                                   distance_measure=self.distance_measure,
                                                                   n_dropped_clusters=self.n_clusters,
                                                                   n_dropped_segments=self.n_segments,
                                                                   run_names=self.run_names)

    def calculate_describe_statistics_for_partitions(self, column=ClusteringQualityMeasures.jaccard_index):
        """
        Calculates the describe statistics for the worst to the best partition for each dataset for the given column
        :returns pd.DataFrame - columns are the 67 partitions, rows are mean, count, sd, etc
        """
        df_column = self.results_for_column(column)
        return df_column.describe().round(2)

    def results_for_column(self, column):
        df_column = None
        for ds in self.partition_outcomes:
            name = ds.iloc[-1][DescribeBadPartCols.name]  # best jaccard is the ground truth which is the ds name
            values = ds[column]
            if df_column is None:
                df_column = pd.DataFrame({name: values})
            else:
                df_column[name] = values
        # transpose so that the columns (=different datasets) become the rows and the rows (=different partitions)
        # become the columns
        df_column = df_column.T
        return df_column

    def plot_describe_statistics_for_partitions_for_column(self, column=ClusteringQualityMeasures.jaccard_index,
                                                           backend: Backends = Backends.none.value):

        reset_matplotlib(backend)
        fig_size = (10, 6)
        fig, ax = plt.subplots(nrows=1,
                               ncols=1,
                               sharey=False,
                               sharex=False,
                               figsize=fig_size)

        self.__plot_for_column(ax, column)
        ax.set_xlabel("Partitions", fontsize=fontsize)

        plt.tight_layout()
        plt.legend(loc="best")
        plt.show()
        return fig

    def plot_describe_statistics_for_partitions(self, backend):
        """Plots Jaccard, sum errors, SCW and PMB"""
        reset_matplotlib(backend)
        fig_size = (15, 12)
        fig, axs = plt.subplots(nrows=2,
                                ncols=2,
                                sharey=False,
                                sharex=True,
                                figsize=fig_size)

        self.__plot_for_column(axs[0, 0], ClusteringQualityMeasures.jaccard_index)
        self.__plot_for_column(axs[0, 1], DescribeBadPartCols.errors)
        self.__plot_for_column(axs[1, 0], ClusteringQualityMeasures.silhouette_score)
        self.__plot_for_column(axs[1, 1], ClusteringQualityMeasures.pmb)

        axs[1, 0].set_xlabel("Partitions", fontsize=fontsize)
        axs[1, 1].set_xlabel("Partitions", fontsize=fontsize)
        plt.tight_layout()
        plt.legend(loc="best")
        plt.show()
        return fig

    def __plot_for_column(self, ax, column):
        df = self.calculate_describe_statistics_for_partitions(column=column)
        ax.plot(df.loc['mean'], label='mean', color='darkgray')
        ax.plot(df.loc['25%'], label='25%', color='dodgerblue')
        ax.plot(df.loc['50%'], label='50%', color='blue', linewidth=2)
        ax.plot(df.loc['75%'], label='75%', color='dodgerblue')
        line = ax.get_lines()
        x_line = line[0].get_xdata()
        y25 = line[1].get_ydata()
        y75 = line[3].get_ydata()
        # fill between 25 and 75
        ax.fill_between(x_line, y25, y75, color='dodgerblue', alpha=.3)
        ax.grid(axis='x', linestyle='dotted')
        ax.grid(axis='y', linestyle='dotted')
        ax.set_ylabel(column, fontsize=fontsize)
        set_axis_label_font_size(ax)

    def plot_multiple_quality_measures(self, columns: [str], backend: str = Backends.none.value):
        reset_matplotlib(backend)

        # Create figure with subplots sharing x axis
        fig, axes = plt.subplots(nrows=len(columns),
                                 ncols=1,
                                 sharex=True,
                                 figsize=(10, 10))

        # Get legend handles and labels from first plot only
        for idx, (ax, column) in enumerate(zip(axes, columns)):
            self.__plot_for_column(ax, column)

            # Add legend only to first subplot
            if idx == 0:
                ax.legend(loc="upper left")

        # Set shared x label only on bottom plot
        axes[-1].set_xlabel("Segmented Clusterings", fontsize=fontsize)

        plt.tight_layout()
        plt.show()
        return fig
