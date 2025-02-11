import pandas as pd

from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.evaluation.run_cluster_quality_measures_calculation import read_clustering_quality_measures
from src.utils.configurations import SYNTHETIC_DATA_DIR, RunInformationCols
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends


class DescribeClusteringQualityForDataVariant:
    """
    Use this class to describe the clustering quality subjects that are part of a data variant.
    The class can do both give overall statistics - e.g. when segment length is analysed from all segment lengths
    in all subjects in the collection - as well as variations across a subjects for which we usually have a single
    number that describes each subject and calculate statistics from that number across the subjects
    (if we did this for segment length we would calculate a mean segment length per subject - this is none sense for o
    ur data as the mean is the same for each subject).
    Overall analysis makes sense for example for segment length, MAE...
    Variations across ds makes sense for n observations, MAE (this time we see how much this varies), n segments
    outside tolerance...
    We will recalculate the number from the labels files
    """

    def __init__(self, wandb_run_file: str, overall_ds_name: str, data_type: str, data_dir: str, results_root_dir: str,
                 distance_measure: str, clustering_quality_measures: [str], round_to: int = 3):
        """
        :param wandb_run_file: full path to the wandb run file
        :param overall_ds_name: a string for the ds - this is mainly used to safe results in
        :param data_type: type of data to load all datasets for, see SyntheticDataType for options
        :param data_dir: directory where the data is stored
        :param results_root_dir: directory where the results are stored
        :param distance_measure: what distance measure to load the results for
        :param clustering_quality_measures: which clustering_quality_measures to load
        :param round_to: what to round the results to
        """
        self.__wandb_run_file = wandb_run_file
        self.__overall_ds_name = overall_ds_name
        self.__data_type = data_type
        self.__data_dir = data_dir
        self.__results_root_dir = results_root_dir
        self.__distance_measure = distance_measure
        self.__internal_measures = clustering_quality_measures
        self.__round_to = round_to
        self.run_names = pd.read_csv(self.__wandb_run_file)['Name'].tolist()

        # load the various result data into dictionaries with key being run name and value being the results df

        # calculations results for the quality measures for each subject
        self.quality_measures_results = {
            name: read_clustering_quality_measures(overall_ds_name=self.__overall_ds_name, data_type=self.__data_type,
                                                   root_results_dir=self.__results_root_dir, data_dir=self.__data_dir,
                                                   distance_measure=self.__distance_measure, run_names=[name])[0] for
            name in self.run_names}

    def all_values_for_clustering_quality_measure(self, clustering_quality_measure: str):
        """
        Returns all values for the given clustering quality measure across all subjects
        :param clustering_quality_measure: see ClusteringQualityMeasures for options
        :return: Series of values
        """
        # create list of measures series
        combined_series = pd.concat([df.set_index(DescribeBadPartCols.name)[clustering_quality_measure] for df in
                                     self.quality_measures_results.values()])
        return combined_series
