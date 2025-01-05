from dataclasses import dataclass
from itertools import chain
from os import path

import pandas as pd

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR, dataset_description_dir, \
    MULTIPLE_DS_SUMMARY_FILE, GENERATED_DATASETS_FILE_PATH, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, \
    get_irregular_folder_name_from, base_dataset_result_folder_for_type, ResultsType
from src.utils.labels_utils import calculate_n_segments_outside_tolerance_for
from src.utils.load_synthetic_data import SyntheticDataType, load_labels
from src.utils.plots.matplotlib_helper_functions import Backends


@dataclass
class SummaryStatistics:
    mae = "mae"  # mean of each labels file across datasets
    overall_mae = "overall mae"  # all maes
    seg_outside_tol = "segments outside tolerance"
    observations = "observations"
    segments = "n segments"
    patterns = "n patterns"
    segment_lengths = "segment lengths"  # mean of each labels file across datasets
    overall_segment_lengths = "overall segment lengths"  # all segment lengths


def combine_all_ds_variations_multiple_description_summary_dfs(result_root_dir: str,
                                                               overall_ds_name: str = "n30",
                                                               dataset_types: [str] = [SyntheticDataType.raw,
                                                                                       SyntheticDataType.normal_correlated,
                                                                                       SyntheticDataType.non_normal_correlated,
                                                                                       SyntheticDataType.rs_1min],
                                                               data_dirs: str = [SYNTHETIC_DATA_DIR,
                                                                                 IRREGULAR_P30_DATA_DIR,
                                                                                 IRREGULAR_P90_DATA_DIR],
                                                               save_combined_results: bool = True):
    """Combines all the dataset description summaries for the given types and runs into one table
    :param result_root_dir: root dir for the results - will write to the dataset-description folder in this dir,
    create it if it does not exist
    :param overall_ds_name: just a string to identify the results
    :param dataset_types: list of dataset types to read, see SyntheticDataType
    :param data_dirs: list of str of directories from which the data was read
    :param save_combined_results: will save the combined resulting csv in dataset-description of the result root
    dir
    :return: MultiIndex pd.Dataframe, access results like e.g. combined_df['dataset_name']['n_segments']['mean']
    """
    results = {}
    for data_dir in data_dirs:
        irr_folder_str = get_irregular_folder_name_from(data_dir)
        for ds_type in dataset_types:
            folder = dataset_description_dir(overall_ds_name, ds_type, result_root_dir, data_dir)
            file_name = path.join(folder, MULTIPLE_DS_SUMMARY_FILE)
            df = pd.read_csv(file_name, index_col=0)
            name = SyntheticDataType.get_dataset_variation_name(ds_type, irr_folder_str)
            results[name] = df
    combined_result = pd.concat(results, axis=1)
    if save_combined_results:
        folder = base_dataset_result_folder_for_type(result_root_dir, ResultsType.dataset_description)
        combined_result.to_csv(path.join(folder, MULTIPLE_DS_SUMMARY_FILE))
    return combined_result


class DescribeMultipleDatasets:
    """
    Use this class to describe multiple datasets that are part of a variation. The class can do both give
    overall statistics - e.g. when segment length is analysed from all segment lengths in all datasets in the collection
    - as well as variations across the dataset for which we usually have a single number that describes the dataset
    and calculate statistics from that number across the datasets (if we did this for segment length we would calculate
    a mean segment length per dataset - this is none sense for our data as the mean is the same for each dataset).
    Overall analysis makes sense for example for segment length, MAE...
    Variations across ds makes sense for n observations, MAE (this time we see how much this varies), n segments
    outside tolerance...
    We will recalculate the number from the labels files
    """

    def __init__(self, wandb_run_file: str, overall_ds_name: str,
                 data_type: str = SyntheticDataType.non_normal_correlated,
                 data_dir: str = SYNTHETIC_DATA_DIR, backend: str = Backends.none.value, round_to: int = 3):
        """
        :param wandb_run_file: full path to the wandb run file
        :param overall_ds_name: a string for the ds - this is mainly used to safe results in
        :param data_type: type of data to load all datasets for, see SyntheticDataType for options
        :param data_dir: directory where the data is stored
        :param backend: backend for matplotlib
        :param round_to: what to round the results to
        """
        self.__wandb_run_file = wandb_run_file
        self.__overall_ds_name = overall_ds_name
        self.__data_type = data_type
        self.__data_dir = data_dir
        self.__backend = backend
        self.__round_to = round_to
        # dictionary with key run name and value labels file for the run
        self.run_names = pd.read_csv(wandb_run_file)['Name'].tolist()
        self.labels = {name: load_labels(name, self.__data_type, self.__data_dir) for name in self.run_names}

    def mae_stats(self):
        """ Return stats for mae
        This is to see variations across the different datasets, therefore we use the mean of each dataset
        :returns pandas describe series
        """
        mean_values = [label[SyntheticDataSegmentCols.mae].mean() for label in self.labels.values()]
        return pd.Series(mean_values).describe().round(3)

    def overall_mae_stats(self):
        """
        Return stats for mae overall, here we consider all segment's mae without calculating the mean
        first. This will give a better impression on how mae varies on a segment level.
        :returns pandas describe series
        """
        # list of list
        values = [label[SyntheticDataSegmentCols.mae] for label in self.labels.values()]
        values_flat = list(chain.from_iterable(values))
        return pd.Series(values_flat).describe().round(3)

    def n_segments_outside_tolerance_stats(self):
        """
        Return stats for n segment outside tolerance
        This is naturally a per dataset number
        :returns pandas describe series
        """
        values = [calculate_n_segments_outside_tolerance_for(label) for label in self.labels.values()]
        return pd.Series(values).describe().round(3)

    def observations_stats(self):
        """
        Return stats for n observations
        This is naturally a per dataset number
        :returns pandas describe series
        """
        values = [label[SyntheticDataSegmentCols.length].sum() for label in self.labels.values()]
        return pd.Series(values).describe().round(3)

    def n_patterns_stats(self):
        """
        Return stats for unique number of patterns, for the raw dataset there are none, but it will
        still use the ones that later will be modeled as this makes most sense overall
        This is naturally a per dataset number
        :returns pandas describe series
        """
        values = [len(label[SyntheticDataSegmentCols.pattern_id].unique()) for label in self.labels.values()]
        return pd.Series(values).describe().round(3)

    def n_segments_stats(self):
        """
        Return stats for n segments, this will be commonly 100 but might change for a few datasets
        This is naturally a per dataset number
        :returns pandas describe series
        """
        values = [label.shape[0] for label in self.labels.values()]
        return pd.Series(values).describe().round(3)

    def segment_length_stats(self):
        """
        Return stats for segment length, here we first calculate the mean per dataset and then the statistics
        of the mean. For many variations the mean will be the same but this might not be true for all
        :returns pandas describe series
        """
        values = [label[SyntheticDataSegmentCols.length].mean() for label in self.labels.values()]
        return pd.Series(values).describe().round(3)

    def overall_segment_length_stats(self):
        """
        Return stats for segment length overall, here we consider all segment lengths without calculating the mean
        first. This will give a better impression on how segment lengths vary.
        :returns pandas describe series
        """
        # list of list
        values = [label[SyntheticDataSegmentCols.length] for label in self.labels.values()]
        values_flat = list(chain.from_iterable(values))
        return pd.Series(values_flat).describe().round(3)

    def summary(self):
        """Returns summary pd.Dataframe of all statistics
        access like df[SummaryStatistics]['mean'], the second can be any value in pandas describe
        """
        return pd.DataFrame({
            SummaryStatistics.mae: self.mae_stats(),
            SummaryStatistics.overall_mae: self.overall_mae_stats(),
            SummaryStatistics.seg_outside_tol: self.n_segments_outside_tolerance_stats(),
            SummaryStatistics.observations: self.observations_stats(),
            SummaryStatistics.segments: self.n_segments_stats(),
            SummaryStatistics.patterns: self.n_patterns_stats(),
            SummaryStatistics.segment_lengths: self.segment_length_stats(),
            SummaryStatistics.overall_segment_lengths: self.overall_segment_length_stats()
        })

    def save_summary(self, root_results_dir: str):
        """
        Saves summary df in the given root_result dir
        :param root_results_dir: directory where to save the results
        """
        df = self.summary()

        folder = dataset_description_dir(overall_dataset_name=self.__overall_ds_name, data_type=self.__data_type,
                                         root_results_dir=root_results_dir, data_dir=self.__data_dir)
        file_name = path.join(folder, MULTIPLE_DS_SUMMARY_FILE)
        df.to_csv(file_name)


if __name__ == '__main__':
    # create summary for a dataset variation
    run_file = GENERATED_DATASETS_FILE_PATH
    overall_ds_name = "n30"
    root_results_dir = ROOT_RESULTS_DIR

    dataset_types = [SyntheticDataType.raw, SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated, SyntheticDataType.rs_1min]

    # do regular sampled ones
    for ds_type in dataset_types:
        ds = DescribeMultipleDatasets(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=ds_type,
                                      data_dir=SYNTHETIC_DATA_DIR)
        ds.save_summary(root_results_dir)

    # do irregular p30 sampled ones
    for ds_type in dataset_types:
        ds = DescribeMultipleDatasets(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=ds_type,
                                      data_dir=IRREGULAR_P30_DATA_DIR)
        ds.save_summary(root_results_dir)

    # do irregular p90 sampled ones
    for ds_type in dataset_types:
        ds = DescribeMultipleDatasets(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=ds_type,
                                      data_dir=IRREGULAR_P90_DATA_DIR)
        ds.save_summary(root_results_dir)

    # write combined results (this also reads all files and then writes a result)
    combine_all_ds_variations_multiple_description_summary_dfs(result_root_dir=root_results_dir)
