from itertools import chain

import pandas as pd

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import SYNTHETIC_DATA_DIR
from src.utils.labels_utils import calculate_n_segments_outside_tolerance_for
from src.utils.load_synthetic_data import SyntheticDataType, load_labels
from src.utils.plots.matplotlib_helper_functions import Backends


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

    def __init__(self, wandb_run_file: str, data_type: str = SyntheticDataType.non_normal_correlated,
                 data_dir: str = SYNTHETIC_DATA_DIR, backend: str = Backends.none.value, round_to: int = 3):
        """
        :param wandb_run_file: full path to the wandb run file
        :param data_type: type of data to load all datasets for, see SyntheticDataType for options
        :param data_dir: directory where the data is stored
        :param backend: backend for matplotlib
        :param round_to: what to round the results to
        """
        self.__wandb_run_file = wandb_run_file
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
