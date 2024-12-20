import ast
from dataclasses import dataclass

import pandas as pd

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import SYNTHETIC_DATA_DIR, GeneralisedCols, dir_for_data_type

from pathlib import Path


@dataclass
class SyntheticDataSets:
    v1: str = "efficient-microwave-1"
    v1_1min_sampling: str = "1min-efficient-microwave-1"
    v_test: str = "flowing-elevator-7"
    v_test_1min_sampling: str = "1min-flowing-elevator-7"
    splendid_sunset: str = "splendid-sunset-12"
    blooming_donkey: str = "blooming-donkey-23"
    perfect_run_1min_sampling: str = "1min-splendid-sunset-12"


@dataclass
class SyntheticDataType:
    """ Don't change the string values as they have to match the dir name! """
    raw: str = "raw"
    normal_correlated: str = "normal"
    non_normal_correlated: str = "non_normal"
    irregular_p30_drop: str = "irregular_p30"
    irregular_p90_drop: str = "irregular_p90"
    rs_1min: str = "resampled_1min"


@dataclass
class SyntheticFileTypes:
    data: str = "-data.csv"
    labels: str = "-labels.csv"
    normal_data: str = "-normal-data.csv"
    normal_correlated_data: str = "-normal-correlated-data.csv"
    scaled_data: str = "-scaled-data.csv"
    normal_scaled_data: str = "-normal-scaled-data.csv"
    normal_correlated_scaled_data: str = "-normal-correlated-scaled-data.csv"

    def all_data_types(self):
        return [self.data, self.normal_data, self.normal_correlated_data]


def load_synthetic_data(run_id: str, data_type: str = SyntheticDataType.normal_correlated,
                        data_dir: str = SYNTHETIC_DATA_DIR):
    """Returns data df and labels df for synthetic data with specified wandb run name
    :param run_id: name of run that generated the dataset
    :param data_type: select type from SyntheticDataType, defaults to normal correlated data
    optional, if not given labels for run_id will be loaded
    :param data_dir: full path to directory where data is stored, defaults to SYNTHETIC_DATA_DIR

    Data df has rows as observations and columns as variants. It has an additional column called datetime
    Labels df is a segment value result df it has the columns specified in the SyntheticDataSegmentCols
    """
    data_file = SyntheticFileTypes.data
    labels_file = SyntheticFileTypes.labels
    file_dir = dir_for_data_type(data_type, data_dir)
    data_file_name = Path(file_dir, run_id + data_file)
    labels_file_name = Path(file_dir, run_id + labels_file)

    print("Load data file with name: " + str(data_file_name))
    print("Load labels data file with name: " + str(labels_file_name))

    assert data_file_name.exists(), "No data files with name " + str(data_file_name)
    assert labels_file_name.exists(), "No labels data files with name " + str(labels_file_name)

    # load labels file
    labels_df = pd.read_csv(labels_file_name, index_col=0)
    # change types from string to arrays
    labels_df[SyntheticDataSegmentCols.correlation_to_model] = labels_df[
        SyntheticDataSegmentCols.correlation_to_model].apply(lambda x: ast.literal_eval(x))
    labels_df[SyntheticDataSegmentCols.actual_correlation] = labels_df[
        SyntheticDataSegmentCols.actual_correlation].apply(lambda x: ast.literal_eval(x))
    labels_df[SyntheticDataSegmentCols.actual_within_tolerance] = labels_df[
        SyntheticDataSegmentCols.actual_within_tolerance].apply(lambda x: ast.literal_eval(x))

    # load data file
    data_df = pd.read_csv(data_file_name, index_col=0)
    data_df[GeneralisedCols.datetime] = pd.to_datetime(data_df[GeneralisedCols.datetime])

    return data_df, labels_df
