from dataclasses import dataclass

import pandas as pd

from src.utils.configurations import SYNTHETIC_DATA_DIR, GeneralisedCols

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


def load_synthetic_data(run_id: str, data_type: str = SyntheticFileTypes.data, labels_dataset: str = "",
                        gt_labels: str = ""):
    """Returns data df and labels df for synthetic data with specified wandb run name
    :param run_id: name of run that generated the dataset
    :param data_type: -data or -normal-data, etc to load (optional, default data)
    :param labels_dataset: name of the labels csv to load if not the same as the run_id, e.g to provide worse clustering,
    optional, if not given labels for run_id will be loaded
    :param gt_labels: name of the labels csv to load that is treated as ground truth, if not provided it is
    the same as the labels_dataset

    Data df is of has rows as observations and columns as time series. It has an additional column called datetime
    Labels df is a segment value result df it has the columns specified in the SyntheticDataSegmentCols

    """
    data_file_name = Path(SYNTHETIC_DATA_DIR, run_id + data_type)
    labels_dataset = labels_dataset if labels_dataset else run_id
    labels_file_name = Path(SYNTHETIC_DATA_DIR, labels_dataset + SyntheticFileTypes.labels)

    print("Load data file with name: " + str(data_file_name))
    print("Load labels data file with name: " + str(labels_file_name))

    assert data_file_name.exists(), "No data files with name " + str(data_file_name)
    assert labels_file_name.exists(), "No labels data files with name " + str(labels_file_name)

    # load labels file
    segment_df = pd.read_csv(labels_file_name, index_col=0)

    # load ground truth labels file
    if gt_labels == "":
        gt_segment_df = segment_df
    else:
        gt_labels_file_name = Path(SYNTHETIC_DATA_DIR, gt_labels + SyntheticFileTypes.labels)
        assert gt_labels_file_name.exists(), "No gt labels files with name " + str(gt_labels_file_name)
        print("Load gt labels data file with name: " + str(gt_labels_file_name))
        gt_segment_df = pd.read_csv(gt_labels_file_name, index_col=0)

    # load data file
    data_df = pd.read_csv(data_file_name, index_col=0)
    data_df[GeneralisedCols.datetime] = pd.to_datetime(data_df[GeneralisedCols.datetime])

    return data_df, segment_df, gt_segment_df
