import ast
import glob
from dataclasses import dataclass
from os import path

import pandas as pd
from pyarrow import ArrowInvalid

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import SYNTHETIC_DATA_DIR, GeneralisedCols, dir_for_data_type, \
    bad_partition_dir_for_data_type

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
    """
    Don't change the string values as they match the directory name!
    The data loading will no longer work if you rename this unless you rename all data directories
    """
    raw: str = "raw"
    normal_correlated: str = "normal"
    non_normal_correlated: str = "non_normal"
    rs_1min: str = "resampled_1min"

    @staticmethod
    def is_resample_type(data_type: str):
        return "resampled_" in data_type

    @staticmethod
    def resample(rule: str):
        return "resampled_" + rule

    @staticmethod
    def rule_from_resample_type(resample_type: str):
        return resample_type.replace("resampled_", "")

    @staticmethod
    def get_log_key_for_data_type(data_type: str):
        if data_type == SyntheticDataType.raw:
            return "RAW"
        if data_type == SyntheticDataType.normal_correlated:
            return "NC"  # correlated
        if data_type == SyntheticDataType.non_normal_correlated:
            return "NN"  # non-normal
        if SyntheticDataType.is_resample_type(data_type):
            return "RS"  # downsampled
        return data_type

    @staticmethod
    def get_display_name_for_data_type(data_type: str):
        if data_type == SyntheticDataType.raw:
            return "Raw"
        if data_type == SyntheticDataType.normal_correlated:
            return "Correlated"
        if data_type == SyntheticDataType.non_normal_correlated:
            return "Non-normal"
        if SyntheticDataType.is_resample_type(data_type):
            return "Downsampled"
        return data_type

    @staticmethod
    def get_dataset_variation_name(data_type: str, extension: str):
        """
        Returns the ds variation name for the given datatype. The extension is
        p30 or p90 for irregular datasets. Values will be RAW if extensions is '' or RAW p30 if extension
        'p30' or RS p30 1min or RS 1min, etc
        """
        resample_rule = SyntheticDataType.rule_from_resample_type(data_type) if SyntheticDataType.is_resample_type(
            data_type) else ''
        return " ".join(
            filter(None, [SyntheticDataType.get_log_key_for_data_type(data_type), extension, resample_rule]))


@dataclass
class SyntheticFileTypes:
    data: str = "-data.parquet"
    labels: str = "-labels.parquet"
    bad_labels: str = "-bad-labels.parquet"


def load_labels(run_id: str, data_type: str = SyntheticDataType.normal_correlated, data_dir: str = SYNTHETIC_DATA_DIR):
    """Returns labels df for synthetic data with specified wandb run name
    :param run_id: name of run that generated the dataset
    :param data_type: select type from SyntheticDataType, defaults to normal correlated data
    optional, if not given labels for run_id will be loaded
    :param data_dir: full path to directory where data is stored, defaults to SYNTHETIC_DATA_DIR

    Labels df has the columns specified in the SyntheticDataSegmentCols
    """
    labels_file = SyntheticFileTypes.labels
    file_dir = dir_for_data_type(data_type, data_dir)
    labels_file_name = Path(file_dir, run_id + labels_file)

    print("Load labels data file with name: " + str(labels_file_name))

    assert labels_file_name.exists(), "No labels files with name " + str(labels_file_name)

    # load labels file
    labels_df = load_labels_file_for(labels_file_name)

    return labels_df


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
    file_dir = dir_for_data_type(data_type, data_dir)
    data_file_name = Path(file_dir, run_id + data_file)

    print("Load data file with name: " + str(data_file_name))

    assert data_file_name.exists(), "No data files with name " + str(data_file_name)

    # load labels file
    labels_df = load_labels(run_id, data_type, data_dir)

    # load data file
    try:
        data_df = pd.read_parquet(data_file_name, engine="pyarrow")
    except ArrowInvalid:
        print("Arrow invalid for file: " + str(data_file_name))
        raise

    data_df[GeneralisedCols.datetime] = pd.to_datetime(data_df[GeneralisedCols.datetime])

    return data_df, labels_df


def safely_load_lists(value):
    """
    This is required due to csv files storing lists as string and parquet naturally handling
    nested types, therefore we check the type first before translating to a list
    """
    if isinstance(value, list):
        return value
    elif isinstance(value, str):
        return ast.literal_eval(value)
    else:
        return value


def load_labels_file_for(labels_file_name: Path):
    """Loads labels df from full path to file
    """

    assert labels_file_name.exists(), "No labels data files with name " + str(labels_file_name)

    try:
        labels_df = pd.read_parquet(labels_file_name, engine="pyarrow")
    except ArrowInvalid:
        print("Arrow invalid for file: " + str(labels_file_name))
        raise

    # change types from string to arrays
    if SyntheticDataSegmentCols.correlation_to_model in labels_df.columns:
        labels_df[SyntheticDataSegmentCols.correlation_to_model] = labels_df[
            SyntheticDataSegmentCols.correlation_to_model].apply(safely_load_lists)
    labels_df[SyntheticDataSegmentCols.actual_correlation] = labels_df[
        SyntheticDataSegmentCols.actual_correlation].apply(safely_load_lists)
    if SyntheticDataSegmentCols.actual_within_tolerance in labels_df.columns:
        labels_df[SyntheticDataSegmentCols.actual_within_tolerance] = labels_df[
            SyntheticDataSegmentCols.actual_within_tolerance].apply(safely_load_lists)
    return labels_df


def load_synthetic_data_and_labels_for_bad_partitions(run_id: str,
                                                      data_type: str = SyntheticDataType.non_normal_correlated,
                                                      data_dir: str = SYNTHETIC_DATA_DIR, load_only: int = None):
    """
    Method to load ground truth data and labels and all bad-partition labels df for the given run id
    :param run_id: which run id to load this data for
    :param data_type: select type from SyntheticDataType, defaults to normal correlated data
    optional, if not given labels for run_id will be loaded
    :param data_dir: full path to directory where data is stored, defaults to SYNTHETIC_DATA_DIR
    :param load_only: give numbers of partitions to load, None if all bad partitions should be loaded
    :return: data_df np.DataFrame, labels_df np.DataFrame, bad_partition_labels {file_name : np.DataFrame}
    """
    # 1. load ground truth data and labels
    gt_data, gt_labels = load_synthetic_data(run_id, data_type, data_dir)

    # 2. load all bad partitions labels files for this run_id and data type
    bad_partitions_dir = bad_partition_dir_for_data_type(data_type, data_dir)
    bad_partitions_path = Path(bad_partitions_dir)
    assert bad_partitions_path.exists(), "No bad partitions folder with name " + str(bad_partitions_path)
    file_path = path.join(bad_partitions_path, run_id + SyntheticFileTypes.bad_labels)
    all_partitions_df = load_labels_file_for(Path(file_path))
    # find all different quality clusterings
    clustering_desc = all_partitions_df[SyntheticDataSegmentCols.cluster_desc].unique()

    # for test runs only load 3 datasets to speed up testing
    if load_only is not None:
        clustering_desc = clustering_desc[:load_only]

    # Split the consolidated bad clustering into dictionary with key orig filename and value pandas df
    results = {}
    for desc in clustering_desc:
        subset_df = all_partitions_df[all_partitions_df[SyntheticDataSegmentCols.cluster_desc] == desc]
        orig_file_name = f"{run_id}-{desc}-{SyntheticFileTypes.labels}"
        results[orig_file_name] = subset_df

    return gt_data, gt_labels, results
