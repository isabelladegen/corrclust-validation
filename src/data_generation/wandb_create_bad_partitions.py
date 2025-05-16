import re
import traceback
from os import path
import random

import pandas as pd

import wandb
from dataclasses import dataclass, field, asdict

from src.data_generation.create_bad_partitions import CreateBadSyntheticPartitions
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.evaluation.describe_bad_partitions import DescribeBadPartitions, DescribeBadPartCols
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import WandbConfiguration, SYNTHETIC_DATA_DIR, SyntheticDataVariates, \
    GENERATED_DATASETS_FILE_PATH, bad_partition_dir_for_data_type, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR
from src.utils.load_synthetic_data import SyntheticDataType, SyntheticFileTypes
from src.utils.plots.matplotlib_helper_functions import Backends


@dataclass
class CreateBadPartitionsConfig:
    wandb_project_name: str = ''
    wandb_entity: str = WandbConfiguration.wandb_entity
    wandb_mode: str = 'online'
    wandb_notes: str = "creates bad partitions of synthetic data"
    tags = ['Synthetic']

    # Load and store results from dir
    data_dir: str = ''
    # Runs to create bad partitions from
    csv_of_runs: str = ''
    # Data type to load and create bad partitions from
    data_type: str = ''
    # data cols to use
    data_cols: [str] = field(default_factory=lambda: SyntheticDataVariates.columns())
    # seed to use for random
    seed: int = 666  # we use the same for each dataset to generate the same bad partitions
    # backend to use for visualisations
    backend: str = Backends.none.value

    # Configure partition creation
    # how many observations to max shift
    leave_obs: int = 100  # how many observations to leave in segment
    # how many partitions to generate per strategy, will result in 3x this number of datasets
    n_partitions: int = 22

    def as_dict(self):
        return asdict(self)


def log_bad_partition_dataset_description(describe: DescribeBadPartitions):
    """"
    Logs information about the dataset using key_id as a prefix for each key
    """
    wandb.log({
        "mean n segments within tolerance": describe.n_segment_within_tolerance_stats()['mean'],
        "std n segments within tolerance": describe.n_segment_within_tolerance_stats()['std'],
        "median n segments within tolerance": describe.n_segment_within_tolerance_stats()['50%'],
        "min n segments within tolerance": describe.n_segment_within_tolerance_stats()['min'],
        "max n segments within tolerance": describe.n_segment_within_tolerance_stats()['max'],
        "mean n segments outside tolerance": describe.n_segment_outside_tolerance_stats()[
            'mean'],
        "std n segments outside tolerance": describe.n_segment_outside_tolerance_stats()['std'],
        "median n segments outside tolerance": describe.n_segment_outside_tolerance_stats()[
            '50%'],
        "min n segments outside tolerance": describe.n_segment_outside_tolerance_stats()['min'],
        "max n segments outside tolerance": describe.n_segment_outside_tolerance_stats()['max'],
        "mean MAE": describe.mae_stats()['mean'],
        "std MAE": describe.mae_stats()['std'],
        "median MAE": describe.mae_stats()['50%'],
        "min MAE": describe.mae_stats()['min'],
        "max MAE": describe.mae_stats()['max'],
        "mean segment length": describe.segment_length_stats()['mean'],
        "std segment length": describe.segment_length_stats()['std'],
        "median segment length": describe.segment_length_stats()['50%'],
        "min segment length": describe.segment_length_stats()['min'],
        "max segment length": describe.segment_length_stats()['max'],
        "mean Jaccard": describe.jaccard_stats()['mean'],
        "std Jaccard": describe.jaccard_stats()['std'],
        "median Jaccard": describe.jaccard_stats()['50%'],
        "min Jaccard": describe.jaccard_stats()['min'],
        "max Jaccard": describe.jaccard_stats()['max'],
        "mean n wrong clusters": describe.n_wrong_cluster_stats()['mean'],
        "std n wrong clusters": describe.n_wrong_cluster_stats()['std'],
        "median n wrong clusters": describe.n_wrong_cluster_stats()['50%'],
        "min n wrong clusters": describe.n_wrong_cluster_stats()['min'],
        "max n wrong clusters": describe.n_wrong_cluster_stats()['max'],
        "mean n obs shifted": describe.n_obs_shifted_stats()['mean'],
        "std n obs shifted": describe.n_obs_shifted_stats()['std'],
        "median n obs shifted": describe.n_obs_shifted_stats()['50%'],
        "min n obs shifted": describe.n_obs_shifted_stats()['min'],
        "max n obs shifted": describe.n_obs_shifted_stats()['max'],
    })


def create_bad_partitions(config: CreateBadPartitionsConfig, ds_name: str, idx: int):
    """
    Wandb generate bad partitions according to the config provided
    :param config: CreateBadPartitionsConfig that configures the creation
    :param ds_name: the name of the dataset to create bad partitions from, will become the run name
    :param idx: the index of the dataset to vary the seed for each dataset slightly
    :return: results_summary dict with key ds name and value bad_part_description, wandb summary dict
    """
    summary = None

    try:
        # check if data dir ends in _p<number>
        match = re.search(r'(_p\d+)$', config.data_dir)
        irr_name = match.group(1) if match else ""

        project_name = config.wandb_project_name + "_" + SyntheticDataType.get_log_key_for_data_type(
            config.data_type) + irr_name
        wandb.init(project=project_name,
                   entity=config.wandb_entity,
                   name=ds_name,
                   mode=config.wandb_mode,
                   notes=config.wandb_notes,
                   tags=config.tags,
                   config=config.as_dict())

        exit_code = 0
        n_partitions = config.n_partitions

        # setup bad partitions path
        bad_partitions_path = bad_partition_dir_for_data_type(config.data_type, config.data_dir)

        resulting_partitions = []

        print("GENERATE BAD PARTITIONS FOR DS: " + ds_name)
        bp = CreateBadSyntheticPartitions(run_name=ds_name, data_type=config.data_type,
                                          data_cols=config.data_cols, data_dir=config.data_dir,
                                          backend=config.backend, seed=config.seed)
        max_seg = bp.labels.shape[0]
        max_obs = max(bp.labels[SyntheticDataSegmentCols.length].min() - config.leave_obs, config.leave_obs)
        # First create partitions where we assign a wrong cluster
        # select random number of segments but ensure one partition changes all 100 segments
        possible_n = list(range(1, max_seg + 1))
        random.seed(66 + idx)
        n_segments = random.sample(possible_n, n_partitions - 1)
        n_segments.append(max_seg)
        n_segments.sort()  # so that lower partitions have fewer errors than higher ones
        print("1. CREATE PARTITIONS WITH WRONGLY ASSIGNED PATTERNS")
        print("... changing patterns for n segments: " + str(n_segments))
        wrong_clusters = bp.randomly_assign_wrong_cluster(n_partitions=n_partitions, n_segments=n_segments)

        # save csv
        print("... saving csvs ...")
        for p in range(n_partitions):
            cluster_desc = "wrong-clusters-" + str(p)
            df = wrong_clusters[p]
            df.insert(1, SyntheticDataSegmentCols.cluster_desc, cluster_desc)
            resulting_partitions.append(df)

        print("2. CREATE PARTITIONS WITH SHIFTED OBSERVATIONS")
        # ensure one partition will shift by the max of 800 observations
        possible_obs = list(range(1, max_obs + 1))
        random.seed(101 + idx)
        n_observations = random.sample(possible_obs, n_partitions - 1)
        print("... shifting n observations: " + str(n_observations))
        n_observations.append(max_obs)
        n_observations.sort()  # so that lower partitions have fewer errors than higher ones
        shifted_end_idx = bp.shift_segments_end_index(n_partitions=n_partitions, n_observations=n_observations)

        # save csv
        print("... saving csvs ...")
        for p in range(n_partitions):
            cluster_desc = "shifted-end-idx-" + str(p)
            df = shifted_end_idx[p]
            df.insert(1, SyntheticDataSegmentCols.cluster_desc, cluster_desc)
            resulting_partitions.append(df)

        print("3. CREATE PARTITIONS WITH SHIFTED OBSERVATIONS AND WRONG CLUSTER ASSIGNMENTS")
        random.seed(6306 + idx)
        n_segments_both = random.sample(possible_n, n_partitions - 1)
        n_segments_both.append(max_seg)
        n_segments_both.sort()
        n_observations_both = random.sample(possible_obs, n_partitions - 1)
        n_observations_both.append(max_obs)
        n_observations_both.sort()
        print("... changing patterns for n segments: " + str(n_segments_both))
        print("... shifting n observations: " + str(n_observations_both))

        shift_and_wrong_clusters = bp.shift_segments_end_index_and_assign_wrong_clusters(n_partitions=n_partitions,
                                                                                         n_observations=n_observations_both,
                                                                                         n_segments=n_segments_both)
        # save csv
        print("... saving csvs ...")
        for p in range(n_partitions):
            cluster_desc = "shifted-and-wrong-cluster-" + str(p)
            df = shift_and_wrong_clusters[p]
            df.insert(1, SyntheticDataSegmentCols.cluster_desc, cluster_desc)
            resulting_partitions.append(df)

        consolidated_df = pd.concat(resulting_partitions, ignore_index=True)
        file_name = path.join(bad_partitions_path, ds_name + SyntheticFileTypes.bad_labels)
        save_bad_partitions_labels_file(consolidated_df, file_name)

        wandb.log({
            "Patterns changed n segments": n_segments,
            "Shifted n obs": n_observations,
            "Both patterns changed n segments": n_segments_both,
            "Both shifted n obs": n_observations_both,
        })

        print("4. SAVE SUMMARY TABLE OF PARTITIONS FOR DS")
        # note distance measure is not used therefore set to ""
        bp = DescribeBadPartitions(ds_name=ds_name, distance_measure="", data_type=config.data_type,
                                   internal_measures=[], external_measures=[ClusteringQualityMeasures.jaccard_index],
                                   data_cols=config.data_cols, data_dir=config.data_dir)
        summary = bp.summary_df
        summary_table = wandb.Table(dataframe=summary, allow_mixed_types=True)
        wandb.log({"Summary Partitions " + ds_name: summary_table})

        print("5. LOG SUMMARY OF RESULTS FOR DS")
        log_bad_partition_dataset_description(bp)


    except Exception as e:
        tb = traceback.format_exc()
        print(tb)  # try to get error into wandb log
        exit_code = 1

    wandb_summary_dic = dict(wandb.run.summary)
    wandb.finish(exit_code=exit_code)
    if exit_code == 1:
        raise
    return summary, wandb_summary_dic


def save_bad_partitions_labels_file(df, file_name):
    for col in [SyntheticDataSegmentCols.correlation_to_model, SyntheticDataSegmentCols.actual_correlation,
                SyntheticDataSegmentCols.actual_within_tolerance]:
        df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    df.to_parquet(file_name, index=False, engine="pyarrow")


if __name__ == "__main__":
    dataset_types = [SyntheticDataType.raw,
                     SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    # dataset_types = [SyntheticDataType.rs_1min] # just resampled
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]
    # data_dirs = [IRREGULAR_P30_DATA_DIR, # only partial and sparse
    #              IRREGULAR_P90_DATA_DIR]
    config = CreateBadPartitionsConfig()
    # config.wandb_mode = "offline"  # don't log the rs bad partition regeneration
    config.wandb_project_name = WandbConfiguration.wandb_partitions_project_name
    config.seed = 666
    config.csv_of_runs = GENERATED_DATASETS_FILE_PATH

    for data_dir in data_dirs:
        for data_type in dataset_types:
            config.data_dir = data_dir
            config.data_type = data_type

            run_names = pd.read_csv(config.csv_of_runs)['Name'].tolist()
            n_datasets = len(run_names)
            n_partitions = config.n_partitions
            # *3 for the different three strategies
            print("Generating " + str(n_partitions * 3) + " bad partitions for " + str(n_datasets) + " datasets")

            for idx, ds_name in enumerate(run_names):
                create_bad_partitions(config, ds_name=ds_name, idx=idx)
