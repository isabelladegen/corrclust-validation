import os
import traceback
from os import path
import random

import pandas as pd

import wandb
from dataclasses import dataclass, field, asdict

from src.data_generation.create_bad_partitions import CreateBadSyntheticPartitions
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.data_generation.wandb_create_synthetic_data import log_dataset_description
from src.evaluation.describe_bad_partitions import DescribeBadPartitions, DescribeBadPartCols
from src.evaluation.describe_synthetic_dataset import DescribeSyntheticDataset
from src.utils.configurations import WandbConfiguration, SYNTHETIC_DATA_DIR, SyntheticDataVariates, \
    GENERATED_DATASETS_FILE_PATH, bad_partition_dir_for_data_type
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends


@dataclass
class CreateBadPartitionsConfig:
    wandb_project_name: str = WandbConfiguration.wandb_project_name
    wandb_entity: str = WandbConfiguration.wandb_entity
    wandb_mode: str = 'online'
    wandb_notes: str = "creates bad partitions of synthetic data"
    tags = ['Synthetic']

    # Load and store results from dir
    data_dir: str = SYNTHETIC_DATA_DIR
    # Runs to create bad partitions from
    csv_of_runs: str = GENERATED_DATASETS_FILE_PATH
    # Data type to load and create bad partitions from
    data_type: str = SyntheticDataType.non_normal_correlated
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


def create_bad_partitions(config: CreateBadPartitionsConfig):
    """
    Wandb generate bad partitions according to the config provided
    :param config: CreateBadPartitionsConfig that configures the creation
    :return: results_summary dict with key ds name and value bad_part_description, wandb summary dict
    """
    try:
        wandb.init(project=config.wandb_project_name,
                   entity=config.wandb_entity,
                   mode=config.wandb_mode,
                   notes=config.wandb_notes,
                   tags=config.tags,
                   config=config.as_dict())

        exit_code = 0
        run_name = wandb.run.name
        if run_name == "":
            run_name = 'test-run'

        print("1. LOAD DATASETS TO GENERATE BAD PARTITIONS FOR")
        csv_file = config.csv_of_runs
        generated_ds = pd.read_csv(csv_file)['Name'].tolist()

        n_datasets = len(generated_ds)
        leave_obs = config.leave_obs
        n_partitions = config.n_partitions
        print("... generating " + str(n_partitions * 3) + " bad partitions for " + str(n_datasets) + " datasets")

        # setup bad partitions path
        bad_partitions_path = bad_partition_dir_for_data_type(config.data_type, config.data_dir)
        # create directory if it does not exist
        os.makedirs(bad_partitions_path, exist_ok=True)

        results_summary = {}

        for idx, ds_name in enumerate(generated_ds):
            print(str(idx) + " GENERATE BAD PARTITIONS")
            print("...for dataset: " + ds_name)
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
            print(str(idx) + ".1 CREATE PARTITIONS WITH WRONGLY ASSIGNED PATTERNS")
            print("... changing patterns for n segments: " + str(n_segments))
            wrong_clusters = bp.randomly_assign_wrong_cluster(n_partitions=n_partitions, n_segments=n_segments)

            # save csv
            print("... saving csvs ...")
            for p in range(n_partitions):
                file_name = path.join(bad_partitions_path + "/", ds_name + "-wrong-clusters-" + str(p) + "-labels.csv")
                wrong_clusters[p].to_csv(file_name)

            print(str(idx) + ".2 CREATE PARTITIONS WITH SHIFTED OBSERVATIONS")
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
                file_name = path.join(bad_partitions_path + "/", ds_name + "-shifted-end-idx-" + str(p) + "-labels.csv")
                shifted_end_idx[p].to_csv(file_name)

            print(str(idx) + ".3 CREATE PARTITIONS WITH SHIFTED OBSERVATIONS AND WRONG CLUSTER ASSIGNMENTS")
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
                file_name = path.join(bad_partitions_path + "/",
                                      ds_name + "-shifted-and-wrong-cluster-" + str(p) + "-labels.csv")
                shift_and_wrong_clusters[p].to_csv(file_name)

            wandb.log({
                "Patterns changed n segments": n_segments,
                "Shifted n obs": n_observations,
                "Both patterns changed n segments": n_segments_both,
                "Both shifted n obs": n_observations_both,
            })

            print(str(idx) + ".4 SAVE SUMMARY TABLE OF PARTITIONS FOR DS")
            # note distance measure is not used therefore set to ""
            bp = DescribeBadPartitions(ds_name=ds_name, data_type=config.data_type, distance_measure="",
                                       data_cols=config.data_cols, internal_measures=[],
                                       external_measures=[DescribeBadPartCols.jaccard_index], seed=config.seed,
                                       drop_n_segments=0, drop_n_clusters=0, data_dir=config.data_dir)
            summary = bp.summary_df
            results_summary[ds_name] = summary
            summary_table = wandb.Table(dataframe=summary, allow_mixed_types=True)
            wandb.log({"Summary Partitions " + ds_name: summary_table})


    except Exception as e:
        tb = traceback.format_exc()
        print(tb)  # try to get error into wandb log
        exit_code = 1

    summary = dict(wandb.run.summary)
    wandb.finish(exit_code=exit_code)
    if exit_code == 1:
        raise
    return results_summary, summary


if __name__ == "__main__":
    config = CreateBadPartitionsConfig()
    create_bad_partitions(config)
