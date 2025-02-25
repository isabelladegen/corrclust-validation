from dataclasses import dataclass, asdict

import pandas as pd

from src.data_generation.wandb_create_bad_partitions import CreateBadPartitionsConfig, create_bad_partitions
from src.utils.configurations import DataCompleteness, ROOT_REDUCED_SYNTHETIC_DATA_DIR, \
    get_data_dir, get_root_folder_for_reduced_cluster, get_root_folder_for_reduced_segments, \
    GENERATED_DATASETS_FILE_PATH
from src.utils.load_synthetic_data import SyntheticDataType


@dataclass
class ReducedBadPartitionConfig(CreateBadPartitionsConfig):
    wandb_mode: str = 'offline'  # don't log online
    wandb_notes: str = "creates bad partitions of reduced synthetic data"
    tags = ['Synthetic', 'Reduced']

    # Load and store results from dir
    root_reduced_data_dir: str = ''
    data_completeness: str = DataCompleteness.complete
    data_dir: str = ''  # this needs to be set in run
    # Data type to load and create bad partitions from
    data_type: str = SyntheticDataType.non_normal_correlated
    # data cols to use
    # seed to use for random
    seed: int = 666  # we use the same for each dataset to generate the same bad partitions

    def as_dict(self):
        return asdict(self)


def run_create_bad_partitions(config: ReducedBadPartitionConfig, data_dir: str, data_type: str):
    # update config for this datatype
    config.data_dir = data_dir
    config.data_type = data_type
    run_names = pd.read_csv(config.csv_of_runs)['Name'].tolist()
    n_datasets = len(run_names)
    n_partitions = config.n_partitions
    # *3 for the different three strategies
    print("Generating " + str(n_partitions * 3) + " bad partitions for " + str(n_datasets) + " datasets")
    for idx, ds_name in enumerate(run_names):
        create_bad_partitions(config, ds_name=ds_name, idx=idx)


def created_bad_partitions_for_clusters_and_segments(data_types: [str], completeness: [str], run_file: str, seed: int,
                                                     root_reduced_data_dir: str):
    # drop 50% and 75% of clusters and segments
    n_dropped_clusters = [12, 17]
    n_dropped_segments = [50, 75]
    # create for clusters
    for dropped_cluster in n_dropped_clusters:
        for comp in completeness:
            for data_type in data_types:
                config = ReducedBadPartitionConfig()
                config.seed = seed
                config.root_reduced_data_dir = root_reduced_data_dir
                config.csv_of_runs = run_file

                # create path for data folder
                root_folder = get_root_folder_for_reduced_cluster(config.root_reduced_data_dir, dropped_cluster)
                data_dir = get_data_dir(root_folder, comp)
                # create partitions
                run_create_bad_partitions(config, data_dir, data_type)
    # create for segments
    for dropped_segments in n_dropped_segments:
        for comp in completeness:
            for data_type in data_types:
                config = ReducedBadPartitionConfig()
                config.seed = seed
                config.root_reduced_data_dir = root_reduced_data_dir
                config.csv_of_runs = run_file

                # create path for data folder
                root_folder = get_root_folder_for_reduced_segments(config.root_reduced_data_dir, dropped_segments)
                data_dir = get_data_dir(root_folder, comp)
                # create partitions
                run_create_bad_partitions(config, data_dir, data_type)


if __name__ == "__main__":
    _seed = 777
    _data_types = [SyntheticDataType.normal_correlated,
                   SyntheticDataType.non_normal_correlated]
    _completeness = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]

    created_bad_partitions_for_clusters_and_segments(data_types=_data_types, completeness=_completeness,
                                                     run_file=GENERATED_DATASETS_FILE_PATH,
                                                     seed=_seed, root_reduced_data_dir=ROOT_REDUCED_SYNTHETIC_DATA_DIR)
