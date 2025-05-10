import pandas as pd

from src.data_generation.wandb_create_bad_partitions import CreateBadPartitionsConfig, create_bad_partitions
from src.utils.configurations import WandbConfiguration, CONFIRMATORY_DATASETS_FILE_PATH, DataCompleteness, \
    get_data_dir, CONFIRMATORY_SYNTHETIC_DATA_DIR
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    # Create bad partitions for confirmatory dataset
    dataset_types = [SyntheticDataType.raw,
                     SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    # dataset_types = [SyntheticDataType.rs_1min] # recreate for resampled
    data_completeness = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]
    # data_completeness = [DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]
    config = CreateBadPartitionsConfig()
    config.wandb_project_name = WandbConfiguration.wandb_confirmatory_partitions_project_name
    config.seed = 2122
    config.csv_of_runs = CONFIRMATORY_DATASETS_FILE_PATH

    for comp in data_completeness:
        for data_type in dataset_types:
            config.data_dir = get_data_dir(root_data_dir=CONFIRMATORY_SYNTHETIC_DATA_DIR, extension_type=comp)
            config.data_type = data_type

            run_names = pd.read_csv(config.csv_of_runs)['Name'].tolist()
            n_datasets = len(run_names)
            n_partitions = config.n_partitions
            # *3 for the different three strategies
            print("Generating " + str(n_partitions * 3) + " bad partitions for " + str(n_datasets) + " datasets")

            for idx, ds_name in enumerate(run_names):
                create_bad_partitions(config, ds_name=ds_name, idx=idx)
