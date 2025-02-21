import numpy as np
import pandas as pd

from src.data_generation.wandb_create_irregular_datasets import CreateIrregularDSConfig, create_irregular_datasets
from src.utils.configurations import DataCompleteness, get_data_dir, CONFIRMATORY_SYNTHETIC_DATA_DIR, \
    WandbConfiguration, CONFIRMATORY_DATASETS_FILE_PATH

if __name__ == "__main__":
    irregular_pairs = [(0.3, DataCompleteness.irregular_p30), (0.9, DataCompleteness.irregular_p90)]

    for p, data_comp in irregular_pairs:
        config = CreateIrregularDSConfig()
        config.p = p
        config.root_result_data_dir = get_data_dir(root_data_dir=CONFIRMATORY_SYNTHETIC_DATA_DIR,
                                                   extension_type=data_comp)
        config.wandb_project_name = WandbConfiguration.wandb_confirmatory_irregular_project_name

        generated_ds = pd.read_csv(CONFIRMATORY_DATASETS_FILE_PATH)['Name'].tolist()
        # we create a run for each ds and we name the run consistently
        for idx, ds_name in enumerate(generated_ds):
            np.random.seed(99 + idx)
            dataset_seed = np.random.randint(low=100, high=1000000)
            create_irregular_datasets(config, ds_name, dataset_seed)
