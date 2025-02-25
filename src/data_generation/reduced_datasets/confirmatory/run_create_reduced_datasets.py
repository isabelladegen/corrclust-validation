import pandas as pd

from src.data_generation.reduced_datasets.run_create_reduced_datasets import create_reduced_datasets
from src.utils.configurations import DataCompleteness, CONFIRMATORY_DATASETS_FILE_PATH, CONFIRMATORY_SYNTHETIC_DATA_DIR, \
    CONFIRMATORY_ROOT_REDUCED_SYNTHETIC_DATA_DIR
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    _run_names = pd.read_csv(CONFIRMATORY_DATASETS_FILE_PATH)['Name'].tolist()
    _seed = 84

    _data_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated]
    _completeness = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]

    _root_reduced_data_dir = CONFIRMATORY_ROOT_REDUCED_SYNTHETIC_DATA_DIR

    create_reduced_datasets(root_data_dir=CONFIRMATORY_SYNTHETIC_DATA_DIR, data_types=_data_types,
                            completeness=_completeness, run_names=_run_names, seed=_seed,
                            root_reduced_data_dir=_root_reduced_data_dir)
