from src.data_generation.reduced_datasets.run_create_bad_partitions import \
    created_bad_partitions_for_clusters_and_segments
from src.utils.configurations import DataCompleteness, CONFIRMATORY_ROOT_REDUCED_SYNTHETIC_DATA_DIR, \
    CONFIRMATORY_DATASETS_FILE_PATH
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    _seed = 1981
    _data_types = [SyntheticDataType.normal_correlated,
                   SyntheticDataType.non_normal_correlated]
    _completeness = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]

    created_bad_partitions_for_clusters_and_segments(data_types=_data_types, completeness=_completeness,
                                                     run_file=CONFIRMATORY_DATASETS_FILE_PATH,
                                                     seed=_seed,
                                                     root_reduced_data_dir=CONFIRMATORY_ROOT_REDUCED_SYNTHETIC_DATA_DIR)
