import glob
from os import path
from pathlib import Path

import pandas as pd

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import bad_partition_dir_for_data_type, dir_for_data_type, ROOT_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, IRREGULAR_P30_DATA_DIR, SYNTHETIC_DATA_DIR, \
    CONFIRMATORY_SYNTHETIC_DATA_DIR, CONF_IRREGULAR_P30_DATA_DIR, CONF_IRREGULAR_P90_DATA_DIR, \
    CONFIRMATORY_DATASETS_FILE_PATH, DataCompleteness
from src.utils.load_synthetic_data import load_labels_file_for, SyntheticDataType, SyntheticFileTypes


def load_bad_partitions_into_one_file(run_id: str,
                                      data_type: str,
                                      data_dir: str, ):
    bad_partitions_dir = bad_partition_dir_for_data_type(data_type, data_dir)
    bad_partitions_path = Path(bad_partitions_dir)
    assert bad_partitions_path.exists(), "No bad partitions folder with name " + str(bad_partitions_path)
    paths = glob.glob(str(bad_partitions_path) + "/" + run_id + "*-labels.parquet")

    results = [load_labels_file_for(Path(file)) for file in paths]

    return results


if __name__ == "__main__":
    # Confirmatory
    # csts_dir = path.join(ROOT_DIR, 'csts/confirmatory')
    # dataset_types = {SyntheticDataType.raw,
    #                  SyntheticDataType.normal_correlated,
    #                  SyntheticDataType.non_normal_correlated,
    #                  SyntheticDataType.rs_1min}
    # data_dirs = {
    #     CONFIRMATORY_SYNTHETIC_DATA_DIR: csts_dir,
    #     CONF_IRREGULAR_P30_DATA_DIR: path.join(csts_dir, "irregular_p30"),
    #     CONF_IRREGULAR_P90_DATA_DIR: path.join(csts_dir, "irregular_p90")
    # }
    # run_names = pd.read_csv(CONFIRMATORY_DATASETS_FILE_PATH)['Name'].tolist()
    # Exploratory
    csts_dir = path.join(ROOT_DIR, 'csts/exploratory')
    dataset_types = {
                     SyntheticDataType.rs_1min}
    data_dirs = {
        SYNTHETIC_DATA_DIR: csts_dir,
    }
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    # reduced
    # base_dir = path.join("reduced-data", "clusters_dropped_12")
    # base_dir = path.join("reduced-data", "clusters_dropped_17")
    # base_dir = path.join("reduced-data", "segments_dropped_50")
    # base_dir = path.join("reduced-data", "segments_dropped_75")
    # csts_dir = path.join(ROOT_DIR, 'csts/confirmatory', base_dir)
    # synt_dir = path.join(ROOT_DIR, 'parquet_data/confirmatory', base_dir)
    # dataset_types = {
    #     SyntheticDataType.normal_correlated,
    #     SyntheticDataType.non_normal_correlated,
    # }
    # data_dirs = {
    #     synt_dir: csts_dir,
    #     path.join(synt_dir, DataCompleteness.irregular_p30): path.join(csts_dir, DataCompleteness.irregular_p30),
    #     path.join(synt_dir, DataCompleteness.irregular_p90): path.join(csts_dir, DataCompleteness.irregular_p90)
    # }

    for data_dir in data_dirs:
        for data_type in dataset_types:
            for run_id in run_names:
                # load bad partition
                all_files = load_bad_partitions_into_one_file(run_id=run_id, data_type=data_type, data_dir=data_dir)
                # combine
                all_labels = pd.concat(all_files, ignore_index=True)

                # save as one file
                file_dir = path.join(data_dirs[data_dir], data_type, 'bad_partitions')
                Path(file_dir).mkdir(parents=True, exist_ok=True)
                labels_file_name = Path(file_dir, run_id + SyntheticFileTypes.bad_labels)
                # convert arrays to strings for human readability
                for col in [SyntheticDataSegmentCols.correlation_to_model, SyntheticDataSegmentCols.actual_correlation,
                            SyntheticDataSegmentCols.actual_within_tolerance]:
                    all_labels[col] = all_labels[col].apply(lambda x: str(x) if isinstance(x, list) else x)
                # convert datetime to string
                all_labels.to_parquet(labels_file_name, index=False, engine="pyarrow")
