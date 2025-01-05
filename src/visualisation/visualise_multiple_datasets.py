import re

from src.evaluation.describe_multiple_datasets import DescribeMultipleDatasets
from src.utils.configurations import get_irregular_folder_name_from
from src.utils.load_synthetic_data import SyntheticDataType


def get_row_name_from(folder):
    irr_folder_name = get_irregular_folder_name_from(folder)
    if irr_folder_name is "":
        return "standard"
    # change _p30 to irregular p 0.3
    match_no = re.search(r'p(\d+)$', irr_folder_name)
    if match_no:
        number = int(match_no.group(1)) / 100
        return "irregular p " + str(number)
    assert False, "Unknown folder extension in: " + folder


class VisualiseMultipleDatasets:
    """Use this class for visualising properties from multiple datasets"""

    def __init__(self, run_file: str, overall_ds_name: str, dataset_types: [str], data_dirs: [str]):
        self.run_file = run_file
        self.overall_ds_name = overall_ds_name
        self.dataset_types = dataset_types
        self.data_dirs = data_dirs
        self.row_names = []
        all_datasets_dict = {}
        for folder in data_dirs:
            row_name = get_row_name_from(folder)
            self.row_names.append(row_name)
            for ds_type in dataset_types:
                column_results = {}
                column_name = SyntheticDataType.get_log_key_for_data_type(ds_type)
                ds = DescribeMultipleDatasets(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                              data_type=ds_type, data_dir=folder)
                column_results[column_name] = ds
            all_datasets_dict[row_name] = column_results
        self.col_names = [SyntheticDataType.get_log_key_for_data_type(ds_type) for ds_type in dataset_types]

