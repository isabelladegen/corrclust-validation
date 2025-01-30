from os import path

import pandas as pd

from src.evaluation.describe_synthetic_dataset import DescribeSyntheticDataset
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, dataset_description_dir, EMPIRICAL_CORRELATION_IMAGE
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends


def plot_correlation_patters_for_datasets(root_result_dir, dataset_types, data_dirs, run_names, save_fig, backend):
    for data_dir in data_dirs:
        for data_type in dataset_types:
            for run_name in run_names:
                dst = DescribeSyntheticDataset(run_name=run_name, data_type=data_type, data_dir=data_dir,
                                               backend=backend)
                fig = dst.plot_correlation_matrix_for_each_pattern()

                if save_fig:
                    folder = dataset_description_dir("images", data_type, root_result_dir, data_dir)
                    image_name = path.join(folder, run_name + "_" + EMPIRICAL_CORRELATION_IMAGE)
                    fig.savefig(image_name, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # heatmap ov average ranking for each dataset in the N30
    # y = data variant, x = distance measure, lower ranks are better
    backend = Backends.none.value
    save_fig = True
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.raw,
                     SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    plot_correlation_patters_for_datasets(root_result_dir, dataset_types, data_dirs, run_names, save_fig,
                                          backend)
