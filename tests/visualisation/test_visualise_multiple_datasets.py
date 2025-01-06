from hamcrest import *
from matplotlib import pyplot as plt

from src.utils.configurations import ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, SYNTHETIC_DATA_DIR, \
    IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_multiple_datasets import VisualiseMultipleDatasets

# backend = Backends.visible_tests.value
backend = Backends.none.value
vds = VisualiseMultipleDatasets(run_file=GENERATED_DATASETS_FILE_PATH, overall_ds_name="n30",
                                dataset_types=[SyntheticDataType.raw, SyntheticDataType.normal_correlated,
                                               SyntheticDataType.non_normal_correlated, SyntheticDataType.rs_1min],
                                data_dirs=[SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR],
                                backend=backend)


def test_can_visualise_overall_segment_lengths_distributions():
    column_names = [SyntheticDataType.get_log_key_for_data_type(SyntheticDataType.raw),
                    SyntheticDataType.get_log_key_for_data_type(SyntheticDataType.normal_correlated),
                    SyntheticDataType.get_log_key_for_data_type(SyntheticDataType.non_normal_correlated),
                    SyntheticDataType.get_log_key_for_data_type(SyntheticDataType.rs_1min)]
    assert_that(vds.col_names, contains_exactly(*column_names))
    assert_that(vds.row_names, contains_exactly('Standard', 'Irregular p 0.3', 'Irregular p 0.9'))

    fig = vds.violin_plots_of_overall_segment_lengths(save_fig=False, root_result_dir=ROOT_RESULTS_DIR)
    assert_that(fig, is_not(None))


def test_can_visualise_overall_mae_distributions():
    fig = vds.violin_plots_of_overall_mae(save_fig=False, root_result_dir=ROOT_RESULTS_DIR)
    assert_that(fig, is_not(None))
