import pandas as pd
from hamcrest import *

from src.utils.configurations import DataCompleteness, GENERATED_DATASETS_FILE_PATH, SYNTHETIC_DATA_DIR, \
    CONFIRMATORY_SYNTHETIC_DATA_DIR, CONFIRMATORY_DATASETS_FILE_PATH
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_exploratory_confirmatory_datasets import VisualiseExploratoryConfirmatoryDatasets

backend = Backends.visible_tests.value
# backend = Backends.none.value

data_types = [SyntheticDataType.raw,
              SyntheticDataType.normal_correlated,
              SyntheticDataType.non_normal_correlated,
              SyntheticDataType.rs_1min]
data_completeness_levels = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]

run_names_exp = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
run_names_conf = pd.read_csv(CONFIRMATORY_DATASETS_FILE_PATH)['Name'].tolist()

vc = VisualiseExploratoryConfirmatoryDatasets(run_namex_exploratory=run_names_exp,
                                              run_name_confirmatory=run_names_conf,
                                              overall_ds_name="n30",
                                              dataset_types=data_types,
                                              completeness_levels=data_completeness_levels,
                                              exploratory_data_dir=SYNTHETIC_DATA_DIR,
                                              confirmatory_data_dir=CONFIRMATORY_SYNTHETIC_DATA_DIR,
                                              backend=backend)


def test_loads_labels_files_of_exploratory_and_confirmatory_dataset():
    exp_labels = vc.exploratory_labels[data_completeness_levels[0]]
    assert_that(len(exp_labels[SyntheticDataType.raw]), is_(30))
    assert_that(len(exp_labels[SyntheticDataType.normal_correlated]), is_(30))
    assert_that(len(exp_labels[SyntheticDataType.non_normal_correlated]), is_(30))
    assert_that(len(exp_labels[SyntheticDataType.rs_1min]), is_(30))

    con_labels = vc.confirmatory_labels[data_completeness_levels[0]]
    assert_that(len(con_labels[SyntheticDataType.raw]), is_(30))
    assert_that(len(con_labels[SyntheticDataType.normal_correlated]), is_(30))
    assert_that(len(con_labels[SyntheticDataType.non_normal_correlated]), is_(30))
    assert_that(len(con_labels[SyntheticDataType.rs_1min]), is_(30))


def test_plots_scatter_plot_of_relaxed_subject_mae():
    fig = vc.plot_relaxed_mae_per_subject_scatter_plot()

    assert_that(fig, is_not(none()))


def test_plots_scatter_plot_of_pattern_id_for_each_segment():
    fig = vc.plot_pattern_id_for_each_segment()

    assert_that(fig, is_not(none()))
