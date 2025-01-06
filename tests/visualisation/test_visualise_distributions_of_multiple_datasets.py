from hamcrest import *

from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_distributions_of_multiple_datasets import VisualiseDistributionsOfMultipleDatasets

# backend = Backends.none.value
backend = Backends.visible_tests.value


def test_can_visualise_standard_variations_of_distributions():
    vds = VisualiseDistributionsOfMultipleDatasets(run_file=GENERATED_DATASETS_FILE_PATH, overall_ds_name="n30",
                                                   dataset_types=[SyntheticDataType.raw,
                                                                  SyntheticDataType.normal_correlated,
                                                                  SyntheticDataType.non_normal_correlated,
                                                                  SyntheticDataType.rs_1min],
                                                   data_dir=SYNTHETIC_DATA_DIR, backend=backend)

    ds_variates = vds.dataset_variates
    assert_that(len(ds_variates), is_(4))
    assert_that(ds_variates[vds.col_names[0]].get_list_of_xtrain_of_all_datasets()[0].shape[0], greater_than(1200000))


def test_can_visualise_irregular_variations_of_distributions():
    vds = VisualiseDistributionsOfMultipleDatasets(run_file=GENERATED_DATASETS_FILE_PATH, overall_ds_name="n30",
                                                   dataset_types=[SyntheticDataType.raw,
                                                                  SyntheticDataType.normal_correlated,
                                                                  SyntheticDataType.non_normal_correlated,
                                                                  SyntheticDataType.rs_1min],
                                                   data_dir=IRREGULAR_P90_DATA_DIR,
                                                   backend=backend)

    ds_variates = vds.dataset_variates
    assert_that(len(ds_variates), is_(4))
    assert_that(ds_variates[vds.col_names[0]].get_list_of_xtrain_of_all_datasets()[0].shape[0], less_than(200000))

    fig = vds.plot_as_standard_distribution()
    fig.savefig("test.png", dpi=300, bbox_inches='tight')
    assert_that(fig, is_not(None))
