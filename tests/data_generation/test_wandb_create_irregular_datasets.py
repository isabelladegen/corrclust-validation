from hamcrest import *

from src.data_generation.wandb_create_irregular_datasets import create_irregular_datasets, CreateIrregularDSConfig
from src.evaluation.describe_synthetic_dataset import DescribeSyntheticDataset
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.wandb_utils import set_test_configurations
from tests.test_utils.configurations_for_testing import TEST_IRREGULAR_P90_DATA_DIR


def test_wandb_create_irregular_datasets_version_for_one_run():
    config = CreateIrregularDSConfig()
    set_test_configurations(config)
    config.root_result_data_dir = TEST_IRREGULAR_P90_DATA_DIR
    config.p = 0.9

    # evaluation is None if the run fails
    ds_name = "misty-forest-56"
    results, wandb_summary = create_irregular_datasets(config=config, ds_name=ds_name, seed=10)
    raw_describe = results['raw']
    nc_describe = results['nc']
    nn_describe = results['nn']
    rs_describe = results['rs']

    # Description of original data
    orig_raw_ds = DescribeSyntheticDataset(ds_name, data_type=SyntheticDataType.raw, data_dir=config.data_dir)
    orig_nn_ds = DescribeSyntheticDataset(ds_name, data_type=SyntheticDataType.normal_correlated,
                                          data_dir=config.data_dir)
    orig_nc_ds = DescribeSyntheticDataset(ds_name, data_type=SyntheticDataType.non_normal_correlated,
                                          data_dir=config.data_dir)
    orig_rs_ds = DescribeSyntheticDataset(ds_name, data_type=SyntheticDataType.rs_1min, data_dir=config.data_dir)

    # check differences between the original and irregular descriptions
    # less observations in irregular versions
    assert_that(raw_describe.number_of_observations, less_than(orig_raw_ds.number_of_observations))
    assert_that(nc_describe.number_of_observations, less_than(orig_nc_ds.number_of_observations))
    assert_that(nn_describe.number_of_observations, less_than(orig_nn_ds.number_of_observations))
    assert_that(rs_describe.number_of_observations, less_than(orig_rs_ds.number_of_observations))

    # mean correlation error is higher in irregular versions
    assert_that(raw_describe.mae_stats['mean'], greater_than(orig_raw_ds.mae_stats['mean']))
    assert_that(nc_describe.mae_stats['mean'], greater_than(orig_nc_ds.mae_stats['mean']))
    assert_that(nn_describe.mae_stats['mean'], greater_than(orig_nn_ds.mae_stats['mean']))
    assert_that(rs_describe.mae_stats['mean'], greater_than(orig_rs_ds.mae_stats['mean']))

    # test wandb logs created, we only test some, and that they come from the right version of datasets
    assert_that(wandb_summary["seed"], is_(10))
    assert_that(wandb_summary["n observations RAW"], is_(raw_describe.number_of_observations))
    assert_that(wandb_summary["frequency RAW"], is_(raw_describe.frequency))
    assert_that(wandb_summary["max MAE RAW"], is_(raw_describe.mae_stats['max']))
    assert_that(wandb_summary["n observations NC"], is_(nc_describe.number_of_observations))
    assert_that(wandb_summary["mean MAE NC"], is_(nc_describe.mae_stats['mean']))
    assert_that(wandb_summary["n observations NN"], is_(nn_describe.number_of_observations))
    assert_that(wandb_summary["frequency NN"], is_(nn_describe.frequency))
    assert_that(wandb_summary["min MAE NN"], is_(nn_describe.mae_stats['min']))
    assert_that(wandb_summary["n observations RS"], is_(rs_describe.number_of_observations))
    assert_that(wandb_summary["frequency RS"], is_(rs_describe.frequency))
    assert_that(wandb_summary["mean MAE RS"], is_(rs_describe.mae_stats['mean']))
