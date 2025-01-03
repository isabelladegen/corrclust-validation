from hamcrest import *

from src.evaluation.describe_multiple_datasets import DescribeMultipleDatasets
from src.utils.configurations import SYNTHETIC_DATA_DIR, GENERATED_DATASETS_FILE_PATH, IRREGULAR_P90
from src.utils.load_synthetic_data import SyntheticDataType

run_file = GENERATED_DATASETS_FILE_PATH
data_dir = SYNTHETIC_DATA_DIR
ds_raw = DescribeMultipleDatasets(wandb_run_file=run_file, data_type=SyntheticDataType.raw, data_dir=data_dir)


# these tests read real data, but they save results in a test result folder!
def test_can_load_base_raw_datasets_and_return_ds_variation_mae():
    assert_that(len(ds_raw.run_names), is_(30))  # 30 files
    assert_that(len(ds_raw.labels), is_(30))  # 30 files
    assert_that(ds_raw.labels[ds_raw.run_names[0]].shape[0], is_(100))  # loaded the data properly

    mae_stats = ds_raw.mae_stats()
    assert_that(mae_stats["mean"], is_(0.613))
    assert_that(mae_stats["std"], is_(0.006))


def test_calculates_various_stats_on_across_the_datasets():
    n_outside = ds_raw.n_segments_outside_tolerance_stats()
    assert_that(n_outside["mean"], is_(95.6))
    assert_that(n_outside["std"], is_(0.498))
    assert_that(n_outside["min"], is_(95))

    assert_that(ds_raw.overall_mae_stats()["mean"], is_(0.613))
    assert_that(ds_raw.overall_mae_stats()["min"], is_(0.0))
    assert_that(ds_raw.mae_stats()["min"], is_(0.602))

    assert_that(ds_raw.observations_stats()["mean"], is_(1264010.0))
    assert_that(ds_raw.n_patterns_stats()["mean"], is_(23.0))
    assert_that(ds_raw.segment_length_stats()["mean"], is_(12640.1))
    assert_that(ds_raw.segment_length_stats()["min"], is_(12438.0))  # across ds
    assert_that(ds_raw.overall_segment_length_stats()["mean"], is_(12640.1))
    assert_that(ds_raw.overall_segment_length_stats()["min"], is_(900))  # considering all segment lengths


def test_can_load_base_non_normal_datasets():
    ds_nn = DescribeMultipleDatasets(wandb_run_file=run_file, data_type=SyntheticDataType.non_normal_correlated,
                                     data_dir=data_dir)

    assert_that(len(ds_nn.run_names), is_(30))  # 30 files
    assert_that(len(ds_nn.labels), is_(30))  # 30 files
    assert_that(ds_nn.labels[ds_nn.run_names[0]].shape[0], is_(100))  # loaded the data properly

    mae_stats = ds_nn.mae_stats()
    assert_that(mae_stats["mean"], is_(0.116))
    assert_that(mae_stats["min"], is_(0.11))


def test_can_irregular_p90_non_normal_dataset():
    ds_irr_p90nn = DescribeMultipleDatasets(wandb_run_file=run_file, data_type=SyntheticDataType.non_normal_correlated,
                                            data_dir=IRREGULAR_P90)

    assert_that(len(ds_irr_p90nn.run_names), is_(30))  # 30 files
    assert_that(len(ds_irr_p90nn.labels), is_(30))  # 30 files
    assert_that(ds_irr_p90nn.labels[ds_irr_p90nn.run_names[0]].shape[0], is_(100))  # loaded the data properly

    mae_stats = ds_irr_p90nn.mae_stats()
    assert_that(mae_stats["mean"], is_(0.123))
