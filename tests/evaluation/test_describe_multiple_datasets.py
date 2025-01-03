from os import path

import pandas as pd
from hamcrest import *

from src.evaluation.describe_multiple_datasets import DescribeMultipleDatasets, SummaryStatistics
from src.utils.configurations import SYNTHETIC_DATA_DIR, GENERATED_DATASETS_FILE_PATH, IRREGULAR_P90, \
    dataset_description_dir, MULTIPLE_DS_SUMMARY_FILE
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_ROOT_RESULTS_DIR

run_file = GENERATED_DATASETS_FILE_PATH
data_dir = SYNTHETIC_DATA_DIR
raw = SyntheticDataType.raw
overall_ds_name = "test_stuff"
results_dir = TEST_ROOT_RESULTS_DIR
ds_raw = DescribeMultipleDatasets(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=raw,
                                  data_dir=data_dir)


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
    ds_nn = DescribeMultipleDatasets(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                     data_type=SyntheticDataType.non_normal_correlated,
                                     data_dir=data_dir)

    assert_that(len(ds_nn.run_names), is_(30))  # 30 files
    assert_that(len(ds_nn.labels), is_(30))  # 30 files
    assert_that(ds_nn.labels[ds_nn.run_names[0]].shape[0], is_(100))  # loaded the data properly

    mae_stats = ds_nn.mae_stats()
    assert_that(mae_stats["mean"], is_(0.116))
    assert_that(mae_stats["min"], is_(0.11))


def test_can_irregular_p90_non_normal_dataset():
    ds_irr_p90nn = DescribeMultipleDatasets(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                            data_type=SyntheticDataType.non_normal_correlated,
                                            data_dir=IRREGULAR_P90)

    assert_that(len(ds_irr_p90nn.run_names), is_(30))  # 30 files
    assert_that(len(ds_irr_p90nn.labels), is_(30))  # 30 files
    assert_that(ds_irr_p90nn.labels[ds_irr_p90nn.run_names[0]].shape[0], is_(100))  # loaded the data properly

    mae_stats = ds_irr_p90nn.mae_stats()
    assert_that(mae_stats["mean"], is_(0.123))


def test_creates_summary_df_of_statistics():
    df = ds_raw.summary()

    assert_that(df[SummaryStatistics.mae]["count"], is_(30))
    assert_that(df[SummaryStatistics.overall_mae]["count"], is_(30 * 100))
    assert_that(df[SummaryStatistics.seg_outside_tol]["count"], is_(30))
    assert_that(df[SummaryStatistics.observations]["count"], is_(30))
    assert_that(df[SummaryStatistics.segments]["count"], is_(30))
    assert_that(df[SummaryStatistics.patterns]["count"], is_(30))
    assert_that(df[SummaryStatistics.segment_lengths]["count"], is_(30))
    assert_that(df[SummaryStatistics.overall_segment_lengths]["count"], is_(30 * 100))


def test_saves_summary_df_of_statistics_in_provide_results_root_using_ds_description():
    ds_raw.save_summary(root_results_dir=results_dir)

    res_dir = dataset_description_dir(overall_ds_name, raw, results_dir)
    file_name = path.join(res_dir, MULTIPLE_DS_SUMMARY_FILE)
    df = pd.read_csv(file_name, index_col=0)
    assert_that(df[SummaryStatistics.mae]["count"], is_(30))
    assert_that(df[SummaryStatistics.overall_mae]["count"], is_(30 * 100))
