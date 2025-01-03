from hamcrest import *

from src.data_generation.wandb_create_bad_partitions import CreateBadPartitionsConfig, create_bad_partitions
from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.utils.wandb_utils import set_test_configurations


def test_wandb_create_bad_partitions():
    config = CreateBadPartitionsConfig()
    # this sets it to test data dir
    set_test_configurations(config)
    # config.data_dir = TEST_IRREGULAR_P90
    config.data_type = SyntheticDataType.normal_correlated  # we don't use this usually
    config.backend = Backends.none.value
    # Configure partition creation to make very few just for testing
    config.n_partitions = 3
    config.seed = 10

    # don't use misty-forest or splendid-sunrise otherwise other tests will start failing
    ds_name = "amber-glade-10"
    bad_part_summary, wandb_summary = create_bad_partitions(config, ds_name=ds_name, idx=0)

    assert_that(bad_part_summary.shape[0], is_(10))
    assert_that(bad_part_summary.loc[0, DescribeBadPartCols.jaccard_index], is_(1))
    assert_that(bad_part_summary.loc[0, DescribeBadPartCols.n_wrong_clusters], is_(0))
    assert_that(bad_part_summary.loc[0, DescribeBadPartCols.n_obs_shifted], is_(0))

    # test wandb log summary
    assert_that(wandb_summary["mean n segments within tolerance"], is_(51.0))
    assert_that(wandb_summary["median n segments outside tolerance"], is_(43.5))
    assert_that(wandb_summary["median MAE"], is_(0.249))
    assert_that(wandb_summary["std segment length"], is_(0.0))
    assert_that(wandb_summary["min Jaccard"], is_(0.0))
    assert_that(wandb_summary["mean n wrong clusters"], is_(35.3))
    assert_that(wandb_summary["std n obs shifted"], is_(333.706))
