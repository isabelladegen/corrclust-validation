from hamcrest import *

from src.data_generation.wandb_create_bad_partitions import CreateBadPartitionsConfig, create_bad_partitions
from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.utils.wandb_utils import set_test_configurations
from tests.test_utils.configurations_for_testing import TEST_GENERATED_DATASETS_FILE_PATH


def test_wandb_create_bad_partitions():
    config = CreateBadPartitionsConfig()
    # this sets it to test data dir
    set_test_configurations(config)
    config.csv_of_runs = TEST_GENERATED_DATASETS_FILE_PATH
    config.data_type = SyntheticDataType.normal_correlated  # we don't use this usually
    config.backend = Backends.none.value
    # Configure partition creation to make very few just for testing
    config.n_partitions = 3
    config.seed = 10

    results_tables, wandb_summary = create_bad_partitions(config)

    assert_that(len(results_tables), is_(2))

    tables = list(results_tables.values())
    assert_that(tables[0].shape[0], is_(10))
    assert_that(tables[0].loc[0, DescribeBadPartCols.jaccard_index], is_(1))
    assert_that(tables[0].loc[0, DescribeBadPartCols.n_wrong_clusters], is_(0))
    assert_that(tables[0].loc[0, DescribeBadPartCols.n_obs_shifted], is_(0))

    assert_that(tables[1].shape[0], is_(10))
    assert_that(tables[1].loc[0, DescribeBadPartCols.jaccard_index], is_(1))
    assert_that(tables[1].loc[0, DescribeBadPartCols.n_wrong_clusters], is_(0))
    assert_that(tables[1].loc[0, DescribeBadPartCols.n_obs_shifted], is_(0))

