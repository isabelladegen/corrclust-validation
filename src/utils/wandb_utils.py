from src.utils.plots.matplotlib_helper_functions import Backends
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR


def set_test_configurations(config):
    """Does the necessary configuration changes for tests"""
    config.wandb_mode = "offline"  # don't log test runs
    config.wandb_notes = "unit testing"
    config.local_image_folder_name = "wandb-images/unit-tests"
    config.backend = Backends.none.value  # change this if debugging plots
    # Store results
    config.data_dir = TEST_DATA_DIR
