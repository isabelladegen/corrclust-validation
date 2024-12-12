from os import path

from hamcrest import *

from src.utils.configurations import PATTERNS_TO_MODEL_PATH, WandbConfiguration


def test_wandb_configuration_loaded():
    # if this test fails you have not added a correct private.yaml file where you
    # entered your wandb credits --> see private-yaml-template.yaml
    # This is only necessary if you want to generate your own synthetic data
    assert_that(WandbConfiguration.wandb_project_name, is_(not_(empty())))
    assert_that(WandbConfiguration.wandb_entity, is_(not_(empty())))


def test_patterns_to_model_file_exists():
    assert_that(path.exists(PATTERNS_TO_MODEL_PATH))
