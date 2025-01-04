from os import path
from pathlib import Path

from src.utils.configurations import ROOT_DIR

TEST_DATA_DIR = path.join(ROOT_DIR, 'tests/test_data/synthetic_data')
TEST_IRREGULAR_P90_DATA_DIR = path.join(TEST_DATA_DIR, 'irregular_p90')
TEST_IRREGULAR_P30_DATA_DIR = path.join(TEST_DATA_DIR, 'irregular_p30')
TEST_IMAGES_DIR = path.join(ROOT_DIR, 'tests/images')
Path(TEST_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
TEST_ROOT_RESULTS_DIR = path.join(ROOT_DIR, 'tests/results')
TEST_IRREGULAR_P30_ROOT_RESULTS_DIR = path.join(TEST_ROOT_RESULTS_DIR, 'irregular_p30')
TEST_IRREGULAR_P90_ROOT_RESULTS_DIR = path.join(TEST_ROOT_RESULTS_DIR, 'irregular_p90')
Path(TEST_ROOT_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
TEST_GENERATED_DATASETS_FILE_PATH = path.join(ROOT_DIR, 'tests/test_data/config/test-create-bad-partitions.csv')
TEST_GENERATED_DATASETS_FILE_PATH_1 = path.join(ROOT_DIR, 'tests/test_data/config/test-create-bad-partitions_1.csv')


