from os import path
from pathlib import Path

from src.utils.configurations import ROOT_DIR

TEST_DATA_DIR = path.join(ROOT_DIR, 'tests/test_data/synthetic_data/')
TEST_IMAGES_DIR = path.join(ROOT_DIR, 'tests/images/')
Path(TEST_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
TEST_TABLES_DIR = path.join(ROOT_DIR, 'tests/tables/')
Path(TEST_TABLES_DIR).mkdir(parents=True, exist_ok=True)

