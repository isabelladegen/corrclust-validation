import os

import pandas as pd
import pytest
from hamcrest import *
from pyarrow import ArrowInvalid

from src.utils.configurations import SYNTHETIC_DATA_DIR, CONFIRMATORY_SYNTHETIC_DATA_DIR

@pytest.mark.skip(reason="takes a long time and is not required everytime")
def test_can_read_all_exploratory_data_and_labels_files():
    source_dir = SYNTHETIC_DATA_DIR

    corrupt_files = read_all_files_in(source_dir)

    print(corrupt_files)
    assert_that(len(corrupt_files), is_(0))

@pytest.mark.skip(reason="takes a long time and is not required everytime")
def test_can_read_all_confirmatory_data_and_labels_files():
    source_dir = CONFIRMATORY_SYNTHETIC_DATA_DIR

    corrupt_files = read_all_files_in(source_dir)

    print(corrupt_files)
    assert_that(len(corrupt_files), is_(0))


def read_all_files_in(source_dir):
    corrupt_files = []
    # check files recursively
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # test if we can read the parquet file
            if file.endswith('.parquet'):
                # Get full path of the CSV file
                file_path = os.path.join(root, file)

                try:
                    pd.read_parquet(str(file_path)).head(0)
                except ArrowInvalid:
                    corrupt_files.append(file_path)
    return corrupt_files
