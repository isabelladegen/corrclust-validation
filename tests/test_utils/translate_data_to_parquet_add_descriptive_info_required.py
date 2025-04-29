import os
from os import path

import pandas as pd
import pytest
from pandas.errors import EmptyDataError

from src.utils.configurations import ROOT_DIR, CONFIRMATORY_SYNTHETIC_DATA_DIR

@pytest.mark.skip(reason="this is a once off calculation to add subject id and transform to parquet")
def test_translate_a_single_file_to_parquet():
    # Source and destination directories
    source_file = path.join(ROOT_DIR, 'tests/old_test_data/synthetic_data/raw/test-run-data.csv')
    parquet_file = path.join(ROOT_DIR, 'tests/test_data/synthetic_data/raw/test-run-data.parquet')
    subject_id = 'misty-forest-56'

    # Read the CSV file
    try:
        df = pd.read_csv(source_file, index_col=0)
    except EmptyDataError:
        print(source_file)

    # Add subject_id
    df.insert(0, 'subject_id', subject_id)

    # Save files as csv
    # df.to_csv(dest_path, index=False)

    # Save files as parquet
    df.to_parquet(parquet_file, index=False, engine='pyarrow')



@pytest.mark.skip(reason="this is a once off calculation to add subject id and transform to parquet")
def test_transform_test_data_to_parquet():
    # Source and destination directories
    source_dir = path.join(ROOT_DIR, 'tests/test_data/synthetic_data')
    parquet_dest_dir = path.join(ROOT_DIR, 'tests/parquet_test_data/synthetic_data')

    os.makedirs(parquet_dest_dir, exist_ok=True)

    # Find all CSV files recursively
    csv_files = []  # list of tuples with first element being the subject name
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.csv'):
                # Get full path of the CSV file
                csv_path = os.path.join(root, file)
                filename_no_ext = os.path.splitext(file)[0]  # Remove .csv extension
                parts = filename_no_ext.split('-')
                bad_parts = False
                if "shifted" in parts or "wrong" in parts:
                    join_parts = 3
                    bad_parts = True
                elif "data" in parts:
                    join_parts = parts.index("data")
                elif "labels" in parts:
                    join_parts = parts.index("labels")
                else:
                    assert False, "Opps"
                # all words before -data, or -label
                bad_desc = '-'.join(parts[join_parts:-1]) if bad_parts else None
                subject_id = '-'.join(parts[:join_parts])  # e.g., first three parts"apricot-waterfall-16"
                csv_files.append((subject_id, bad_desc, csv_path))

    # add subject name to each file
    for subject_id, bad_desc, csv_path in csv_files:
        # Create the relative path to preserve directory structure
        rel_path = os.path.relpath(csv_path, source_dir)
        dest_path_parquet = os.path.join(parquet_dest_dir, rel_path)
        dest_path_parquet = dest_path_parquet.replace('.csv', '.parquet')

        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path_parquet), exist_ok=True)

        # Read the CSV file
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except EmptyDataError:
            print(csv_path)
            continue

        # Add subject_id
        df.insert(0, 'subject_id', subject_id)
        if bad_desc:
            df.insert(1, 'clustering_desc', bad_desc)

        # Save files as csv
        # df.to_csv(dest_path, index=False)

        # Save files as parquet
        df.to_parquet(dest_path_parquet, index=False, engine='pyarrow')

@pytest.mark.skip(reason="this is a once off calculation to add subject id and transform to parquet")
def test_load_all_files_in_exploratory_add_info_for_hugging_face():
    # Source and destination directories
    # source_dir = SYNTHETIC_DATA_DIR
    source_dir = CONFIRMATORY_SYNTHETIC_DATA_DIR
    # dest_dir = path.join(ROOT_DIR, 'new_data/exploratory')
    dest_dir = path.join(ROOT_DIR, 'new_data/confirmatory')
    parquet_dest_dir = path.join(ROOT_DIR, 'parquet_data/confirmatory')
    # parquet_dest_dir = path.join(ROOT_DIR, 'parquet_data/exploratory')

    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(parquet_dest_dir, exist_ok=True)

    # Find all CSV files recursively
    csv_files = []  # list of tuples with first element being the subject name
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if "synthetic-correlated-data-n30" in file:
                continue
            if "confirmatory-synthetic-correlated-data-n30" in file:
                continue
            if file.endswith('.csv'):
                # Get full path of the CSV file
                csv_path = os.path.join(root, file)
                filename_no_ext = os.path.splitext(file)[0]  # Remove .csv extension
                parts = filename_no_ext.split('-')
                bad_parts = False
                if "shifted" in parts or "wrong" in parts:
                    join_parts = 3
                    bad_parts = True
                elif "data" in parts:
                    join_parts = parts.index("data")
                elif "labels" in parts:
                    join_parts = parts.index("labels")
                else:
                    directory = os.path.dirname(csv_path)
                    last_folder = os.path.basename(directory)
                    if 'reduced-data' in last_folder:
                        continue
                    print(csv_path)
                    assert False, "Opps"
                # all words before -data, or -label
                bad_desc = '-'.join(parts[join_parts:-1]) if bad_parts else None
                subject_id = '-'.join(parts[:join_parts])  # e.g., first three parts"apricot-waterfall-16"
                csv_files.append((subject_id, bad_desc, csv_path))

    # add subject name to each file
    for subject_id, bad_desc, csv_path in csv_files:
        # Create the relative path to preserve directory structure
        rel_path = os.path.relpath(csv_path, source_dir)
        dest_path = os.path.join(dest_dir, rel_path)
        dest_path_parquet = os.path.join(parquet_dest_dir, rel_path)
        dest_path_parquet = dest_path_parquet.replace('.csv', '.parquet')

        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        os.makedirs(os.path.dirname(dest_path_parquet), exist_ok=True)

        # Read the CSV file
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except EmptyDataError:
            print(csv_path)
            continue

        # Add subject_id
        df.insert(0, 'subject_id', subject_id)
        if bad_desc:
            df.insert(1, 'clustering_desc', bad_desc)

        # # Save files as csv
        # df.to_csv(dest_path, index=False)

        # Save files as parquet
        df.to_parquet(dest_path_parquet, index=False, engine='pyarrow')

@pytest.mark.skip(reason="this is a once off calculation to add subject id and transform to parquet")
def test_transform_ticc_use_case_data_to_same_format():
    # Source and destination directories
    source_dir = path.join(ROOT_DIR, 'tests', 'use_case', 'ticc_test_result_labels')
    parquet_dest_dir = source_dir

    # Find all CSV files recursively
    csv_files = []  # list of tuples with first element being the subject name
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if "synthetic-correlated-data-n30" in file:
                continue
            if "confirmatory-synthetic-correlated-data-n30" in file:
                continue
            if file.endswith('.csv'):
                # Get full path of the CSV file
                csv_path = os.path.join(root, file)
                bad_desc = None
                subject_id = 'test-subject'
                csv_files.append((subject_id, bad_desc, csv_path))

    # add subject name to each file
    for subject_id, bad_desc, csv_path in csv_files:
        # Create the relative path to preserve directory structure
        rel_path = os.path.relpath(csv_path, source_dir)
        dest_path_parquet = os.path.join(parquet_dest_dir, rel_path)
        dest_path_parquet = dest_path_parquet.replace('.csv', '.parquet')

        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path_parquet), exist_ok=True)

        # Read the CSV file
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except EmptyDataError:
            print(csv_path)
            continue

        # Add subject_id
        df.insert(0, 'subject_id', subject_id)
        if bad_desc:
            df.insert(1, 'clustering_desc', bad_desc)

        # # Save files as csv
        # df.to_csv(dest_path, index=False)

        # Save files as parquet
        df.to_parquet(dest_path_parquet, index=False, engine='pyarrow')