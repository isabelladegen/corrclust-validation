from os import path

import pandas as pd

from src.data_generation.generate_synthetic_segmented_dataset import CorrType, SyntheticDataSegmentCols
from src.evaluation.describe_subjects_for_data_variant import DescribeSubjectsForDataVariant
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, \
    IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, get_data_completeness_from, base_dataset_result_folder_for_type, \
    ResultsType
from src.utils.load_synthetic_data import SyntheticDataType
from src.visualisation.run_average_rank_visualisations import data_variant_description


def calculate_some_stats(descriptor, compl_dir, data_type):
    variant_desc = data_variant_description[(get_data_completeness_from(compl_dir), data_type)]
    mae_df = descriptor.overall_per_pattern_mae_stats(SyntheticDataSegmentCols.relaxed_mae)
    mae_df.reset_index(inplace=True)
    mae_df.insert(0, 'Data variant', variant_desc)
    n_seg_df = descriptor.per_pattern_n_segments_outside_tolerance_stats()
    n_seg_df.reset_index(inplace=True)
    n_seg_df.insert(0, 'Data variant', variant_desc)
    return mae_df, n_seg_df


if __name__ == '__main__':
    # create mae and out of segment out of tolerance table summary per pattern
    run_file = GENERATED_DATASETS_FILE_PATH
    overall_ds_name = "n30"
    root_results_dir = ROOT_RESULTS_DIR

    dataset_types = [SyntheticDataType.raw, SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated, SyntheticDataType.rs_1min]

    # descriptive names
    overall_mae_dfs = []
    n_seg_out_tol_dfs = []

    # do regular sampled ones
    for ds_type in dataset_types:
        data_dir = SYNTHETIC_DATA_DIR
        ds = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=ds_type,
                                            data_dir=data_dir, load_data=False)
        mae_df, n_seg_df = calculate_some_stats(ds, data_dir, ds_type)
        overall_mae_dfs.append(mae_df)
        n_seg_out_tol_dfs.append(n_seg_df)

    # do irregular p30 sampled ones
    for ds_type in dataset_types:
        data_dir = IRREGULAR_P30_DATA_DIR
        ds = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=ds_type,
                                            data_dir=data_dir,
                                            load_data=False)
        mae_df, n_seg_df = calculate_some_stats(ds, data_dir, ds_type)
        overall_mae_dfs.append(mae_df)
        n_seg_out_tol_dfs.append(n_seg_df)

    # do irregular p90 sampled ones
    for ds_type in dataset_types:
        data_dir = IRREGULAR_P90_DATA_DIR
        ds = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=ds_type,
                                            data_dir=data_dir, load_data=False)
        mae_df, n_seg_df = calculate_some_stats(ds, data_dir, ds_type)
        overall_mae_dfs.append(mae_df)
        n_seg_out_tol_dfs.append(n_seg_df)

    # create one big df
    combined_mae_df = pd.concat(overall_mae_dfs, ignore_index=True)
    combined_n_seg_df = pd.concat(n_seg_out_tol_dfs, ignore_index=True)

    # save results
    folder = base_dataset_result_folder_for_type(root_results_dir, ResultsType.dataset_description)
    combined_mae_df.to_csv(str(path.join(folder, "overall_mae_per_pattern_per_data_variant.csv")))
    combined_n_seg_df.to_csv(str(path.join(folder, "overall_seg_out_tolerance_per_pattern_per_data_variant.csv")))
