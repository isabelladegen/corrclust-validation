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
    for cor in all_cor:
        overall_mae_per_cor_per_data_variant[cor][variant_desc] = descriptor.overall_mae_stats(
            SyntheticDataSegmentCols.relaxed_mae, corr_type=cor)
        n_seg_out_tol_per_cor_per_data_variant[cor][variant_desc] = descriptor.n_segments_outside_tolerance_stats(
            corr_type=cor)


def combine_into_pivoted_df(nested_dict):
    rows = []

    # Iterate through the nested dictionary
    for cor in nested_dict:  # correlation level
        for data_variant in nested_dict[cor]:  # data variant level
            # Get the describe series
            describe_series = nested_dict[cor][data_variant]

            # Create a row with multi-index
            for stat_name, stat_value in describe_series.items():
                rows.append({
                    'correlation': cor,
                    'data_variant': data_variant,
                    'statistic': stat_name,
                    'value': stat_value
                })

    # Create the DataFrame from the list of dictionaries
    df = pd.DataFrame(rows)

    # You can reshape this into a more readable format if needed
    pivoted_df = df.pivot_table(
        index=['correlation', 'data_variant'],
        columns='statistic',
        values='value'
    )

    # Reset index for a flatter structure if preferred
    final_df = pivoted_df.reset_index()
    return final_df


if __name__ == '__main__':
    # create summary for a dataset variation
    run_file = GENERATED_DATASETS_FILE_PATH
    overall_ds_name = "n30"
    root_results_dir = ROOT_RESULTS_DIR
    additional_cor = [CorrType.pearson, CorrType.kendall]
    all_cor = [CorrType.spearman, CorrType.pearson, CorrType.kendall]

    dataset_types = [SyntheticDataType.raw, SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated, SyntheticDataType.rs_1min]

    # descriptive names
    overall_mae_per_cor_per_data_variant = {cor: {} for cor in all_cor}
    n_seg_out_tol_per_cor_per_data_variant = {cor: {} for cor in all_cor}

    # do regular sampled ones
    for ds_type in dataset_types:
        data_dir = SYNTHETIC_DATA_DIR
        ds = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=ds_type,
                                            data_dir=data_dir, additional_corr=additional_cor, load_data=True)
        calculate_some_stats(ds, data_dir, ds_type)

    # do irregular p30 sampled ones
    for ds_type in dataset_types:
        data_dir = IRREGULAR_P30_DATA_DIR
        ds = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=ds_type,
                                            data_dir=data_dir, additional_corr=additional_cor,
                                            load_data=True)
        calculate_some_stats(ds, data_dir, ds_type)

    # do irregular p90 sampled ones
    for ds_type in dataset_types:
        data_dir = IRREGULAR_P90_DATA_DIR
        ds = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=ds_type,
                                            data_dir=data_dir, additional_corr=additional_cor,
                                            load_data=True)
        calculate_some_stats(ds, data_dir, ds_type)

    # save results
    folder = base_dataset_result_folder_for_type(root_results_dir, ResultsType.dataset_description)
    overall_mae_df = combine_into_pivoted_df(overall_mae_per_cor_per_data_variant)
    overall_mae_df.to_csv(str(path.join(folder, "overall_mae_per_cor_per_data_variant.csv")))
    overall_seg_tol_df = combine_into_pivoted_df(n_seg_out_tol_per_cor_per_data_variant)
    overall_seg_tol_df.to_csv(str(path.join(folder, "overall_seg_out_tolerance_per_cor_per_data_variant.csv")))
