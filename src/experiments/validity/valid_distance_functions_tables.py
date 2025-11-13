import os
from os import path

import numpy as np
import pandas as pd

from src.evaluation.distance_metric_evaluation import EvaluationCriteria, read_csv_of_raw_values_for_all_criteria
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, ResultsType
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

threshold_values = {
    EvaluationCriteria.inter_i: 0.1,
    EvaluationCriteria.inter_ii: 1.0,
    EvaluationCriteria.inter_iii: 0.7,
    EvaluationCriteria.disc_i: 4,
    EvaluationCriteria.disc_ii: 3,
    EvaluationCriteria.disc_iii: 0.98,
}

df_thresholds = {
    EvaluationCriteria.inter_i: lambda x: x <= threshold_values[EvaluationCriteria.inter_i],
    EvaluationCriteria.inter_ii: lambda x: x == threshold_values[EvaluationCriteria.inter_ii],
    EvaluationCriteria.inter_iii: lambda x: x > threshold_values[EvaluationCriteria.inter_iii],
    EvaluationCriteria.disc_i: lambda x: x > threshold_values[EvaluationCriteria.disc_i],
    EvaluationCriteria.disc_ii: lambda x: x < threshold_values[EvaluationCriteria.disc_i],
    EvaluationCriteria.disc_iii: lambda x: x > threshold_values[EvaluationCriteria.disc_iii],
}


def calculate_mean_sd(measures, run_names, data_type, data_dir, root_results_dir):
    # Load all raw_criteria_data for this data variant
    measures = measures
    criteria = [EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii, EvaluationCriteria.inter_iii,
                EvaluationCriteria.disc_i, EvaluationCriteria.disc_ii, EvaluationCriteria.disc_iii]
    raw_dfs = []
    for run_name in run_names:
        raw_criteria_df = read_csv_of_raw_values_for_all_criteria(run_name=run_name, data_type=data_type,
                                                                  data_dir=data_dir,
                                                                  base_results_dir=root_results_dir)
        # filter measures and criteria
        raw_dfs.append(raw_criteria_df.loc[criteria, measures])

    # Stack all DataFrames along a new axis
    stacked_data = np.stack([df.values for df in raw_dfs])

    # Calculate mean along the first axis (across DataFrames - subjects)
    mean_values = np.round(np.mean(stacked_data, axis=0).astype(float), 2)

    # Create new DataFrame with median values
    # Using the column names and index from the first DataFrame
    mean_df = pd.DataFrame(
        mean_values,
        columns=raw_dfs[0].columns,
        index=raw_dfs[0].index
    ).T

    # Calculate SD
    sd_values = np.round(np.std(stacked_data.astype(float), axis=0, ddof=1), 2)
    sd_df = pd.DataFrame(
        sd_values,
        columns=raw_dfs[0].columns,
        index=raw_dfs[0].index
    ).T

    result_df = mean_df.copy()

    for col in mean_df.columns:
        for idx in mean_df.index:
            mean_val = mean_df.loc[idx, col]
            sd_val = sd_df.loc[idx, col]

            add_star = df_thresholds.get(col, lambda x: False)(mean_val)
            star = "*" if add_star else ""

            result_df.loc[idx, col] = f"{mean_val} (SD {sd_val}){star}"

    # rename columns
    column_names = {
        EvaluationCriteria.inter_i: "d L0",
        EvaluationCriteria.inter_ii: "avg d",
        EvaluationCriteria.inter_iii: "r",
        EvaluationCriteria.disc_i: "H_D",
        EvaluationCriteria.disc_ii: "H_L",
        EvaluationCriteria.disc_iii: "F1",
    }

    result_df = result_df.rename(columns=column_names)

    return result_df


if __name__ == "__main__":
    root_result_dir = ROOT_RESULTS_DIR

    # this is an extensive list
    distance_measures = [DistanceMeasures.l1_cor_dist,  # lp norms
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.linf_cor_dist,
                         DistanceMeasures.l1_with_ref,  # lp norms with reference vector
                         DistanceMeasures.l2_with_ref,
                         DistanceMeasures.l3_with_ref,
                         DistanceMeasures.l5_with_ref,
                         DistanceMeasures.linf_with_ref,
                         DistanceMeasures.dot_transform_l1,  # dot transform + lp norms
                         DistanceMeasures.dot_transform_l2,
                         DistanceMeasures.dot_transform_linf,
                         DistanceMeasures.log_frob_cor_dist,  # correlation metrics
                         DistanceMeasures.foerstner_cor_dist]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    save_to_folder = path.join(root_result_dir, ResultsType.distance_measure_evaluation, 'validity-outcomes')
    os.makedirs(save_to_folder, exist_ok=True)

    mean_sd_df_normal_100 = calculate_mean_sd(distance_measures, run_names, SyntheticDataType.normal_correlated,
                                              SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR)
    mean_sd_df_normal_100.to_csv(path.join(save_to_folder, 'construct-mean_sd_normal_100.csv'))

    # discriminant
    mean_sd_df_raw_100 = calculate_mean_sd(distance_measures, run_names, SyntheticDataType.raw, SYNTHETIC_DATA_DIR,
                                           ROOT_RESULTS_DIR)
    mean_sd_df_raw_100.to_csv(path.join(save_to_folder, 'discriminant-mean_sd_raw_100.csv'))

    mean_sd_df_downsampled_100 = calculate_mean_sd(distance_measures, run_names, SyntheticDataType.rs_1min,
                                                   SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR)
    mean_sd_df_downsampled_100.to_csv(path.join(save_to_folder, 'discriminant-mean_sd_downsampled_100.csv'))

    # external validity
    mean_sd_df_normal_70 = calculate_mean_sd(distance_measures, run_names, SyntheticDataType.normal_correlated,
                                             IRREGULAR_P30_DATA_DIR, ROOT_RESULTS_DIR)
    mean_sd_df_normal_70.to_csv(path.join(save_to_folder, 'external-mean_sd_normal_70.csv'))

    mean_sd_df_normal_10 = calculate_mean_sd(distance_measures, run_names, SyntheticDataType.normal_correlated,
                                             IRREGULAR_P90_DATA_DIR, ROOT_RESULTS_DIR)
    mean_sd_df_normal_10.to_csv(path.join(save_to_folder, 'external-mean_sd_normal_10.csv'))

    mean_sd_df_non_normal_100 = calculate_mean_sd(distance_measures, run_names, SyntheticDataType.non_normal_correlated,
                                                  SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR)
    mean_sd_df_non_normal_100.to_csv(path.join(save_to_folder, 'external-mean_sd_non_normal_100.csv'))

    mean_sd_df_non_normal_10 = calculate_mean_sd(distance_measures, run_names, SyntheticDataType.non_normal_correlated,
                                                 IRREGULAR_P90_DATA_DIR, ROOT_RESULTS_DIR)
    mean_sd_df_non_normal_10.to_csv(path.join(save_to_folder, 'external-mean_sd_non_normal_10.csv'))

