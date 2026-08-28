import os
from os import path

import numpy as np
import pandas as pd

from src.evaluation.distance_metric_evaluation import read_csv_of_raw_values_for_all_criteria
from src.experiments.validity.distance_measure_validity import DistanceMeasureValidity, DM_THRESHOLDS, \
    INITIAL_PAPER_RULE, DROPPING_OVERALL_ENTROPY, STRICT_MUST_PASS_RULES, REVIEWED_RULES
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, ResultsType, Aggregators
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType


def calculate_mean_sd_min_max(measures, run_names, data_type, data_dir, root_results_dir):
    """Mean, SD, min, max across subjects, for one data variant. Returns a MultiIndex-column
        df: level 0 = EvaluationCriteria, level 1 = Aggregators stat, index = distance_measure."""
    # Load all raw_criteria_data for this data variant
    measures = measures

    raw_dfs = []
    for run_name in run_names:
        raw_criteria_df = read_csv_of_raw_values_for_all_criteria(run_name=run_name, data_type=data_type,
                                                                  data_dir=data_dir,
                                                                  base_results_dir=root_results_dir)
        # filter measures and criteria
        raw_dfs.append(raw_criteria_df.loc[list(DM_THRESHOLDS.keys()), measures])

    # Stack all DataFrames along a new axis
    stacked_data = np.stack([df.values for df in raw_dfs]).astype(float)  # (subjects, criteria, measures)

    stat_values = {
        Aggregators.mean: np.mean(stacked_data, axis=0),
        Aggregators.std: np.std(stacked_data, axis=0, ddof=1),
        Aggregators.min: np.min(stacked_data, axis=0),
        Aggregators.max: np.max(stacked_data, axis=0),
    }

    # calculate all stats
    stat_dfs = []
    for stat, values in stat_values.items():
        df = pd.DataFrame(np.round(values, 2), columns=raw_dfs[0].columns, index=raw_dfs[0].index).T
        df.columns = pd.MultiIndex.from_product([df.columns, [stat]])
        stat_dfs.append(df)

    return pd.concat(stat_dfs, axis=1).sort_index(axis=1, level=0)


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

    variants = {
        'normal_100': (SyntheticDataType.normal_correlated, SYNTHETIC_DATA_DIR),
        'normal_70': (SyntheticDataType.normal_correlated, IRREGULAR_P30_DATA_DIR),
        'normal_10': (SyntheticDataType.normal_correlated, IRREGULAR_P90_DATA_DIR),
        'non_normal_100': (SyntheticDataType.non_normal_correlated, SYNTHETIC_DATA_DIR),
        'non_normal_10': (SyntheticDataType.non_normal_correlated, IRREGULAR_P90_DATA_DIR),
        'raw_100': (SyntheticDataType.raw, SYNTHETIC_DATA_DIR),
        'downsampled_100': (SyntheticDataType.rs_1min, SYNTHETIC_DATA_DIR),
    }

    # calculate and save stats df
    stats = {}
    for name, (data_type, data_dir) in variants.items():
        stats[name] = calculate_mean_sd_min_max(distance_measures, run_names, data_type, data_dir, root_result_dir)
        stats[name].to_csv(path.join(save_to_folder, f'summary_statistics_{name}.csv'))

    # create validity assessment class
    validity = DistanceMeasureValidity(validity_rule=REVIEWED_RULES,
                                       normal_100=stats['normal_100'],
                                       normal_70=stats['normal_70'],
                                       normal_10=stats['normal_10'], non_normal_100=stats['non_normal_100'],
                                       non_normal_10=stats['non_normal_10'], raw_100=stats['raw_100'],
                                       downsampled_100=stats['downsampled_100'])

    # save mean (sd) * table for each variant considered
    for name in variants:
        validity.mean_sd_valid_summary_table(stats[name]).to_csv(
            path.join(save_to_folder, f'mean_sd_valid_{name}.csv'))

    # overall validity results
    validity.overall_validity().to_csv(path.join(save_to_folder, 'overall_validity_results.csv'))
    validity.external_validity_details().to_csv(path.join(save_to_folder, 'external_validity_results.csv'))
    validity.discriminant_validity_details().to_csv(path.join(save_to_folder, 'discriminant_validity_results.csv'))
