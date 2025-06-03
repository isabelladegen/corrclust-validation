import os
from os import path

import pandas as pd

from src.evaluation.interpretation_distance_metric_ranking import DistanceMetricInterpretation

from src.utils.configurations import CONFIRMATORY_DATASETS_FILE_PATH, CONF_ROOT_RESULTS_DIR, \
    CONF_IRREGULAR_P90_DATA_DIR, CONF_IRREGULAR_P30_DATA_DIR, CONFIRMATORY_SYNTHETIC_DATA_DIR, \
    get_data_completeness_from, ResultsType, DISTANCE_MEASURE_AVG_RANK_STATS_VALIDATION
from src.utils.distance_measures import DistanceMeasures, short_distance_measure_names
from src.utils.load_synthetic_data import SyntheticDataType
from src.visualisation.run_average_rank_visualisations import data_variant_description


def run_wilcox_signed_rank_tests_for_hypotheses(prereg_hypotheses: [], run_names: [], root_results_dir: str,
                                                overall_ds_name: str):
    stats_results = []
    alpha = 0.05
    target_power = 0.8
    bonferroni_adjust = 1  # no adjustment due to hierarchical testing
    alternative = "less"  # one-sided confirmatory tests (h[2] better ranked than h[3]

    # Calculate stats
    for h in prereg_hypotheses:
        # h e.g  (downsampled, sparse, DistanceMeasures.l1_cor_dist, DistanceMeasures.l2_cor_dist)
        data_type = h[0]
        data_dir = h[1]
        m1 = h[2]
        m2 = h[3]
        interpretation = DistanceMetricInterpretation(run_names=run_names, overall_ds_name=overall_ds_name,
                                                      data_type=data_type,
                                                      data_dir=data_dir,
                                                      root_results_dir=root_results_dir,
                                                      measures=[m1, m2])

        # run one-sided tests predicting h[2] lower ranked than h[3]
        wilcox_result = interpretation.statistical_validation_of_two_measures_based_on_average_ranking(measure1=m1,
                                                                                                       measure2=m2,
                                                                                                       alternative=alternative)

        # calculate statistical significance
        data_variant = data_variant_description[(get_data_completeness_from(data_dir), data_type)]
        results = wilcox_result.as_series(variant_name=data_variant, target_power=target_power, alpha=alpha,
                                         bonferroni_adjust=bonferroni_adjust, alternative=alternative)
        results["H"] = short_distance_measure_names[m1] + " < " + short_distance_measure_names[m2]

        stats_results.append(results)

    # Save result
    stats_df = pd.DataFrame(stats_results)
    folder_name = path.join(root_results_dir, ResultsType.distance_measure_evaluation)
    folder_name_res = path.join(folder_name, overall_ds_name)
    os.makedirs(folder_name_res, exist_ok=True)
    full_path = path.join(folder_name_res, DISTANCE_MEASURE_AVG_RANK_STATS_VALIDATION)
    print("save results in: " + str(full_path))
    stats_df.to_csv(str(full_path))


if __name__ == "__main__":
    # Confirm preregistered tests from exploratory phase for confirmatory data
    overall_dataset_name = "n30"
    run_names = pd.read_csv(CONFIRMATORY_DATASETS_FILE_PATH)['Name'].tolist()
    root_result_dir = CONF_ROOT_RESULTS_DIR

    downsampled = SyntheticDataType.rs_1min
    non_normal = SyntheticDataType.non_normal_correlated
    normal = SyntheticDataType.normal_correlated
    sparse = CONF_IRREGULAR_P90_DATA_DIR
    partial = CONF_IRREGULAR_P30_DATA_DIR
    complete = CONFIRMATORY_SYNTHETIC_DATA_DIR

    # preregistered hypotheses, sequential list of tuples (data_type, data_dir, measure 1, measure 2)
    hypotheses = [
        (downsampled, sparse, DistanceMeasures.l1_cor_dist, DistanceMeasures.l2_cor_dist),
        (non_normal, partial, DistanceMeasures.l1_with_ref, DistanceMeasures.l1_cor_dist),
        (normal, partial, DistanceMeasures.l1_cor_dist, DistanceMeasures.l1_with_ref),
        (downsampled, partial, DistanceMeasures.l1_cor_dist, DistanceMeasures.l2_cor_dist),
        (downsampled, complete, DistanceMeasures.l1_cor_dist, DistanceMeasures.l2_cor_dist),
        (non_normal, complete, DistanceMeasures.l1_with_ref, DistanceMeasures.l1_cor_dist),
        (normal, sparse, DistanceMeasures.l1_cor_dist, DistanceMeasures.l1_with_ref),
        (non_normal, sparse, DistanceMeasures.l1_with_ref, DistanceMeasures.l1_cor_dist),
        (normal, complete, DistanceMeasures.l1_cor_dist, DistanceMeasures.l1_with_ref),
    ]

    # evaluate all hypotheses
    run_wilcox_signed_rank_tests_for_hypotheses(prereg_hypotheses=hypotheses, run_names=run_names,
                                                root_results_dir=root_result_dir, overall_ds_name=overall_dataset_name)
