import os
from os import path

import pandas as pd

from src.evaluation.interpretation_distance_metric_ranking import DistanceMetricInterpretation
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, get_filename_for_statistical_validation_between_measures, \
    ResultsType
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.visualisation.run_average_rank_visualisations import data_variant_description


def interpret_distance_metric_for(top_x: [], data_dirs: [], dataset_types: [], run_names: [], root_results_dir: str,
                                  distance_measures: [], overall_ds_name: str):
    stats_results = []
    # measures_to_compare
    measure1 = DistanceMeasures.l1_with_ref
    measure2 = DistanceMeasures.l1_cor_dist
    # for l2
    # measure1 = DistanceMeasures.l1_cor_dist
    # measure2 = DistanceMeasures.l2_cor_dist
    # for l3
    # measure1 = DistanceMeasures.l2_cor_dist
    # measure2 = DistanceMeasures.l3_cor_dist
    # number of tests run to adjust alpha
    # bonferroni_adjust = len(distance_measures)
    bonferroni_adjust = 1

    # run all calculation for each of the 12 data variant
    for data_dir in data_dirs:
        for data_type in dataset_types:
            interpretation = DistanceMetricInterpretation(run_names=run_names, overall_ds_name=overall_ds_name,
                                                          data_type=data_type,
                                                          data_dir=data_dir,
                                                          root_results_dir=root_results_dir,
                                                          measures=distance_measures)

            wilcox_result = interpretation.statistical_validation_of_two_measures_based_on_average_ranking(measure1,
                                                                                                           measure2)

            # calculate statistical significance
            alpha = 0.05
            target_power = 0.8
            data_variant = data_variant_description[(data_dir, data_type)]
            stats_results.append(wilcox_result.as_series(variant_name=data_variant, target_power=target_power,
                                                         alpha=alpha, bonferroni_adjust=bonferroni_adjust))

            # calculate top and bottom x distance mesures by average and criteria ranks
            for x in top_x:
                interpretation.top_and_bottom_x_distance_measures_ranks(x=x, save_results=True)

    # save statistical result
    stats_df = pd.DataFrame(stats_results)
    file_name = get_filename_for_statistical_validation_between_measures(measure1, measure2)
    folder_name = path.join(root_results_dir, ResultsType.distance_measure_evaluation)
    folder_name_res = path.join(folder_name, overall_ds_name)
    os.makedirs(folder_name_res, exist_ok=True)
    full_path = path.join(folder_name_res, file_name)
    print("save results in: " + str(full_path))
    stats_df.to_csv(str(full_path))


if __name__ == "__main__":
    # statistics and top/bottom distance measures
    top_x = [1, 2, 4]
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [
        SyntheticDataType.raw,
        SyntheticDataType.normal_correlated,
        SyntheticDataType.non_normal_correlated,
        SyntheticDataType.rs_1min
    ]
    data_dirs = [
        SYNTHETIC_DATA_DIR,
        IRREGULAR_P30_DATA_DIR,
        IRREGULAR_P90_DATA_DIR
    ]

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
                         DistanceMeasures.log_frob_cor_dist,  # correlation metrix
                         DistanceMeasures.foerstner_cor_dist]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    interpret_distance_metric_for(top_x=top_x, data_dirs=data_dirs, dataset_types=dataset_types, run_names=run_names,
                                  root_results_dir=root_result_dir, distance_measures=distance_measures,
                                  overall_ds_name="n30")
