import os
from os import path

import pandas as pd

from src.evaluation.distance_metric_evaluation import criteria_short_names
from src.evaluation.internal_measure_ground_truth_assessment import InternalMeasureGroundTruthAssessment, \
    internal_measure_ranking_method
from src.evaluation.interpretation_distance_metric_ranking import DistanceMetricInterpretation
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, base_dataset_result_folder_for_type, ResultsType, \
    AVERAGE_RANK_DISTRIBUTION, HEATMAP_OF_RANKS, HEATMAP_OF_BEST_MEASURES_RAW_VALUES, GROUND_TRUTH_HEATMAP_OF_RANKS, \
    GROUND_TRUTH_HEATMAP_RAW_VALUES
from src.utils.distance_measures import DistanceMeasures, short_distance_measure_names
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.run_average_rank_visualisations import data_variant_description
from src.visualisation.run_heatmap_of_average_distance_measure_ranks import pattern_keys_ordered, plot_ranking_heat_map
from src.visualisation.visualise_distance_measure_rank_distributions import heatmap_of_ranks, heatmap_of_raw_values


def ground_truth_heatmap_for_all_variants(data_dirs, dataset_types, root_results_dir, distance_measures,
                                          internal_measures, overall_ds_name, backend, stats_value="50%",
                                          save_fig=True):
    # create variant description that serve as keys
    variant_descriptions = []
    for data_dir in data_dirs:
        for data_type in dataset_types:
            variant_desc = data_variant_description[(data_dir, data_type)]

    # create variant description dictionary
    # key= variant name, value = None (will be df once created)
    dict_for_variant = {desc: None for desc in variant_descriptions}

    # initialise results
    # key=internal measure name, value = dict with key=variant description, value will be rank df
    all_ranks = {measure: dict_for_variant.copy() for measure in internal_measures}
    # key=internal measure name, value = dict with key=variant description, value will be raw values df
    all_raw_values = {measure: dict_for_variant.copy() for measure in internal_measures}

    # calculate ranks & raw values
    for data_dir in data_dirs:
        for data_type in dataset_types:
            ga = InternalMeasureGroundTruthAssessment(overall_ds_name=overall_ds_name,
                                                      internal_measures=internal_measures,
                                                      distance_measures=distance_measures,
                                                      data_dir=data_dir,
                                                      data_type=data_type,
                                                      root_results_dir=root_results_dir)
            variant_desc = data_variant_description[(data_dir, data_type)]
            ranks_for_variant = ga.stats_for_ranks_across_all_runs()
            raw_values_for_variant = ga.stats_for_raw_values_across_all_runs()

            for internal_measure in internal_measures:
                all_ranks[internal_measure][variant_desc] = ranks_for_variant[internal_measure].loc[stats_value]
                all_raw_values[internal_measure][variant_desc] = raw_values_for_variant[internal_measure].loc[
                    stats_value]

    # plot data
    for measure in internal_measures:
        # plot ranks
        rank_fig, top_two_dist = plot_ranking_heat_map(backend, all_ranks[measure], pattern_keys_ordered)

        # plot raw values
        raw_value_fig, _ = plot_ranking_heat_map(backend, all_raw_values[measure], pattern_keys_ordered,
                                                 bar_label="Raw", low_is_best=internal_measure_ranking_method[measure])

        # save figures
        if save_fig:
            folder = base_dataset_result_folder_for_type(root_results_dir, ResultsType.distance_measure_evaluation)
            folder = path.join(folder, "images")
            os.makedirs(folder, exist_ok=True)
            rank_file_name = path.join(folder, measure + "_" + GROUND_TRUTH_HEATMAP_OF_RANKS)
            raw_values_file_name = path.join(folder, measure + "_" + GROUND_TRUTH_HEATMAP_RAW_VALUES)
            rank_fig.savefig(rank_file_name, dpi=300, bbox_inches='tight')
            raw_value_fig.savefig(raw_values_file_name, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # heatmap ov average ranking for each dataset in the N30
    # y = data variant, x = distance measure, lower ranks are better
    backend = Backends.visible_tests.value
    save_fig = True
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.raw,
                     SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]
    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi]

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

    ground_truth_heatmap_for_all_variants(data_dirs=data_dirs, dataset_types=dataset_types, overall_ds_name="n30",
                                          root_results_dir=root_result_dir, distance_measures=distance_measures,
                                          backend=backend, save_fig=save_fig, internal_measures=internal_measures)
