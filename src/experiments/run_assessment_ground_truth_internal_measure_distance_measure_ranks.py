import os
from os import path

import pandas as pd

from src.evaluation.distance_metric_evaluation import criteria_short_names
from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.evaluation.internal_measure_ground_truth_assessment import InternalMeasureGroundTruthAssessment, \
    internal_measure_lower_values_best
from src.evaluation.interpretation_distance_metric_ranking import DistanceMetricInterpretation
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, base_dataset_result_folder_for_type, ResultsType, \
    AVERAGE_RANK_DISTRIBUTION, HEATMAP_OF_RANKS, HEATMAP_OF_BEST_MEASURES_RAW_VALUES, GROUND_TRUTH_HEATMAP_OF_RANKS, \
    GROUND_TRUTH_HEATMAP_RAW_VALUES, internal_measure_evaluation_dir_for
from src.utils.distance_measures import DistanceMeasures, short_distance_measure_names
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.run_average_rank_visualisations import data_variant_description
from src.visualisation.run_heatmap_of_average_distance_measure_ranks import pattern_keys_ordered, plot_ranking_heat_map
from src.visualisation.visualise_distance_measure_rank_distributions import heatmap_of_ranks, heatmap_of_raw_values


def raw_values_ranks_heatmaps_for_ground_truth(data_dirs, dataset_types, root_results_dir, distance_measures,
                                               internal_measures, overall_ds_name, backend, stats_value,
                                               distance_measures_for_summary,
                                               save_fig=True):
    # create variant description that serve as keys
    variant_descriptions = []
    for data_dir in data_dirs:
        for data_type in dataset_types:
            variant_desc = data_variant_description[(data_dir, data_type)]
            variant_descriptions.append(variant_desc)

    # create variant description dictionary
    # key= variant name, value = None (will be df once created)
    dict_for_variant = {desc: None for desc in variant_descriptions}

    # initialise results
    # key=internal measure name, value = dict with key=variant description, value will be rank df
    all_ranks = {measure: dict_for_variant.copy() for measure in internal_measures}
    # key=internal measure name, value = dict with key=variant description, value will be raw values df
    all_raw_values = {measure: dict_for_variant.copy() for measure in internal_measures}

    # key = internal measure name, value = list of stats df for each variant
    all_stats_results = {measure: [] for measure in internal_measures}

    # create reasonable_distance_measures_ranges_for_ground_truth.csv
    reasonable_mins = []
    reasonable_maxs = []
    reasonable_data_variant = []
    reasonable_index = []

    # create per_distance_measures_descriptive_stats_for_ground_truth.csv
    dist_mins = []
    dist_maxs = []
    dist_mean = []
    dist_medians = []
    dist_stds = []
    dist_data_variant = []
    dist_index = []
    dist_distances = []

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
            stats_results = ga.wilcoxons_signed_rank_until_all_significant()

            for internal_measure in internal_measures:
                all_ranks[internal_measure][variant_desc] = ranks_for_variant[internal_measure].loc[stats_value]
                all_raw_values[internal_measure][variant_desc] = raw_values_for_variant[internal_measure].loc[
                    stats_value]

                # change stat results into a df for this variant
                df = stats_results[internal_measure]
                df.insert(0, "Data Variant", variant_desc)
                all_stats_results[internal_measure].append(df)

                # describe median min-max ranges for reasonable distance measures
                stats_df = raw_values_for_variant[internal_measure]
                reasonable_mins.append(stats_df[reasonable_distance_measures].loc['50%'].min())
                reasonable_maxs.append(stats_df[reasonable_distance_measures].loc['50%'].max())
                reasonable_data_variant.append(variant_desc)
                reasonable_index.append(internal_measure)

                # describe median min-max ranges for all distance measures separately
                for distance_measure in distance_measures:
                    dist_mins.append(stats_df[distance_measure].loc['min'])
                    dist_maxs.append(stats_df[distance_measure].loc['max'])
                    dist_mean.append(stats_df[distance_measure].loc['mean'])
                    dist_medians.append(stats_df[distance_measure].loc['50%'])
                    dist_stds.append(stats_df[distance_measure].loc['std'])
                    dist_data_variant.append(variant_desc)
                    dist_index.append(internal_measure)
                    dist_distances.append(distance_measure)

    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type='',  # all datatypes as rows
        results_dir=root_results_dir,
        data_dir='',  # all data data comp as rows
        distance_measure='')  # all distances as columns

    # build describe df: reasonable_distance_measures_ranges_for_ground_truth.csv
    reasonable_summary_df = pd.DataFrame(
        {'Data variant': reasonable_data_variant,
         'Internal index': reasonable_index,
         'min': reasonable_mins,
         'max': reasonable_maxs,
         })
    reasonable_summary_df.to_csv(
        str(os.path.join(store_results_in, IAResultsCSV.reasonable_distance_measures_median_ranges_for_ground_truth)))

    # build per distance measure df
    per_distance_measure_df = pd.DataFrame(
        {'Data variant': dist_data_variant,
         'Internal index': dist_index,
         'Distance': dist_distances,
         'mean': dist_mean,
         'std': dist_stds,
         'median': dist_medians,
         'min': dist_mins,
         'max': dist_maxs,
         })
    per_distance_measure_df.to_csv(
        str(os.path.join(store_results_in, IAResultsCSV.per_distance_measures_descriptive_stats_for_ground_truth)))

    # plot data
    for measure in internal_measures:
        # save stats results
        stats_for_measure = all_stats_results[measure]
        combined_stats_df = pd.concat(stats_for_measure, ignore_index=True)
        combined_stats_df.to_csv(str(os.path.join(store_results_in,
                                                  measure + "_" + IAResultsCSV.distance_measures_stat_results_for_ground_truth)))

        # plot ranks
        rank_fig, rank_matrix = plot_ranking_heat_map(backend, all_ranks[measure], pattern_keys_ordered)

        # save rank data
        rank_matrix.to_csv(
            str(os.path.join(store_results_in, measure + "_" + IAResultsCSV.distance_measures_ranks_for_ground_truth)))

        # plot raw values
        raw_value_fig, raw_value_matrix = plot_ranking_heat_map(backend, all_raw_values[measure], pattern_keys_ordered,
                                                                bar_label="Raw",
                                                                low_is_best=internal_measure_lower_values_best[measure])
        # save raw value data
        raw_value_matrix.to_csv(
            str(os.path.join(store_results_in,
                             measure + "_" + IAResultsCSV.distance_measures_raw_values_for_ground_truth)))

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
    # calculates raw and rank matrix
    # calculates statistics for groups until all significant
    # plot heatmap ov average ranking for each dataset in the N30
    # y = data variant, x = distance measure, lower ranks are better
    backend = Backends.none.value
    save_fig = True
    root_result_dir = ROOT_RESULTS_DIR
    stats_value = '50%'  # use median
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

    reasonable_distance_measures = [DistanceMeasures.l1_cor_dist,  # lp norms
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
                                    DistanceMeasures.dot_transform_linf, ]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    raw_values_ranks_heatmaps_for_ground_truth(data_dirs=data_dirs, dataset_types=dataset_types, overall_ds_name="n30",
                                               root_results_dir=root_result_dir, distance_measures=distance_measures,
                                               backend=backend, save_fig=save_fig, internal_measures=internal_measures,
                                               stats_value=stats_value,
                                               distance_measures_for_summary=reasonable_distance_measures)
