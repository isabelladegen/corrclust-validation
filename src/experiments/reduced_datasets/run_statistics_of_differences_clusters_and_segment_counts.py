import os

import pandas as pd

from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.evaluation.internal_measure_ground_truth_cluster_segment_count_differences import \
    InternalMeasureGroundTruthClusterSegmentCount
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, ROOT_REDUCED_SYNTHETIC_DATA_DIR, \
    ROOT_REDUCED_RESULTS_DIR, DataCompleteness, get_root_folder_for_reduced_cluster, get_data_dir, \
    get_root_folder_for_reduced_segments, internal_measure_evaluation_dir_for, \
    SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.run_average_rank_visualisations import data_variant_description


def calculate_count_compare_statistics(dropped_dirs, completeness, dataset_types, reduced_root_results_dir,
                                       original_root_result_dir, distance_measure,
                                       internal_measures, overall_ds_name, stats_value, type):
    # create variant description that serve as keys
    variant_descriptions = []
    for comp in completeness:
        for data_type in dataset_types:
            variant_desc = data_variant_description[(comp, data_type)]
            variant_descriptions.append(variant_desc)

    # create variant description dictionary
    # key= variant name, value = None (will be df once created)
    dict_for_variant = {desc: None for desc in variant_descriptions}

    # initialise results
    # key=internal measure name, value = dict with key=variant description, value will be raw values df
    all_raw_values = {measure: dict_for_variant.copy() for measure in internal_measures}

    # key = internal measure name, value = list of stats df for each variant
    all_stats_results = {measure: [] for measure in internal_measures}

    # create per_distance_measures_descriptive_stats_for_ground_truth.csv
    dist_mins = []
    dist_maxs = []
    dist_mean = []
    dist_medians = []
    dist_stds = []
    dist_data_variant = []
    dist_index = []
    count_names = []

    # calculate raw values and stats for each data variant
    for comp in completeness:
        # create data dirs for comp type
        data_dirs = []
        for drop_dir in dropped_dirs:
            data_dirs.append(get_data_dir(drop_dir, comp))

        # for each data type
        for data_type in dataset_types:
            ga = InternalMeasureGroundTruthClusterSegmentCount(overall_ds_name=overall_ds_name,
                                                               internal_measures=internal_measures,
                                                               distance_measure=distance_measure,
                                                               data_dirs=data_dirs,
                                                               data_type=data_type,
                                                               reduced_root_result=reduced_root_results_dir,
                                                               original_root_result=original_root_result_dir)
            variant_desc = data_variant_description[(comp, data_type)]
            raw_values_for_variant = ga.stats_for_raw_values_across_all_runs()
            stats_results = ga.wilcoxons_signed_rank_between_all_counts()

            for internal_measure in internal_measures:
                all_raw_values[internal_measure][variant_desc] = raw_values_for_variant[internal_measure].loc[
                    stats_value]

                # change stat results into a df for this variant
                df = stats_results[internal_measure]
                df.insert(0, "Data Variant", variant_desc)
                all_stats_results[internal_measure].append(df)

                # describe median min-max ranges for all distance measures separately
                stats_df = raw_values_for_variant[internal_measure]
                for count_name in ga.count_names:
                    dist_mins.append(stats_df[count_name].loc['min'])
                    dist_maxs.append(stats_df[count_name].loc['max'])
                    dist_mean.append(stats_df[count_name].loc['mean'])
                    dist_medians.append(stats_df[count_name].loc['50%'])
                    dist_stds.append(stats_df[count_name].loc['std'])
                    dist_data_variant.append(variant_desc)
                    dist_index.append(internal_measure)
                    count_names.append(count_name)

    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type='',  # all datatypes as rows
        results_dir=reduced_root_results_dir,
        data_dir='',  # all data data comp as rows
        distance_measure='')  # all distances as columns

    # build per distance measure df
    per_distance_measure_df = pd.DataFrame(
        {'Data variant': dist_data_variant,
         'Internal index': dist_index,
         'Counts': count_names,
         'mean': dist_mean,
         'std': dist_stds,
         'median': dist_medians,
         'min': dist_mins,
         'max': dist_maxs,
         })
    filename = "_".join([type, distance_measure, IAResultsCSV.per_distance_measures_descriptive_stats_for_ground_truth])
    per_distance_measure_df.to_csv(str(os.path.join(store_results_in, filename)))

    for measure in internal_measures:
        # save stats results
        stats_for_measure = all_stats_results[measure]
        combined_stats_df = pd.concat(stats_for_measure, ignore_index=True)
        filename = "_".join([type, measure, distance_measure,
                             IAResultsCSV.distance_measures_stat_results_for_ground_truth])
        combined_stats_df.to_csv(str(os.path.join(store_results_in, filename)))


if __name__ == "__main__":
    # Calculations for reduced datasets ground truth
    # calculates raw and rank matrix
    # calculates statistics for groups until all significant
    # plot heatmap ov average ranking for each dataset in the N30
    # y = data variant, x = distance measure, lower ranks are better
    n_dropped_clusters = [12, 17]
    n_dropped_segments = [50, 75]
    root_reduced_dir = ROOT_REDUCED_SYNTHETIC_DATA_DIR
    base_results_dir = ROOT_REDUCED_RESULTS_DIR

    overall_dataset_name = "n30"
    run_file = GENERATED_DATASETS_FILE_PATH
    distance_measure = DistanceMeasures.l5_cor_dist
    clustering_quality_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi]
    data_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated]
    data_completeness = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]

    backend = Backends.none.value
    save_fig = True
    stats_value = '50%'  # use median

    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi]

    # Evaluate for clusters
    print("CLUSTERS")
    dropped_dirs = [SYNTHETIC_DATA_DIR]
    for dropped_cluster in n_dropped_clusters:
        dir_for_cluster = get_root_folder_for_reduced_cluster(root_reduced_dir, dropped_cluster)
        dropped_dirs.append(dir_for_cluster)

    calculate_count_compare_statistics(dropped_dirs=dropped_dirs, completeness=data_completeness,
                                       dataset_types=data_types,
                                       overall_ds_name=overall_dataset_name,
                                       reduced_root_results_dir=base_results_dir,
                                       original_root_result_dir=ROOT_RESULTS_DIR,
                                       distance_measure=distance_measure,
                                       internal_measures=internal_measures,
                                       stats_value=stats_value, type='clusters')

    # Evaluate for segments
    print("DROPPED SEGMENTS")
    dropped_dirs = [SYNTHETIC_DATA_DIR]
    for dropped_segments in n_dropped_segments:
        dir_for = get_root_folder_for_reduced_segments(root_reduced_dir, dropped_segments)
        dropped_dirs.append(dir_for)

    calculate_count_compare_statistics(dropped_dirs=dropped_dirs, completeness=data_completeness,
                                       dataset_types=data_types,
                                       overall_ds_name=overall_dataset_name,
                                       reduced_root_results_dir=base_results_dir,
                                       original_root_result_dir=ROOT_RESULTS_DIR,
                                       distance_measure=distance_measure,
                                       internal_measures=internal_measures,
                                       stats_value=stats_value, type='segments')
