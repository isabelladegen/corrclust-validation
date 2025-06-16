from os import path

import pandas as pd
from matplotlib import pyplot as plt

from src.experiments.run_calculate_internal_measures_for_ground_truth import \
    read_ground_truth_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import DataCompleteness, GENERATED_DATASETS_FILE_PATH, \
    ROOT_REDUCED_RESULTS_DIR, get_root_folder_for_reduced_cluster, SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR, \
    get_root_folder_for_reduced_segments, get_clustering_quality_multiple_data_variants_result_folder, ResultsType, \
    GROUND_TRUTH_CI_PLOT, get_image_results_path, get_data_dir, get_data_completeness_from
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.run_average_rank_visualisations import data_variant_description
from src.visualisation.visualise_clustering_quality_measures_for_multiple_data_variants import create_ci_grid
from src.visualisation.visualise_multiple_data_variants import get_row_name_from


def ci_for(results_dirs: {str}, data_completenesses: [str], data_types: [str], overall_dataset_name: str, run_file: str,
           distance_measure: str, internal_measures: [str], drop_type: str):
    x_values = list(results_dirs.keys())
    # load data for all sparsity levels and data types for all numbers of clusters
    # data_dict : {completeness levels, {data_type,{internal measure{pd.DataFrame}}}} Nested dictionary containing data for each plot square
    data_dict = {}
    internal_ms = {}
    segments_stats = []
    cluster_stats = []
    for comp in data_completenesses:
        data_type_d = {}
        for data_type in data_types:
            data_desc = data_variant_description[(comp, data_type)]
            for count_val, results_dir in results_dirs.items():
                per_distance_measure_df = read_ground_truth_clustering_quality_measures(
                    overall_ds_name=overall_dataset_name,
                    data_type=data_type,
                    root_results_dir=results_dir,
                    data_dir=get_data_dir(SYNTHETIC_DATA_DIR, comp), distance_measure=distance_measure)
                n_pat = per_distance_measure_df['n patterns']
                n_seg = per_distance_measure_df['n segments']
                cluster_stats.append({
                    'data variant': data_desc,
                    'count': count_val,
                    'mean': n_pat.mean().round(2),
                    'std': n_pat.std().round(2),
                    'type': drop_type})
                segments_stats.append({
                    'data variant': data_desc,
                    'count': count_val,
                    'mean': n_seg.mean().round(2),
                    'std': n_seg.std().round(2),
                    'type': drop_type})
                for im in internal_measures:
                    df = per_distance_measure_df[['file name', im]].copy()
                    df.rename(columns={im: count_val}, inplace=True)
                    df.set_index('file name', inplace=True)
                    if im in internal_ms:
                        append_to = internal_ms[im]
                        append_to[count_val] = df[count_val]
                        df = append_to

                    internal_ms[im] = df
            data_type_d[SyntheticDataType.get_display_name_for_data_type(data_type)] = internal_ms
        data_dict[get_row_name_from(comp)] = data_type_d

    # create fig
    fig = create_ci_grid(data_dict=data_dict, internal_indices=internal_measures, distance_measures=x_values,
                         backend=Backends.none.value, figsize=(8, 8), annotation_idx=[0, 2],
                         for_distances=False, loc='center left')
    plt.show()

    # save figure
    folder = get_clustering_quality_multiple_data_variants_result_folder(
        results_type=ResultsType.internal_measure_evaluation,
        overall_dataset_name=overall_dataset_name,
        results_dir=ROOT_REDUCED_RESULTS_DIR,
        distance_measure='')
    # add an image results folder
    join_to_file = [drop_type] + internal_measures.copy() + [distance_measure] + [GROUND_TRUTH_CI_PLOT]
    file_name = get_image_results_path(folder, '_'.join(join_to_file))
    fig.savefig(file_name, dpi=300, bbox_inches='tight')

    # save stats
    pd.DataFrame(cluster_stats).to_csv(path.join(folder, '_'.join([drop_type, 'n_patterns', 'stats.csv'])))
    pd.DataFrame(segments_stats).to_csv(path.join(folder, '_'.join([drop_type, 'n_segments', 'stats.csv'])))


if __name__ == "__main__":
    # ci plots for 100%, 50% and 25% cluster count respectively segment count
    # reduced datasets
    save_fig = True

    # drop 50% and 75% of clusters and segments
    cluster_results_dir = {23: ROOT_RESULTS_DIR,
                           11: get_root_folder_for_reduced_cluster(ROOT_REDUCED_RESULTS_DIR, 12),
                           6: get_root_folder_for_reduced_cluster(ROOT_REDUCED_RESULTS_DIR, 17),
                           }
    segment_results_dir = {100: ROOT_RESULTS_DIR,
                           50: get_root_folder_for_reduced_segments(ROOT_REDUCED_RESULTS_DIR, 50),
                           25: get_root_folder_for_reduced_segments(ROOT_REDUCED_RESULTS_DIR, 75),
                           }
    data_completeness = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]
    data_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated]

    overall_dataset_name = "n30"
    run_file = GENERATED_DATASETS_FILE_PATH
    distance_measure = DistanceMeasures.l5_cor_dist
    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi]

    # Evaluate for clusters
    print("CI FOR DROPPED CLUSTERS")
    ci_for(results_dirs=cluster_results_dir, data_completenesses=data_completeness, data_types=data_types,
           overall_dataset_name=overall_dataset_name, run_file=run_file, distance_measure=distance_measure,
           internal_measures=internal_measures, drop_type='clusters')

    print("CI FOR DROPPED SEGMENTS")
    ci_for(results_dirs=segment_results_dir, data_completenesses=data_completeness, data_types=data_types,
           overall_dataset_name=overall_dataset_name, run_file=run_file, distance_measure=distance_measure,
           internal_measures=internal_measures, drop_type='segments')
