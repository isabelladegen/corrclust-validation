import os
from os import path

import matplotlib.pyplot as plt

from src.experiments.validity.visualise_mean_sd_icvi_clusters import plot_errorbars_for_data, \
    load_ground_truth_quality_measures
from src.experiments.validity.visualise_raw_values import dist_labels
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ResultsType, VALID_ROOT_RESULTS_DIR, ROOT_REDUCED_SYNTHETIC_DATA_DIR, \
    ROOT_REDUCED_RESULTS_DIR, DataCompleteness, SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR, \
    get_root_folder_for_reduced_segments
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    legend_t = 'Segments'
    cond = ['Normal 100%', 'Normal 10%']
    root_reduced_dir = ROOT_REDUCED_SYNTHETIC_DATA_DIR
    base_results_dir = ROOT_REDUCED_RESULTS_DIR

    distance_measures = [
        DistanceMeasures.l1_cor_dist,
        DistanceMeasures.l2_cor_dist,
        DistanceMeasures.l3_cor_dist,
        DistanceMeasures.l5_cor_dist,
        DistanceMeasures.l1_with_ref,
        DistanceMeasures.dot_transform_l2
    ]

    clustering_quality_measures = [
        ClusteringQualityMeasures.silhouette_score,
        ClusteringQualityMeasures.dbi,
        ClusteringQualityMeasures.vrc,
        ClusteringQualityMeasures.pmb
    ]
    overall_dataset_name = "n30"

    data_completeness = [
        DataCompleteness.complete,
        DataCompleteness.irregular_p90
    ]

    data_types = [
        SyntheticDataType.normal_correlated,
    ]

    n_dropped_segments_to_count_label = {0: "100", 50: "50", 75: "25"}
    data_dir_and_result_root_by_n_dropped = {}
    for n_dropped, count_label in n_dropped_segments_to_count_label.items():
        if n_dropped == 0:
            base_data_dir, result_root = SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR
        else:
            base_data_dir = get_root_folder_for_reduced_segments(root_reduced_dir, n_dropped)
            result_root = get_root_folder_for_reduced_segments(base_results_dir, n_dropped)
        data_dir_and_result_root_by_n_dropped[n_dropped] = (base_data_dir, result_root, count_label)

    results_by_count = load_ground_truth_quality_measures(
        data_dir_and_result_root_by_n_dropped=data_dir_and_result_root_by_n_dropped,
        data_completeness=data_completeness,
        data_types=data_types,
        distance_measures=distance_measures,
        quality_measures=clustering_quality_measures,
        overall_dataset_name=overall_dataset_name,
    )
    data_100 = results_by_count["100"]
    data_50 = results_by_count["50"]
    data_25 = results_by_count["25"]

    invalid_dms = {
        cond[0]: {
            ClusteringQualityMeasures.silhouette_score: [],
            ClusteringQualityMeasures.dbi: [],
            ClusteringQualityMeasures.vrc: [],
            ClusteringQualityMeasures.pmb: [
                dist_labels[DistanceMeasures.l1_cor_dist],
                dist_labels[DistanceMeasures.l2_cor_dist],
                dist_labels[DistanceMeasures.l3_cor_dist],
                dist_labels[DistanceMeasures.l5_cor_dist],
                dist_labels[DistanceMeasures.dot_transform_l2]],
        },
        cond[1]: {
            ClusteringQualityMeasures.silhouette_score: [
                dist_labels[DistanceMeasures.l1_with_ref]],
            ClusteringQualityMeasures.dbi: [
                dist_labels[DistanceMeasures.l1_cor_dist],
                dist_labels[DistanceMeasures.l1_with_ref]],
            ClusteringQualityMeasures.vrc: [],
            ClusteringQualityMeasures.pmb: [dist_labels[dm] for dm in distance_measures],
        }
    }
    plot_errorbars_for_data(data_100, data_50, data_25, legend_t, cond, (18, 6),
                            distance_measures=distance_measures,
                            quality_measures=clustering_quality_measures,
                            invalid_distance_measures_by_condition_and_quality_measure=invalid_dms)
    results_folder = path.join(VALID_ROOT_RESULTS_DIR, ResultsType.internal_measure_evaluation, 'images')
    os.makedirs(results_folder, exist_ok=True)
    plt.savefig(path.join(results_folder, 'structural_test_1_4_segments.png'), dpi=300, bbox_inches='tight')
    plt.show()
