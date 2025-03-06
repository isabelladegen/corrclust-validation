import os

import pandas as pd

from src.evaluation.calculate_internal_measures_ground_truth import CalculateInternalMeasuresGroundTruth
from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.experiments.run_calculate_internal_measures_for_ground_truth import \
    run_internal_measure_calculation_for_ground_truth
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, \
    internal_measure_calculation_dir_for, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, \
    ROOT_REDUCED_SYNTHETIC_DATA_DIR, ROOT_REDUCED_RESULTS_DIR, DataCompleteness, get_root_folder_for_reduced_cluster, \
    get_data_dir, get_root_folder_for_reduced_segments
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    # Calculate the internal measure for all distance measures but only ground truth ds for reduced
    overall_dataset_name = "n30"
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    n_dropped_clusters = [12, 17]
    n_dropped_segments = [50, 75]
    root_reduced_dir = ROOT_REDUCED_SYNTHETIC_DATA_DIR
    base_results_dir = ROOT_REDUCED_RESULTS_DIR

    distance_measures = [DistanceMeasures.l1_cor_dist,  # lp norms
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.linf_cor_dist]
    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi]
    data_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated]

    data_completeness = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]

    # Evaluate for clusters
    print("CLUSTERS")
    for dropped_cluster in n_dropped_clusters:
        dir_for_cluster = get_root_folder_for_reduced_cluster(root_reduced_dir, dropped_cluster)
        results_dir = get_root_folder_for_reduced_cluster(base_results_dir, dropped_cluster)
        data_dirs = []
        for complete in data_completeness:
            data_dir = get_data_dir(dir_for_cluster, complete)
            data_dirs.append(data_dir)
        for data_dir in data_dirs:
            for data_type in data_types:
                for distance_measure in distance_measures:
                    print("Calculate Ground truth Clustering Quality Measures for completeness:")
                    print(data_dir)
                    print("and data type: " + data_type)
                    print("and distance measure: " + distance_measure)
                    run_internal_measure_calculation_for_ground_truth(overall_dataset_name, run_names=run_names,
                                                                      distance_measure=distance_measure,
                                                                      data_type=data_type,
                                                                      data_dir=data_dir, results_dir=results_dir,
                                                                      internal_measures=internal_measures)

    # Evaluate for segments
    print("DROPPED SEGMENTS")
    for dropped_segments in n_dropped_segments:
        dir_for = get_root_folder_for_reduced_segments(root_reduced_dir, dropped_segments)
        results_dir = get_root_folder_for_reduced_segments(base_results_dir, dropped_segments)
        data_dirs = []
        for complete in data_completeness:
            data_dir = get_data_dir(dir_for, complete)
            data_dirs.append(data_dir)

        for data_dir in data_dirs:
            for data_type in data_types:
                for distance_measure in distance_measures:
                    print("Calculate Ground truth Clustering Quality Measures for completeness:")
                    print(data_dir)
                    print("and data type: " + data_type)
                    print("and distance measure: " + distance_measure)
                    run_internal_measure_calculation_for_ground_truth(overall_dataset_name, run_names=run_names,
                                                                      distance_measure=distance_measure,
                                                                      data_type=data_type,
                                                                      data_dir=data_dir, results_dir=results_dir,
                                                                      internal_measures=internal_measures)
