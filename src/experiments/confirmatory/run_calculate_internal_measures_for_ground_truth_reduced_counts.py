import pandas as pd

from src.experiments.run_calculate_internal_measures_for_ground_truth import \
    run_internal_measure_calculation_for_ground_truth, load_all_ground_truth_data_for_all_subjects_and_data_type
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import DataCompleteness, get_root_folder_for_reduced_cluster, \
    get_data_dir, get_root_folder_for_reduced_segments, CONFIRMATORY_DATASETS_FILE_PATH, \
    CONFIRMATORY_ROOT_REDUCED_SYNTHETIC_DATA_DIR, CONF_ROOT_REDUCED_RESULTS_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    # Calculate the internal measure for all distance measures but only ground truth ds for reduced
    overall_dataset_name = "n30"
    run_names = pd.read_csv(CONFIRMATORY_DATASETS_FILE_PATH)['Name'].tolist()

    n_dropped_clusters = [12, 17]
    n_dropped_segments = [50, 75]
    root_reduced_dir = CONFIRMATORY_ROOT_REDUCED_SYNTHETIC_DATA_DIR
    base_results_dir = CONF_ROOT_REDUCED_RESULTS_DIR

    distance_measure = DistanceMeasures.l5_cor_dist
    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi]
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
                print("Calculate Ground truth Clustering Quality Measures for completeness:")
                print(data_dir)
                print("and data type: " + data_type)
                print("and distance measure: " + distance_measure)
                data_dict, gt_labels_dict = load_all_ground_truth_data_for_all_subjects_and_data_type(
                    run_ids=run_names, data_type=data_type, data_dir=data_dir)
                run_internal_measure_calculation_for_ground_truth(overall_dataset_name, run_names=run_names,
                                                                  data_dict=data_dict,
                                                                  gt_labels_dict=gt_labels_dict,
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
                print("Calculate Ground truth Clustering Quality Measures for completeness:")
                print(data_dir)
                print("and data type: " + data_type)
                print("and distance measure: " + distance_measure)
                data_dict, gt_labels_dict = load_all_ground_truth_data_for_all_subjects_and_data_type(
                    run_ids=run_names, data_type=data_type, data_dir=data_dir)
                run_internal_measure_calculation_for_ground_truth(overall_dataset_name, run_names=run_names,
                                                                  data_dict=data_dict,
                                                                  gt_labels_dict=gt_labels_dict,
                                                                  distance_measure=distance_measure,
                                                                  data_type=data_type,
                                                                  data_dir=data_dir, results_dir=results_dir,
                                                                  internal_measures=internal_measures)
