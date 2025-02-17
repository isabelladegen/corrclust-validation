import pandas as pd

from src.experiments.run_internal_measure_assessment import run_internal_measure_assessment_datasets
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, ROOT_REDUCED_SYNTHETIC_DATA_DIR, DataCompleteness, \
    ROOT_REDUCED_RESULTS_DIR, get_root_folder_for_reduced_cluster, \
    get_data_dir, get_root_folder_for_reduced_segments
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    """Run internal measure assessment for reduced dataset"""
    # drop 50% and 75% of clusters and segments
    n_dropped_clusters = [12, 17]
    n_dropped_segments = [50, 75]

    root_reduced_dir = ROOT_REDUCED_SYNTHETIC_DATA_DIR
    data_completeness = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]

    overall_dataset_name = "n30"
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    distance_measures = [DistanceMeasures.l1_cor_dist]
    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi]

    data_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated]

    base_results_dir = ROOT_REDUCED_RESULTS_DIR

    # Evaluate for clusters
    print("CALCULATE FOR DROPPED CLUSTERS")
    for dropped_cluster in n_dropped_clusters:
        root_data_dir = get_root_folder_for_reduced_cluster(root_reduced_dir, dropped_cluster)
        results_dir = get_root_folder_for_reduced_cluster(base_results_dir, dropped_cluster)
        for cmp in data_completeness:
            data_dir = get_data_dir(root_data_dir, cmp)
            for data_type in data_types:
                for distance_measure in distance_measures:
                    print(
                        "CALCULATE FOR CLUSTER: Distance measure: " + distance_measure + " , Dataset type: " + data_type +
                        ", Compactness: " + data_dir)
                    run_internal_measure_assessment_datasets(overall_ds_name="n30", run_names=run_names,
                                                             distance_measure=distance_measure, data_type=data_type,
                                                             data_dir=data_dir, results_dir=results_dir,
                                                             internal_measures=internal_measures)

    # Evaluate for segments
    print("CALCULATE FOR DROPPED SEGMENTS")
    for dropped_seg in n_dropped_segments:
        root_data_dir = get_root_folder_for_reduced_segments(root_reduced_dir, dropped_seg)
        results_dir = get_root_folder_for_reduced_segments(base_results_dir, dropped_seg)
        for cmp in data_completeness:
            data_dir = get_data_dir(root_data_dir, cmp)
            for data_type in data_types:
                for distance_measure in distance_measures:
                    print("CALCULATE FOR SEGMENT: Distance measure: " + distance_measure + " , Dataset type: "
                          + data_type + ", Compactness: " + data_dir)
                    run_internal_measure_assessment_datasets(overall_ds_name="n30", run_names=run_names,
                                                             distance_measure=distance_measure, data_type=data_type,
                                                             data_dir=data_dir, results_dir=results_dir,
                                                             internal_measures=internal_measures)
