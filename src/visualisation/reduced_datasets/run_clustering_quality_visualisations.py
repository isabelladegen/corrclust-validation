from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, ROOT_REDUCED_SYNTHETIC_DATA_DIR, DataCompleteness, \
    ROOT_REDUCED_RESULTS_DIR, get_root_folder_for_reduced_cluster, get_data_dir, get_root_folder_for_reduced_segments
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.run_clustering_quality_visualisations import clustering_quality_visualisations

if __name__ == "__main__":
    # violin plots and scatter plots for all clustering quality measures for each dataset in the N30 for the
    # reduced datasets
    # backend = Backends.visible_tests.value
    backend = Backends.none.value
    save_fig = True

    # drop 50% and 75% of clusters and segments
    n_dropped_clusters = [12, 17]
    n_dropped_segments = [50, 75]
    root_reduced_dir = ROOT_REDUCED_SYNTHETIC_DATA_DIR
    data_completeness = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]

    overall_dataset_name = "n30"
    run_file = GENERATED_DATASETS_FILE_PATH
    distance_measures = [DistanceMeasures.l1_cor_dist]
    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi]
    data_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated]

    base_results_dir = ROOT_REDUCED_RESULTS_DIR

    # Evaluate for clusters
    print("CALCULATE FOR DROPPED CLUSTERS")
    for dropped_cluster in n_dropped_clusters:
        dir_for_cluster = get_root_folder_for_reduced_cluster(root_reduced_dir, dropped_cluster)
        results_dir = get_root_folder_for_reduced_cluster(base_results_dir, dropped_cluster)
        data_dirs = []
        for complete in data_completeness:
            data_dir = get_data_dir(dir_for_cluster, complete)
            data_dirs.append(data_dir)
        print("Visualise for reduced cluster data:")
        print(dropped_cluster)
        clustering_quality_visualisations(data_dirs=data_dirs, data_types=data_types, run_file=run_file,
                                          root_results_dir=results_dir,
                                          distance_measures=distance_measures,
                                          clustering_quality_measures=internal_measures,
                                          overall_ds_name=overall_dataset_name,
                                          backend=backend, save_fig=save_fig, figsize1=(7.5, 10), figsize2=(9, 10))

    # Evaluate for segments
    print("CALCULATE FOR DROPPED SEGMENTS")
    for dropped_segments in n_dropped_segments:
        dir_for = get_root_folder_for_reduced_segments(root_reduced_dir, dropped_segments)
        results_dir = get_root_folder_for_reduced_segments(base_results_dir, dropped_segments)
        data_dirs = []
        for complete in data_completeness:
            data_dir = get_data_dir(dir_for, complete)
            data_dirs.append(data_dir)
        print("Visualise for reduced cluster data:")
        print(dropped_segments)
        clustering_quality_visualisations(data_dirs=data_dirs, data_types=data_types, run_file=run_file,
                                          root_results_dir=results_dir,
                                          distance_measures=distance_measures,
                                          clustering_quality_measures=internal_measures,
                                          overall_ds_name=overall_dataset_name,
                                          backend=backend, save_fig=save_fig, figsize1=(7.5, 10), figsize2=(9, 10))
