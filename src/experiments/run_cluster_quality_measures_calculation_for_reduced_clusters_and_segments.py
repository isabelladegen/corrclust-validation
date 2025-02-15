import pandas as pd

from src.experiments.run_cluster_quality_measures_calculation import run_internal_measure_calculation_for_dataset
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, ROOT_RESULTS_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    overall_dataset_name = "n30"
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    distance_measures = [DistanceMeasures.l1_cor_dist]
    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi]
    data_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated]
    data_dirs = [SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR]

    # drop 50% and 75% of clusters and segments
    n_dropped_clusters = [12, 6]
    n_dropped_segments = [50, 25]

    results_dir = ROOT_RESULTS_DIR

    for data_dir in data_dirs:
        for data_type in data_types:
            for distance_measure in distance_measures:
                print("Calculate Clustering Quality Measures for completeness:")
                print(data_dir)
                print("and data type: " + data_type)
                run_internal_measure_calculation_for_dataset(overall_dataset_name, run_names=run_names,
                                                             distance_measure=distance_measure, data_type=data_type,
                                                             data_dir=data_dir, results_dir=results_dir,
                                                             internal_measures=internal_measures,
                                                             n_dropped_clusters=n_dropped_clusters,
                                                             n_dropped_segments=n_dropped_segments)
