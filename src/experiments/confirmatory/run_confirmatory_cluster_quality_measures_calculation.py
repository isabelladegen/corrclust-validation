import pandas as pd

from src.experiments.run_cluster_quality_measures_calculation import \
    load_all_clustering_data_for_subjects_and_data_type, run_internal_measure_calculation_for_dataset
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import CONFIRMATORY_DATASETS_FILE_PATH, CONFIRMATORY_SYNTHETIC_DATA_DIR, \
    CONF_IRREGULAR_P30_DATA_DIR, CONF_IRREGULAR_P90_DATA_DIR, CONF_ROOT_RESULTS_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    overall_dataset_name = "n30"
    run_names = pd.read_csv(CONFIRMATORY_DATASETS_FILE_PATH)['Name'].tolist()

    # all variants but raw
    data_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated,
                  SyntheticDataType.rs_1min]
    data_dirs = [CONFIRMATORY_SYNTHETIC_DATA_DIR, CONF_IRREGULAR_P30_DATA_DIR, CONF_IRREGULAR_P90_DATA_DIR]

    root_result_dir = CONF_ROOT_RESULTS_DIR

    # distance measures that appear in a preregistered hypothesis
    distance_measures = [DistanceMeasures.l1_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.linf_cor_dist,
                         DistanceMeasures.l1_with_ref,
                         DistanceMeasures.l5_with_ref,
                         DistanceMeasures.foerstner_cor_dist]

    # internal measures that appear in a preregistration form
    internal_measures = [ClusteringQualityMeasures.silhouette_score,  ClusteringQualityMeasures.dbi]

    for data_dir in data_dirs:
        for data_type in data_types:
            variant = (data_dir, data_type)
            print(f"Load all data for data type {data_type} and data dir {data_dir}")
            data_dict, gt_labels_dict, partitions_dict = load_all_clustering_data_for_subjects_and_data_type(
                run_ids=run_names, data_type=data_type, data_dir=data_dir)
            for distance_measure in distance_measures:
                print(f"Calculate Clustering Quality for distance measure {distance_measure}")
                run_internal_measure_calculation_for_dataset(overall_dataset_name, data_dict=data_dict,
                                                             gt_labels_dict=gt_labels_dict,
                                                             partitions_dict=partitions_dict,
                                                             distance_measure=distance_measure, data_type=data_type,
                                                             data_dir=data_dir, results_dir=root_result_dir,
                                                             internal_measures=internal_measures,
                                                             n_cores=6)
