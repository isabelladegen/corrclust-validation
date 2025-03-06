import pandas as pd

from src.evaluation.impact_of_distance_measure_assessment import ImpactDistanceMeasureAssessment
from src.evaluation.internal_measure_assessment import get_full_filename_for_results_csv, IAResultsCSV, \
    InternalMeasureAssessment, get_name_paired_t_test_between_distance_measures
from src.experiments.run_cluster_quality_measures_calculation import read_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, internal_measure_evaluation_dir_for, \
    SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType


def run_impact_of_distance_measure_on_data_variant(overall_ds_name: str, run_names: [str], distance_measures: [str],
                                                   data_type: str, data_dir: str, results_dir: str,
                                                   internal_measures: [str], n_dropped_clusters: [int] = [],
                                                   n_dropped_segments: [int] = []):
    store_results_in = internal_measure_evaluation_dir_for(overall_dataset_name=overall_ds_name,
                                                           data_type=data_type, distance_measure="",
                                                           results_dir=results_dir, data_dir=data_dir)

    for index in internal_measures:
        da = ImpactDistanceMeasureAssessment(overall_ds_name=overall_ds_name, root_result_dir=results_dir,
                                             data_type=data_type, data_dir=data_dir, internal_measure=index,
                                             distance_measures=distance_measures)
        df = da.paired_samples_t_test_on_fisher_transformed_correlation_coefficients()

        # save the result
        file_name = get_name_paired_t_test_between_distance_measures(index)
        df.to_csv(get_full_filename_for_results_csv(store_results_in, file_name))


if __name__ == "__main__":
    overall_ds_name = "n30"
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    distance_measures = [DistanceMeasures.l1_cor_dist,
                         DistanceMeasures.l1_with_ref,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.l5_with_ref,
                         DistanceMeasures.linf_cor_dist]

    # Config for L2 only ran for downsampled, complete data
    # distance_measures = [DistanceMeasures.l1_cor_dist, DistanceMeasures.l1_with_ref,
    #                      DistanceMeasures.foerstner_cor_dist, DistanceMeasures.l2_cor_dist]
    # dataset_types = [SyntheticDataType.rs_1min]
    # data_dirs = [SYNTHETIC_DATA_DIR]

    # Config for L3 only ran for downsampled, complete data
    # distance_measures = [DistanceMeasures.l1_cor_dist, DistanceMeasures.l1_with_ref,
    #                      DistanceMeasures.foerstner_cor_dist, DistanceMeasures.l2_cor_dist, DistanceMeasures.l3_cor_dist]
    # dataset_types = [SyntheticDataType.rs_1min]
    # data_dirs = [SYNTHETIC_DATA_DIR]

    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.pmb]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    for data_dir in data_dirs:
        for data_type in dataset_types:
            run_impact_of_distance_measure_on_data_variant(overall_ds_name=overall_ds_name,
                                                           run_names=run_names,
                                                           distance_measures=distance_measures,
                                                           data_type=data_type,
                                                           data_dir=data_dir,
                                                           results_dir=root_result_dir,
                                                           internal_measures=internal_measures)
