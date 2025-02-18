from os import path
from pathlib import Path

from src.evaluation.impact_of_reduction_on_internal_indices import ImpactReductionOnInternalIndices, ReductionType
from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_REDUCED_SYNTHETIC_DATA_DIR, DataCompleteness, ROOT_REDUCED_RESULTS_DIR, \
    get_root_folder_for_reduced_cluster, get_data_dir, get_root_folder_for_reduced_segments, ROOT_RESULTS_DIR, \
    SYNTHETIC_DATA_DIR, internal_measure_evaluation_dir_for
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType


def run_impact_of_reduction_for_data_variant(overall_ds_name: str,
                                             reduced_root_result_dir: str,
                                             unreduced_root_result_dir: str,
                                             data_type: str,
                                             root_reduced_data_dir: str,
                                             unreduced_data_dir: str,
                                             data_completeness: str,
                                             n_dropped: [int],
                                             reduction_type: str,
                                             internal_measure: str,
                                             distance_measure: str):
    ir = ImpactReductionOnInternalIndices(overall_ds_name=overall_ds_name,
                                          reduced_root_result_dir=reduced_root_result_dir,
                                          unreduced_root_result_dir=unreduced_root_result_dir,
                                          data_type=data_type,
                                          root_reduced_data_dir=root_reduced_data_dir,
                                          unreduced_data_dir=unreduced_data_dir,
                                          data_completeness=data_completeness,
                                          n_dropped=n_dropped,
                                          reduction_type=reduction_type,
                                          internal_measure=internal_measure,
                                          distance_measure=distance_measure)

    # calculate t-tests between correlations: these are the results for all 0 and n_dropped version of this data variant
    paired_t_test_df = ir.paired_samples_t_test_on_fisher_transformed_correlation_coefficients()
    data_dir = get_data_dir(root_reduced_data_dir, data_completeness)
    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type=data_type,
        results_dir=reduced_root_result_dir, data_dir=data_dir,
        distance_measure=distance_measure)
    # filename needs to have data variant in it
    filename = IAResultsCSV.paired_t_test_reduced_datasets
    filename = "_".join([reduction_type, internal_measure, filename])
    complete_path = path.join(store_results_in, filename)
    print(complete_path)
    paired_t_test_df.to_csv(complete_path)


if __name__ == "__main__":
    """Run p tests for impact of reduction between 100%, 50% and 25% of clusters/segments"""
    n_dropped_clusters = [12, 17]
    n_dropped_segments = [50, 75]

    overall_dataset_name = "n30"

    distance_measure = DistanceMeasures.l1_cor_dist
    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi]

    data_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated]

    data_completeness = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]
    # root dir for data and results for reduced datasets
    root_reduced_data_dir = ROOT_REDUCED_SYNTHETIC_DATA_DIR
    root_reduced_results_dir = ROOT_REDUCED_RESULTS_DIR
    # root dirs for data and results for unreduced datasets
    root_results_dir = ROOT_RESULTS_DIR
    unreduced_data_dir = SYNTHETIC_DATA_DIR

    # Evaluate for clusters
    print("CALCULATE FOR DROPPED CLUSTERS")
    for cmp in data_completeness:
        for data_type in data_types:
            for measure in internal_measures:
                run_impact_of_reduction_for_data_variant(overall_ds_name=overall_dataset_name,
                                                         reduced_root_result_dir=root_reduced_results_dir,
                                                         unreduced_root_result_dir=root_results_dir,
                                                         data_type=data_type,
                                                         root_reduced_data_dir=root_reduced_data_dir,
                                                         unreduced_data_dir=unreduced_data_dir,
                                                         data_completeness=cmp,
                                                         n_dropped=n_dropped_clusters,
                                                         reduction_type=ReductionType.clusters,
                                                         internal_measure=measure,
                                                         distance_measure=distance_measure)

    # Evaluate for segments
    print("CALCULATE FOR DROPPED SEGMENTS")
    for dropped_seg in n_dropped_segments:
        for cmp in data_completeness:
            for data_type in data_types:
                for measure in internal_measures:
                    run_impact_of_reduction_for_data_variant(overall_ds_name=overall_dataset_name,
                                                             reduced_root_result_dir=root_reduced_results_dir,
                                                             unreduced_root_result_dir=root_results_dir,
                                                             data_type=data_type,
                                                             root_reduced_data_dir=root_reduced_data_dir,
                                                             unreduced_data_dir=unreduced_data_dir,
                                                             data_completeness=cmp,
                                                             n_dropped=n_dropped_segments,
                                                             reduction_type=ReductionType.segments,
                                                             internal_measure=measure,
                                                             distance_measure=distance_measure)
