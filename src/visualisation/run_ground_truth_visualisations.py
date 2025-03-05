from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_clustering_quality_measures_for_multiple_data_variants import \
    VisualiseGroundTruthClusteringQualityMeasuresForDataVariants


def ground_truth_visualisations(data_dirs: [str], data_types: [str], root_results_dir: str,
                                distance_measures: [str], clustering_quality_measures: [str],
                                overall_ds_name: str, backend: str, save_fig=True):
    gtv = VisualiseGroundTruthClusteringQualityMeasuresForDataVariants(overall_ds_name=overall_ds_name,
                                                                       dataset_types=data_types,
                                                                       data_dirs=data_dirs,
                                                                       result_root_dir=root_results_dir,
                                                                       internal_measures=clustering_quality_measures,
                                                                       distance_measures=distance_measures,
                                                                       backend=backend)
    gtv.ci_mean_ground_truth_for_quality_measures(save_fig=save_fig)


if __name__ == "__main__":
    # confidence interval
    # backend = Backends.visible_tests.value
    backend = Backends.none.value
    save_fig = True
    overall_ds_name = "n30"
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    distance_measures = [DistanceMeasures.l5_cor_dist, DistanceMeasures.linf_cor_dist,
                         DistanceMeasures.dot_transform_linf,
                         DistanceMeasures.l3_cor_dist, DistanceMeasures.l1_cor_dist, DistanceMeasures.l1_with_ref,
                         DistanceMeasures.l2_cor_dist, DistanceMeasures.log_frob_cor_dist,
                         DistanceMeasures.foerstner_cor_dist]

    ground_truth_visualisations(data_dirs=data_dirs, data_types=dataset_types,
                                root_results_dir=root_result_dir, distance_measures=distance_measures,
                                clustering_quality_measures=[ClusteringQualityMeasures.silhouette_score,
                                                             ClusteringQualityMeasures.dbi],
                                overall_ds_name=overall_ds_name,
                                backend=backend, save_fig=save_fig)

    ground_truth_visualisations(data_dirs=data_dirs, data_types=dataset_types,
                                root_results_dir=root_result_dir, distance_measures=distance_measures,
                                clustering_quality_measures=[ClusteringQualityMeasures.vrc],
                                overall_ds_name=overall_ds_name,
                                backend=backend, save_fig=save_fig)

    ground_truth_visualisations(data_dirs=data_dirs, data_types=dataset_types,
                                root_results_dir=root_result_dir, distance_measures=distance_measures,
                                clustering_quality_measures=[ClusteringQualityMeasures.pmb],
                                overall_ds_name=overall_ds_name,
                                backend=backend, save_fig=save_fig)
