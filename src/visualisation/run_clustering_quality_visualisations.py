from matplotlib import pyplot as plt

from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, DataCompleteness
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_clustering_quality_measures_for_multiple_data_variants import \
    VisualiseClusteringQualityMeasuresForDataVariants


def clustering_quality_visualisations(data_dirs: [str], data_types: [str], run_file: str, root_results_dir: str,
                                      distance_measures: [str], clustering_quality_measures: [str],
                                      overall_ds_name: str, backend: str, save_fig=True, figsize1=(15, 10),
                                      figsize2=(18, 10)):
    for distance_measure in distance_measures:
        vds = VisualiseClusteringQualityMeasuresForDataVariants(run_file=run_file,
                                                                overall_ds_name=overall_ds_name,
                                                                dataset_types=data_types, data_dirs=data_dirs,
                                                                result_root_dir=root_results_dir,
                                                                distance_measure=distance_measure,
                                                                backend=backend)
        vds.scatter_plots_for_multiple_quality_measures(reference_measure=ClusteringQualityMeasures.jaccard_index,
                                                        quality_measures=[ClusteringQualityMeasures.silhouette_score,
                                                                          ClusteringQualityMeasures.dbi,
                                                                          ClusteringQualityMeasures.vrc,
                                                                          ClusteringQualityMeasures.pmb],
                                                        data_type=SyntheticDataType.non_normal_correlated,
                                                        completeness=DataCompleteness.irregular_p30,
                                                        save_fig=True)
        for quality_measure in clustering_quality_measures:
            vds.violin_plots_for_quality_measure(quality_measure=quality_measure, save_fig=save_fig, figsize=figsize1)
            if quality_measure != ClusteringQualityMeasures.jaccard_index:
                vds.violin_plots_for_correlation_coefficients(quality_measure=quality_measure, save_fig=save_fig,
                                                              figsize=figsize1)
                vds.scatter_plots_for_quality_measures(
                    quality_measures=[ClusteringQualityMeasures.jaccard_index, quality_measure], save_fig=save_fig,
                    figsize=figsize2)
            plt.close('all')


if __name__ == "__main__":
    # violin plots for all clustering quality measures for each dataset in the N30 and each distance measure
    # backend = Backends.visible_tests.value
    backend = Backends.none.value
    save_fig = True
    overall_ds_name = "n30"
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.raw,
                     SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    distance_measures = [DistanceMeasures.l1_cor_dist,
                         DistanceMeasures.l1_with_ref,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.foerstner_cor_dist]

    # Config for L2 only ran for downsampled, complete data
    # distance_measures = [DistanceMeasures.l2_cor_dist]
    # dataset_types = [SyntheticDataType.rs_1min]
    # data_dirs = [SYNTHETIC_DATA_DIR]

    clustering_quality_measures = [ClusteringQualityMeasures.jaccard_index,
                                   ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi,
                                   ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.pmb]

    run_file = GENERATED_DATASETS_FILE_PATH

    clustering_quality_visualisations(data_dirs=data_dirs, data_types=dataset_types, run_file=run_file,
                                      root_results_dir=root_result_dir, distance_measures=distance_measures,
                                      clustering_quality_measures=clustering_quality_measures,
                                      overall_ds_name=overall_ds_name,
                                      backend=backend, save_fig=save_fig)
