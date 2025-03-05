from hamcrest import *

from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, SYNTHETIC_DATA_DIR, \
    IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_clustering_quality_measures_for_multiple_data_variants import \
    VisualiseClusteringQualityMeasuresForDataVariants, VisualiseGroundTruthClusteringQualityMeasuresForDataVariants

# backend = Backends.visible_tests.value
backend = Backends.none.value

root_results_dir = ROOT_RESULTS_DIR
data_types = [SyntheticDataType.raw,
              SyntheticDataType.normal_correlated,
              SyntheticDataType.non_normal_correlated,
              SyntheticDataType.rs_1min]
data_dirs = [SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR,
             IRREGULAR_P90_DATA_DIR]
clustering_quality_measures = []
vds = VisualiseClusteringQualityMeasuresForDataVariants(run_file=GENERATED_DATASETS_FILE_PATH, overall_ds_name="n30",
                                                        dataset_types=data_types, data_dirs=data_dirs,
                                                        result_root_dir=root_results_dir,
                                                        distance_measure=DistanceMeasures.l1_with_ref, backend=backend)


def test_can_visualise_quality_measures_for_all_data_variants():
    column_names = [SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.raw),
                    SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.normal_correlated),
                    SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.non_normal_correlated),
                    SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.rs_1min)]
    assert_that(vds.col_names, contains_exactly(*column_names))
    assert_that(vds.row_names, contains_exactly('Complete 100%', 'Partial 70%', 'Sparse 10%'))
    fig = vds.violin_plots_for_quality_measure(quality_measure=ClusteringQualityMeasures.jaccard_index, save_fig=False)
    assert_that(fig, is_not(None))


def test_can_visualise_extreme_valued_dbi_measure_for_all_data_variants():
    # dbi has extremely large numbers that screw the plots - use log scale instead
    fig = vds.violin_plots_for_quality_measure(quality_measure=ClusteringQualityMeasures.dbi, save_fig=False)
    assert_that(fig, is_not(None))


def test_can_visualise_correlation_coefficient_distributions_for_all_data_variants():
    column_names = [SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.raw),
                    SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.normal_correlated),
                    SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.non_normal_correlated),
                    SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.rs_1min)]
    assert_that(vds.col_names, contains_exactly(*column_names))
    assert_that(vds.row_names, contains_exactly('Complete 100%', 'Partial 70%', 'Sparse 10%'))
    fig = vds.violin_plots_for_correlation_coefficients(quality_measure=ClusteringQualityMeasures.silhouette_score,
                                                        save_fig=False)
    assert_that(fig, is_not(None))


def test_can_visualise_scatter_plots_for_the_quality_measures_across_data_variants():
    column_names = [SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.raw),
                    SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.normal_correlated),
                    SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.non_normal_correlated),
                    SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.rs_1min)]
    assert_that(vds.col_names, contains_exactly(*column_names))
    assert_that(vds.row_names, contains_exactly('Complete 100%', 'Partial 70%', 'Sparse 10%'))
    fig = vds.scatter_plots_for_quality_measures(
        quality_measures=[ClusteringQualityMeasures.jaccard_index,
                          ClusteringQualityMeasures.vrc],
        save_fig=False)
    vds.scatter_plots_for_quality_measures(
        quality_measures=[ClusteringQualityMeasures.jaccard_index,
                          ClusteringQualityMeasures.silhouette_score],
        save_fig=False)
    vds.scatter_plots_for_quality_measures(
        quality_measures=[ClusteringQualityMeasures.jaccard_index,
                          ClusteringQualityMeasures.dbi],
        save_fig=False)
    vds.scatter_plots_for_quality_measures(
        quality_measures=[ClusteringQualityMeasures.jaccard_index,
                          ClusteringQualityMeasures.pmb],
        save_fig=False)
    assert_that(fig, is_not(None))


def test_can_visualise_ci_of_mean_ground_truth_for_measures():
    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi]
    dist_measures = [DistanceMeasures.l5_cor_dist, DistanceMeasures.linf_cor_dist, DistanceMeasures.dot_transform_linf,
                     DistanceMeasures.l3_cor_dist, DistanceMeasures.l1_cor_dist, DistanceMeasures.l1_with_ref,
                     DistanceMeasures.log_frob_cor_dist, DistanceMeasures.foerstner_cor_dist]
    gt_data_types = [SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    gtv = VisualiseGroundTruthClusteringQualityMeasuresForDataVariants(overall_ds_name="n30",
                                                                       dataset_types=gt_data_types, data_dirs=data_dirs,
                                                                       result_root_dir=root_results_dir,
                                                                       internal_measures=internal_measures,
                                                                       distance_measures=dist_measures,
                                                                       backend=backend)
    fig = gtv.ci_mean_ground_truth_for_quality_measures(save_fig=False)
    assert_that(fig, is_not(none()))
