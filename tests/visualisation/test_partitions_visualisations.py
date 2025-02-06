import pandas as pd
from hamcrest import *

from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import internal_measure_evaluation_dir_for, \
    get_image_results_path
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.visualisation.partitions_visualisation import PartitionVisualisation
from tests.test_utils.configurations_for_testing import TEST_IMAGES_DIR, TEST_GENERATED_DATASETS_FILE_PATH, \
    TEST_ROOT_RESULTS_DIR, TEST_DATA_DIR

images_dir = TEST_IMAGES_DIR
results_dir = TEST_ROOT_RESULTS_DIR
data_dir = TEST_DATA_DIR
data_type = SyntheticDataType.normal_correlated
overall_dataset_name = "test_stuff"
run_names = pd.read_csv(TEST_GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
l1ref_results_dir = internal_measure_evaluation_dir_for(overall_dataset_name, data_type, results_dir, data_dir,
                                                        DistanceMeasures.l1_with_ref)
pal1ref = PartitionVisualisation(overall_dataset_name, data_type, results_dir, data_dir, DistanceMeasures.l1_with_ref,
                                 run_names=run_names)
backend = Backends.none.value


# backend = Backends.visible_tests.value


# these tests need results to run - run the quality measures calculation first
def test_loads_all_partitions_for_each_datasets_ordered_by_worst_jaccard_to_best():
    # read all datasets
    assert_that(len(pal1ref.partition_outcomes), is_(2))

    # has results for each partition
    a_df = pal1ref.partition_outcomes[0]
    assert_that(a_df.shape[0], is_(8))

    # sorted partitions from worst to best
    worst_partition = a_df.iloc[0]
    best_partition = a_df.iloc[7]
    assert_that(worst_partition[ClusteringQualityMeasures.jaccard_index], is_(0.0))
    assert_that(best_partition[ClusteringQualityMeasures.jaccard_index], is_(1.0))
    assert_that(a_df.index[0], is_(0))

    # includes all information
    assert_that(a_df.columns.to_list(), contains_exactly(DescribeBadPartCols.name,
                                                         DescribeBadPartCols.n_patterns,
                                                         DescribeBadPartCols.n_segments,
                                                         DescribeBadPartCols.n_observations,
                                                         DescribeBadPartCols.errors,
                                                         DescribeBadPartCols.n_seg_outside_tol,
                                                         DescribeBadPartCols.n_wrong_clusters,
                                                         DescribeBadPartCols.n_obs_shifted,
                                                         ClusteringQualityMeasures.jaccard_index,
                                                         ClusteringQualityMeasures.silhouette_score,
                                                         ClusteringQualityMeasures.dbi,
                                                         ClusteringQualityMeasures.vrc,
                                                         ClusteringQualityMeasures.pmb))


def test_calculates_statistics_per_partition_across_n_30_datasets():
    df = pal1ref.calculate_describe_statistics_for_partitions(column=ClusteringQualityMeasures.jaccard_index)

    assert_that(len(df.columns), is_(8))  # partitions are now columns, describe is across the 30 ds
    assert_that(df.loc['mean', 0], is_(0))
    assert_that(df.loc['mean', 7], is_(0.98))


def test_plot_descriptive_statistics_for_partitions_for_column():
    fig = pal1ref.plot_describe_statistics_for_partitions_for_column(column=ClusteringQualityMeasures.jaccard_index,
                                                                     backend=backend)

    assert_that(fig, is_not(None))
    fig.savefig(get_image_results_path(l1ref_results_dir, 'bad_partitions_jaccard.png'))

    fig = pal1ref.plot_describe_statistics_for_partitions_for_column(column=ClusteringQualityMeasures.silhouette_score,
                                                                     backend=backend)

    assert_that(fig, is_not(None))
    fig.savefig(get_image_results_path(l1ref_results_dir, 'bad_partitions_scw.png'))

    fig = pal1ref.plot_describe_statistics_for_partitions_for_column(column=ClusteringQualityMeasures.pmb,
                                                                     backend=backend)

    assert_that(fig, is_not(None))
    fig.savefig(get_image_results_path(l1ref_results_dir, 'bad_partitions_pmb.png'))

    fig = pal1ref.plot_describe_statistics_for_partitions_for_column(column=DescribeBadPartCols.errors,
                                                                     backend=backend)

    assert_that(fig, is_not(None))
    fig.savefig(get_image_results_path(l1ref_results_dir, 'bad_partitions_sum_errors.png'))

    fig = pal1ref.plot_describe_statistics_for_partitions_for_column(column=DescribeBadPartCols.n_wrong_clusters,
                                                                     backend=backend)

    assert_that(fig, is_not(None))
    fig.savefig(get_image_results_path(l1ref_results_dir, 'bad_partitions_n_wrong_clusters.png'))

    fig = pal1ref.plot_describe_statistics_for_partitions_for_column(column=DescribeBadPartCols.n_obs_shifted,
                                                                     backend=backend)

    assert_that(fig, is_not(None))
    fig.savefig(get_image_results_path(l1ref_results_dir, 'bad_partitions_n_obs_shifted.png'))


def test_plot_describe_statistics_for_partitions():
    fig = pal1ref.plot_describe_statistics_for_partitions(backend=backend)

    assert_that(fig, is_not(None))
    fig.savefig(get_image_results_path(l1ref_results_dir, 'bad_partitions.png'))


def test_plot_descriptive_statistics_for_partitions_for_column_for_l2_measure():
    l2_results_dir = internal_measure_evaluation_dir_for(overall_dataset_name, data_type, results_dir, data_dir,
                                                         DistanceMeasures.l2_cor_dist)

    pal2 = PartitionVisualisation(overall_dataset_name, data_type, results_dir, data_dir, DistanceMeasures.l2_cor_dist,
                                  run_names=run_names)
    fig = pal2.plot_describe_statistics_for_partitions_for_column(column=ClusteringQualityMeasures.silhouette_score,
                                                                  backend=backend)
    assert_that(fig, is_not(None))
    fig.savefig(get_image_results_path(l2_results_dir, 'bad_partitions_scw_L2.png'))

    fig = pal2.plot_describe_statistics_for_partitions_for_column(column=ClusteringQualityMeasures.pmb,
                                                                  backend=backend)
    assert_that(fig, is_not(None))
    fig.savefig(get_image_results_path(l2_results_dir, 'bad_partitions_pmb_L2.png'))


def test_plot_descriptive_statistics_for_partitions_for_column_for_l1_measure():
    l1_results_dir = internal_measure_evaluation_dir_for(overall_dataset_name, data_type,
                                                         results_dir, data_dir,
                                                         DistanceMeasures.l1_cor_dist)

    pal1 = PartitionVisualisation(overall_dataset_name, data_type, results_dir, data_dir, DistanceMeasures.l1_cor_dist,
                                  run_names=run_names)
    fig = pal1.plot_describe_statistics_for_partitions_for_column(column=ClusteringQualityMeasures.silhouette_score,
                                                                  backend=backend)
    assert_that(fig, is_not(None))
    fig.savefig(get_image_results_path(l1_results_dir, 'bad_partitions_scw_L1.png'))

    fig = pal1.plot_describe_statistics_for_partitions_for_column(column=ClusteringQualityMeasures.pmb,
                                                                  backend=backend)
    assert_that(fig, is_not(None))
    fig.savefig(get_image_results_path(l1_results_dir, 'bad_partitions_pmb_L1.png'))


def test_plots_all_three_descriptive_measures_in_one_column_for_paper():
    columns = [ClusteringQualityMeasures.jaccard_index,
               DescribeBadPartCols.n_wrong_clusters,
               DescribeBadPartCols.n_obs_shifted]
    fig = pal1ref.plot_multiple_quality_measures(columns=columns, backend=backend)
    assert_that(fig, is_not(None))
