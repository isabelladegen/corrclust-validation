from hamcrest import *

from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.evaluation.describe_clustering_quality_for_data_variant import DescribeClusteringQualityForDataVariant, \
    IntSummaryCols
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.visualisation.visualise_multiple_data_variants import get_row_name_from

run_file = GENERATED_DATASETS_FILE_PATH
data_dir = SYNTHETIC_DATA_DIR
non_normal = SyntheticDataType.non_normal_correlated
overall_ds_name = "n30"
results_dir = ROOT_RESULTS_DIR
distance_measure = DistanceMeasures.l1_with_ref
describe = DescribeClusteringQualityForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                                   data_type=non_normal, data_dir=data_dir,
                                                   results_root_dir=results_dir, distance_measure=distance_measure)


def test_returns_overall_clustering_quality_measure_for_data_variant():
    values = describe.all_values_for_clustering_quality_measure(ClusteringQualityMeasures.jaccard_index)
    # load the result for each segmented clustering (67) for each subject (30)
    assert_that(len(values), is_(30 * 67))


def test_returns_overall_correlation_coefficients_of_quality_measure_with_jaccard_index_for_data_variant():
    values = describe.all_values_for_correlations_with_jaccard_index_for_quality_measure(
        ClusteringQualityMeasures.silhouette_score)
    # load the result for each segmented clustering (67) for each subject (30) and correlated it with jaccard
    assert_that(len(values), is_(30))
    assert_that(values.min(), is_(0.877))
    assert_that(values.max(), is_(0.93))


def test_returns_mean_and_sd_of_correlation_for_data_variant_and_distance_level():
    df = describe.mean_sd_correlation_for(
        quality_measures=[ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi])
    assert_that(df[IntSummaryCols.data_stage][0], is_(SyntheticDataType.get_display_name_for_data_type(non_normal)))
    assert_that(df[IntSummaryCols.data_completeness][0], is_(get_row_name_from(data_dir)))
    assert_that(df[ClusteringQualityMeasures.silhouette_score][0], is_('0.90 (0.01)'))
    assert_that(df[ClusteringQualityMeasures.dbi][0], is_('-0.89 (0.08)'))


def test_returns_p_value_and_effect_size_of_correlation_for_data_variant_and_distance_level():
    measures = [ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi,
                ClusteringQualityMeasures.silhouette_score]
    df = describe.p_value_and_effect_size_of_correlation_for(quality_measures=measures)
    assert_that(df[IntSummaryCols.data_stage][0], is_(SyntheticDataType.get_display_name_for_data_type(non_normal)))
    assert_that(df[IntSummaryCols.data_completeness][0], is_(get_row_name_from(data_dir)))
    assert_that(df[(ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi)][0], is_('0.71 (-0.07)'))
    assert_that(df[(ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.vrc)][0], is_('0.00 (14.83)'))
    assert_that(df[(ClusteringQualityMeasures.dbi, ClusteringQualityMeasures.vrc)][0], is_('0.00 (4.68)'))


def test_returns_measures_values_for_ground_truth_and_lowest_jaccard_index():
    measures = [ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi,
                ClusteringQualityMeasures.silhouette_score]
    df = describe.mean_sd_measure_values_for_ground_truth_and_lowest_jaccard_index(quality_measures=measures)
    assert_that(df[IntSummaryCols.data_stage][0], is_(SyntheticDataType.get_display_name_for_data_type(non_normal)))
    assert_that(df[IntSummaryCols.data_completeness][0], is_(get_row_name_from(data_dir)))
    assert_that(df[(ClusteringQualityMeasures.vrc, IntSummaryCols.gt)][0], is_('14582.24 (4622.20)'))
    assert_that(df[(ClusteringQualityMeasures.silhouette_score, IntSummaryCols.gt)][0], is_('0.97 (0.00)'))
    assert_that(df[(ClusteringQualityMeasures.dbi, IntSummaryCols.gt)][0], is_('0.05 (0.01)'))
    assert_that(df[(ClusteringQualityMeasures.vrc, IntSummaryCols.worst)][0], is_('1.00 (0.22)'))
    assert_that(df[(ClusteringQualityMeasures.silhouette_score, IntSummaryCols.worst)][0], is_('-0.45 (0.04)'))
    assert_that(df[(ClusteringQualityMeasures.dbi, IntSummaryCols.worst)][0], is_('9.39 (2.02)'))

def test_returns_benchmark_summary_df_across_subjects():
    df = describe.summary_benchmark_df()

    # best partition across 30 subjects
    assert_that(df.loc[0, (ClusteringQualityMeasures.jaccard_index, 'mean')], is_(1))
    assert_that(df.loc[0, (ClusteringQualityMeasures.silhouette_score, 'mean')], is_(0.969))
    assert_that(df.loc[0, (ClusteringQualityMeasures.dbi, 'mean')], is_(0.047))
    assert_that(df.loc[0, (DescribeBadPartCols.errors, 'mean')], is_(0.024))
    assert_that(df.loc[0, DescribeBadPartCols.n_obs_shifted][0], is_(0))
    assert_that(df.loc[0, DescribeBadPartCols.n_wrong_clusters][0], is_(0))

    # worst partition across 30 subjects
    assert_that(df.loc[df.index[-1], (ClusteringQualityMeasures.jaccard_index, 'mean')], is_(0))
    assert_that(df.loc[df.index[-1], (ClusteringQualityMeasures.silhouette_score, 'mean')], is_(-0.446))
    assert_that(df.loc[df.index[-1], (ClusteringQualityMeasures.dbi, 'mean')], is_(9.112))
    assert_that(df.loc[df.index[-1], (DescribeBadPartCols.errors, 'mean')], is_(0.77))
    assert_that(df.loc[df.index[-1], DescribeBadPartCols.n_obs_shifted][0], is_(0))
    assert_that(df.loc[df.index[-1], DescribeBadPartCols.n_wrong_clusters][0], is_(100))

