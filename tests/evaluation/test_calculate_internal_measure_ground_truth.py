import pandas as pd
from hamcrest import *

from src.evaluation.calculate_internal_measures_ground_truth import CalculateInternalMeasuresGroundTruth
from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR, TEST_GENERATED_DATASETS_FILE_PATH


def test_calculates_internal_measures_for_provided_distance_measure_and_ground_truth_dataset():
    run_names = pd.read_csv(TEST_GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.dbi, ClusteringQualityMeasures.vrc]
    distance_measure = DistanceMeasures.dot_transform_l2
    test_data_dir = TEST_DATA_DIR
    data_type = SyntheticDataType.normal_correlated
    gt = CalculateInternalMeasuresGroundTruth(run_names=run_names, internal_measures=internal_measures,
                                              distance_measure=distance_measure, data_type=data_type,
                                              data_dir=test_data_dir)

    df = gt.ground_truth_summary_df
    row_1 = df.iloc[0]
    assert_that(df.shape[0], is_(len(run_names)))
    assert_that(row_1[DescribeBadPartCols.name], is_(run_names[0]))
    assert_that(row_1[DescribeBadPartCols.n_patterns], is_(23))
    assert_that(row_1[DescribeBadPartCols.n_segments], is_(100))
    assert_that(row_1[DescribeBadPartCols.n_observations], is_(1226400))
    assert_that(row_1[DescribeBadPartCols.errors].round(2), is_(0.02))
    assert_that(df.columns, has_item(ClusteringQualityMeasures.silhouette_score))
    assert_that(df.columns, has_item(ClusteringQualityMeasures.dbi))
    assert_that(df.columns, has_item(ClusteringQualityMeasures.pmb))
    assert_that(df.columns, has_item(ClusteringQualityMeasures.vrc))
    assert_that(df.isna().sum().sum(), is_(0))  # no nans


def test_run_wilxocon_signed_rank_test_for_given_internal_measure_and_distance_measure_pairs():
    pass