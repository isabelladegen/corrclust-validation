from hamcrest import *

from src.evaluation.describe_clustering_quality_for_data_variant import DescribeClusteringQualityForDataVariant
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

run_file = GENERATED_DATASETS_FILE_PATH
data_dir = SYNTHETIC_DATA_DIR
non_normal = SyntheticDataType.non_normal_correlated
overall_ds_name = "n30"
results_dir = ROOT_RESULTS_DIR
distance_measure = DistanceMeasures.l1_with_ref
clustering_quality_measures = [ClusteringQualityMeasures.jaccard_index, ClusteringQualityMeasures.silhouette_score]
describe = DescribeClusteringQualityForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                                   data_type=non_normal, data_dir=data_dir,
                                                   results_root_dir=results_dir, distance_measure=distance_measure)


def test_returns_overall_clustering_quality_measure_for_data_variant():
    values = describe.all_values_for_clustering_quality_measure(ClusteringQualityMeasures.jaccard_index)
    # load the result for each segmented clustering (67) for each subject (30)
    assert_that(len(values), is_(30 * 67))
