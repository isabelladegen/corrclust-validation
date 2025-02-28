from hamcrest import *

from src.evaluation.internal_measure_ground_truth_assessment import InternalMeasureGroundTruthAssessment
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                     ClusteringQualityMeasures.dbi, ClusteringQualityMeasures.vrc]
distance_measures = [DistanceMeasures.l1_cor_dist, DistanceMeasures.l3_cor_dist,
                     DistanceMeasures.foerstner_cor_dist]
data_dir = SYNTHETIC_DATA_DIR
data_type = SyntheticDataType.raw
name = "n30"
results_dir = ROOT_RESULTS_DIR

ga = InternalMeasureGroundTruthAssessment(overall_ds_name=name, internal_measures=internal_measures,
                                          distance_measures=distance_measures, data_dir=data_dir,
                                          data_type=data_type, root_results_dir=results_dir)


def test_internal_measure_ground_truth_assessment_loads_all_distance_measures():
    dfs = ga.ground_truth_calculation_dfs

    assert_that(len(dfs), is_(len(distance_measures)))
    l1_results = dfs[DistanceMeasures.l1_cor_dist]
    # one ground truth result for all runs
    assert_that(l1_results.shape[0], is_(30))
    assert_that(l1_results[ClusteringQualityMeasures.silhouette_score].isna().sum(), is_(0))


