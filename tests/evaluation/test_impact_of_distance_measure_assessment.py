import pandas as pd
from hamcrest import *

from src.evaluation.impact_of_distance_measure_assessment import ImpactDistanceMeasureAssessment, \
    get_col_name_distance_internal_corr, get_col_distances_compared
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.stats import StatsCols
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR, TEST_ROOT_RESULTS_DIR

distance_measures = [DistanceMeasures.l1_cor_dist, DistanceMeasures.l1_with_ref, DistanceMeasures.foerstner_cor_dist]
internal_measure = ClusteringQualityMeasures.silhouette_score
da = ImpactDistanceMeasureAssessment(overall_ds_name="test_stuff", root_result_dir=TEST_ROOT_RESULTS_DIR,
                                     data_type=SyntheticDataType.normal_correlated, data_dir=TEST_DATA_DIR,
                                     internal_measure=internal_measure, distance_measures=distance_measures)


def test_loads_correlation_coefficients_for_given_internal_measure_and_different_distance_measures():
    df = da.correlations_df

    # loads all the correlation values for each subject (2) and each distance measure
    assert_that(df.shape, is_((2, len(distance_measures) + 1)))

    # check values for Förstner
    foerstner_vals = df[
        get_col_name_distance_internal_corr(internal_measure=internal_measure, distance_measure=distance_measures[-1])]
    assert_that(foerstner_vals.iloc[0], is_(0.984))
    assert_that(foerstner_vals.iloc[1], is_(0.532))


def test_calculates_fisher_t_paired_test_for_each_distance_measure_comb():
    df = da.paired_samples_t_test_on_fisher_transformed_correlation_coefficients()

    assert_that(df.shape, is_((4, len(distance_measures))))

    ps = df.loc[StatsCols.p_value]
    statistics = df.loc[StatsCols.statistic]
    effect_sizes = df.loc[StatsCols.effect_size]
    powers = df.loc[StatsCols.achieved_power]

    col_name = get_col_distances_compared(distance_measures[0], distance_measures[1])
    assert_that(ps[col_name], is_(0.5))
    assert_that(statistics[col_name], is_(1))
    assert_that(effect_sizes[col_name], is_(0.707))
    assert_that(powers[col_name], is_(0.073))

