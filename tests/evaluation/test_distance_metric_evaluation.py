from os import path
from pathlib import Path

import pandas as pd
from hamcrest import *

from src.evaluation.describe_synthetic_dataset import DescribeSyntheticDataset
from src.evaluation.distance_metric_assessment import DistanceMeasureCols
from src.evaluation.distance_metric_evaluation import DistanceMetricEvaluation, EvaluationCriteria
from src.utils.configurations import DISTANCE_MEASURE_ASSESSMENT_RESULTS_FOLDER_NAME, Aggregators
from src.utils.distance_measures import DistanceMeasures
from src.utils.plots.matplotlib_helper_functions import Backends
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR, TEST_IMAGES_DIR, TEST_ROOT_RESULTS_DIR, \
    TEST_IRREGULAR_P90_DATA_DIR

backend = Backends.none.value
# backend = Backends.visible_tests.value
a_ds_name = "misty-forest-56"
test_data_dir = TEST_IRREGULAR_P90_DATA_DIR  # so we have much less data for speed
images_dir = TEST_IMAGES_DIR
ds = DescribeSyntheticDataset(a_ds_name, data_dir=test_data_dir)
sel_measures = [DistanceMeasures.l2_cor_dist, DistanceMeasures.log_frob_cor_dist,
                DistanceMeasures.foerstner_cor_dist]
ev = DistanceMetricEvaluation(ds, measures=sel_measures, backend=backend)


def test_calculates_distances_for_all_empirical_correlations_to_all_canonical_patterns():
    df = ev.distances_df
    # wrong should not be 10000
    df.to_csv("irregularp90_distances.csv")
    assert_that(df.shape[0], is_(23 * 100))  # each segment compared to each canonical pattern
    # todo write more asserts
    # todo investigate nan cases for Förstner as they are still many


def test_calculate_ci_of_mean_differences_between_adjacent_level_sets_for_each_distance_measure():
    df = ev.ci_for_mean_differences

    l2_df = df[df[DistanceMeasureCols.type] == DistanceMeasures.l2_cor_dist]
    frob_df = df[df[DistanceMeasureCols.type] == DistanceMeasures.log_frob_cor_dist]
    foer_df = df[df[DistanceMeasureCols.type] == DistanceMeasures.foerstner_cor_dist]

    # calculated correct number of ci intervals
    n_ci_intervals = len(ev.level_set_indices) - 1
    assert_that(l2_df.shape[0], is_(n_ci_intervals))
    assert_that(frob_df.shape[0], is_(n_ci_intervals))
    assert_that(foer_df.shape[0], is_(n_ci_intervals))

    # check ci (measures are ordered Foerstner, L2, Log Frob)
    assert_that(df[df[DistanceMeasureCols.compared] == (0, 1)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "lower", "lower"))
    assert_that(df[df[DistanceMeasureCols.compared] == (1, 2)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("overlap", "lower", "higher"))
    assert_that(df[df[DistanceMeasureCols.compared] == (2, 3)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "lower", "lower"))
    assert_that(df[df[DistanceMeasureCols.compared] == (3, 4)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "lower", "higher"))
    assert_that(df[df[DistanceMeasureCols.compared] == (4, 5)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "lower", "lower"))


def test_per_level_set_statistics_calculation():
    df = ev.per_level_set_distance_statistics_df

    # all measures have been calculated
    assert_that(len(df[DistanceMeasureCols.type].unique()), is_(len(sel_measures)))
    # all level sets calculated
    assert_that(len(df[DistanceMeasureCols.level_set].unique()), is_(6))
    # each distance measure has the right number of rows
    assert_that(df[DistanceMeasureCols.level_set].shape[0], is_(len(sel_measures) * 6))

    # calculate a mean for one of the measures for level set 4
    m = sel_measures[1]
    data = ev.distances_df[[DistanceMeasureCols.level_set, m]]
    levels_set4_data = data[data[DistanceMeasureCols.level_set] == 4]
    actual_l4 = df[df[DistanceMeasureCols.level_set] == 4].set_index(DistanceMeasureCols.type)

    assert_that(actual_l4.loc[m, Aggregators.mean], is_(levels_set4_data[m].mean().round(3)))
    assert_that(actual_l4.loc[m, Aggregators.std], is_(levels_set4_data[m].std().round(3)))
    assert_that(actual_l4.loc[m, Aggregators.min], is_(levels_set4_data[m].min().round(3)))
    assert_that(actual_l4.loc[m, Aggregators.max], is_(levels_set4_data[m].max().round(3)))
    assert_that(actual_l4.loc[m, Aggregators.count], is_(levels_set4_data[m].count().round(3)))


def test_can_calculate_level_rate_of_increase_between_adjacent_levels():
    result_df = ev.rate_of_increase_between_level_sets()

    # row for each measure and each level set pair
    assert_that(result_df.shape[0], is_(len(sel_measures) * (len(ev.level_sets) - 1)))

    # check some values
    m = sel_measures[0]
    df_m = result_df.loc[(result_df[DistanceMeasureCols.type] == m)]
    result_l01 = df_m.loc[(df_m[DistanceMeasureCols.compared] == (0, 1))][DistanceMeasureCols.rate_of_increase].values[
        0]
    result_l45 = df_m.loc[(df_m[DistanceMeasureCols.compared] == (4, 5))][DistanceMeasureCols.rate_of_increase].values[
        0]

    # calculate by hand here
    mean_for_m = ev.per_level_set_distance_statistics_df.loc[
        (ev.per_level_set_distance_statistics_df[DistanceMeasureCols.type] == m)][
        [DistanceMeasureCols.level_set, Aggregators.mean]]
    l0 = mean_for_m[mean_for_m[DistanceMeasureCols.level_set] == 0][Aggregators.mean].values[0]
    l1 = mean_for_m[mean_for_m[DistanceMeasureCols.level_set] == 1][Aggregators.mean].values[0]
    l4 = mean_for_m[mean_for_m[DistanceMeasureCols.level_set] == 4][Aggregators.mean].values[0]
    l5 = mean_for_m[mean_for_m[DistanceMeasureCols.level_set] == 5][Aggregators.mean].values[0]

    # check results are the same
    assert_that(result_l01, is_((l1 - l0).round(3)))
    assert_that(result_l45, is_((l5 - l4).round(3)))


def test_calculates_raw_results_for_each_criteria_and_each_distance_measure():
    df = ev.raw_results_for_each_criteria()
    assert_that(df.shape, is_((8, len(sel_measures))))
    # check value added for each measure
    assert_that(df.loc[EvaluationCriteria.inter_i, sel_measures[0]], is_(0.023))
    assert_that(df.loc[EvaluationCriteria.inter_i, sel_measures[1]], is_(1.352))
    assert_that(df.loc[EvaluationCriteria.inter_i, sel_measures[2]], is_(1.2))
    # check each criterion is calculated
    assert_that(df.loc[EvaluationCriteria.inter_ii, sel_measures[0]], is_(True))
    assert_that(df.loc[EvaluationCriteria.inter_iii, sel_measures[0]], is_(0.515))
    assert_that(df.loc[EvaluationCriteria.disc_i, sel_measures[0]], is_(1))
    assert_that(df.loc[EvaluationCriteria.disc_ii, sel_measures[0]], is_(1))
    assert_that(df.loc[EvaluationCriteria.disc_iii, sel_measures[0]], is_(1))
    assert_that(df.loc[EvaluationCriteria.stab_i, sel_measures[0]], is_(1))
    assert_that(df.loc[EvaluationCriteria.stab_ii, sel_measures[0]], is_(1))
