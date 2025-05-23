import pandas as pd
import pandas.testing as tm
import pytest
from hamcrest import *

from src.evaluation.distance_metric_evaluation import DistanceMetricEvaluation, EvaluationCriteria, \
    read_csv_of_raw_values_for_all_criteria, DistanceMeasureCols
from src.utils.configurations import Aggregators
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from tests.test_utils.configurations_for_testing import TEST_IMAGES_DIR, TEST_IRREGULAR_P90_DATA_DIR, \
    TEST_ROOT_RESULTS_DIR

backend = Backends.none.value
# backend = Backends.visible_tests.value
a_ds_name = "misty-forest-56"
test_data_dir = TEST_IRREGULAR_P90_DATA_DIR  # so we have much less data for speed
images_dir = TEST_IMAGES_DIR
sel_measures = [DistanceMeasures.l2_cor_dist, DistanceMeasures.log_frob_cor_dist,
                DistanceMeasures.foerstner_cor_dist]
ev = DistanceMetricEvaluation(run_name=a_ds_name, data_type=SyntheticDataType.non_normal_correlated,
                              data_dir=test_data_dir, measures=sel_measures, backend=backend)


def test_calculates_distances_for_all_empirical_correlations_to_all_canonical_patterns():
    df = ev.distances_df
    # df.to_csv("irregularp90_distances.csv")
    assert_that(df.shape[0], is_(23 * 100))  # each segment compared to each canonical pattern
    # todo write more asserts


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
                contains_exactly("higher", "lower", "higher"))
    assert_that(df[df[DistanceMeasureCols.compared] == (2, 3)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "lower", "lower"))
    assert_that(df[df[DistanceMeasureCols.compared] == (3, 4)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("overlap", "lower", "higher"))
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


def test_calculate_normalised_distances_for_each_distance_level():
    df = ev.normalised_distance_df

    for measure in sel_measures:
        assert_that(df[measure].min(), equal_to(0))
        assert_that(df[measure].max(), equal_to(1))


def test_calculate_overall_shannon_entropy():
    result = ev.calculate_overall_shannon_entropy()

    # we have a result for each distance measure
    assert_that(len(result), is_(len(sel_measures)))

    assert_that(result[sel_measures[0]], is_(4.519))  # l2
    assert_that(result[sel_measures[1]], is_(4.645))  # Frobenious
    assert_that(result[sel_measures[2]], is_(3.813))  # Förstner


def test_calculate_shannon_entropy_per_level_set():
    result = ev.calculate_level_set_shannon_entropy()

    # we have a result for each level set
    assert_that(result.shape[0], is_(len(ev.level_set_indices)))
    # we have a column of results for each measure and the level set column
    assert_that(result.shape[1], is_(len(sel_measures) + 1))


def test_calculate_n_nan_inf_distances():
    result = ev.count_nan_inf_distance_for_measures()

    # we have a result for each measure
    assert_that(len(result), is_(len(sel_measures)))

    assert_that(result[sel_measures[0]], is_(0))  # l2
    assert_that(result[sel_measures[1]], is_(0))  # Frobenious
    assert_that(result[sel_measures[2]], is_(0))  # Förstner


@pytest.mark.skip(reason="just an experiment with different bin sizes")
def test_different_bins_for_overall_entropy():
    n_bins_list = [3, 8, 12, 50, 100, 150, 200, 253, 2500]

    # Create list to store results
    results = []

    # Calculate metrics for each n_bins value
    for n_bins in n_bins_list:
        metrics = ev.calculate_overall_shannon_entropy(n_bins=n_bins)
        # Add n_bins to the metrics dict
        metrics['n_bins'] = n_bins
        results.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(results)
    df.to_csv('overall_entropy_for_different_bins.csv')


@pytest.mark.skip(reason="just an experiment with different bin sizes")
def test_different_bins_for_entropy_per_level_set():
    n_bins_list = [3, 8, 12, 50, 100, 150, 200, 253, 2500]

    # Create list to store results
    results = []

    # Calculate metrics for each n_bins value
    for n_bins in n_bins_list:
        df_result = ev.calculate_level_set_shannon_entropy(n_bins)
        # Add n_bins as a column
        df_result.insert(0, 'n_bins', n_bins)
        # Append to results list
        results.append(df_result)

    # Concatenate all results into a single DataFrame
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv('level_set_entropy_for_different_bins.csv')


def test_calculates_raw_results_for_each_criteria_and_each_distance_measure():
    df = ev.raw_results_for_each_criteria()
    assert_that(df.shape, is_((7, len(sel_measures))))
    # check value added for each measure
    # level 0 avg distances
    assert_that(df.loc[EvaluationCriteria.inter_i, sel_measures[0]], is_(0.072))  # l2
    assert_that(df.loc[EvaluationCriteria.inter_i, sel_measures[1]], is_(7.219))  # log frob
    assert_that(df.loc[EvaluationCriteria.inter_i, sel_measures[2]], is_(7.549))  # Förstner
    # check each criterion is calculated
    # means of all adjacent level sets are sig different
    assert_that(df.loc[EvaluationCriteria.inter_ii, sel_measures[0]], is_(True))  # l2
    assert_that(df.loc[EvaluationCriteria.inter_ii, sel_measures[1]], is_(False))  # log frob
    assert_that(df.loc[EvaluationCriteria.inter_ii, sel_measures[2]], is_(False))  # Förstner
    # avg rate of increase between level set
    assert_that(df.loc[EvaluationCriteria.inter_iii, sel_measures[0]], is_(0.506))
    # overall entropy
    assert_that(df.loc[EvaluationCriteria.disc_i, sel_measures[0]], is_(4.519))
    # average level set entropy
    assert_that(df.loc[EvaluationCriteria.disc_ii, sel_measures[0]], is_(2.374))
    # F1 score
    assert_that(df.loc[EvaluationCriteria.disc_iii, sel_measures[0]], is_(1))  # l2
    assert_that(df.loc[EvaluationCriteria.disc_iii, sel_measures[1]], is_(0.509))  # log Frob
    assert_that(df.loc[EvaluationCriteria.disc_iii, sel_measures[2]], is_(0.044))  # Förstner
    # number of nan's
    assert_that(df.loc[EvaluationCriteria.stab_ii, sel_measures[0]], is_(0))


def test_saves_results(tmp_path):
    # if you want to check the structure it creates use the TEST_ROOT_RESULTS_DIR,
    # tmp_path handles the dir itself and will delete it after the test
    # base_results_dir = TEST_ROOT_RESULTS_DIR
    base_results_dir = str(tmp_path)
    ds_name = "test-criteria"
    ev.save_csv_of_raw_values_for_all_criteria(ds_name, base_results_dir)

    # read csv
    loaded_results = read_csv_of_raw_values_for_all_criteria(ds_name, ev.data_type, ev.data_dir, base_results_dir)

    # check two frames are the same
    calculated_results = ev.raw_results_for_each_criteria()
    tm.assert_frame_equal(loaded_results, calculated_results)
