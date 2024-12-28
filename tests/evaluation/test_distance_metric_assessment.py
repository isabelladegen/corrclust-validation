import itertools
from os import path
from pathlib import Path

import numpy as np
from hamcrest import *
from scipy.linalg import eigvals

from src.utils.configurations import Aggregators, DISTANCE_MEASURE_ASSESSMENT_RESULTS_FOLDER_NAME
from src.utils.distance_measures import calculate_foerstner_matrices_distance_between
from src.utils.plots.matplotlib_helper_functions import Backends
from src.utils.stats import number_of_unique_two_combinations
from src.evaluation.distance_metric_assessment import DistanceMeasureCols, DistanceMetricAssessment, \
    default_order
from src.evaluation.describe_synthetic_dataset import DescribeSyntheticDataset
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR, TEST_IMAGES_DIR, TEST_ROOT_RESULTS_DIR

backend = Backends.none.value
# backend = Backends.visible_tests.value
a_ds_name = "misty-forest-56"
test_data_dir = TEST_DATA_DIR
images_dir = TEST_IMAGES_DIR
# todo undo this when we store proper results for the distance measure assessment
tables_dir = path.join(TEST_ROOT_RESULTS_DIR, DISTANCE_MEASURE_ASSESSMENT_RESULTS_FOLDER_NAME)
Path(tables_dir).mkdir(parents=True, exist_ok=True)
ds = DescribeSyntheticDataset(a_ds_name, data_dir=test_data_dir)
da = DistanceMetricAssessment(ds, backend=backend)
lp_measures = [DistanceMeasureCols.l1_cor_dist, DistanceMeasureCols.l2_cor_dist, DistanceMeasureCols.l10_cor_dist,
               DistanceMeasureCols.linf_cor_dist]
lp_da = DistanceMetricAssessment(ds, measures=lp_measures, backend=backend)
ts_measures = [DistanceMeasureCols.dtw]


def test_calculate_distances_between_each_segment_pair_in_each_group_for_all_distances():
    segment_pair_distance_df = da.segment_pair_distance_df
    assert_that(len(segment_pair_distance_df[DistanceMeasureCols.group].unique()), is_(6))
    # check all segment pairs distances have been calculated
    assert_that(segment_pair_distance_df.shape[0],
                is_(len(list(itertools.chain.from_iterable(ds.segment_pairs_for_group.values())))))

    # euclidian
    euc = segment_pair_distance_df[DistanceMeasureCols.l2_cor_dist]
    euc_nan = euc.isna().sum()
    euc_inf = np.isinf(euc).values.ravel().sum()
    assert_that(int(euc.sum()), is_(7447))
    assert_that(euc_nan, is_(0))
    assert_that(euc_inf, is_(0))

    # Frobenious
    frob = segment_pair_distance_df[DistanceMeasureCols.log_frob_cor_dist]
    frob_nan = frob.isna().sum()
    frob_inf = np.isinf(frob).values.ravel().sum()
    assert_that(int(frob.sum()), is_(181308))
    assert_that(frob_nan, is_(0))
    assert_that(frob_inf, is_(0))

    # Foerstner
    foer = segment_pair_distance_df[DistanceMeasureCols.foerstner_cor_dist]
    foer_nan = foer.isna().sum()
    foer_inf = np.isinf(foer).values.ravel().sum()
    assert_that(int(foer.replace([np.inf, -np.inf], np.nan).dropna().sum()), is_(65565))
    assert_that(foer_nan, is_(0))  # fixed by +1 and absolute value for negative numbers
    assert_that(foer_inf, is_(0))  # fixed by epsilon 1e-10 to diagonal to make all matrices full rank


def test_of_foerstner_nan_and_inf():
    # this is not a proper test but it shows problematic matrices for Förstner dist
    # nan results
    # 1) near zero eigenvalues -> fix with lambda + 1
    # 2) negative eigenvalues of the generalised problem -> which is the case for the generalised problem when
    # comparing exactly opposite matrices which is often the case for us (like below) -> fix also through regularisation
    # 0.989899+0.000000j
    # 1.009901+0.000000j
    # -0.424803+0.000000j -> this is likely an imprecision! one of the eigenvalues of each matrix should have been 0
    # but instead is extremely small leading in the generalised problem to this completely wrong eigenvalue of -0.425!
    # with regularisation these become
    # 0.989899
    # 1.009901
    # 1.000001 -> which is correct for the not full rank
    cor1 = np.array([[1., -0.02, -0.02], [-0.02, 1., 1.], [-0.02, 1., 1.]])
    cor2 = np.array([[1., -0.01, -0.01], [-0.01, 1., 1.], [-0.01, 1., 1.]])
    dist = calculate_foerstner_matrices_distance_between(cor1, cor2)
    print(dist)
    assert_that(np.isnan(dist), is_(False))
    assert_that(np.isinf(dist), is_(False))

    # inf -> extremely frequent
    # this happens when B matrix is singular (here cor 1 and 2) = det=0 and the other is not singular
    # this happens e.g. when two rows are the same
    # or one row is a multiple of another row
    # 3) we fix this by adding epsilon to the diagonal of all correlation matrices
    cor1_i = np.array([[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]])
    cor2_i = np.array([[1., -0.01, 1.], [-0.01, 1., - 0.01], [1., -0.01, 1.]])
    disti = calculate_foerstner_matrices_distance_between(cor1_i, cor2_i)
    print(disti)
    assert_that(np.isnan(disti), is_(False))
    assert_that(np.isinf(disti), is_(False))


def test_eigenvalues_for_weird_matrices_for_write_up_and_bug_report():
    # not well behaved that don't go to inf but to a negative eigenvalue instead - shocking!!
    A = np.array([[1., -0.02, -0.02], [-0.02, 1., 1.], [-0.02, 1., 1.]])
    B = np.array([[1., -0.01, -0.01], [-0.01, 1., 1.], [-0.01, 1., 1.]])
    # well behaved that go to inf
    # A = np.array([[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]])
    # B = np.array([[1., -0.01, 1.], [-0.01, 1., - 0.01], [1., -0.01, 1.]])

    print("Eig A")
    print(eigvals(A))
    print("Eig B")
    print(eigvals(B))
    vals = eigvals(A, b=B)
    real_vals = np.real(vals)
    print("Generalised")
    print(real_vals)

    reg_m = np.identity(A.shape[0]) * 1e-10
    Areg = A + reg_m
    Breg = B + reg_m
    print("Reg Eig A")
    print(eigvals(Areg))
    print("Reg Eig B")
    print(eigvals(Breg))

    # calculate generalised eigenvalues
    vals = eigvals(Areg, b=Breg)
    real_vals = np.real(vals)
    print("Regulated generalised")
    print(real_vals)


def test_calculates_mean_std_count_per_group_for_each_distance_measure():
    # Euclidian
    group_euc_stats = da.per_group_distance_statistics_df[
        da.per_group_distance_statistics_df[DistanceMeasureCols.type] == DistanceMeasureCols.l2_cor_dist]
    assert_that(group_euc_stats.shape[0], is_(6))  # statistics for group 0-5
    g0_euc = group_euc_stats[group_euc_stats[DistanceMeasureCols.group] == 0]
    g1_euc = group_euc_stats[group_euc_stats[DistanceMeasureCols.group] == 1]
    assert_that(g0_euc[Aggregators.count].values[0], is_(170))
    assert_that(g1_euc[Aggregators.count].values[0], is_(822))
    assert_that(g0_euc[Aggregators.mean].values[0], is_(less_than(g1_euc[Aggregators.mean].values[0])))
    assert_that(g0_euc[Aggregators.min].values[0], is_(less_than(g1_euc[Aggregators.min].values[0])))
    assert_that(g0_euc[Aggregators.max].values[0], is_(less_than(g1_euc[Aggregators.max].values[0])))

    # Frobenious
    group_frob_stats = da.per_group_distance_statistics_df[
        da.per_group_distance_statistics_df[DistanceMeasureCols.type] == DistanceMeasureCols.log_frob_cor_dist]
    assert_that(group_frob_stats.shape[0], is_(6))  # statistics for group 0-5
    g0_frob = group_frob_stats[group_frob_stats[DistanceMeasureCols.group] == 0]
    g1_frob = group_frob_stats[group_frob_stats[DistanceMeasureCols.group] == 1]
    assert_that(g0_frob[Aggregators.count].values[0], is_(170))
    assert_that(g1_frob[Aggregators.count].values[0], is_(822))
    assert_that(g0_frob[Aggregators.mean].values[0], is_(less_than(g1_frob[Aggregators.mean].values[0])))
    assert_that(g0_frob[Aggregators.min].values[0], is_(less_than(g1_frob[Aggregators.min].values[0])))
    assert_that(g0_frob[Aggregators.max].values[0], is_(less_than(g1_frob[Aggregators.max].values[0])))

    # Foerstner
    group_foer_stats = da.per_group_distance_statistics_df[
        da.per_group_distance_statistics_df[DistanceMeasureCols.type] == DistanceMeasureCols.foerstner_cor_dist]
    assert_that(group_foer_stats.shape[0], is_(6))  # statistics for group 0-5
    g0_foer = group_foer_stats[group_foer_stats[DistanceMeasureCols.group] == 0]
    g1_foer = group_foer_stats[group_foer_stats[DistanceMeasureCols.group] == 1]
    assert_that(g0_foer[Aggregators.count].values[0], is_(170))  # now fixed with new implementation of Förstner
    assert_that(g1_foer[Aggregators.count].values[0], is_(822))  # now fixed with new implementation of Förstner
    assert_that(g0_foer[Aggregators.mean].values[0], is_(less_than(g1_foer[Aggregators.mean].values[0])))
    assert_that(g0_foer[Aggregators.min].values[0], is_(less_than(g1_foer[Aggregators.min].values[0])))
    assert_that(g0_foer[Aggregators.max].values[0], is_(less_than(g1_foer[Aggregators.max].values[0])))


def test_calculates_effect_size_for_all_differences_in_means_between_groups_per_distance_measure():
    euc_effect_sizes_df = da.effect_sizes_between_groups_df[
        da.effect_sizes_between_groups_df[DistanceMeasureCols.type] == DistanceMeasureCols.l2_cor_dist]
    assert_that(round(euc_effect_sizes_df[DistanceMeasureCols.effect_size].abs().min(), 2), is_(1.02))

    frob_effect_sizes_df = da.effect_sizes_between_groups_df[
        da.effect_sizes_between_groups_df[DistanceMeasureCols.type] == DistanceMeasureCols.log_frob_cor_dist]
    assert_that(round(frob_effect_sizes_df[DistanceMeasureCols.effect_size].abs().min(), 2), is_(0.17))

    foer_effect_sizes_df = da.effect_sizes_between_groups_df[
        da.effect_sizes_between_groups_df[DistanceMeasureCols.type] == DistanceMeasureCols.foerstner_cor_dist]
    assert_that(round(foer_effect_sizes_df[DistanceMeasureCols.effect_size].abs().min(), 2), is_(0.04))


def test_returns_order_of_group_by_smallest_first():
    l2_order, euc_dist = da.ordered_groups_and_mean_distances_by_smallest_first(DistanceMeasureCols.l2_cor_dist)
    frob_order, frob_dist = da.ordered_groups_and_mean_distances_by_smallest_first(
        DistanceMeasureCols.log_frob_cor_dist)

    assert_that(l2_order, contains_exactly(0, 1, 2, 3, 4, 5))
    assert_that(frob_order, contains_exactly(0, 2, 4, 1, 3, 5))


def test_calculate_ci_of_mean_differences_between_groups_for_each_distance_measure():
    df = da.ci_for_mean_differences

    euc_df = df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l2_cor_dist]
    frob_df = df[df[DistanceMeasureCols.type] == DistanceMeasureCols.log_frob_cor_dist]
    foer_df = df[df[DistanceMeasureCols.type] == DistanceMeasureCols.foerstner_cor_dist]

    # 15 group comparisons for each measure
    assert_that(euc_df.shape[0], is_(15))
    assert_that(frob_df.shape[0], is_(15))
    assert_that(foer_df.shape[0], is_(15))

    # distances in group 0 are lower than group 1 (foerstner measure fails to compute)
    assert_that(df[df[DistanceMeasureCols.compared] == (0, 1)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "lower", "lower"))
    # distances in group 0 are lower than group 5
    assert_that(df[df[DistanceMeasureCols.compared] == (0, 5)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "lower", "lower"))
    assert_that(df[df[DistanceMeasureCols.compared] == (1, 2)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "higher", "higher"))
    assert_that(df[df[DistanceMeasureCols.compared] == (2, 3)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "lower", "lower"))
    assert_that(df[df[DistanceMeasureCols.compared] == (3, 4)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "higher", "higher"))
    assert_that(df[df[DistanceMeasureCols.compared] == (4, 5)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "lower", "lower"))
    assert_that(df[df[DistanceMeasureCols.compared] == (1, 3)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "lower", "lower"))
    assert_that(df[df[DistanceMeasureCols.compared] == (1, 4)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "higher", "overlap"))
    assert_that(df[df[DistanceMeasureCols.compared] == (1, 5)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "lower", "lower"))
    assert_that(df[df[DistanceMeasureCols.compared] == (2, 4)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "lower", "lower"))
    assert_that(df[df[DistanceMeasureCols.compared] == (2, 5)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "lower", "lower"))
    assert_that(df[df[DistanceMeasureCols.compared] == (3, 5)][DistanceMeasureCols.stat_diff].tolist(),
                contains_exactly("lower", "lower", "lower"))


def test_calculate_distances_within_groups():
    df = da.distances_statistics_for_each_pattern()
    assert_that(df.shape[0], is_(23 * 3))  # 23 patterns for each distance measure

    # Euclidean
    euc_s = df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l2_cor_dist]
    assert_that(euc_s.shape[0], is_(23))  # 23 patterns
    p0_eu = euc_s[euc_s[SyntheticDataSegmentCols.pattern_id] == 0]
    p23_eu = euc_s[euc_s[SyntheticDataSegmentCols.pattern_id] == 23]
    assert_that(p0_eu[Aggregators.count].values[0], is_(10))
    assert_that(round(p0_eu[Aggregators.mean].values[0], 2), is_(0.03))
    assert_that(round(p0_eu[Aggregators.min].values[0], 2), is_(0.01))
    assert_that(round(p0_eu[Aggregators.max].values[0], 2), is_(0.06))
    assert_that(p23_eu[Aggregators.count].values[0], is_(6))
    assert_that(round(p23_eu[Aggregators.mean].values[0], 2), is_(0.0))
    assert_that(round(p23_eu[Aggregators.min].values[0], 2), is_(0.0))
    assert_that(round(p23_eu[Aggregators.max].values[0], 2), is_(0.0))


def test_ci_of_differences_between_patterns():
    stats = da.distances_statistics_for_each_pattern()
    df, alpha = da.ci_of_differences_between_patterns(stats)
    combinations = number_of_unique_two_combinations(23)
    assert_that(alpha, is_(0.05 / combinations))
    assert_that(df.shape[0], is_(combinations * 3))  # for three distance measures


def test_plot_ci_of_differences_between_groups():
    fig = da.plot_ci_of_differences_between_groups_for_measures(
        measures=[DistanceMeasureCols.l2_cor_dist, DistanceMeasureCols.log_frob_cor_dist])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci-distance-measures-between-groups.png'))

    fig2 = lp_da.plot_ci_for_ordered_groups_for_measures(
        measures=[DistanceMeasureCols.l1_cor_dist, DistanceMeasureCols.linf_cor_dist])
    assert_that(fig2, is_not(None))
    fig.savefig(path.join(images_dir, 'ci-distance-measures-between-groups_l1_linf.png'))


def test_plot_correlation_matrix_for_given_pattern_pairs():
    # group 2/3 - surprising
    fig = da.plot_correlation_matrices_of_biggest_distances_for_groups(2, 3, plot_diagonal=False, what="biggest")
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'correlation_for_group_2_3_biggest_distances-misty-forest-56.png'))

    fig = da.plot_correlation_matrices_of_biggest_distances_for_groups(2, 3, plot_diagonal=False, what="smallest")
    fig.savefig(path.join(images_dir, 'correlation_for_group_2_3_smallest_distances-misty-forest-56.png'))
    assert_that(fig, is_not(None))

    # group 2/5 - surprising
    fig = da.plot_correlation_matrices_of_biggest_distances_for_groups(2, 5, plot_diagonal=False, what="biggest")
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'correlation_for_group_2_5_biggest_distances-misty-forest-56.png'))

    fig = da.plot_correlation_matrices_of_biggest_distances_for_groups(2, 5, plot_diagonal=False, what="smallest")
    fig.savefig(path.join(images_dir, 'correlation_for_group_2_5_smallest_distances-misty-forest-56.png'))
    assert_that(fig, is_not(None))

    # group 2/4  - expected
    fig = da.plot_correlation_matrices_of_biggest_distances_for_groups(2, 4, plot_diagonal=False, what="biggest")
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'correlation_for_group_2_4_biggest_distances-misty-forest-56.png'))

    fig = da.plot_correlation_matrices_of_biggest_distances_for_groups(2, 4, plot_diagonal=False, what="smallest")
    fig.savefig(path.join(images_dir, 'correlation_for_group_2_4_smallest_distances-misty-forest-56.png'))
    assert_that(fig, is_not(None))

    # group 4/5 - surprising
    fig = da.plot_correlation_matrices_of_biggest_distances_for_groups(4, 5, plot_diagonal=False, what="biggest")
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'correlation_for_group_4_5_biggest_distances-misty-forest-56.png'))

    fig = da.plot_correlation_matrices_of_biggest_distances_for_groups(4, 5, plot_diagonal=False, what="smallest")
    fig.savefig(path.join(images_dir, 'correlation_for_group_4_5_smallest_distances-misty-forest-56.png'))
    assert_that(fig, is_not(None))


def test_plot_box_diagrams_of_distances_for_all_groups():
    fig = da.plot_box_diagrams_of_distances_for_all_groups()
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-distances-all-groups-all-measures-misty-forest-56.png'))

    # differences between patterns
    df = da.calculate_df_of_pattern_pair_groups()
    df.to_csv(path.join(tables_dir, 'sig_different_pattern_pairs_for_log_frob_l2.csv'))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l2_cor_dist].shape[0], is_(25))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.log_frob_cor_dist].shape[0], is_(47))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.foerstner_cor_dist].shape[0], is_(31))

    fig = da.plot_box_diagrams_of_distances_for_all_groups(measures=[DistanceMeasureCols.l2_cor_dist],
                                                           order=default_order)
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-distances-all-groups-l2-corr-misty-forest-56.png'))


def test_box_diagrams_for_lp_measures():
    fig = lp_da.plot_box_diagrams_of_distances_for_all_groups(order=default_order)
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-distances-all-groups-lp-measures-misty-forest-56.png'))

    fig = lp_da.plot_box_diagrams_of_distances_for_all_groups(measures=[DistanceMeasureCols.l1_cor_dist],
                                                              order=default_order)
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-distances-all-groups-l1-corr-misty-forest-56.png'))

    fig = lp_da.plot_box_diagrams_of_distances_for_all_groups(measures=[DistanceMeasureCols.l10_cor_dist],
                                                              order=default_order)
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-distances-all-groups-l10-corr-misty-forest-56.png'))

    fig = lp_da.plot_box_diagrams_of_distances_for_all_groups(measures=[DistanceMeasureCols.linf_cor_dist],
                                                              order=default_order)
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-distances-all-groups-linf-corr-misty-forest-56.png'))


def test_plot_confidence_intervals_of_mean_difference():
    fig = da.plot_ci_of_differences_between_groups_for_measures(
        measures=[DistanceMeasureCols.l2_cor_dist, DistanceMeasureCols.log_frob_cor_dist])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci-distance-measures-between-groups.png'))

    fig = da.plot_ci_for_ordered_groups_for_measures(
        measures=[DistanceMeasureCols.l2_cor_dist, DistanceMeasureCols.log_frob_cor_dist])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci_and_distances_for_ordered_groups_distance_measure.png'))

    fig = lp_da.plot_ci_of_differences_between_groups_for_measures(
        measures=[DistanceMeasureCols.l1_cor_dist, DistanceMeasureCols.linf_cor_dist])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci-distance-measures-between-groups_l1_linf.png'))

    fig = lp_da.plot_ci_for_ordered_groups_for_measures(
        measures=[DistanceMeasureCols.l1_cor_dist, DistanceMeasureCols.linf_cor_dist])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci_for_ordered_groups_l1_linf.png'))


def test_find_min_or_max_distances_for_each_group_for_a_measure():
    min_distances = da.find_min_or_max_distances_for_each_group_for_a_measure("min", DistanceMeasureCols.l2_cor_dist)
    max_distances = da.find_min_or_max_distances_for_each_group_for_a_measure("max", DistanceMeasureCols.l2_cor_dist)

    assert_that(min_distances.shape[0], is_(len(da.groups)))
    assert_that(min_distances.iloc[0][DistanceMeasureCols.pattern_pairs], is_((1, 1)))
    assert_that(max_distances.shape[0], is_(len(da.groups)))
    assert_that(max_distances.iloc[0][DistanceMeasureCols.pattern_pairs], is_((4, 4)))


def test_statistics_of_differences_between_pattern_pairs():
    df = lp_da.calculate_statistics_for_pattern_pairs_for_measure()
    assert_that(df.shape[0], is_(len(lp_measures) * 276))
    assert_that(len(df[DistanceMeasureCols.group].unique()), is_(6))


def test_dict_of_pattern_pairs_with_overlapping_ci():
    result = lp_da.calculate_df_of_pattern_pair_groups()
    l2 = result[result[DistanceMeasureCols.type] == DistanceMeasureCols.l2_cor_dist].reset_index(drop=True)
    assert_that(l2.iloc[0][DistanceMeasureCols.group], contains_exactly(0))  # first group all group 0
    assert_that(l2.iloc[0][DistanceMeasureCols.pattern_pairs],
                contains_exactly((25, 25), (23, 23), (13, 13), (17, 17)))  # smallest distances
    assert_that(l2.iloc[2][DistanceMeasureCols.group], contains_exactly(1))  # third group all group 1
    assert_that(l2.iloc[2][DistanceMeasureCols.pattern_pairs],
                contains_exactly((1, 7), (9, 15), (18, 19), (3, 21), (3, 5), (9, 11), (1, 4), (1, 10), (2, 11), (6, 24),
                                 (18, 21), (6, 15), (2, 8), (2, 20), (18, 24), (9, 12), (18, 20), (6, 8), (6, 7),
                                 (3, 12), (9, 10), (2, 5), (3, 4), (1, 19)))  # first group all group 0
    assert_that(l2.iloc[4][DistanceMeasureCols.group], contains_exactly(1, 2))  # third group all group 1

    # check the number of pairs is correct
    group0_pairs = len(ds.patterns_by_group[0])
    group1_pairs = len(ds.patterns_by_group[1])
    group2_pairs = len(ds.patterns_by_group[2])
    group3_pairs = len(ds.patterns_by_group[3])
    group4_pairs = len(ds.patterns_by_group[4])
    group5_pairs = len(ds.patterns_by_group[5])
    total_pairs = sum([group0_pairs, group1_pairs, group2_pairs, group3_pairs, group4_pairs, group5_pairs])

    for measure in result[DistanceMeasureCols.type].unique():
        pp = list(result[result[DistanceMeasureCols.type] == measure][DistanceMeasureCols.pattern_pairs])
        # collapse lists of lists and make unique
        pp = list(set(itertools.chain.from_iterable(pp)))
        assert_that(len(pp), is_(total_pairs))


def test_calculate_ci_mean_differences_between_pattern_pairs():
    df = lp_da.calculate_ci_mean_differences_between_pattern_pairs_for_each_group()
    assert_that(len(df[DistanceMeasureCols.group].unique()), is_(6))
    assert_that(len(df[DistanceMeasureCols.type].unique()), is_(4))
    assert_that(len(df[DistanceMeasureCols.stat_diff].unique()), is_(3))  # all three outcomes


def test_find_pattern_pairs_for_within_group_differences_where_the_distances_dont_agree():
    dict_of_lists = lp_da.find_within_group_differences_where_the_distances_dont_agree()
    assert_that(len(dict_of_lists.keys()), is_(6))
    assert_that(len(dict_of_lists[0]), is_(11))
    assert_that(len(dict_of_lists[1]), is_(228))
    assert_that(len(dict_of_lists[2]), is_(548))
    assert_that(len(dict_of_lists[3]), is_(1031))
    assert_that(len(dict_of_lists[4]), is_(266))
    assert_that(len(dict_of_lists[5]), is_(19))


def test_plot_ci_of_mean_differences_between_pattern_pairs():
    fig = lp_da.plot_ci_per_group_for_pattern_pairs_where_distances_do_not_agree(
        measures=[DistanceMeasureCols.l1_cor_dist, DistanceMeasureCols.l2_cor_dist])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci_all_groups_pattern_pairs_l1_l2.png'))


def test_plot_pattern_pairs_ci_difference_heat_map():
    for group in lp_da.groups:
        fig = lp_da.plot_ci_pattern_pair_heat_map(group=group, measure=DistanceMeasureCols.l1_cor_dist)
        assert_that(fig, is_not(None))
        fig.savefig(path.join(images_dir, 'ci_pattern_pair_diff_heat_map_group_' + str(group) + '_distance_l1.png'))

    for group in lp_da.groups:
        fig = lp_da.plot_ci_pattern_pair_heat_map(group=group, measure=DistanceMeasureCols.l2_cor_dist)
        assert_that(fig, is_not(None))
        fig.savefig(path.join(images_dir, 'ci_pattern_pair_diff_heat_map_group_' + str(group) + '_distance_l2.png'))


def test_calculate_experimental_distances():
    angle_m = [DistanceMeasureCols.l1_with_ref, DistanceMeasureCols.l2_with_ref, DistanceMeasureCols.dot_transform_l1,
               DistanceMeasureCols.dot_transform_l2,
               DistanceMeasureCols.dot_transform_linf, DistanceMeasureCols.cosine]
    a_da = DistanceMetricAssessment(ds, measures=angle_m, backend=backend)

    # box diagram
    fig = a_da.plot_box_diagrams_of_distances_for_all_groups(order=default_order)
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-distances-experimental-measures-misty-forest-56.png'))

    # distances for each row are significantly different to other rows, the higher the better from this
    # perspective, still important that same pattern is 0
    df = a_da.calculate_df_of_pattern_pair_groups()
    # L1 with transformation
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.dot_transform_l1].shape[0], is_(42))
    # L2 better than Linf
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.dot_transform_l2].shape[0], is_(47))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.dot_transform_linf].shape[0], is_(15))
    # Cosine is worse than L2 dot transform
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.cosine].shape[0], is_(19))

    # L1 with ref is also good!
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l1_with_ref].shape[0], is_(50))

    # L2 with ref is best!
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l2_with_ref].shape[0], is_(64))


def test_compare_ref_distances():
    m = [DistanceMeasureCols.l1_with_ref, DistanceMeasureCols.l1_cor_dist, DistanceMeasureCols.l2_with_ref,
         DistanceMeasureCols.l2_cor_dist]
    a_da = DistanceMetricAssessment(ds, measures=m, backend=backend)

    # box diagram
    fig = a_da.plot_box_diagrams_of_distances_for_all_groups(order=default_order)
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-distances-l-measures-with-ref-misty-forest-56.png'))


def test_box_plots_for_various_p():
    p_m = [DistanceMeasureCols.l1_cor_dist, DistanceMeasureCols.l2_cor_dist, DistanceMeasureCols.l5_cor_dist,
           DistanceMeasureCols.l10_cor_dist, DistanceMeasureCols.l50_cor_dist, DistanceMeasureCols.l100_cor_dist,
           DistanceMeasureCols.linf_cor_dist]
    a_da = DistanceMetricAssessment(ds, measures=p_m, backend=backend)

    fig = a_da.plot_box_diagrams_of_distances_as_function_of_p(groups=['all'])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-for-lp-all-groups-misty-forest-56.png'))

    fig = a_da.plot_box_diagrams_of_distances_as_function_of_p(groups=a_da.groups)
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-for-lp-per-group-misty-forest-56.png'))

    df = a_da.calculate_df_of_pattern_pair_groups()
    df.to_csv(path.join(tables_dir, 'sig_different_pattern_pairs_for_lp.csv'))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l1_cor_dist].shape[0], is_(23))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l2_cor_dist].shape[0], is_(25))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l5_cor_dist].shape[0], is_(24))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l10_cor_dist].shape[0], is_(23))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l50_cor_dist].shape[0], is_(17))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l100_cor_dist].shape[0], is_(15))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.linf_cor_dist].shape[0], is_(15))


def test_box_plots_for_various_p_with_reference_vector():
    p_m = [DistanceMeasureCols.l1_with_ref, DistanceMeasureCols.l2_with_ref, DistanceMeasureCols.l5_with_ref,
           DistanceMeasureCols.l10_with_ref, DistanceMeasureCols.l50_with_ref, DistanceMeasureCols.l100_with_ref,
           DistanceMeasureCols.linf_with_ref]
    a_da = DistanceMetricAssessment(ds, measures=p_m, backend=backend)

    fig = a_da.plot_box_diagrams_of_distances_as_function_of_p(groups=['all'])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-for-lp-with-ref-all-groups-misty-forest-56.png'))

    fig = a_da.plot_box_diagrams_of_distances_as_function_of_p(groups=a_da.groups)
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-for-lp-with-ref-per-group-misty-forest-56.png'))

    # CI diagrams with groups as column and row for two measures
    fig = a_da.plot_ci_of_differences_between_groups_for_measures(
        measures=[DistanceMeasureCols.l1_with_ref, DistanceMeasureCols.linf_with_ref])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci-distance-measures-between-groups_l1_vs_linf_with_ref.png'))

    fig = a_da.plot_ci_of_differences_between_groups_for_measures(
        measures=[DistanceMeasureCols.l1_with_ref, DistanceMeasureCols.l2_with_ref])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci-distance-measures-between-groups_l1_vs_l2_with_ref.png'))

    # CI diagrams with columns being the two distance measures compared, no rows
    fig = a_da.plot_ci_for_ordered_groups_for_measures(
        measures=[DistanceMeasureCols.l1_with_ref, DistanceMeasureCols.linf_with_ref])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci_and_distances_for_ordered_groups_l1_and_linf_with_ref.png'))

    fig = a_da.plot_ci_for_ordered_groups_for_measures(
        measures=[DistanceMeasureCols.l1_with_ref, DistanceMeasureCols.linf_with_ref])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci_and_distances_for_ordered_groups_l1_and_linf_with_ref.png'))

    # differences between patterns
    df = a_da.calculate_df_of_pattern_pair_groups()
    df.to_csv(path.join(tables_dir, 'sig_different_pattern_pairs_for_lnorms_with_ref.csv'))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l1_with_ref].shape[0], is_(50))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l2_with_ref].shape[0], is_(64))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l5_with_ref].shape[0], is_(61))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l10_with_ref].shape[0], is_(61))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l50_with_ref].shape[0], is_(43))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.l100_with_ref].shape[0], is_(43))
    assert_that(df[df[DistanceMeasureCols.type] == DistanceMeasureCols.linf_with_ref].shape[0], is_(32))


def test_ci_plot_for_various_numbers_of_measures():
    fig = lp_da.plot_ci_of_differences_between_groups_for_measures(measures=[DistanceMeasureCols.l1_cor_dist])
    assert_that(fig, is_not(None))

    fig = lp_da.plot_ci_of_differences_between_groups_for_measures(
        measures=[DistanceMeasureCols.l1_cor_dist, DistanceMeasureCols.l2_cor_dist])
    assert_that(fig, is_not(None))

    fig = lp_da.plot_ci_of_differences_between_groups_for_measures(
        measures=[DistanceMeasureCols.l1_cor_dist, DistanceMeasureCols.l2_cor_dist, DistanceMeasureCols.l10_cor_dist])
    assert_that(fig, is_not(None))

    fig = lp_da.plot_ci_of_differences_between_groups_for_measures(
        measures=[DistanceMeasureCols.l1_cor_dist, DistanceMeasureCols.l2_cor_dist, DistanceMeasureCols.l10_cor_dist,
                  DistanceMeasureCols.linf_cor_dist])
    assert_that(fig, is_not(None))


def test_can_calculate_discriminative_power_criteria():
    result_df = lp_da.discriminative_power_criteria()

    assert_that(result_df.shape[0], is_(len(lp_measures)))

    m = lp_measures[0]
    assert_that(result_df.loc[m, DistanceMeasureCols.avg_rate], is_(1.93))
    assert_that(result_df.loc[m, DistanceMeasureCols.monotonic], is_(True))
    assert_that(result_df.loc[m, DistanceMeasureCols.cv], is_(0.4))
    assert_that(result_df.loc[m, DistanceMeasureCols.rc], is_(3.41))


def test_box_plots_and_ci_for_mixed_p_distances():
    p_m = [DistanceMeasureCols.l1_cor_dist, DistanceMeasureCols.l1_with_ref, DistanceMeasureCols.l2_cor_dist,
           DistanceMeasureCols.l2_with_ref, DistanceMeasureCols.linf_cor_dist, DistanceMeasureCols.linf_with_ref]
    a_da = DistanceMetricAssessment(ds, measures=p_m, backend=backend)

    fig = a_da.plot_box_diagrams_of_distances_as_function_of_p(groups=['all'])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-for-lp-mixed-all-groups-misty-forest-56.png'))

    fig = a_da.plot_box_diagrams_of_distances_as_function_of_p(groups=a_da.groups)
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-for-lp-mixed-per-group-misty-forest-56.png'))

    # single box diagrams
    fig = a_da.plot_box_diagrams_of_distances_for_all_groups(measures=[DistanceMeasureCols.l1_with_ref],
                                                             order=default_order)
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-distances-all-groups-l1-with-ref-misty-forest-56.png'))

    fig = a_da.plot_box_diagrams_of_distances_for_all_groups(measures=[DistanceMeasureCols.l2_with_ref],
                                                             order=default_order)
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'box-diagram-distances-all-groups-l2-with-ref-misty-forest-56.png'))

    # CI diagrams with groups as column and row for two measures
    fig = a_da.plot_ci_of_differences_between_groups_for_measures(
        measures=[DistanceMeasureCols.l1_cor_dist, DistanceMeasureCols.l1_with_ref])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci-distance-measures-between-groups_l1_vs_l1_with_ref.png'))

    fig = a_da.plot_ci_of_differences_between_groups_for_measures(
        measures=[DistanceMeasureCols.l2_cor_dist, DistanceMeasureCols.l2_with_ref])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci-distance-measures-between-groups_l2_vs_l2_with_ref.png'))

    fig = a_da.plot_ci_of_differences_between_groups_for_measures(
        measures=[DistanceMeasureCols.linf_cor_dist, DistanceMeasureCols.linf_with_ref])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci-distance-measures-between-groups_linf_vs_linf_with_ref.png'))

    fig = a_da.plot_ci_of_differences_between_groups_for_measures(
        measures=[DistanceMeasureCols.l1_with_ref, DistanceMeasureCols.l2_with_ref])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci-distance-measures-between-groups_l1_with_ref_vs_l2_with_ref.png'))

    # CI diagrams with columns being the two distance measures compared, no rows
    fig = a_da.plot_ci_for_ordered_groups_for_measures(
        measures=[DistanceMeasureCols.l1_cor_dist, DistanceMeasureCols.l1_with_ref])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci_and_distances_for_ordered_groups_l1_vs_l1_with_ref.png'))

    fig = a_da.plot_ci_for_ordered_groups_for_measures(
        measures=[DistanceMeasureCols.l2_cor_dist, DistanceMeasureCols.l2_with_ref])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci_and_distances_for_ordered_groups_l2_vs_l2_with_ref.png'))

    fig = a_da.plot_ci_for_ordered_groups_for_measures(
        measures=[DistanceMeasureCols.linf_cor_dist, DistanceMeasureCols.linf_with_ref])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci_and_distances_for_ordered_groups_linf_vs_linf_with_ref.png'))

    fig = a_da.plot_ci_for_ordered_groups_for_measures(
        measures=[DistanceMeasureCols.l1_with_ref, DistanceMeasureCols.l2_with_ref])
    assert_that(fig, is_not(None))
    fig.savefig(path.join(images_dir, 'ci_and_distances_for_ordered_groups_l1_with_ref_vs_l2_with_ref.png'))
