from dataclasses import dataclass

import numpy as np

from hamcrest import *

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.use_case.ticc_result import TICCResult, check_if_two_matrices_are_the_same, access_lower_block_diagonals
from src.utils.plots.matplotlib_helper_functions import Backends
from tests.use_case.ticc.test_ticc_runs_on_original_test_data import TICCSettings, get_ticc_example_result

backend = Backends.none.value


@dataclass
class StandardResultTest:
    test_settings = TICCSettings()
    test_settings.backend = backend
    __result = None  # lazy loaded

    def result(self) -> TICCResult:
        if not self.__result:
            self.__result = get_ticc_example_result(ticc_settings=self.test_settings)
        return self.__result


@dataclass
class WindowSizeOneResultTest:
    __result = None  # lazy loaded

    def result(self):
        if not self.__result:
            settings = TICCSettings()
            settings.backend = backend
            settings.window_size = 1
            self.__result = get_ticc_example_result(ticc_settings=settings)
        return self.__result


def test_return_matrices_on_and_off_diagonal():
    result = StandardResultTest().result()
    cluster = result.clusters()[0]
    diagonal = result.adjacency_matrices_for_cluster(cluster, off_diagonal=0)
    diagonal1 = result.adjacency_matrices_for_cluster(cluster, off_diagonal=1)
    diagonal2 = result.adjacency_matrices_for_cluster(cluster, off_diagonal=2)
    unique_mfr = result.adjacency_matrices_for_each_time_relationship(cluster)

    # check length and shape of block matrices
    assert_that(len(unique_mfr), is_(result.window_size))
    [assert_that(m.shape, is_((result.number_of_time_series, result.number_of_time_series))) for m in unique_mfr]

    assert_that(len(diagonal), is_(result.window_size))
    [assert_that(m.shape, is_((result.number_of_time_series, result.number_of_time_series))) for m in diagonal]

    assert_that(len(diagonal1), is_(result.window_size - 1))
    [assert_that(m.shape, is_((result.number_of_time_series, result.number_of_time_series))) for m in diagonal1]

    assert_that(len(diagonal2), is_(result.window_size - 2))
    [assert_that(m.shape, is_((result.number_of_time_series, result.number_of_time_series))) for m in diagonal2]

    # check that the first matrix in each of the diagonals fits with the unique one for that relationship
    assert_that(check_if_two_matrices_are_the_same(diagonal[0], unique_mfr[0]))
    assert_that(check_if_two_matrices_are_the_same(diagonal1[0], unique_mfr[1]))
    assert_that(check_if_two_matrices_are_the_same(diagonal2[0], unique_mfr[2]))


def test_verifies_all_the_paper_assumptions():
    result = StandardResultTest().result()
    check = result.verify_all_paper_assumptions_for_mfr()
    assert_that(check, is_(False))  # sadly this is not the case for the test data


def test_plots_mrf_for_given_cluster_as_heatmap():
    result = StandardResultTest().result()
    cluster = result.clusters()[0]

    result.plot_mrf_for_cluster_as_heatmap(cluster)


def test_mean_max_min_segment_length():
    result = StandardResultTest().result()
    assert_that(result.mean_segment_length(round_to=2), is_(529.92))
    assert_that(result.max_segment_length(), is_(4056))
    assert_that(result.min_segment_length(), is_(13))


def test_returns_results_as_labels_df():
    result = StandardResultTest().result()
    df = result.to_labels_df(subject_id="ticc example")

    assert_that(result.number_of_observations, is_(19607))
    assert_that(df.shape[0], is_(result.number_of_segments()))
    assert_that(df[SyntheticDataSegmentCols.start_idx].iloc[0], is_(0))
    assert_that(df[SyntheticDataSegmentCols.end_idx].iloc[-1], is_(result.number_of_observations - 1))
    assert_that(len(df[SyntheticDataSegmentCols.pattern_id].unique()), is_(result.number_of_clusters))
    assert_that(df[SyntheticDataSegmentCols.length].sum(), is_(result.number_of_observations))
    assert_that(df[SyntheticDataSegmentCols.length].min(), is_(result.min_segment_length()))
    assert_that(df[SyntheticDataSegmentCols.length].max(), is_(result.max_segment_length()))
    assert_that(round(df[SyntheticDataSegmentCols.length].mean(), 3), is_(result.mean_segment_length()))


def test_returns_mrf_as_temporal_network_array():
    result = StandardResultTest().result()

    cluster = 0
    mrfs = result.adjacency_matrices_for_cluster(cluster, off_diagonal=0)

    temporal_array = result.get_mrf_for_cluster_as_3d_array(cluster)

    for t in range(len(mrfs)):
        assert_that(np.array_equal(temporal_array[:, :, t], mrfs[t]))


def test_calculates_betweenness_centrality_for_mfr_and_plots_split_graph_over_clusters_for_window_size_3():
    result = StandardResultTest().result()
    betweenness = result.betweenness_centrality_for_all_cluster()

    assert_that(betweenness.shape, is_((result.number_of_none_zero_clusters, result.number_of_time_series)))
    result.plot_network_slice_plot_over_all_clusters()


def test_can_calculate_betweenness_centrality_for_real_data_and_window_size_1():
    result1 = WindowSizeOneResultTest().result()
    betweenness_per_cluster = result1.betweenness_centrality_for_all_cluster()
    assert_that(betweenness_per_cluster.shape,
                is_((result1.number_of_none_zero_clusters, result1.number_of_time_series)))


def test_can_calculate_degree_and_closeness_centrality_for_clusters():
    result1 = WindowSizeOneResultTest().result()
    closeness_centrality_per_cluster = result1.closeness_centrality_for_all_clusters()
    degree_centrality_per_cluster = result1.degree_centrality_for_all_clusters()
    assert_that(closeness_centrality_per_cluster.shape,
                is_((result1.number_of_none_zero_clusters, result1.number_of_time_series)))
    assert_that(degree_centrality_per_cluster.shape,
                is_((result1.number_of_none_zero_clusters, result1.number_of_time_series)))


def test_can_calculate_degree_betweenness_and_closeness_centrality_when_only_one_cluster_is_not_zero():
    no_ts = 3
    window_size = 1
    clusters = 4
    made_up_cov = np.ones((3, 3))
    mrf = {"0": made_up_cov}

    res = TICCResult(data=np.array([]), cluster_assignment=[], dictionary_of_mrf_for_clusters=mrf, empirical_covariances=made_up_cov,
                     number_of_clusters=clusters, number_of_time_series=no_ts, window_size=window_size,
                     has_converged=False, backend=backend)
    closeness_centrality_per_cluster = res.closeness_centrality_for_all_clusters()
    degree_centrality_per_cluster = res.degree_centrality_for_all_clusters()
    betweenness_centrality_per_cluster = res.betweenness_centrality_for_all_cluster()
    assert_that(betweenness_centrality_per_cluster.shape,
                is_((1, res.number_of_time_series)))
    assert_that(closeness_centrality_per_cluster.shape,
                is_((1, res.number_of_time_series)))
    assert_that(degree_centrality_per_cluster.shape,
                is_((1, res.number_of_time_series)))


def test_accessing_diagonal_matrices():
    window_size = 5
    number_of_ts = 4

    diagonal = np.zeros((number_of_ts, number_of_ts))
    first = np.ones((number_of_ts, number_of_ts))
    second = np.ones((number_of_ts, number_of_ts)) * 2
    third = np.ones((number_of_ts, number_of_ts)) * 3
    fourth = np.ones((number_of_ts, number_of_ts)) * 4

    temp1 = np.vstack((diagonal, first))
    temp1 = np.vstack((temp1, second))
    temp1 = np.vstack((temp1, third))
    temp1 = np.vstack((temp1, fourth))

    temp2 = np.vstack((first, diagonal))
    temp2 = np.vstack((temp2, first))
    temp2 = np.vstack((temp2, second))
    temp2 = np.vstack((temp2, third))

    temp3 = np.vstack((second, first))
    temp3 = np.vstack((temp3, diagonal))
    temp3 = np.vstack((temp3, first))
    temp3 = np.vstack((temp3, second))

    temp4 = np.vstack((third, second))
    temp4 = np.vstack((temp4, first))
    temp4 = np.vstack((temp4, diagonal))
    temp4 = np.vstack((temp4, first))

    temp5 = np.vstack((fourth, third))
    temp5 = np.vstack((temp5, second))
    temp5 = np.vstack((temp5, first))
    temp5 = np.vstack((temp5, diagonal))

    test_array = np.hstack((temp1, temp2))
    test_array = np.hstack((test_array, temp3))
    test_array = np.hstack((test_array, temp4))
    test_array = np.hstack((test_array, temp5))

    resulting_diag = access_lower_block_diagonals(test_array, window_size, sub_diagonal=0)
    assert_that(len(resulting_diag), is_(window_size))
    for ar in resulting_diag:
        unique_values = list(np.unique(ar))
        assert_that(len(unique_values), is_(1))  # all numbers equal
        assert_that(unique_values[0], is_(0))  # all numbers 0

    first_diag = access_lower_block_diagonals(test_array, window_size, sub_diagonal=1)
    assert_that(len(first_diag), is_(window_size - 1))
    for ar in first_diag:
        unique_values = list(np.unique(ar))
        assert_that(len(unique_values), is_(1))  # all numbers equal
        assert_that(unique_values[0], is_(1))  # all numbers 1

    second_diag = access_lower_block_diagonals(test_array, window_size, sub_diagonal=2)
    assert_that(len(second_diag), is_(window_size - 2))
    for ar in second_diag:
        unique_values = list(np.unique(ar))
        assert_that(len(unique_values), is_(1))  # all numbers equal
        assert_that(unique_values[0], is_(2))  # all numbers 2

    third_diag = access_lower_block_diagonals(test_array, window_size, sub_diagonal=3)
    assert_that(len(third_diag), is_(window_size - 3))
    for ar in third_diag:
        unique_values = list(np.unique(ar))
        assert_that(len(unique_values), is_(1))  # all numbers equal
        assert_that(unique_values[0], is_(3))  # all numbers 3

    fourth_diag = access_lower_block_diagonals(test_array, window_size, sub_diagonal=4)
    assert_that(len(fourth_diag), is_(window_size - 4))
    for ar in fourth_diag:
        unique_values = list(np.unique(ar))
        assert_that(len(unique_values), is_(1))  # all numbers equal
        assert_that(unique_values[0], is_(4))  # all numbers 4
