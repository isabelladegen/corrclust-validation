from dataclasses import dataclass
from os import path

import numpy as np
from hamcrest import *

from src.use_case.ticc.TICC_solver import TICC
from src.use_case.ticc_result import TICCResult
from src.utils.configurations import ROOT_DIR
from src.utils.plots.matplotlib_helper_functions import Backends


def load_ticc_example_data_as_x_train():
    ticc_example_data_file = path.join(ROOT_DIR, 'tests/use_case/ticc/test-data/example_data.txt')
    return np.loadtxt(ticc_example_data_file, delimiter=",")


@dataclass
class TICCSettings:  # attention changing these defaults will make the comparison test fail
    window_size = 3
    number_of_clusters = 8
    switch_penalty = 600
    lambda_var = 11e-2
    max_iter = 100
    threshold = 2e-5
    allow_zero_cluster_inbetween = True
    use_gmm_initialisation = True
    reassign_points_to_zero_clusters = True
    biased = False
    do_training_split = False
    keep_track_of_assignments = False
    cluster_reassignment = 20
    backend = Backends.none.value


def get_ticc_example_result(x_train: np.ndarray = load_ticc_example_data_as_x_train(),
                            ticc_settings: TICCSettings = TICCSettings()) -> TICCResult:
    """
    Creates a TICC result from syntetic test data for testing
    :param x_train: optional numpy array with rows being observations and columns being the different time series,
    if not provided than the default example data is used
    :param ticc_settings: optional TICCSettings class for the various parameters,
    if not provided default settings are used
    :return: TICCResult from training
    """
    ticc = TICC(window_size=ticc_settings.window_size, number_of_clusters=ticc_settings.number_of_clusters,
                lambda_parameter=ticc_settings.lambda_var, beta=ticc_settings.switch_penalty,
                max_iters=ticc_settings.max_iter,
                threshold=ticc_settings.threshold,
                allow_zero_cluster_inbetween=ticc_settings.allow_zero_cluster_inbetween,
                do_training_split=ticc_settings.do_training_split,
                keep_track_of_assignments=ticc_settings.keep_track_of_assignments,
                backend=ticc_settings.backend)
    return ticc.fit(data=x_train,
                    use_gmm_initialisation=ticc_settings.use_gmm_initialisation,
                    reassign_points_to_zero_clusters=ticc_settings.reassign_points_to_zero_clusters)


settings = TICCSettings()
window_size = settings.window_size
number_of_clusters = settings.number_of_clusters
beta = settings.switch_penalty
lambda_var = settings.lambda_var
max_iter = settings.max_iter
threshold = settings.threshold
x_train = load_ticc_example_data_as_x_train()
numer_of_ts_in_example_data = 10
do_training_split = True


def test_ticc_using_original_configuration_returns_same_results_on_test_data():
    allow_zero_cluster_inbetween = True
    reassign_points_to_zero_clusters = True
    use_gmm_initialisation = True

    ticc = TICC(window_size=window_size, number_of_clusters=number_of_clusters, lambda_parameter=lambda_var, beta=beta,
                max_iters=max_iter, threshold=threshold, allow_zero_cluster_inbetween=allow_zero_cluster_inbetween,
                do_training_split=do_training_split)
    training_result = ticc.fit(data=x_train,
                               use_gmm_initialisation=use_gmm_initialisation,
                               reassign_points_to_zero_clusters=reassign_points_to_zero_clusters)

    assert_that(training_result.has_converged, is_(True))
    assert_that(training_result.number_of_none_zero_clusters, is_(8))
    assert_that(training_result.number_of_segments(), is_(31))
    assert_that(training_result.number_of_times_each_cluster_is_used(), is_([2, 5, 9, 1, 5, 1, 1, 7]))
    observations_per_segment = training_result.number_of_observations_per_segment()
    assert_that(observations_per_segment[0], is_((2, 40)))
    assert_that(observations_per_segment[7], is_((7, 35)))
    assert_that(observations_per_segment[-1], is_((5, 8065)))
    assert_that(round(training_result.bic(), 3), is_(58024.795))

    # check mrf results
    assert_that(len(training_result.mrf.keys()), is_(number_of_clusters))
    assert_that(training_result.mrf[0].shape,
                is_((window_size * numer_of_ts_in_example_data, window_size * numer_of_ts_in_example_data)))

    are_diags_the_same = training_result.check_diagonal_matrices_are_the_same()
    assert_that(all(are_diags_the_same[0]), is_(True))

    adjacency_are_undirected_graphs = training_result.check_adjacency_matrices_are_undirected_graphs()
    assert_that(all(adjacency_are_undirected_graphs[2]), is_(True))  # again sadly this is not the case for all clusters


def test_ticc_using_modified_configuration_that_finds_no_cluster_returns_same_results_as_in_original_run_on_test_data():
    allow_zero_cluster_inbetween = False
    reassign_points_to_zero_clusters = False
    use_gmm_initialisation = False

    ticc = TICC(window_size=window_size, number_of_clusters=number_of_clusters, lambda_parameter=lambda_var, beta=beta,
                max_iters=max_iter, threshold=threshold, allow_zero_cluster_inbetween=allow_zero_cluster_inbetween,
                do_training_split=False)
    training_result = ticc.fit(data=x_train,
                               use_gmm_initialisation=use_gmm_initialisation,
                               reassign_points_to_zero_clusters=reassign_points_to_zero_clusters)

    assert_that(training_result.has_converged, is_(True))
    assert_that(training_result.number_of_none_zero_clusters, is_(6))
    assert_that(training_result.number_of_segments(), is_(13))
    assert_that(training_result.number_of_times_each_cluster_is_used(), is_([6, 1, 1, 1, 1, 3, 0, 0]))

    observations_per_segment = training_result.number_of_observations_per_segment()
    assert_that(observations_per_segment[0], is_((0, 40)))
    assert_that(observations_per_segment[7], is_((4, 61)))
    assert_that(observations_per_segment[-1], is_((0, 8067)))
    assert_that(round(training_result.bic(), 3), is_(24156.524))

    # check mrf results
    assert_that(len(training_result.mrf.keys()), is_(training_result.number_of_none_zero_clusters))
    assert_that(training_result.mrf[0].shape,
                is_((window_size * numer_of_ts_in_example_data, window_size * numer_of_ts_in_example_data)))
    # not all clusters return toeplitz matrices, the first one does
    assert_that(all(training_result.check_diagonal_matrices_are_the_same()[0]), is_(True))
