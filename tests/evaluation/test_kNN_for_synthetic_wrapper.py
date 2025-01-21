import numpy as np
from hamcrest import *

from src.evaluation.knn_for_synthetic_wrapper import KNNForSyntheticWrapper
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR

backend = Backends.visible_tests.value

n_neighbors = 1
ds_1 = 'misty-forest-56'
test_data_dir = TEST_DATA_DIR
nn_dt = SyntheticDataType.non_normal_correlated


def test_knn_classification_of_segments_l2():
    w = KNNForSyntheticWrapper(measure=DistanceMeasures.l2_cor_dist, n_neighbors=n_neighbors, data_dir=test_data_dir,
                               backend=backend)
    x_test, y_true, y_pred = w.load_data_and_predict(ds_1)

    fig = w.plot_confusion_matrix()
    assert_that(fig, is_not(None))

    assert_that(all(np.equal(y_pred, y_true)))

    accuracy, precision, recall, f1 = w.evaluate_scores(average="macro")
    assert_that(accuracy, is_(1.0))
    assert_that(precision, is_(1.0))
    assert_that(recall, is_(1.0))
    assert_that(f1, is_(1.0))


def test_knn_classification_of_segments_l2_with_ref():
    w = KNNForSyntheticWrapper(measure=DistanceMeasures.l2_with_ref, n_neighbors=n_neighbors, data_dir=test_data_dir,
                               backend=backend)
    x_test, y_true, y_pred = w.load_data_and_predict(ds_1)

    fig = w.plot_confusion_matrix()
    assert_that(fig, is_not(None))

    accuracy, precision, recall, f1 = w.evaluate_scores(average="macro")
    assert_that(all(np.equal(y_pred, y_true)))

    assert_that(accuracy, is_(1.0))
    assert_that(precision, is_(1.0))
    assert_that(recall, is_(1.0))
    assert_that(f1, is_(1.0))


def test_knn_classification_of_segments_log_frobenious():
    w = KNNForSyntheticWrapper(measure=DistanceMeasures.log_frob_cor_dist, n_neighbors=n_neighbors,
                               data_dir=test_data_dir, backend=backend)
    x_test, y_true, y_pred = w.load_data_and_predict(ds_1)

    fig = w.plot_confusion_matrix()
    assert_that(fig, is_not(None))

    assert_that(all(np.equal(y_pred, y_true)), is_(False))  # makes mistakes


    accuracy, precision, recall, f1 = w.evaluate_scores(average="macro")
    assert_that(accuracy, is_(0.15))
    assert_that(precision, is_(0.04))
    assert_that(recall, is_(0.13))
    assert_that(f1, is_(0.06))


def test_can_handle_no_errors():
    w = KNNForSyntheticWrapper(measure=DistanceMeasures.l1_cor_dist, n_neighbors=n_neighbors, data_dir=test_data_dir,
                               backend=backend)
    x_test, y_true, y_pred = w.load_data_and_predict(ds_1)

    errors = [(true_label, y_pred[idx]) for idx, true_label in enumerate(y_true) if
              true_label != y_pred[idx]]

    # check that there are errors for this ds
    assert_that(len(errors), is_(0))
    error_sample, error_cor, correct_cor, nn_classes, correct_classes, d_nn, d_correct = w.error_information()
    assert_that(len(error_sample), is_(0))
    assert_that(len(error_cor), is_(0))
    assert_that(len(correct_cor), is_(0))
    assert_that(len(nn_classes), is_(0))
    assert_that(len(correct_classes), is_(0))
    assert_that(len(d_nn), is_(0))
    assert_that(len(d_correct), is_(0))


def test_returns_errors_correct_nn_d_nn_and_d_correct():
    w = KNNForSyntheticWrapper(measure=DistanceMeasures.l1_cor_dist, n_neighbors=n_neighbors, data_dir=test_data_dir,
                               backend=backend)
    x_test, y_true, y_pred = w.load_data_and_predict(ds_1, data_type=SyntheticDataType.rs_1min)

    errors = [(true_label, y_pred[idx]) for idx, true_label in enumerate(y_true) if
              true_label != y_pred[idx]]

    # check that there are errors for this ds
    assert_that(len(errors), greater_than(1))

    # check error analysis
    error_sample, error_cor, correct_cor, nn_classes, correct_classes, d_nn, d_correct = w.error_information()
    possible_values = [-1, 0, 1]
    for cor in error_cor:
        # knn is trained on ideal pattern values
        assert_that(len(set(cor) - set(possible_values)), is_(0),
                    f"Error correlation with none ideal training values {cor}")
    assert_that(len(error_sample), is_(len(errors)))
    assert_that(len(error_sample[0]), is_(3))
    assert_that(len(error_cor), is_(len(errors)))
    assert_that(len(error_cor[0]), is_(3))
    assert_that(len(correct_cor), is_(len(errors)))
    assert_that(len(correct_cor[0]), is_(3))
    assert_that(len(nn_classes), is_(len(errors)))
    assert_that(isinstance(nn_classes[0], (int, np.integer)))
    assert_that(len(correct_classes), is_(len(errors)))
    assert_that(isinstance(correct_classes[0], (int, np.integer)))
    assert_that(len(d_nn), is_(len(errors)))
    assert_that(isinstance(d_nn[0], (float, np.floating)))
    assert_that(len(d_correct), is_(len(errors)))
    assert_that(isinstance(d_correct[0], (float, np.floating)))
