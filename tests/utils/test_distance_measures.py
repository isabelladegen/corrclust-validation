import numpy as np
from hamcrest import *

from src.data_generation.generate_synthetic_correlated_data import generate_correlation_matrix
from src.data_generation.model_correlation_patterns import check_if_psd
from src.utils.distance_measures import l1_distance_from_matrices, l2_distance_from_matrices, l5_distance_from_matrices, \
    l10_distance_from_matrices, linf_distance_from_matrices, l100_distance_from_matrices, l50_distance_from_matrices, \
    cosine_similarity, linf_with_ref_distance_from_matrices, l100_with_ref_distance_from_matrices, \
    l50_with_ref_distance_from_matrices, l10_with_ref_distance_from_matrices, l5_with_ref_distance_from_matrices, \
    l2_with_ref_distance_from_matrices, l1_with_ref_distance_from_matrices, \
    calculate_foerstner_matrices_distance_between, calculate_log_matrix_frobenius_distance_between

corr1 = [1, 0, 0]
corr2 = [0, 1, 0]

m1 = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
m2 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]


def test_generates_proper_correlation_matrix_from_upper_triu_vector():
    matrix_corr1 = generate_correlation_matrix(corr1)
    matrix_corr2 = generate_correlation_matrix(corr2)
    assert_that(np.array_equal(matrix_corr1, np.array(m1)))
    assert_that(np.array_equal(matrix_corr2, np.array(m2)))


def test_l1_distance():
    nd_matrix = l1_distance_from_matrices(np.array(m1), np.array(m2))
    l1m = l1_distance_from_matrices(m1, m2)
    l1v = l1_distance_from_matrices(corr1, corr2)
    assert_that(l1m, is_(l1v))  # should give the same result
    assert_that(l1m, is_(nd_matrix))  # should give the same result
    assert_that(l1v, is_(2))


def test_l2_distance():
    nd_matrix = l2_distance_from_matrices(np.array(m1), np.array(m2))
    matrix_dist = l2_distance_from_matrices(m1, m2)
    vector_dist = l2_distance_from_matrices(corr1, corr2)
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(1.414))


def test_l5_distance():
    nd_matrix = l5_distance_from_matrices(np.array(m1), np.array(m2))
    matrix_dist = l5_distance_from_matrices(m1, m2)
    vector_dist = l5_distance_from_matrices(corr1, corr2)
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(1.149))


def test_l10_distance():
    nd_matrix = l10_distance_from_matrices(np.array(m1), np.array(m2))
    matrix_dist = l10_distance_from_matrices(m1, m2)
    vector_dist = l10_distance_from_matrices(corr1, corr2)
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(1.072))


def test_l50_distance():
    nd_matrix = l50_distance_from_matrices(np.array(m1), np.array(m2))
    matrix_dist = l50_distance_from_matrices(m1, m2)
    vector_dist = l50_distance_from_matrices(corr1, corr2)
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(1.014))


def test_l100_distance():
    nd_matrix = l100_distance_from_matrices(np.array(m1), np.array(m2))
    matrix_dist = l100_distance_from_matrices(m1, m2)
    vector_dist = l100_distance_from_matrices(corr1, corr2)
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(1.007))


def test_linf_distance():
    nd_matrix = linf_distance_from_matrices(np.array(m1), np.array(m2))
    matrix_dist = linf_distance_from_matrices(m1, m2)
    vector_dist = linf_distance_from_matrices(corr1, corr2)
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(1))


def test_l1_with_ref_distance():
    nd_matrix = l1_with_ref_distance_from_matrices(np.array(m1), np.array(m2))
    l1m = l1_with_ref_distance_from_matrices(m1, m2)
    l1v = l1_with_ref_distance_from_matrices(corr1, corr2)
    assert_that(l1m, is_(l1v))  # should give the same result
    assert_that(l1m, is_(nd_matrix))  # should give the same result
    assert_that(round(l1v, 3), is_(6.309))


def test_l2_with_ref_distance():
    nd_matrix = l2_with_ref_distance_from_matrices(np.array(m1), np.array(m2))
    matrix_dist = l2_with_ref_distance_from_matrices(m1, m2)
    vector_dist = l2_with_ref_distance_from_matrices(corr1, corr2)
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(2.6))


def test_l5_with_ref_distance():
    nd_matrix = l5_with_ref_distance_from_matrices(np.array(m1), np.array(m2))
    matrix_dist = l5_with_ref_distance_from_matrices(m1, m2)
    vector_dist = l5_with_ref_distance_from_matrices(corr1, corr2)
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(1.554))


def test_l10_with_ref_distance():
    nd_matrix = l10_with_ref_distance_from_matrices(np.array(m1), np.array(m2))
    matrix_dist = l10_with_ref_distance_from_matrices(m1, m2)
    vector_dist = l10_with_ref_distance_from_matrices(corr1, corr2)
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(1.329))


def test_l50_with_ref_distance():
    nd_matrix = l50_with_ref_distance_from_matrices(np.array(m1), np.array(m2))
    matrix_dist = l50_with_ref_distance_from_matrices(m1, m2)
    vector_dist = l50_with_ref_distance_from_matrices(corr1, corr2)
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(1.187))


def test_l100_with_ref_distance():
    nd_matrix = l100_with_ref_distance_from_matrices(np.array(m1), np.array(m2))
    matrix_dist = l100_with_ref_distance_from_matrices(m1, m2)
    vector_dist = l100_with_ref_distance_from_matrices(corr1, corr2)
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(1.171))


def test_linf_with_ref_distance():
    nd_matrix = linf_with_ref_distance_from_matrices(np.array(m1), np.array(m2))
    matrix_dist = linf_with_ref_distance_from_matrices(m1, m2)
    vector_dist = linf_with_ref_distance_from_matrices(corr1, corr2)
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(1.155))


def test_foerstner_distance():
    nd_matrix = calculate_foerstner_matrices_distance_between(np.array(m1), np.array(m2))
    matrix_dist = calculate_foerstner_matrices_distance_between(m1, m2)
    vector_dist = calculate_foerstner_matrices_distance_between(corr1, corr2)
    nd_vector_dist = calculate_foerstner_matrices_distance_between(np.array(corr1), np.array(corr2))
    assert_that(nd_vector_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(23.036))


def test_challenging_foerstner_distances():
    # these lead to nan (p8 and p5)
    p1 = [0, 0, 1]
    p13 = [1, 1, 1]
    p25 = [-1, -1, 1]
    a_x = [-0.08, -0.07, 0.99]
    fdist125 = calculate_foerstner_matrices_distance_between(p1, p25)
    fdist11 = calculate_foerstner_matrices_distance_between(p1, p1)
    fdist113 = calculate_foerstner_matrices_distance_between(p1, p13)
    fdistax_1 = calculate_foerstner_matrices_distance_between(a_x, p1)
    fdistax_13 = calculate_foerstner_matrices_distance_between(a_x, p13)
    fdistax_125 = calculate_foerstner_matrices_distance_between(a_x, p25)

    ax_psd = check_if_psd(a_x)
    assert_that(ax_psd)
    print("dfwere")


def test_frobenious_distance():
    nd_matrix = calculate_log_matrix_frobenius_distance_between(np.array(m1), np.array(m2))
    vector_dist = calculate_log_matrix_frobenius_distance_between(corr1, corr2)
    nd_vector_dist = calculate_log_matrix_frobenius_distance_between(np.array(corr1), np.array(corr2))
    matrix_dist = calculate_log_matrix_frobenius_distance_between(m1, m2)
    assert_that(nd_vector_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(56.69))


def test_cosine_similarity():
    nd_matrix = cosine_similarity(np.array(m1), np.array(m2))
    matrix_dist = cosine_similarity(m1, m2)
    vector_dist = cosine_similarity(corr1, corr2)
    assert_that(matrix_dist, is_(vector_dist))  # should give the same result
    assert_that(matrix_dist, is_(nd_matrix))  # should give the same result
    assert_that(round(vector_dist, 3), is_(0))
