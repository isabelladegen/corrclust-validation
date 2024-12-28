from dataclasses import dataclass
from typing import overload

import numpy as np
import pandas as pd
from scipy.linalg import eigvals, logm, norm

from src.data_generation.generate_synthetic_correlated_data import generate_correlation_matrix


@dataclass
class DistanceMeasures:
    l1_with_ref = "L1 with ref"
    l2_with_ref = "L2 with ref"
    l5_with_ref = "L5 with ref"
    l10_with_ref = "L10 with ref"
    l50_with_ref = "L50 with ref"
    l100_with_ref = "L100 with ref"
    linf_with_ref = "Linf with ref"
    l1_cor_dist: str = "L1 corr dist"
    l2_cor_dist: str = "L2 corr dist"
    l5_cor_dist: str = "L5 corr dist"
    l10_cor_dist: str = "L10 corr dist"
    l50_cor_dist: str = "L50 corr dist"
    l100_cor_dist: str = "L100 corr dist"
    linf_cor_dist: str = "Linf corr dist"
    log_frob_cor_dist: str = "Log Frobenious corr dist"
    foerstner_cor_dist: str = "Foerstner corr dist"


@overload
def calculate_foerstner_matrices_distance_between(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def calculate_foerstner_matrices_distance_between(m1: [], m2: []): ...


def calculate_foerstner_matrices_distance_between(corr1, corr2, epsilon: float = 1e-10):
    """
    Calculates foerstner correlation distance for segment1 and segment 2.
    See https://link.springer.com/chapter/10.1007/978-3-662-05296-9_31 for distance definition
    :param corr1: DataFrame of first segment correlation matrix of shape (n_ts, n_ts) or upper triu vector
    :param corr1: DataFrame of second segment correlation matrix of shape (n_ts, n_ts) or upper triu vector
    :param epsilon: a small identity matrix based number to make all correlation matrices full rank and avoid inf
    eigenvalues in the generalised eigenvalue problem when comparing two singular matrices
    set to na if not both matrices are positive definite and symmetric
    :return: Foerstner distance between the two correlation matrices of each of the segment
    """
    if hasattr(corr1, 'shape'):
        m1 = corr1
        m2 = corr2
    else:
        m1 = generate_correlation_matrix(corr1)
        m2 = generate_correlation_matrix(corr2)

    # cov matrices must have the same shape as all segments have the same number of time series
    assert m1.shape == m2.shape

    # FIX: regularisation to avoid singular matrices
    reg_m = np.identity(m1.shape[0]) * epsilon
    m1 = m1 + reg_m
    m2 = m2 + reg_m

    # calculate generalised eigenvalues
    vals = eigvals(m1, b=m2)
    real_vals = np.real(vals)

    # calculate Foerstner covariance distance metric of real part of eigenvalue
    # 1) FIX small eigenvalues close to 0 with lambda+1 (which is what log1p is doing)
    # 2) FIX negative eigenvalues from a bug in scipy with regularisation,
    # see bug report https://github.com/scipy/scipy/issues/21951
    # 3) FIX singularity using regularised matrices
    distance = np.sqrt(np.sum(np.square(np.log1p(real_vals))))
    return distance


@overload
def calculate_log_matrix_frobenius_distance_between(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def calculate_log_matrix_frobenius_distance_between(m1: [], m2: []): ...


def calculate_log_matrix_frobenius_distance_between(corr1, corr2):
    """
    Calculates log(matrix) frobenius distance between segment1 and segment2.
    See https://link.springer.com/article/10.1007/s10115-017-1098-1 for distance definition
    :param corr1: correlation matrix of shape (n_ts, n_ts) or vector of upper triu
    :param corr2: correlation matrix of shape (n_ts, n_ts) or vector of upper triu
    :return: Log(matrix) frobenius distance between the two correlation matrices of segments
    """
    if hasattr(corr1, 'shape'):
        m1 = corr1
        m2 = corr2
    else:
        m1 = generate_correlation_matrix(corr1)
        m2 = generate_correlation_matrix(corr2)

    # cov matrices must have the same shape
    assert m2.shape == m2.shape

    # calculate matrix logarithm of both covariance/correlation matrices (this moves the cov/corr
    # into tangent space on the Riemannian manifold
    log_m1 = logm(m1)
    log_m2 = logm(m2)

    # calculate the frobenius norm between the two log_m matrices
    matrix_diff = log_m1 - log_m2
    distance = norm(matrix_diff, ord='fro')  # frobenius norm
    return distance


@overload
def l1_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l1_distance_from_matrices(m1: [], m2: []): ...


def l1_distance_from_matrices(m1, m2):
    """
        Calculates L1 corr distance between segment1 and segment 2.
        Assuming the columns are the different time series and the rows are the observations.
        The two df need to have the same number of columns  (observations, n_ts), the covariance/correlation
        matrix for each segment is of shape (columns, columns).
        :param m1: DataFrame of first segment covariance matrix of shape (n_ts, n_ts)
        :param m2: DataFrame of second segment covariance matrix of shape (n_ts, n_ts)
    """
    if hasattr(m1, 'shape'):
        matrix1 = m1
        matrix2 = m2
    else:
        matrix1 = generate_correlation_matrix(m1)
        matrix2 = generate_correlation_matrix(m2)
    corr_diff = calculate_matrix_diff(matrix1, matrix2)
    dist = np.linalg.norm(corr_diff, ord=1)
    return dist


@overload
def l2_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l2_distance_from_matrices(m1: [], m2: []): ...


def l2_distance_from_matrices(corr1, corr2):
    """
        Calculates L2 corr distance between correlations of segment1 and segment 2.
        matrix for each segment is of shape (columns, columns).
        :param corr2: DataFrame of first segment covariance matrix of shape (n_ts, n_ts) or upper triu vector
        :param corr2: DataFrame of second segment covariance matrix of shape (n_ts, n_ts) or upper triu vector
    """
    if hasattr(corr1, 'shape'):
        m1 = corr1
        m2 = corr2
    else:
        m1 = generate_correlation_matrix(corr1)
        m2 = generate_correlation_matrix(corr2)

    corr_diff = calculate_matrix_diff(m1, m2)
    dist = np.linalg.norm(corr_diff, ord=2)
    return dist


@overload
def linf_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def linf_distance_from_matrices(m1: [], m2: []): ...


def linf_distance_from_matrices(corr1, corr2):
    """
        Calculates Linf corr distance between segment1 and segment 2.
        :param corr1: DataFrame of first segment covariance matrix of shape (n_ts, n_ts) or upper triu vector
        :param corr2: DataFrame of second segment covariance matrix of shape (n_ts, n_ts) or upper triu vector
    """
    if hasattr(corr1, 'shape'):
        m1 = corr1
        m2 = corr2
    else:
        m1 = generate_correlation_matrix(corr1)
        m2 = generate_correlation_matrix(corr2)

    corr_diff = calculate_matrix_diff(m1, m2)
    dist = np.linalg.norm(corr_diff, ord=np.inf)
    return dist


@overload
def l1_with_ref_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l1_with_ref_distance_from_matrices(m1: [], m2: []): ...


def l1_with_ref_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame):
    """
        Calculates L1 corr distance with reference between segment1 and segment 2.
        Assuming the columns are the different time series and the rows are the observations.
        The two df need to have the same number of columns  (observations, n_ts), the covariance/correlation
        matrix for each segment is of shape (columns, columns).
        :param m1: DataFrame of first segment covariance matrix of shape (n_ts, n_ts) or list of upper triu
        :param m2: DataFrame of second segment covariance matrix of shape (n_ts, n_ts)
    """
    if hasattr(m1, 'shape'):
        v1 = np.triu(m1, k=1)
        v2 = np.triu(m2, k=1)
    else:
        v1 = m1
        v2 = m2
    return lp_with_reference_vector(v1, v2, p=1)


@overload
def l2_with_ref_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l2_with_ref_distance_from_matrices(m1: [], m2: []): ...


def l2_with_ref_distance_from_matrices(m1, m2):
    """
        Calculates L2 corr distance with reference between segment1 and segment 2.
        Assuming the columns are the different time series and the rows are the observations.
        The two df need to have the same number of columns  (observations, n_ts), the covariance/correlation
        matrix for each segment is of shape (columns, columns).
        :param m1: can be either list or 2d np array or dataframe for m1 matrix
        :param m2: can be either list or 2d np array or dataframe for m2 matrix
    """
    if hasattr(m1, 'shape'):
        v1 = np.triu(m1, k=1)
        v2 = np.triu(m2, k=1)
    else:
        v1 = m1
        v2 = m2
    return lp_with_reference_vector(v1, v2, p=2)


@overload
def linf_with_ref_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def linf_with_ref_distance_from_matrices(m1: [], m2: []): ...


def linf_with_ref_distance_from_matrices(m1, m2):
    """
        Calculates Linf corr distance with reference between segment1 and segment 2.
        :param m1: DataFrame of first segment covariance matrix of shape (n_ts, n_ts) or upper triu vector
        :param m2: DataFrame of second segment covariance matrix of shape (n_ts, n_ts) or upper triu vector
    """
    if hasattr(m1, 'shape'):
        v1 = np.triu(m1, k=1)
        v2 = np.triu(m2, k=1)
    else:
        v1 = m1
        v2 = m2
    dist = lp_with_reference_vector(v1, v2, p=np.inf)
    return dist


def calculate_matrix_diff(m1, m2):
    # matrices must have the same shape as all segments have the same number of variates
    assert m1.shape == m2.shape
    c1 = np.triu(m1, k=1)
    c2 = np.triu(m2, k=1)
    # calculate L1
    diff = c1 - c2
    return diff


def dot_transformation(v):
    """
    To avoid mirror vectors having the same distance e.g. distance [0,0,1] to [0,1,0] naturally the same as
    the distance[0,0,-1] to [0,1,0]. We calculate the dot product between the vector and the normed [1,1,1] reference
    vector, add 1 to it to avoid mistaking -1 with +1 and divide it to 2 to make its max length 1
    :param v: np_array
    :return: np_array with rotation information one dimension added (e.g instead 3D it will be 4D) etc
    """
    dim = len(v)
    ref = np.ones(dim) * 1 / np.sqrt(dim)
    # difference between reference and v, +1 to make a difference between -1 and +1 (orientation), /2 to make it no longer than max 1
    s = (np.dot(ref, v) + 1) / 2
    v_d = np.append(v, s)
    return v_d


def lp_norm(v1: np.array, v2: np.array, p=2):
    dist = np.linalg.norm(v1 - v2, ord=p)
    return dist


def lp_with_reference_vector(v1, v2, p: float = 2):
    if isinstance(v1, list):
        v1 = np.array(v1)
        v2 = np.array(v2)

    # calculate distance between v1 and v2
    dist = lp_norm(v1, v2, p)

    # multiply distance with the sum of the distance to the reference vect
    dim = len(v1)
    ref = np.ones(dim) * 1 / np.sqrt(dim)
    v1_ref = np.linalg.norm(ref - v1, ord=p)
    v2_ref = np.linalg.norm(ref - v2, ord=p)
    result_dist = dist * (v1_ref + v2_ref)
    return result_dist


def distance_calculation_method_for(distance_measure: str):
    """
    Returns method to calculate distance for the given distance measure name
    :param distance_measure: a distance measure that takes two matrices to calculate the distance
    return method for distance calculation
    """
    if distance_measure == DistanceMeasures.l1_cor_dist:
        return l1_distance_from_matrices
    elif distance_measure == DistanceMeasures.l2_cor_dist:
        return l2_distance_from_matrices
    elif distance_measure == DistanceMeasures.linf_cor_dist:
        return linf_distance_from_matrices
    elif distance_measure == DistanceMeasures.l1_with_ref:
        return l1_with_ref_distance_from_matrices
    elif distance_measure == DistanceMeasures.l2_with_ref:
        return l2_with_ref_distance_from_matrices
    elif distance_measure == DistanceMeasures.linf_with_ref:
        return linf_with_ref_distance_from_matrices
    elif distance_measure == DistanceMeasures.foerstner_cor_dist:
        return calculate_foerstner_matrices_distance_between
    elif distance_measure == DistanceMeasures.log_frob_cor_dist:
        return calculate_log_matrix_frobenius_distance_between
    else:
        assert False, "Unknown distance measure with name: " + distance_measure
