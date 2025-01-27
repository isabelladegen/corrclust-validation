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
    l3_with_ref = "L3 with ref"
    l5_with_ref = "L5 with ref"
    l10_with_ref = "L10 with ref"
    l50_with_ref = "L50 with ref"
    l100_with_ref = "L100 with ref"
    linf_with_ref = "Linf with ref"
    l1_cor_dist: str = "L1 dist"
    l2_cor_dist: str = "L2 dist"
    l3_cor_dist: str = "L3 dist"
    l5_cor_dist: str = "L5 dist"
    l10_cor_dist: str = "L10 dist"
    l50_cor_dist: str = "L50 dist"
    l100_cor_dist: str = "L100 dist"
    linf_cor_dist: str = "Linf dist"
    log_frob_cor_dist: str = "Log Frobenious dist"
    foerstner_cor_dist: str = "Foerstner dist"
    cosine: str = "Cosine"
    dot_transform_linf: str = "Dot transf Linf"
    dot_transform_l1: str = "Dot transf L1"
    dot_transform_l2: str = "Dot transf L2"


short_distance_measure_names = {
    DistanceMeasures.l1_cor_dist: 'L1',
    DistanceMeasures.l2_cor_dist: 'L2',
    DistanceMeasures.l3_cor_dist: 'L3',
    DistanceMeasures.l5_cor_dist: 'L5',
    DistanceMeasures.linf_cor_dist: 'Linf',
    DistanceMeasures.l1_with_ref: 'L1 ref',
    DistanceMeasures.l2_with_ref: 'L2 ref',
    DistanceMeasures.l3_with_ref: 'L3 ref',
    DistanceMeasures.l5_with_ref: 'L5 ref',
    DistanceMeasures.linf_with_ref: 'Linf ref',
    DistanceMeasures.dot_transform_l1: 'dt L1',
    DistanceMeasures.dot_transform_l2: 'dt L2',
    DistanceMeasures.dot_transform_linf: 'dt Linf',
    DistanceMeasures.log_frob_cor_dist: 'Log F',
    DistanceMeasures.foerstner_cor_dist: 'Foer',
}


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
    if isinstance(corr1, np.ndarray) and len(corr1.shape) == 2:
        m1 = corr1
        m2 = corr2
    else:
        if isinstance(corr1, list):  # turn into nd.array
            m1 = np.array(corr1)
            m2 = np.array(corr2)
        else:
            m1 = corr1
            m2 = corr2
        if len(m1.shape) != 2:  # it's a vector, turn into matrix
            m1 = generate_correlation_matrix(m1)
            m2 = generate_correlation_matrix(m2)

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


def calculate_log_matrix_frobenius_distance_between(corr1, corr2, epsilon: float = 1e-10):
    """
    Calculates log(matrix) frobenius distance between segment1 and segment2.
    See https://link.springer.com/article/10.1007/s10115-017-1098-1 for distance definition
    :param corr1: correlation matrix of shape (n_ts, n_ts) or vector of upper triu
    :param corr2: correlation matrix of shape (n_ts, n_ts) or vector of upper triu
    :param epsilon: a small identity matrix based number to make all correlation matrices full rank and avoid log
    not being defined
    :return: Log(matrix) frobenius distance between the two correlation matrices of segments
    """
    if isinstance(corr1, np.ndarray) and len(corr1.shape) == 2:
        m1 = corr1
        m2 = corr2
    else:
        if isinstance(corr1, list):  # turn into nd.array
            m1 = np.array(corr1)
            m2 = np.array(corr2)
        else:
            m1 = corr1
            m2 = corr2
        if len(m1.shape) != 2:  # it's a vector, turn into matrix
            m1 = generate_correlation_matrix(m1)
            m2 = generate_correlation_matrix(m2)

    # corr matrices must have the same shape
    assert m2.shape == m2.shape

    # FIX: regularisation to avoid singular matrices
    reg_m = np.identity(m1.shape[0]) * epsilon
    m1 = m1 + reg_m
    m2 = m2 + reg_m

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
    return lp_distance(m1, m2, p=1)


@overload
def l2_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l2_distance_from_matrices(m1: [], m2: []): ...


def l2_distance_from_matrices(corr1, corr2):
    return lp_distance(corr1, corr2, p=2)


@overload
def l3_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l3_distance_from_matrices(m1: [], m2: []): ...


def l3_distance_from_matrices(corr1, corr2):
    return lp_distance(corr1, corr2, p=3)


@overload
def l5_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l5_distance_from_matrices(m1: [], m2: []): ...


def l5_distance_from_matrices(corr1, corr2):
    return lp_distance(corr1, corr2, p=5)


@overload
def l10_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l10_distance_from_matrices(m1: [], m2: []): ...


def l10_distance_from_matrices(corr1, corr2):
    return lp_distance(corr1, corr2, p=10)


@overload
def l50_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l50_distance_from_matrices(m1: [], m2: []): ...


def l50_distance_from_matrices(corr1, corr2):
    return lp_distance(corr1, corr2, p=50)


@overload
def l100_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l100_distance_from_matrices(m1: [], m2: []): ...


def l100_distance_from_matrices(corr1, corr2):
    return lp_distance(corr1, corr2, p=100)


@overload
def linf_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def linf_distance_from_matrices(m1: [], m2: []): ...


def linf_distance_from_matrices(corr1, corr2):
    return lp_distance(corr1, corr2, p=np.inf)


def lp_distance(m1, m2, p):
    """
        Calculates LP corr distance between 2 correlation matrices or their upper half vector.
        :param m1: can be either list or 2d np array for m1 matrix
        :param m2: can be either list or 2d np array for m2 matrix
        :param p: order for distance 1=L1, 2=L2, etc.
    """
    # create numpy array
    v1, v2 = to_numpy_vectors_of_upper_halfs(m1, m2)
    corr_diff = v1 - v2
    return np.linalg.norm(corr_diff, ord=p)


def to_numpy_vectors_of_upper_halfs(m1, m2):
    if isinstance(m1, list):
        m1_nd = np.array(m1)
        m2_nd = np.array(m2)
    else:
        m1_nd = m1
        m2_nd = m2
    if len(m1_nd.shape) == 2:  # it's a 2d array and therefore a matrix -> take upper triangular bit
        assert m1_nd.shape == m2_nd.shape, "m1 and m2 must have the same shape"
        n = len(m1_nd)
        v1 = m1_nd[np.triu_indices(n, k=1)]
        v2 = m2_nd[np.triu_indices(n, k=1)]
    else:  # it's already a vector
        v1 = m1_nd
        v2 = m2_nd
    return v1, v2


@overload
def l1_with_ref_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l1_with_ref_distance_from_matrices(m1: [], m2: []): ...


def l1_with_ref_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame):
    return lp_with_reference_vector(m1, m2, p=1)


@overload
def l2_with_ref_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l2_with_ref_distance_from_matrices(m1: [], m2: []): ...


def l2_with_ref_distance_from_matrices(m1, m2):
    return lp_with_reference_vector(m1, m2, p=2)


@overload
def l3_with_ref_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l3_with_ref_distance_from_matrices(m1: [], m2: []): ...


def l3_with_ref_distance_from_matrices(m1, m2):
    return lp_with_reference_vector(m1, m2, p=3)


@overload
def l5_with_ref_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l5_with_ref_distance_from_matrices(m1: [], m2: []): ...


def l5_with_ref_distance_from_matrices(m1, m2):
    return lp_with_reference_vector(m1, m2, p=5)


@overload
def l10_with_ref_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l10_with_ref_distance_from_matrices(m1: [], m2: []): ...


def l10_with_ref_distance_from_matrices(m1, m2):
    return lp_with_reference_vector(m1, m2, p=10)


@overload
def l50_with_ref_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l50_with_ref_distance_from_matrices(m1: [], m2: []): ...


def l50_with_ref_distance_from_matrices(m1, m2):
    return lp_with_reference_vector(m1, m2, p=50)


@overload
def l100_with_ref_distance_from_matrices(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def l100_with_ref_distance_from_matrices(m1: [], m2: []): ...


def l100_with_ref_distance_from_matrices(m1, m2):
    return lp_with_reference_vector(m1, m2, p=100)


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
    return lp_with_reference_vector(m1, m2, p=np.inf)


def lp_with_reference_vector(m1, m2, p):
    # turn m1 and m2 into vectors of upper half of the correlation matrix
    v1, v2 = to_numpy_vectors_of_upper_halfs(m1, m2)

    # calculate distance between v1 and v2
    dist = lp_distance(v1, v2, p)

    # multiply distance with the sum of the distance to the reference vect
    dim = len(v1)
    ref = np.ones(dim) * 1 / np.sqrt(dim)
    v1_ref = lp_distance(v1, ref, p)
    v2_ref = lp_distance(v2, ref, p)
    result_dist = dist * (v1_ref + v2_ref)
    return result_dist


@overload
def cosine_similarity(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def cosine_similarity(m1: [], m2: []): ...


def cosine_similarity(m1, m2):
    """
        Calculates cosine similarity between segment1 and segment 2.
        :param m1: DataFrame of first segment covariance matrix of shape (n_ts, n_ts) or upper triu vector
        :param m2: DataFrame of second segment covariance matrix of shape (n_ts, n_ts) or upper triu vector
    """
    v1, v2 = to_numpy_vectors_of_upper_halfs(m1, m2)

    dist = np.dot(v1, v2) / (np.linalg.norm(v1, ord=2) * np.linalg.norm(v2, ord=2))
    return dist


@overload
def dot_transform_l1_distance(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def dot_transform_l1_distance(m1: [], m2: []): ...


def dot_transform_l1_distance(m1, m2):
    """
        Calculates dot transform l1 distance between segment1 and segment 2.
        :param m1: DataFrame of first segment covariance matrix of shape (n_ts, n_ts) or upper triu vector
        :param m2: DataFrame of second segment covariance matrix of shape (n_ts, n_ts) or upper triu vector
    """
    v1, v2 = to_numpy_vectors_of_upper_halfs(m1, m2)

    t_corr1 = dot_transformation(v1)
    t_corr2 = dot_transformation(v2)
    t_corr_diff = t_corr1 - t_corr2

    dist = np.linalg.norm(t_corr_diff, ord=1)
    return dist


@overload
def dot_transform_l2_distance(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def dot_transform_l2_distance(m1: [], m2: []): ...


def dot_transform_l2_distance(m1, m2):
    """
        Calculates dot transform l2 distance between segment1 and segment 2.
        :param m1: DataFrame of first segment covariance matrix of shape (n_ts, n_ts) or upper triu vector
        :param m2: DataFrame of second segment covariance matrix of shape (n_ts, n_ts) or upper triu vector
    """
    v1, v2 = to_numpy_vectors_of_upper_halfs(m1, m2)

    t_corr1 = dot_transformation(v1)
    t_corr2 = dot_transformation(v2)
    t_corr_diff = t_corr1 - t_corr2

    dist = np.linalg.norm(t_corr_diff, ord=2)
    return dist


@overload
def dot_transform_linf_distance(m1: pd.DataFrame, m2: pd.DataFrame): ...


@overload
def dot_transform_linf_distance(m1: [], m2: []): ...


def dot_transform_linf_distance(m1, m2):
    """
        Calculates dot transform linf distance between segment1 and segment 2.
        :param m1: DataFrame of first segment covariance matrix of shape (n_ts, n_ts) or upper triu vector
        :param m2: DataFrame of second segment covariance matrix of shape (n_ts, n_ts) or upper triu vector
    """
    v1, v2 = to_numpy_vectors_of_upper_halfs(m1, m2)

    t_corr1 = dot_transformation(v1)
    t_corr2 = dot_transformation(v2)
    t_corr_diff = t_corr1 - t_corr2

    dist = np.linalg.norm(t_corr_diff, ord=np.inf)
    return dist


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
    elif distance_measure == DistanceMeasures.l3_cor_dist:
        return l3_distance_from_matrices
    elif distance_measure == DistanceMeasures.l5_cor_dist:
        return l5_distance_from_matrices
    elif distance_measure == DistanceMeasures.l10_cor_dist:
        return l10_distance_from_matrices
    elif distance_measure == DistanceMeasures.l50_cor_dist:
        return l50_distance_from_matrices
    elif distance_measure == DistanceMeasures.l100_cor_dist:
        return l100_distance_from_matrices
    elif distance_measure == DistanceMeasures.linf_cor_dist:
        return linf_distance_from_matrices
    elif distance_measure == DistanceMeasures.l1_with_ref:
        return l1_with_ref_distance_from_matrices
    elif distance_measure == DistanceMeasures.l2_with_ref:
        return l2_with_ref_distance_from_matrices
    elif distance_measure == DistanceMeasures.l3_with_ref:
        return l3_with_ref_distance_from_matrices
    elif distance_measure == DistanceMeasures.l5_with_ref:
        return l5_with_ref_distance_from_matrices
    elif distance_measure == DistanceMeasures.l10_with_ref:
        return l10_with_ref_distance_from_matrices
    elif distance_measure == DistanceMeasures.l50_with_ref:
        return l50_with_ref_distance_from_matrices
    elif distance_measure == DistanceMeasures.l100_with_ref:
        return l100_with_ref_distance_from_matrices
    elif distance_measure == DistanceMeasures.linf_with_ref:
        return linf_with_ref_distance_from_matrices
    elif distance_measure == DistanceMeasures.foerstner_cor_dist:
        return calculate_foerstner_matrices_distance_between
    elif distance_measure == DistanceMeasures.log_frob_cor_dist:
        return calculate_log_matrix_frobenius_distance_between
    elif distance_measure == DistanceMeasures.cosine:
        return cosine_similarity
    elif distance_measure == DistanceMeasures.dot_transform_l1:
        return dot_transform_l1_distance
    elif distance_measure == DistanceMeasures.dot_transform_l2:
        return dot_transform_l2_distance
    elif distance_measure == DistanceMeasures.dot_transform_linf:
        return dot_transform_linf_distance
    else:
        assert False, "Unknown distance measure with name: " + distance_measure
