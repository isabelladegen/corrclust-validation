import numpy as np
import pandas as pd
from scipy.linalg import eigvals, logm, norm


def calculate_foerstner_matrices_distance_between(m1: pd.DataFrame, m2: pd.DataFrame, epsilon: float = 1e-10):
    """
    Calculates foerstner distance for segment1 and segment 2.
    See https://link.springer.com/chapter/10.1007/978-3-662-05296-9_31 for distance definition
    Assuming the columns are the different time series and the rows
    are the observations. The two df need to have the same number of columns  (observations, n_ts), the covariance/correlation
    matrix for each segment is of shape (columns, columns). You need to make sure cov1 and 2 are positive definite!
    :param m1: DataFrame of first segment covariance matrix of shape (n_ts, n_ts)
    :param m2: DataFrame of second segment covariance matrix of shape (n_ts, n_ts)
    :param epsilon: a small identity matrix based number to make all correlation matrices full rank and avoid inf
    eigenvalues in the generalised eigenvalue problem when comparing two singular matrices
    set to na if not both matrices are positive definite and symmetric
    :return: Foerstner distance between the two matrices of each of the segment
    """
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


def calculate_log_matrix_frobenius_distance_between(m1: pd.DataFrame, m2: pd.DataFrame):
    """
    Calculates log(matrix) frobenius distance between segment1 and segment2.
    See https://link.springer.com/article/10.1007/s10115-017-1098-1 for distance definition
    Assuming the columns are the different time series and the rows
    are the observations. The two df need to have the same number of columns  (observations, n_ts), the covariance
    matrix for each segment is of shape (columns, columns). You need to make sure cov1 and 2 are positive definite!
    :param m1: DataFrame of first segment covariance or correlation matrix of shape (n_ts, n_ts)
    :param m2: DataFrame of second segment covariance or correlation matrix of shape (n_ts, n_ts)
    set to na if not both matrices are positive definite and symmetric
    :return: Log(matrix) frobenius distance between the two covariance/correlation matrices of segments
    """
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


def calculate_covariance_matrix_for_segment_df(df: pd.DataFrame, regularisation: float = 0, ddof: int = 1):
    """
    Calculates the covariance of a dataframe using pandas cov as it is more stable than numpy
    :param df: segment data frame of shape (observations, n_ts)
    :param regularisation: a small identity matrix based number to avoid non positive semi-definite covariance matrices
    :param ddof: how the cov is normed 0=N -> Population Cov, 1=N-1 -> Sample Cov
    :return: pd.Dataframe of covariance matrix of shape (n_ts, n_ts)
    """
    # don't use numpy for cov calculation as the resulting matrix more often nto positive semi-definite which means
    cov_seg = df.cov(ddof=ddof)
    cov = cov_seg.to_numpy()
    if regularisation > 0:
        cov = cov + np.identity(cov.shape[0]) * regularisation
    regularised_cov_df = pd.DataFrame(cov, index=[cov_seg.index, cov_seg.columns])
    return regularised_cov_df.round(4)