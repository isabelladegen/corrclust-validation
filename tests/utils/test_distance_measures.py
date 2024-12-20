import numpy as np
from hamcrest import *
from pandas import DataFrame

from src.utils.distance_measures import calculate_foerstner_matrices_distance_between, \
    calculate_log_matrix_frobenius_distance_between, calculate_covariance_matrix_for_segment_df

observations_seg1 = 10
dist_columns = ['col 1', 'col 2', 'col 3', 'col 4']
segment_1_data = {dist_columns[0]: np.random.randint(3, size=observations_seg1),
                  dist_columns[1]: np.random.randint(4, size=observations_seg1),
                  dist_columns[2]: np.random.randint(3, size=observations_seg1),
                  dist_columns[3]: np.random.randint(5, size=observations_seg1)
                  }
segment1_df = DataFrame.from_dict(segment_1_data)

observations_seg2 = 4
segment_2_data = {dist_columns[0]: np.random.randint(4, size=observations_seg2),
                  dist_columns[1]: np.random.randint(3, size=observations_seg2),
                  dist_columns[2]: np.random.randint(4, size=observations_seg2),
                  dist_columns[3]: np.random.randint(6, size=observations_seg2)
                  }
segment2_df = DataFrame.from_dict(segment_2_data)  # random numbers between 0 and 3

cov1 = calculate_covariance_matrix_for_segment_df(segment1_df)
cov2 = calculate_covariance_matrix_for_segment_df(segment2_df)


def test_calculates_foerstner_distance_between_two_segments():
    distance = calculate_foerstner_matrices_distance_between(cov1, cov2)
    assert_that(distance, is_not(0))
    print("Foerstner dist: " + str(distance))


def test_calculates_log_cov_frobenius_distance_between_two_segments():
    distance = calculate_log_matrix_frobenius_distance_between(cov1, cov2)
    assert_that(distance, is_not(0))
    print("Log(cov) Frobenius dist: " + str(distance))
