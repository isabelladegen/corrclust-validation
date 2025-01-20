import ast
from dataclasses import dataclass
from os import path

import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh

from src.data_generation.generate_synthetic_correlated_data import generate_correlation_matrix
from src.utils.configurations import PATTERNS_TO_MODEL_PATH


@dataclass
class PatternCols:
    id = 'Id'
    canonical_patterns = 'Canonical Pattern'
    relaxed_patterns = 'Relaxed Pattern'
    reg_term = 'Regularisation'
    is_ideal = 'Is Ideal'


def check_if_psd(correlation_coefficients: [], tol=1e-10):
    """ Check if the correlation coefficients result in a positive semi definite matrix:
    - these fulfill both the triangular inequality (conditions on the angles) but also
    - ensure there is no contractions such as from [0, -1, -1] -> vectors 1 and 2 are orthogonal (r12 = 0), vector 3
    is opposite to vector 1 (r13 = -1) but it cannot also be opposite to vector 2 (r23 = -1)
    """
    matrix = generate_correlation_matrix(correlation_coefficients)
    eigenvalues = eigvalsh(matrix)
    return np.all(eigenvalues > -tol)


def check_triangle_inequality(r12, r13, r23):
    """
    Source: Claude Sonnet 3.5
    Check if correlation vector satisfies triangle inequality:
    1 + r12*r23*r13 ≥ (r12^2 + r23^2 + r13^2)/2
    """
    lhs = 1 + r12 * r23 * r13
    rhs = (r12 ** 2 + r23 ** 2 + r13 ** 2) / 2
    return lhs >= rhs


def has_two_strong_correlations(correlation_coefficients: []):
    """ Checks weather they are two strong correlation coefficients
    """
    count = sum(1 for x in correlation_coefficients if abs(x) == 1)
    return count == 2


def relax_canonical_pattern(correlation_coefficients: []):
    """ Relaxes the canonical pattern to make them valid correlation coefficients for a 3x3 matrix"""
    if has_two_strong_correlations(correlation_coefficients):
        relaxed_pattern = correlation_coefficients.copy()
        adjust = 0.71
        for i in range(len(correlation_coefficients)):
            if adjust == 0.69:  # we've already adjusted two
                break
            if abs(correlation_coefficients[i]) == 1:
                relaxed_pattern[i] = adjust * correlation_coefficients[i]  # reduce
                adjust = adjust - 0.01  # next adjust more
        is_valid = check_if_psd(relaxed_pattern)
        assert is_valid, "Relaxation mistake. Relaxed pattern: " + str(relaxed_pattern) + " is not psd."
        return relaxed_pattern
    return []


class ModelCorrelationPatterns:
    def __init__(self):
        file = PATTERNS_TO_MODEL_PATH
        assert (path.exists(file))
        df = pd.read_csv(file)
        df[PatternCols.canonical_patterns] = df[PatternCols.canonical_patterns].apply(lambda x: ast.literal_eval(x))
        df[PatternCols.relaxed_patterns] = df[PatternCols.relaxed_patterns].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) else x)

        df[PatternCols.is_ideal] = df[PatternCols.is_ideal].apply(lambda x: True if x == 'Yes' else False)
        df.astype({PatternCols.is_ideal: 'bool'}, copy=False)
        self.df = df

    def ids_of_patterns_to_model(self):
        return self.df.dropna()[PatternCols.id]

    def relaxed_patterns_and_regularisation_term(self):
        """ Returns dictionary of shape:
        keys: ids of patterns to model
        values: tuple with tuple[0] being list of correlations to model and tuple[1] being float of
         regularisation to use with Cholesky decomposition correlation method
        """
        ids = self.ids_of_patterns_to_model()
        filtered_df = self.df[self.df[PatternCols.id].isin(ids)].copy()
        filtered_df['pattern lists'] = filtered_df[PatternCols.relaxed_patterns].apply(list)
        values = filtered_df[['pattern lists', PatternCols.reg_term]].apply(tuple, axis=1)
        result = dict(zip(filtered_df[PatternCols.id], values))
        return result

    def canonical_patterns(self):
        """ Returns dictionary of shape:
        keys: ids of the canonical pattern
        values: ideal correlation unadjusted
        """
        ids = self.ids_of_patterns_to_model()
        filtered_df = self.df[self.df[PatternCols.id].isin(ids)].copy()
        result = dict(zip(filtered_df[PatternCols.id], filtered_df[PatternCols.canonical_patterns].apply(list)))
        return result

    def relaxed_patterns(self):
        """ Returns dictionary of shape:
        keys: ids of the canonical pattern
        values: relaxed canonical pattern
        """
        ids = self.ids_of_patterns_to_model()
        filtered_df = self.df[self.df[PatternCols.id].isin(ids)].copy()
        result = dict(zip(filtered_df[PatternCols.id], filtered_df[PatternCols.relaxed_patterns].apply(list)))
        return result

    def x_and_y_of_patterns_to_model(self):
        """ Returns feature matrix X and class label vector y. X has each correlation pair as column and
        y has the class label for each row in X. The ideal label and correlations are used.
        """
        patterns = self.canonical_patterns()
        y = np.array(list(patterns.keys()))
        x = np.array(list(patterns.values()))
        return x, y

    def perfect_valid_correlation_patterns(self):
        """
        Returns all patterns that unadjusted are valid correlation matrices:
        - are positive semi definite
        :returns dictionary with key = pattern id and value = pattern that can be modeled without adjustments
        """
        patterns = self.canonical_patterns()
        return {key: value for key, value in patterns.items() if check_if_psd(value)}

    def calculate_relaxed_patterns(self):
        """ Returns all patterns either ideal or relaxed such that they are valid correlation patterns"""
        can_patterns = self.canonical_patterns()
        results = {}
        for key, canonical_pattern in can_patterns.items():
            if check_if_psd(canonical_pattern):  # perfect modeled
                results[key] = canonical_pattern
            else:  # needs relaxation to be valid
                relaxed_pattern = relax_canonical_pattern(canonical_pattern)
                if len(relaxed_pattern) == 0:
                    continue
                results[key] = relaxed_pattern
        return results
