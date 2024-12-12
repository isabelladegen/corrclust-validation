import ast
from dataclasses import dataclass
from os import path

import numpy as np
import pandas as pd

from src.utils.configurations import PATTERNS_TO_MODEL_PATH


@dataclass
class PatternCols:
    id = 'Id'
    ideal_cors = 'Ideal'
    modelable_cors = 'Modelable'
    reg_term = 'Regularisation'
    is_ideal = 'Is Ideal'
    should_model = 'Should be modelled'


def check_triangle_inequality(r12, r13, r23):
    """
    Source: Claude Sonnet 3.5
    Check if correlation vector satisfies triangle inequality:
    1 + r12*r23*r13 ≥ (r12^2 + r23^2 + r13^2)/2
    """
    lhs = 1 + r12 * r23 * r13
    rhs = (r12 ** 2 + r23 ** 2 + r13 ** 2) / 2
    return lhs >= rhs


class ModelCorrelationPatterns:
    def __init__(self):
        file = PATTERNS_TO_MODEL_PATH
        assert (path.exists(file))
        df = pd.read_csv(file, converters={PatternCols.ideal_cors: ast.literal_eval,
                                           PatternCols.modelable_cors: ast.literal_eval})
        df[PatternCols.should_model] = df[PatternCols.should_model].apply(lambda x: True if x == 'Yes' else False)
        df.astype({PatternCols.should_model: 'bool'}, copy=False)
        df[PatternCols.is_ideal] = df[PatternCols.is_ideal].apply(lambda x: True if x == 'Yes' else False)
        df.astype({PatternCols.is_ideal: 'bool'}, copy=False)
        self.df = df

    def ids_of_patterns_to_model(self):
        return self.df[self.df[PatternCols.should_model]][PatternCols.id]

    def patterns_to_model(self):
        """ Returns dictionary of shape:
        keys: ids of patterns to model
        values: tuple with tuple[0] being list of correlations to model and tuple[1] being float of
         regularisation to use with Cholesky decomposition correlation method
        """
        ids = self.ids_of_patterns_to_model()
        filtered_df = self.df[self.df[PatternCols.id].isin(ids)].copy()
        filtered_df['pattern lists'] = filtered_df[PatternCols.modelable_cors].apply(list)
        values = filtered_df[['pattern lists', PatternCols.reg_term]].apply(tuple, axis=1)
        result = dict(zip(filtered_df[PatternCols.id], values))
        return result

    def ideal_correlations(self):
        """ Returns dictionary of shape:
        keys: ids for the ideal correlation
        values: ideal correlation (to be used with loadings correlation method)
        """
        ids = self.ids_of_patterns_to_model()
        filtered_df = self.df[self.df[PatternCols.id].isin(ids)].copy()
        result = dict(zip(filtered_df[PatternCols.id], filtered_df[PatternCols.ideal_cors].apply(list)))
        return result

    def x_and_y_of_patterns_to_model(self):
        """ Returns feature matrix X and class label vector y. X has each correlation pair as column and
        y has the class label for each row in X. The ideal label and correlations are used.
        """
        patterns = self.ideal_correlations()
        y = np.array(list(patterns.keys()))
        x = np.array(list(patterns.values()))
        return x, y
