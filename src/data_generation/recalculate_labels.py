import ast

import numpy as np
import pandas as pd

from src.data_generation.generate_synthetic_correlated_data import calculate_spearman_correlation, \
    check_correlations_are_within_original_strength
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import SyntheticDataVariates


def recalculate_labels_df_from_data(data_df, labels_df):
    achieved_corrs = []
    within_tols = []
    # iterate through all segments and recalculate achieved correlation, keep data ordered by pattern_id
    for idx, row in labels_df.iterrows():
        start_idx = row[SyntheticDataSegmentCols.start_idx]
        end_idx = row[SyntheticDataSegmentCols.end_idx]
        length = row[SyntheticDataSegmentCols.length]

        # select data
        segment_df = data_df[SyntheticDataVariates.columns()].iloc[start_idx:end_idx + 1]
        segment = segment_df.to_numpy()
        assert segment.shape[0] == length, "Mistake with indexing dataframe"

        # calculated
        achieved_cor = calculate_spearman_correlation(segment)
        within_tol = check_correlations_are_within_original_strength(row[SyntheticDataSegmentCols.correlation_to_model],
                                                                     achieved_cor)

        # store results
        achieved_corrs.append(achieved_cor)
        within_tols.append(within_tol)

    # update df
    labels_df[SyntheticDataSegmentCols.actual_correlation] = achieved_corrs
    labels_df[SyntheticDataSegmentCols.actual_within_tolerance] = within_tols
    labels_df[SyntheticDataSegmentCols.mae] = sum_abs_error(labels_df)
    return labels_df


def sum_abs_error(labels_df: pd.DataFrame, round_to: int = 3):
    """Recalculate just the sum of absolute error between correlation to model and correlation achieved
    for each segment
    """
    n = len(labels_df.loc[0, SyntheticDataSegmentCols.correlation_to_model])
    canonical_pattern = np.array(labels_df[SyntheticDataSegmentCols.correlation_to_model].to_list())
    achieved_correlation = np.array(labels_df[SyntheticDataSegmentCols.actual_correlation].to_list())
    mean_absolute_error = np.round(np.sum(abs(canonical_pattern - achieved_correlation), axis=1) / n, round_to)
    return mean_absolute_error
