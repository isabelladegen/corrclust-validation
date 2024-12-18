from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from itertools import cycle

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from tslearn.preprocessing import TimeSeriesScalerMinMax

from src.utils.configurations import GeneralisedCols, SyntheticDataVariates
from src.utils.plots.matplotlib_helper_functions import Backends
from src.data_generation.generate_synthetic_correlated_data import GenerateData, calculate_spearman_correlation, \
    check_correlations_are_within_original_strength


def min_max_scaled_df(df: pd.DataFrame, scale_range: (), columns: []) -> pd.DataFrame:
    df_ = df.copy()
    d = len(columns)
    sz = df.shape[0]
    # X : array-like of shape (n_ts, sz, d) -> (1, n_observation, n_variates)
    x = df[columns].to_numpy().reshape(1, sz, d)

    scaler = TimeSeriesScalerMinMax(value_range=scale_range)
    x_scaled = scaler.fit_transform(x)
    x_scaled = x_scaled.reshape(sz, d)  # reshape back into 2d array

    for idx, col in enumerate(columns):
        df_[col] = x_scaled[:, idx]

    return df_


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
    labels_df[SyntheticDataSegmentCols.mae] = mean_absolute_error_from_labels_df(labels_df)
    return labels_df


def mean_absolute_error_from_labels_df(labels_df: pd.DataFrame, round_to: int = 3):
    """Recalculate just the sum of absolute error between correlation to model and correlation achieved
    for each segment
    """
    n = len(labels_df.loc[0, SyntheticDataSegmentCols.correlation_to_model])
    canonical_pattern = np.array(labels_df[SyntheticDataSegmentCols.correlation_to_model].to_list())
    achieved_correlation = np.array(labels_df[SyntheticDataSegmentCols.actual_correlation].to_list())
    error = np.round(np.sum(abs(canonical_pattern - achieved_correlation), axis=1) / n, round_to)
    return error


def mean_absolute_error(canonical_patterns: [], achieved_correlations: [], round_to: int = 3):
    """
    Calculate the mean absolute error between canonical patterns and achieved correlations
    """
    n = len(canonical_patterns)
    error = np.round(
        np.sum(abs(np.array(canonical_patterns) - np.array(achieved_correlations)), axis=1) / n, round_to)
    return error


@dataclass
class SyntheticDataSegmentCols:  # todo rename to labels cols
    segment_id = "id"
    start_idx = "start idx"
    end_idx = "end idx"
    length = "length"
    pattern_id = "cluster_id"  # canonical pattern id
    correlation_to_model = "correlation to model"  # canonical pattern to model
    regularisation = "corr regularisation"  # todo move to dataset level
    actual_correlation = "correlation achieved"  # this is spearman correlation
    mae = "MAE"  # between canonical pattern and achieved correlation
    actual_within_tolerance = "correlation achieved with tolerance"
    distribution_to_model = 'distribution to model'  # todo move to dataset level
    distribution_args = 'distribution args'  # todo move to dataset level
    distribution_kwargs = 'distribution kwargs'  # todo move to dataset level
    repeats = 'repeated data generation'  # todo move to dataset level


class SyntheticSegmentedData:
    def __init__(self, n_segments: int, n_variates: int,
                 distributions_for_variates: [],
                 distributions_args: [], distributions_kwargs: [], short_segment_durations: [],
                 long_segment_durations: [], patterns_to_model: {}, variate_names: [], cor_method: str = "loadings"):
        self.start_time = datetime(2017, 6, 23, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        self.time_delta = timedelta(seconds=1)  # sample at 1 second by default
        self.distributions_for_variates = distributions_for_variates
        self.distributions_args = distributions_args
        self.distributions_kwargs = distributions_kwargs
        self.short_segment_durations = short_segment_durations
        self.n_draw_short = 4  # take n_draw_short short then n_draw_long than n_draw_short
        self.n_draw_long = 1
        # what method to use to create the correlations (loadings or cholesky)
        self.cor_method = cor_method
        # how many times we reattempt to generate a segment to avoid failures
        self.long_segment_durations = long_segment_durations
        self.patterns_to_model = patterns_to_model
        self.variate_names = variate_names
        self.n_variates = n_variates
        self.n_segments = n_segments
        self.non_normal_data_df = None  # that is the final AID like correlated df
        self.segment_data_generators: [GenerateData] = []
        self.non_normal_labels_df = None
        self.resampled_data = None
        self.resampled_labels_df = None

    def generate(self):
        self.non_normal_data_df = None
        self.segment_data_generators = []
        self.non_normal_labels_df = None
        generated_df = None

        # lists to store results
        segment_ids = []
        start_indices = []
        end_indices = []
        pattern_ids = []
        observation_count = []
        correlations_to_model = []
        regularisations = []  # only for cholesky
        actual_correlations = []
        actual_within_tols = []
        maes = []

        segment_start_idx = 0  # zero based indexing
        pattern_ids_to_model = list(self.patterns_to_model.keys())
        pattern_iter = cycle(pattern_ids_to_model)  # cycle through the patterns
        short_seg_iter = cycle(self.short_segment_durations)
        long_seg_iter = cycle(self.long_segment_durations)

        drawn_short = 0
        drawn_long = 0

        # create segments
        for segment_id in range(self.n_segments):
            repeated = 0
            cov_repeat = 0
            n_observations = 0
            # figure out segment length
            if drawn_short < self.n_draw_short:
                n_observations = next(short_seg_iter)
                drawn_short += 1
            elif drawn_long < self.n_draw_long:
                n_observations = next(long_seg_iter)
                drawn_long += 1
                # reset back to draw short ones
                if drawn_long == self.n_draw_long:
                    drawn_short = 0
                    drawn_long = 0

            pattern_id = next(pattern_iter)
            pattern = self.patterns_to_model[pattern_id]
            distributions = self.distributions_for_variates
            args = self.distributions_args
            kwargs = self.distributions_kwargs

            if isinstance(pattern, tuple):
                correlations = pattern[0]
                regularisation = pattern[1]
                # cholesky decomposition correlation needs regularisation term
                generator = GenerateData(n_observations, self.n_variates, correlations, distributions, args=args,
                                         kwargs=kwargs, covariance_regularisation=regularisation,
                                         method=self.cor_method)
            else:
                # loadings correlation data doesn't need regularisation term
                correlations = pattern
                regularisation = np.nan
                generator = GenerateData(n_observations, self.n_variates, correlations, distributions, args=args,
                                         kwargs=kwargs, method=self.cor_method)

            generator.generate()

            correlations_achieved = generator.achieved_correlations()
            within_tol = generator.check_if_achieved_correlation_is_within_original_strengths()

            df = pd.DataFrame(data=generator.generated_data, columns=self.variate_names)

            # safe results
            self.segment_data_generators.append(generator)

            # safe results for cluster segment df
            segment_end_idx = segment_start_idx + (n_observations - 1)  # -1 as 0 based
            segment_ids.append(segment_id)
            correlations_to_model.append(correlations)
            regularisations.append(regularisation)
            actual_correlations.append(correlations_achieved)
            actual_within_tols.append(within_tol)
            start_indices.append(segment_start_idx)
            end_indices.append(segment_end_idx)
            pattern_ids.append(pattern_id)
            observation_count.append(n_observations)
            segment_start_idx = segment_end_idx + 1  # next segment start idx

            # append data to generated_df
            if generated_df is None:
                generated_df = df
            else:
                generated_df = pd.concat([generated_df, df], axis=0)

        maes = mean_absolute_error(correlations_to_model, actual_correlations)
        segment_dict = {
            SyntheticDataSegmentCols.segment_id: segment_ids,
            SyntheticDataSegmentCols.start_idx: start_indices,
            SyntheticDataSegmentCols.end_idx: end_indices,
            SyntheticDataSegmentCols.length: observation_count,
            SyntheticDataSegmentCols.pattern_id: pattern_ids,
            SyntheticDataSegmentCols.correlation_to_model: correlations_to_model,
            SyntheticDataSegmentCols.actual_correlation: actual_correlations,
            SyntheticDataSegmentCols.actual_within_tolerance: actual_within_tols,
            SyntheticDataSegmentCols.mae: maes,
            SyntheticDataSegmentCols.regularisation: regularisations,
        }
        self.non_normal_labels_df = pd.DataFrame(segment_dict)

        generated_df.reset_index(drop=True, inplace=True)
        self.non_normal_data_df = self.__add_timestamp(generated_df)
        return self.non_normal_data_df

    def plot_distribution_for_segment(self, segment_id: int, backend=Backends.none.value):
        """ Segments id start with 0 to n_segments-1"""
        title = "PDF and histogram of data for segment " + str(segment_id)
        return self.segment_data_generators[segment_id].plot_pdf_and_histogram(title=title, backend=backend)

    def plot_correlation_matrix_for_segment(self, segment_id: int, backend=Backends.none.value):
        """ Segments id start with 0 to n_segments-1"""
        title = "Spearman correlation matrix for segment " + str(segment_id)
        self.segment_data_generators[segment_id].plot_correlation_matrix(title=title, backend=backend)

    def __add_timestamp(self, generated_df):
        times = [self.start_time]
        for x in range(generated_df.shape[0] - 1):
            times.append(times[x] + self.time_delta)

        # add datetime column
        generated_df.insert(loc=0, column=GeneralisedCols.datetime, value=times)
        generated_df[GeneralisedCols.datetime] = pd.to_datetime(generated_df[GeneralisedCols.datetime])
        return generated_df

    def resample(self, rule: str):
        """
        Downsample original generated data with provided rule. Attention this was tested for rule 1min,
        might not work for all rules

        :param rule: pandas resample rule
        :return: pd.Dataframe
        """
        resampled = self.non_normal_data_df.resample(rule, on=GeneralisedCols.datetime).mean()
        self.resampled_data = resampled

        # calculate new cluster segment df for resampled data
        reduction_in_frequency = self.non_normal_data_df.shape[0] / self.resampled_data.shape[0]
        resampled_labels_df = self.non_normal_labels_df.copy()
        resampled_labels_df[SyntheticDataSegmentCols.length] = resampled_labels_df[
            SyntheticDataSegmentCols.length].div(reduction_in_frequency).astype(int)
        new_start_indices = [0]
        new_end_indices = []
        for previous_idx in range(resampled_labels_df.shape[0] - 1):
            previous_length = resampled_labels_df.iloc[previous_idx].length
            previous_start_idx = new_start_indices[previous_idx]
            new_idx = previous_start_idx + previous_length
            new_start_indices.append(new_idx)
            new_end_indices.append(new_idx - 1)
        # add last end index
        new_end_indices.append(new_start_indices[-1] + resampled_labels_df.iloc[-1].length - 1)

        # update labels data
        resampled_labels_df[SyntheticDataSegmentCols.start_idx] = new_start_indices
        resampled_labels_df[SyntheticDataSegmentCols.end_idx] = new_end_indices

        # update actual correlation and correlation within tolerance
        resampled_labels_df = recalculate_labels_df_from_data(resampled, resampled_labels_df)
        self.resampled_labels_df = resampled_labels_df

        return resampled, resampled_labels_df

    def raw_generated_data_labels_df(self):
        """
        Returns the normal data before it got correlated and distribution shifted
        """
        normal_data = [g.normal_data for g in self.segment_data_generators]
        stacked_data = np.vstack(normal_data)
        df = self.non_normal_data_df.copy()
        df[self.variate_names] = stacked_data
        labels_df = self.non_normal_labels_df.copy()
        labels_df = recalculate_labels_df_from_data(df, labels_df)
        return df, labels_df

    def normal_correlated_generated_data_labels_df(self):
        """
        Returns the normal and correlated data before it got distribution shifted
        """
        normal_cor_data = [g.normal_correlated_data for g in self.segment_data_generators]
        stacked_data = np.vstack(normal_cor_data)
        df = self.non_normal_data_df.copy()
        df[self.variate_names] = stacked_data
        labels_df = self.non_normal_labels_df.copy()
        labels_df = recalculate_labels_df_from_data(df, labels_df)
        return df, labels_df
