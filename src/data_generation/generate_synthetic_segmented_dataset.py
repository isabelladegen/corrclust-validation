from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from itertools import cycle

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from tslearn.preprocessing import TimeSeriesScalerMinMax

from src.utils.configurations import GeneralisedCols
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


@dataclass
class SyntheticDataSegmentCols:
    segment_id = "id"
    start_idx = "start idx"
    end_idx = "end idx"
    length = "length"
    pattern_id = "cluster_id"  # correlation pattern id
    correlation_to_model = "correlation to model"
    regularisation = "cov regularisation"
    actual_correlation = "correlation achieved"  # this is spearman correlation
    actual_within_tolerance = "correlation achieved with tolerance"
    distribution_to_model = 'distribution to model'
    distribution_args = 'distribution args'
    distribution_kwargs = 'distribution kwargs'
    repeats = 'repeated data generation'


class SyntheticSegmentedData:
    def __init__(self, n_segments: int, n_variates: int,
                 distributions_for_variates: [],
                 distributions_args: [], distributions_kwargs: [], short_segment_durations: [],
                 long_segment_durations: [], patterns_to_model: {}, variate_names: [], cor_method: str = "loadings",
                 max_repetitions: int = 100):
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
        self.max_repeats = max_repetitions
        self.long_segment_durations = long_segment_durations
        self.patterns_to_model = patterns_to_model
        self.variate_names = variate_names
        self.n_variates = n_variates
        self.n_segments = n_segments
        self.generated_df = None  # that is the final correlated df
        self.segment_data_generators: [GenerateData] = []
        self.generated_segment_df = None  # similar structure like for relclust result
        self.resampled_data = None
        self.resampled_segment_df = None

    def generate(self):
        self.generated_df = None
        self.segment_data_generators = []
        self.generated_segment_df = None
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
        distributions_to_model = []
        distributions_args = []
        distributiosn_kwargs = []
        repeats = []

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

            # repeat generation infinitely to get positive definite cov and max_repeat times for correlation
            correlations_achieved = None
            within_tol = None
            while repeated < self.max_repeats:
                while cov_repeat < self.max_repeats:
                    try:
                        generator.generate()
                        cov_repeat += 1
                    except LinAlgError:
                        # usually happens if cov not positive definite
                        # this will run forever if we cannot get a positiv definite matrix
                        repeated = 0
                        print("Generation failed, reattempting to generate segment with id " + str(segment_id))

                correlations_achieved = generator.achieved_correlations()
                within_tol = generator.check_if_achieved_correlation_is_within_original_strengths()
                if all(within_tol):
                    break
                else:
                    repeated += 1

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
            distributions_to_model.append([dist.name for dist in distributions])
            distributions_args.append(args)
            distributiosn_kwargs.append(kwargs)
            start_indices.append(segment_start_idx)
            end_indices.append(segment_end_idx)
            pattern_ids.append(pattern_id)
            observation_count.append(n_observations)
            repeats.append(repeated)
            segment_start_idx = segment_end_idx + 1  # next segment start idx

            # append data to generated_df
            if generated_df is None:
                generated_df = df
            else:
                generated_df = pd.concat([generated_df, df], axis=0)

        segment_dict = {
            SyntheticDataSegmentCols.segment_id: segment_ids,
            SyntheticDataSegmentCols.start_idx: start_indices,
            SyntheticDataSegmentCols.end_idx: end_indices,
            SyntheticDataSegmentCols.length: observation_count,
            SyntheticDataSegmentCols.pattern_id: pattern_ids,
            SyntheticDataSegmentCols.correlation_to_model: correlations_to_model,
            SyntheticDataSegmentCols.actual_correlation: actual_correlations,
            SyntheticDataSegmentCols.actual_within_tolerance: actual_within_tols,
            SyntheticDataSegmentCols.regularisation: regularisations,
            SyntheticDataSegmentCols.distribution_to_model: distributions_to_model,
            SyntheticDataSegmentCols.distribution_args: distributions_args,
            SyntheticDataSegmentCols.distribution_kwargs: distributiosn_kwargs,
            SyntheticDataSegmentCols.repeats: repeats,
        }
        self.generated_segment_df = pd.DataFrame(segment_dict)  # segment id is the index of this df

        generated_df.reset_index(drop=True, inplace=True)
        self.generated_df = self.__add_timestamp(generated_df)
        return self.generated_df

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
        resampled = self.generated_df.resample(rule, on=GeneralisedCols.datetime).mean()
        self.resampled_data = resampled

        # calculate new cluster segment df for resampled data
        reduction_in_frequency = self.generated_df.shape[0] / self.resampled_data.shape[0]
        resampled_segment_df = self.generated_segment_df.copy()
        resampled_segment_df[SyntheticDataSegmentCols.length] = resampled_segment_df[
            SyntheticDataSegmentCols.length].div(reduction_in_frequency).astype(int)
        new_start_indices = [0]
        new_end_indices = []
        for previous_idx in range(resampled_segment_df.shape[0] - 1):
            previous_length = resampled_segment_df.iloc[previous_idx].length
            previous_start_idx = new_start_indices[previous_idx]
            new_idx = previous_start_idx + previous_length
            new_start_indices.append(new_idx)
            new_end_indices.append(new_idx - 1)
        # add last end index
        new_end_indices.append(new_start_indices[-1] + resampled_segment_df.iloc[-1].length - 1)
        # update dataframe
        resampled_segment_df[SyntheticDataSegmentCols.start_idx] = new_start_indices
        resampled_segment_df[SyntheticDataSegmentCols.end_idx] = new_end_indices

        # update actual correlation and correlation within tolerance
        new_actual_correlations = []
        new_cors_within_tol = []
        for row in resampled_segment_df.iterrows():
            start_idx = row[1][SyntheticDataSegmentCols.start_idx]
            end_idx = row[1][SyntheticDataSegmentCols.end_idx]

            data = resampled.iloc[start_idx:end_idx + 1].to_numpy()
            correlations = calculate_spearman_correlation(data)

            original_cor = row[1][SyntheticDataSegmentCols.correlation_to_model]
            within_tol = check_correlations_are_within_original_strength(original_cor, correlations)

            new_actual_correlations.append(correlations)
            new_cors_within_tol.append(within_tol)

        resampled_segment_df[SyntheticDataSegmentCols.actual_correlation] = new_actual_correlations
        resampled_segment_df[SyntheticDataSegmentCols.actual_within_tolerance] = new_cors_within_tol

        self.resampled_segment_df = resampled_segment_df

        return resampled, resampled_segment_df

    def normal_generated_df(self):
        """ Returns the normal data before it got correlated and distribution shifted"""
        normal_data = [g.normal_data for g in self.segment_data_generators]
        stacked_data = np.vstack(normal_data)
        df = self.generated_df.copy()
        df[self.variate_names] = stacked_data
        return df

    def normal_correlated_generated_df(self):
        """ Returns the normal and correlated data before it got distribution shifted"""
        normal_cor_data = [g.normal_correlated_data for g in self.segment_data_generators]
        stacked_data = np.vstack(normal_cor_data)
        df = self.generated_df.copy()
        df[self.variate_names] = stacked_data
        return df
