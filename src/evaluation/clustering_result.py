import numpy as np
import pandas as pd

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.timeseries_utils import TimeColumns

delta_min_col = "delta min"
hours_of_day_col = TimeColumns.hour
months_col = TimeColumns.month
weekdays_col = TimeColumns.week_day
day_col = "Day"
year_col = TimeColumns.year
cluster_col = "Cluster"
segment_col = "Segment"
segment_length_col = "length"


class SegmentValueClusterResult:
    """ This is the "Segment Value Dataframe" that TICC logs from a clustering result and uses for evaluation.
    We can transform any other clustering result into such a df to then run the evaluations about the results.
    This is a convenience class to access the dataframe for evaluation
    """

    def __init__(self, segment_value_df: pd.DataFrame, original_observations_columns: [str],
                 scaled_observation_columns: [str]):
        """
        :param segment_value_df: a segment value df with columns: "datetime","iob mean","cob mean","bg mean","Cluster","Segment","xtrain iob mean","xtrain cob mean","xtrain bg mean",
        "delta min","hours","months","weekdays","Day","years"
        """
        self.time_cols = [delta_min_col, hours_of_day_col, months_col, weekdays_col, day_col, year_col]
        self.original_columns = original_observations_columns
        self.scaled_columns = scaled_observation_columns
        self.df = segment_value_df
        self.number_of_observations = self.df.shape[0]
        self.segment_lengths = self.__create_segment_length_df()

    def cluster_ids(self) -> list:
        return list(set(self.df[cluster_col]))

    def segment_ids(self) -> list:
        return list(set(self.df[segment_col]))

    def xtrain_with_original_column_names(self):
        x_train_df = self.df[self.scaled_columns]
        new_col_names = {}
        for idx, item in enumerate(self.original_columns):
            new_col_names[self.scaled_columns[idx]] = item
        x_train_df = x_train_df.rename(columns=new_col_names)
        return x_train_df

    def original_observations_df(self):
        return self.df[self.original_columns]

    def data_for_segment(self, segment_id):
        return self.df[self.df[segment_col] == segment_id]

    def data_for_cluster(self, cluster_id):
        return self.df[self.df[cluster_col] == cluster_id]

    def derivative_of_each_segment(self):
        """
        Calculates the derivative of each segment
        :return: pd.Dataframe of derivatives
        """
        derivative_df = self.df.copy()
        segments = self.segment_ids()
        ts_cols = self.original_columns + self.scaled_columns
        # calculate derivative of each segment separately (to deal with edges the same)
        for segment in segments:
            segment_ts = self.df[self.df[segment_col] == segment][ts_cols]
            derivatives = np.gradient(segment_ts, axis=0)
            derivative_df.loc[derivative_df[segment_col] == segment, ts_cols] = derivatives

        # rename the ts columns to derivative columns
        derivative_cols = to_derivative_col_names(ts_cols)
        rename_dict = {ts_cols[i]: derivative_cols[i] for i in range(len(ts_cols))}
        derivative_df.rename(columns=rename_dict, inplace=True)
        return derivative_df

    @classmethod
    def create_from_ticc_run(cls, original_df: pd.DataFrame, x_train: np.ndarray, cluster_assignments: [int],
                             columns: []):
        """
        :param original_df: dataframe with various columns beyond x_train
        :param x_train: ndarray of some of the columns of the original_df
        :param cluster_assignments: list of cluster assignment for each observation
        :param columns: columns of the original_df used to create x_train, assume column oder in columns and x_train
        match: columns[i] ==x_train[:,i]
        """
        # calculate segment ids
        segment_assignments = SegmentValueClusterResult.__calculate_segment_assignments(cluster_assignments)
        xtrain_columns = to_xtrain_col_names(columns)

        # check all data representation have the same number of samples/observations
        assert original_df.shape[0] == x_train.shape[0] == len(
            cluster_assignments) == len(segment_assignments), \
            "Different number of observations in in original df, x_train and cluster assignments"
        # check that x_train and columns are the right shape
        assert len(columns) == x_train.shape[1], "Columns in x_train and columns need to match but don't"
        # check that all columns are in original df
        assert set(columns).issubset(
            list(original_df.columns)), "Columns need to all be part of original_df, but are not"

        new_df = original_df.copy()  # keep all of the original columns
        new_df[cluster_col] = cluster_assignments
        new_df[segment_col] = segment_assignments
        # add both original observations and the observations from x_train which might have been e.g. min max scaled
        for cdx, col in enumerate(xtrain_columns):
            # col used for training
            new_df[col] = x_train[:, cdx]

        # add time information
        if isinstance(original_df.index, pd.DatetimeIndex):
            new_df[delta_min_col] = original_df.index.to_series().diff().values.astype(
                'timedelta64[m]').astype('int')
            # adjust the first time gap to the standard sampling_gap
            new_df.iloc[0, new_df.columns.get_loc(delta_min_col)] = 0
            new_df[hours_of_day_col] = original_df.index.hour
            new_df[months_col] = original_df.index.month
            new_df[weekdays_col] = original_df.index.weekday
            new_df[day_col] = original_df.index.day
            new_df[year_col] = original_df.index.year

        return SegmentValueClusterResult(new_df, columns, to_xtrain_col_names(columns))

    @classmethod
    def __calculate_segment_assignments(cls, cluster_assignments: [int]) -> []:
        """
        Assigns a segment id to each of the observations, similar to cluster_assignments starting with segment_id 0
        """
        result = []
        current_cluster = int(cluster_assignments[0])
        current_segment_number = 0

        for observation_index in range(len(cluster_assignments)):
            current_observations_cluster = int(cluster_assignments[observation_index])
            # increase segment number when cluster changes
            if current_observations_cluster != current_cluster:
                current_segment_number += 1
                current_cluster = current_observations_cluster
            result.append(current_segment_number)
        return result

    @classmethod
    def create_from_segment_df(cls, segment_cluster_result_df: pd.DataFrame, original_df: pd.DataFrame,
                               x_train: np.ndarray,
                               columns: []):
        """"
        :param segment_cluster_result_df: df with columns: start idx, end idx, cluster id, for each column sb cov
        :param original_df: dataframe of the observations' data with various columns beyond x_train. Row for each
        observation, can have a datetime
        :param x_train: ndarray of some of the columns of the original_df. Row for each observation, columns for each
        variate that was used to cluster
        :param columns: columns of the original_df used to create x_train, assume column oder in columns and x_train
        match: columns[i] ==x_train[:,i]
        """
        # need to remove datetimeindex as it will stop with removing -1 rows
        orig_df = original_df.copy()
        seg_clust_result = segment_cluster_result_df.copy()
        xtrain = np.copy(x_train)

        if isinstance(original_df.index, pd.DatetimeIndex):
            orig_df = original_df.reset_index(names='datetime')

        # remove observations for -1 cluster from all df
        last_row = segment_cluster_result_df.iloc[-1]
        if last_row[SyntheticDataSegmentCols.pattern_id] == -1:
            no_observations_to_delete = last_row[SyntheticDataSegmentCols.length]
            # drop last row from segment_custer_result
            seg_clust_result = segment_cluster_result_df.drop(index=segment_cluster_result_df.shape[0] - 1)
            # drop observations from x_train and original df
            for o in range(no_observations_to_delete):
                orig_df.drop(index=orig_df.shape[0] - 1, inplace=True)
                xtrain = np.delete(xtrain, obj=-1, axis=0)

        # calculate cluster assignments from segment_df
        cluster_assignments = []
        seg_clust_result.apply(lambda row: cluster_assignments.extend(
            [row[SyntheticDataSegmentCols.pattern_id]] * row[SyntheticDataSegmentCols.length]), axis=1)

        # set datetime index as expected
        if 'datetime' in orig_df.columns.values:
            orig_df['datetime'] = pd.to_datetime(orig_df['datetime'])
            orig_df = orig_df.set_index('datetime')

        return cls.create_from_ticc_run(orig_df, xtrain, cluster_assignments, columns)

    @classmethod
    def create_from_cpclust_result(cls, original_df, x_train, y_pred, columns):
        """ Used for Gaussian Mixtures - CPClust results"""
        assert original_df.shape[0] == x_train.shape[0] == y_pred.shape[0], \
            "original df, x_train and y_pred must all have the same number of observations"
        # calculate segment_ids
        seg_id = 0
        segment_ids = []
        current_cluster = y_pred[0]  # initialise with first cluster
        for obs_cluster_assignment in y_pred:
            # update current cluster and seg id on cluster change
            if obs_cluster_assignment != current_cluster:
                seg_id += 1
                current_cluster = obs_cluster_assignment
            # add seg_id to segment_ids for each observation
            segment_ids.append(seg_id)

        # create df with all information
        df = original_df.copy()
        df[cluster_col] = y_pred
        df[segment_col] = segment_ids

        scaled_cols = to_xtrain_col_names(columns)
        # add x_train to df
        for cdx, col in enumerate(scaled_cols):
            df[col] = x_train[:, cdx]

        # add date time information
        assert isinstance(df.index, pd.DatetimeIndex), "Original df didn't have datetime index"
        df[delta_min_col] = df.index.to_series().diff().values.astype(
            'timedelta64[m]').astype('int')
        # adjust the first time gap to the standard sampling_gap
        df.iloc[0, df.columns.get_loc(delta_min_col)] = 0
        df[hours_of_day_col] = df.index.hour
        df[months_col] = df.index.month
        df[weekdays_col] = df.index.weekday
        df[day_col] = df.index.day
        df[year_col] = df.index.year

        return SegmentValueClusterResult(df, columns, scaled_cols)

    @classmethod
    def create_from_equal_length_segments_and_y_pred(cls, df, x_train, y_pred, original_columns: [str]):
        """ Used for k-means"""
        y_pred_per_hour = np.repeat(y_pred, 24)
        assert len(y_pred_per_hour) == df.shape[
            0], "You may be using a sampling/segmentation that's not yet implemented for result logging"
        df[cluster_col] = y_pred_per_hour
        df[segment_col] = np.repeat(list(range(len(y_pred))), 24)
        scaled_cols = to_xtrain_col_names(original_columns)
        # reshape scaled columns back to fit with pandas
        for cdx, col in enumerate(scaled_cols):
            df[col] = np.reshape(x_train[:, :, cdx], -1)

        # add time information
        if isinstance(df.index, pd.DatetimeIndex):
            df[delta_min_col] = df.index.to_series().diff().values.astype(
                'timedelta64[m]').astype('int')
            # adjust the first time gap to the standard sampling_gap
            df.iloc[0, df.columns.get_loc(delta_min_col)] = 0
            df[hours_of_day_col] = df.index.hour
            df[months_col] = df.index.month
            df[weekdays_col] = df.index.weekday
            df[day_col] = df.index.day
            df[year_col] = df.index.year

        return SegmentValueClusterResult(df, original_columns, to_xtrain_col_names(original_columns))

    @classmethod
    def create_from_segment_values_results_df(cls, segment_values_df: pd.DataFrame, original_columns: [str]):
        if not isinstance(segment_values_df.index, pd.DatetimeIndex):
            segment_values_df['datetime'] = pd.to_datetime(segment_values_df['datetime'])
            segment_values_df = segment_values_df.set_index('datetime')
        return SegmentValueClusterResult(segment_values_df, original_columns, to_xtrain_col_names(original_columns))

    def number_of_segments(self):
        return self.segment_lengths.shape[0]

    def min_segment_length(self):
        return int(self.segment_lengths.min(axis=0)[segment_length_col])

    def mean_segment_length(self):
        return np.around(self.segment_lengths.mean(axis=0)[segment_length_col], decimals=3)

    def max_segment_length(self):
        return int(self.segment_lengths.max(axis=0)[segment_length_col])

    def __create_segment_length_df(self):
        clusters = []
        segment_ids = []
        segment_lengths = []

        for segment_id in self.segment_ids():
            segment = self.df[self.df[segment_col] == segment_id]
            segment_lengths.append(segment.shape[0])
            segment_ids.append(segment_id)
            clusters.append(segment[cluster_col].values[0])

        return pd.DataFrame({segment_col: segment_ids, cluster_col: clusters, segment_length_col: segment_lengths})

    def number_of_time_each_cluster_is_used(self) -> pd.Series:
        return self.segment_lengths[cluster_col].value_counts()


def to_xtrain_col_names(cols: []) -> []:
    """ Returns column names as xtrain column names"""
    return ["xtrain " + str(col) for col in cols]


def to_derivative_col_names(cols: []) -> []:
    """ Returns column names as derivative column names"""
    return [str(col) + " dt" for col in cols]
