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


class SegmentValueClusterResult:  # TODO rename to segmented and clustered data as it is the data_df with segment id
    # cluster id col
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
    def create_from_labels_and_data_dfs(cls, labels_df: pd.DataFrame, data_df: pd.DataFrame, data_columns: []):
        """"
        :param labels_df: df with columns: start idx, end idx, cluster id, for each column sb cov
        :param data_df: dataframe of the observations' data with various with datetime columns
        :param data_columns: data column names
        """
        # need to remove datetimeindex as it will stop with removing -1 rows
        orig_df = data_df.copy()
        seg_clust_result = labels_df.copy()
        xtrain = orig_df.data[data_columns].to_numpy()

        if isinstance(data_df.index, pd.DatetimeIndex):
            orig_df = data_df.reset_index(names='datetime')

        # remove observations for -1 cluster from all df
        last_row = labels_df.iloc[-1]
        if last_row[SyntheticDataSegmentCols.pattern_id] == -1:
            no_observations_to_delete = last_row[SyntheticDataSegmentCols.length]
            # drop last row from segment_custer_result
            seg_clust_result = labels_df.drop(index=labels_df.shape[0] - 1)
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

        return cls.create_from_ticc_run(orig_df, xtrain, cluster_assignments, data_columns)

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
