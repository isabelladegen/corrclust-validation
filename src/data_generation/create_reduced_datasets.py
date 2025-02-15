import random

import pandas as pd

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import SyntheticDataVariates
from src.utils.load_synthetic_data import load_synthetic_data


class CreateReducedDatasets:
    def __init__(self, run_names: [str], data_type: str, data_dir: str, drop_n_clusters: [int],
                 drop_n_segments: [int], base_seed: int = 600,
                 round_to: int = 3):
        self.run_names = run_names
        self.drop_n_segments = drop_n_segments
        self.drop_n_clusters = drop_n_clusters
        self.data_dir = data_dir
        self.data_type = data_type
        self.base_seed = base_seed
        self.round_to = round_to
        self.orig_data = {}
        self.orig_labels = {}
        # dictionary of dictionary, first level with key n_drop, second level with key run_name and value either labels
        # or data, or lists of patterns or segment
        self.selected_patterns = {drop_n: {} for drop_n in drop_n_clusters}
        self.selected_segments = {drop_n: {} for drop_n in drop_n_segments}

        self.reduced_labels_patterns = {drop_n: {} for drop_n in drop_n_clusters}
        self.reduced_data_patterns = {drop_n: {} for drop_n in drop_n_clusters}

        self.reduced_labels_segments = {drop_n: {} for drop_n in drop_n_segments}
        self.reduced_data_segments = {drop_n: {} for drop_n in drop_n_segments}

        # load all original data
        for run_name in run_names:
            data, labels = load_synthetic_data(run_name, self.data_type, data_dir=self.data_dir)
            self.orig_data[run_name] = data
            self.orig_labels[run_name] = labels

        a_label_file = self.orig_labels[run_names[0]]
        self.patterns = list(a_label_file[SyntheticDataSegmentCols.pattern_id].unique())
        self.segment_ids = list(a_label_file[SyntheticDataSegmentCols.segment_id].unique())

        n_patterns_orig = len(self.patterns)
        n_segments_orig = a_label_file.shape[0]

        # check valid clusters to drop:
        for drop in drop_n_clusters:
            assert n_patterns_orig - drop >= 2, "You need to keep at least two patterns"

        # check valid segments to drop:
        for drop in drop_n_segments:
            assert n_segments_orig - drop >= 4, "You need to keep at least four segments"

        # set random to base_seed
        random.seed(self.base_seed)

        # drop clusters
        for drop_n in drop_n_clusters:
            n_clusters = n_patterns_orig - drop_n
            for run_name in run_names:
                # select different patterns to keep for each run
                selected_patterns = random.sample(self.patterns, n_clusters)
                reduced_labels, reduced_data = self.__select_segments_and_patterns_from(self.orig_data[run_name],
                                                                                        self.orig_labels[run_name],
                                                                                        selected_patterns,
                                                                                        self.segment_ids)
                self.reduced_labels_patterns[drop_n][run_name] = reduced_labels
                self.reduced_data_patterns[drop_n][run_name] = reduced_data
                self.selected_patterns[drop_n][run_name] = selected_patterns

        for drop_n in drop_n_segments:
            n_segment = n_segments_orig - drop_n
            for run_name in run_names:
                selected_segs = random.sample(self.segment_ids, n_segment)
                reduced_labels, reduced_data = self.__select_segments_and_patterns_from(self.orig_data[run_name],
                                                                                        self.orig_labels[run_name],
                                                                                        self.patterns,
                                                                                        selected_segs)
                self.reduced_labels_segments[drop_n][run_name] = reduced_labels
                self.reduced_data_segments[drop_n][run_name] = reduced_data
                self.selected_segments[drop_n][run_name] = selected_segs

    def __select_segments_and_patterns_from(self, data, labels, selected_patterns, selected_segs):
        """
        This method updates all the labels df and data df making them look like they were generated this way to
        work with all other analysis methods
        :param data: pd.DataFrame of timeseries data (observations)
        :param labels: pd.DataFrame of labels
        :param selected_patterns: patterns to keep
        :param selected_segs: segments to keep
        :return: pd.DataFrame for labels, pd.DataFrame for data
        """
        reduced_labels_df = labels.copy()
        # first select patterns
        if len(selected_patterns) < len(self.patterns):
            reduced_labels_df = reduced_labels_df[
                reduced_labels_df[SyntheticDataSegmentCols.pattern_id].isin(selected_patterns)]
        # select the segments
        elif len(selected_segs) < len(self.segment_ids):
            reduced_labels_df = reduced_labels_df[
                reduced_labels_df[SyntheticDataSegmentCols.segment_id].isin(selected_segs)]
        reduced_labels_df = reduced_labels_df.reset_index(drop=True)

        # update the data
        keep_data = []
        select_start_end_idx = list(zip(reduced_labels_df[SyntheticDataSegmentCols.start_idx],
                                        reduced_labels_df[SyntheticDataSegmentCols.end_idx]))
        new_start_idx = new_end_idx = 0
        for idx, (start_idx, end_idx) in enumerate(select_start_end_idx):
            seg_data = data.iloc[start_idx:end_idx + 1, :]
            # check selection
            length = reduced_labels_df.iloc[idx][SyntheticDataSegmentCols.length]
            assert seg_data.shape[0] == length, "Selected wrong indices"

            # update indices in labels to be correct with new data
            new_end_idx = new_start_idx + (length - 1)  # zero based indexing
            reduced_labels_df.iloc[idx][SyntheticDataSegmentCols.start_idx] = new_start_idx
            reduced_labels_df.iloc[idx][SyntheticDataSegmentCols.end_idx] = new_end_idx
            new_start_idx = new_end_idx + 1

            # safe data to list
            keep_data.append(seg_data)

        # update index to be continuous
        reduced_data = pd.concat(keep_data, axis=0).reset_index(drop=True)

        return reduced_labels_df, reduced_data
