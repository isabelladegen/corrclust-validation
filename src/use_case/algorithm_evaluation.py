import pandas as pd

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols


class AlgorithmEvaluation:
    """
    This class can be used to evaluate the results of any algorithm and compare it to the ground truth
    """

    def __init__(self, result_labels_df: pd.DataFrame, gt_labels_df: pd.DataFrame, data: pd.DataFrame, run_name: str,
                 data_dir: str, data_type: str, round_to: int = 3):
        """
        :param result_labels_df: pandas dataframe containing the results of a clustering algorithm, expected columns
        are  SyntheticDataSegmentCols.segment_id, SyntheticDataSegmentCols.start_idx, SyntheticDataSegmentCols.end_idx,
            SyntheticDataSegmentCols.length, SyntheticDataSegmentCols.pattern_id,
            SyntheticDataSegmentCols.actual_correlation. Note that start_idx is 0 based and end_idx is expected to be
            selected!
        :param gt_labels_df: labels_file for matching ground truth
        :param data: data for matching ground truth and algorithms results
        :param run_name: subject name  the files are from
        :param data_dir: data completeness but given as a dir to allow for testing
        :param data_type: generation stage, see SyntheticDataType for options
        :param round_to: decimals results get rounded to
        """
        self._result_labels_df = result_labels_df
        self._gt_labels = gt_labels_df
        self._data = data
        self._run_name = run_name
        self._data_dir = data_dir
        self._data_type = data_type
        self._round_to = round_to

    def segmentation_ratio(self) -> float:
        """
        Calculates ratio between resulting number of segments and ground truth number of segments
        Interpretation:
        - 1 -> same number of segments
        - > 1 -> algorithm over segments (produces more segments than the ground truth has)
        - < <1 -> algorithm under segments
        :return: ratio
        """
        return round(self._result_labels_df.shape[0] / self._gt_labels.shape[0], self._round_to)

    def segmentation_length_ratio(self, stats: str = '50%'):
        """
        Calculates ratio between resulting stats length of segments and ground truth
        Interpretation:
        - 1 -> same segment length
        - > 1 -> algorithm produces longer segments
        - < <1 -> algorithm produces shorter segments
        :param stats: pandas describe stats to use for length, default median
        :return: ratio
        """
        results_stats = self._result_labels_df[SyntheticDataSegmentCols.length].describe()
        gt_stats = self._gt_labels[SyntheticDataSegmentCols.length].describe()
        return round(results_stats[stats] / gt_stats[stats], self._round_to)
