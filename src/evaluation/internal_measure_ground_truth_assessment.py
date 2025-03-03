import pandas as pd

from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.experiments.run_calculate_internal_measures_for_ground_truth import \
    read_ground_truth_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures

# value for ascending, if true lowest value will get rank 1, if falls highest value will get rank 1
internal_measure_ranking_method = {
    ClusteringQualityMeasures.silhouette_score: False,  # higher is better
    ClusteringQualityMeasures.dbi: True,  # lower is better
    ClusteringQualityMeasures.vrc: False,  # higher is better
    ClusteringQualityMeasures.pmb: False,  # higher is better
}


class InternalMeasureGroundTruthAssessment:
    """Calculates per data variant and per internal measure which distance measure works best and worst"""

    def __init__(self, overall_ds_name: str, internal_measures: [str], distance_measures: [str], data_type: str,
                 data_dir: str, root_results_dir: str, round_to: int = 3):
        self.overall_ds_name = overall_ds_name
        self.distance_measures = distance_measures
        self.internal_measures = internal_measures
        self.data_dir = data_dir
        self.data_type = data_type
        self.root_results_dir = root_results_dir
        self.round_to = round_to
        # key=distance measure, value df of ground truth calculation
        self.ground_truth_calculation_dfs = {}

        for distance_measure in distance_measures:
            # read all internal measures for a distance measure
            per_distance_measure_df = read_ground_truth_clustering_quality_measures(
                overall_ds_name=self.overall_ds_name,
                data_type=self.data_type,
                root_results_dir=self.root_results_dir,
                data_dir=self.data_dir, distance_measure=distance_measure)
            self.ground_truth_calculation_dfs[distance_measure] = per_distance_measure_df

    def rank_distance_measures_for_each_internal_measure(self):
        """
        Ranks each distance measure for each internal measure and returns dictionary of ranks keyed
        by internal measure name
        :return dictionary{key=internal measure name: values= df with rows=run-names, columns= distance measures,
        cells=rank for that distance measure for that run
        """
        ranked_dict = {}
        raw_scores = self.raw_scores_for_each_internal_measure()

        for internal_measure, df in raw_scores.items():
            # Rank across columns (axis=1) for each row
            ranked_df = df.rank(axis=1, method='dense', ascending=internal_measure_ranking_method[internal_measure])
            ranked_dict[internal_measure] = ranked_df
        return ranked_dict

    def raw_scores_for_each_internal_measure(self):
        """
        Reshapes data into dictionary of raw score per internal measure
        :return dictionary{key=internal measure name: values= df with rows=run-names, columns=distance measures, values=
        internal measure score for that distance measure and run}
        """
        result_dict = {}
        run_names = self.ground_truth_calculation_dfs[self.distance_measures[0]][DescribeBadPartCols.name].to_list()

        for measure in self.internal_measures:
            # Create empty dataframe with runs as index
            measure_df = pd.DataFrame(index=run_names)
            # Fill in values for each distance measure
            for distance_measure, df in self.ground_truth_calculation_dfs.items():
                if measure in df.columns:
                    # Set the run name as index for easier merging
                    temp_df = df.set_index(DescribeBadPartCols.name)
                    measure_df[distance_measure] = temp_df[measure]

            result_dict[measure] = measure_df

        return result_dict

    def stats_for_ranks_across_all_runs(self):
        stats_results = {}
        ranks = self.rank_distance_measures_for_each_internal_measure()
        for measure in self.internal_measures:
            stats_results[measure] = ranks[measure].describe().round(self.round_to)
        return stats_results

    def stats_for_raw_values_across_all_runs(self):
        stats_results = {}
        raw_values = self.raw_scores_for_each_internal_measure()
        for measure in self.internal_measures:
            stats_results[measure] = raw_values[measure].describe().round(self.round_to)
        return stats_results

    def grouping_for_each_internal_measure(self, stats_value: str):
        """
        Calculates the grouping of distance measures per internal measure. Lower groups
        are better performing distance measures, higher worse.
        :param stats_value: which rank stats value to use, e.g. 50% = median ranks across the n=30 pairs
        :return:
        """
        rank_stats = self.stats_for_ranks_across_all_runs()
        groupings = {}
        for measure in self.internal_measures:
            ranks = rank_stats[measure].loc[stats_value]
            unique_ranks = sorted(ranks.unique())

            # Create a mapping from actual rank to group numbers (1, 2, 3...)
            rank_to_group = {rank: i + 1 for i, rank in enumerate(unique_ranks)}

            # Group indices (distance measure) by group numbers
            result = {}
            for distance_measure, rank in ranks.items():
                group_num = rank_to_group[rank]
                if group_num not in result:
                    result[group_num] = []
                result[group_num].append(distance_measure)

            # Sort the dictionary by keys
            result = dict(sorted(result.items()))
            groupings[measure] = result
        return groupings

