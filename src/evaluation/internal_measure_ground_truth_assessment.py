from src.experiments.run_calculate_internal_measures_for_ground_truth import \
    read_ground_truth_clustering_quality_measures


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