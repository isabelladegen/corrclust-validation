import os

import pandas as pd

from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.evaluation.internal_measure_ground_truth_assessment import InternalMeasureGroundTruthAssessment
from src.experiments.run_internal_indices_family2_validation import \
    rank_distance_measures_by_raw_values_run_wilcox_signed_rank_tests
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, internal_measure_evaluation_dir_for, get_data_completeness_from, VALID_ROOT_RESULTS_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.run_average_rank_visualisations import data_variant_description

if __name__ == "__main__":
    # Calculates which distance measures are significantly better than others for given internal measures
    backend = Backends.none.value  #
    save_fig = False
    save_results_dir = VALID_ROOT_RESULTS_DIR
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]
    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi]

    # valid
    distance_measures = [DistanceMeasures.l1_cor_dist,
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.dot_transform_l2
                         ]

    rank_distance_measures_by_raw_values_run_wilcox_signed_rank_tests(data_dirs=data_dirs, dataset_types=dataset_types,
                                                                      overall_ds_name="n30",
                                                                      root_results_dir=root_result_dir,
                                                                      distance_measures=distance_measures,
                                                                      internal_measures=internal_measures,
                                                                      alpha=0.05,
                                                                      alternative="two-sided",
                                                                      bf_adjust=1,
                                                                      non_zero=0.0001,
                                                                      save_results_dir=save_results_dir
                                                                      )
