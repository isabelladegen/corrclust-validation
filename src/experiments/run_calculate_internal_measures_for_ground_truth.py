import os

import pandas as pd

from src.evaluation.calculate_internal_measures_ground_truth import CalculateInternalMeasuresGroundTruth
from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, \
    internal_measure_calculation_dir_for, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType


def read_ground_truth_clustering_quality_measures(overall_ds_name: str, data_type: str, root_results_dir: str,
                                                  data_dir: str,
                                                  distance_measure: str):
    """ Reads the summary results files from the internal measure calculation for the datasets in the overall_ds_name
    :param overall_ds_name: a name for the dataset we're using e.g. n30 or n2
    :param data_type: which datatype to use see SyntheticDataType
    :param root_results_dir: directory where to store the results, it will use a subdirectory based on
    the distance measure, and the data type and other information
    """
    result_dir = internal_measure_calculation_dir_for(overall_ds_name,
                                                      data_type,
                                                      root_results_dir,
                                                      data_dir,
                                                      distance_measure)
    file_name = os.path.join(result_dir, IAResultsCSV.internal_measures_for_ground_truth)
    df = pd.read_csv(str(file_name), index_col=0)
    return df


def run_internal_measure_calculation_for_ground_truth(overall_ds_name: str, run_names: [str], distance_measure: str,
                                                      data_type: str, data_dir: str, results_dir: str,
                                                      internal_measures: [str]):
    """
    Calculates the internal measure assessment for all ds in the csv files of the generated runs
    :param overall_ds_name: a name for the dataset we're using e.g. n30 or n2
    :param run_names: list of names for the runs to calculate the clustering quality measures for
    :param distance_measure: name of distance measure to run assessment for
    :param data_type: which datatype to use see SyntheticDataType
    :param data_dir: where to read the data from
    :param results_dir: directory where to store the results, it will use a subdirectory based on the distance measure,
    and the data type
    :param internal_measures: list of internal measures to calculate
    """
    store_results_in = internal_measure_calculation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type=data_type,
        results_dir=results_dir,
        data_dir=data_dir,
        distance_measure=distance_measure)
    gt = CalculateInternalMeasuresGroundTruth(run_names=run_names, internal_measures=internal_measures,
                                              distance_measure=distance_measure, data_type=data_type,
                                              data_dir=data_dir)

    df = gt.ground_truth_summary_df
    # store results
    df.to_csv(str(os.path.join(store_results_in, IAResultsCSV.internal_measures_for_ground_truth)))


if __name__ == "__main__":
    # Calculate the internal measure for all distance measures but only ground truth ds
    overall_dataset_name = "n30"
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    # all measures
    distance_measures = [DistanceMeasures.l1_cor_dist,  # lp norms
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.linf_cor_dist,
                         DistanceMeasures.l1_with_ref,  # lp norms with reference vector
                         DistanceMeasures.l2_with_ref,
                         DistanceMeasures.l3_with_ref,
                         DistanceMeasures.l5_with_ref,
                         DistanceMeasures.linf_with_ref,
                         DistanceMeasures.dot_transform_l1,  # dot transform + lp norms
                         DistanceMeasures.dot_transform_l2,
                         DistanceMeasures.dot_transform_linf,
                         DistanceMeasures.log_frob_cor_dist,  # correlation metrix
                         DistanceMeasures.foerstner_cor_dist]
    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi]
    data_types = [SyntheticDataType.raw, SyntheticDataType.normal_correlated,
                  SyntheticDataType.non_normal_correlated, SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR]
    results_dir = ROOT_RESULTS_DIR

    for data_dir in data_dirs:
        for data_type in data_types:
            for distance_measure in distance_measures:
                print("Calculate Ground truth Clustering Quality Measures for completeness:")
                print(data_dir)
                print("and data type: " + data_type)
                print("and distance measure: " + distance_measure)
                run_internal_measure_calculation_for_ground_truth(overall_dataset_name, run_names=run_names,
                                                                  distance_measure=distance_measure,
                                                                  data_type=data_type,
                                                                  data_dir=data_dir, results_dir=results_dir,
                                                                  internal_measures=internal_measures)
