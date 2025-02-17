import os

import pandas as pd

from src.evaluation.describe_bad_partitions import DescribeBadPartitions
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, \
    get_internal_measures_summary_file_name, internal_measure_calculation_dir_for, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType


def read_clustering_quality_measures(overall_ds_name: str, data_type: str, root_results_dir: str, data_dir: str,
                                     distance_measure: str, run_names: [str]):
    """ Reads the summary results files from the internal measure calculation for the datasets in the overall_ds_name
    :param overall_ds_name: a name for the dataset we're using e.g. n30 or n2
    :param data_type: which datatype to use see SyntheticDataType
    :param root_results_dir: directory where to store the results, it will use a subdirectory based on
    the distance measure, and the data type and other information
    :param run_names: names of all the runs to read
    """
    results_dir = internal_measure_calculation_dir_for(overall_ds_name,
                                                       data_type,
                                                       root_results_dir,
                                                       data_dir,
                                                       distance_measure)

    results_for_run = []
    for run_name in run_names:
        file_name = get_internal_measures_summary_file_name(run_name)
        file_to_read = os.path.join(results_dir, file_name)
        df = pd.read_csv(str(file_to_read), index_col=0)
        # sort by Jaccard index
        df = df.sort_values(by=ClusteringQualityMeasures.jaccard_index)
        df.reset_index(drop=True, inplace=True)
        results_for_run.append(df)
    return results_for_run


def run_internal_measure_calculation_for_dataset(overall_ds_name: str, run_names: [str], distance_measure: str,
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
    :param internal_measures: list of internal measures to assess
    """
    store_results_in = internal_measure_calculation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type=data_type,
        results_dir=results_dir,
        data_dir=data_dir,
        distance_measure=distance_measure)
    partitions = []
    for ds_name in run_names:
        print(ds_name)
        # we don't vary the seed so all datasets will select the same clusters and segments
        sum_df = DescribeBadPartitions(ds_name=ds_name, distance_measure=distance_measure, data_type=data_type,
                                       internal_measures=internal_measures, data_dir=data_dir).summary_df.copy()
        file_name = get_internal_measures_summary_file_name(ds_name)
        sum_df.to_csv(str(os.path.join(store_results_in, file_name)))
        partitions.append(sum_df)


if __name__ == "__main__":
    overall_dataset_name = "n30"
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    distance_measures = [DistanceMeasures.l1_cor_dist, DistanceMeasures.l1_with_ref,
                         DistanceMeasures.foerstner_cor_dist]
    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi]
    data_types = [SyntheticDataType.raw, SyntheticDataType.normal_correlated,
                  SyntheticDataType.non_normal_correlated, SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR]
    # Config for L2 only ran for downsampled, complete data
    # distance_measures = [DistanceMeasures.l2_cor_dist]
    # data_types = [SyntheticDataType.rs_1min]
    # data_dirs = [SYNTHETIC_DATA_DIR]
    results_dir = ROOT_RESULTS_DIR

    for data_dir in data_dirs:
        for data_type in data_types:
            for distance_measure in distance_measures:
                print("Calculate Clustering Quality Measures for completeness:")
                print(data_dir)
                print("and data type: " + data_type)
                run_internal_measure_calculation_for_dataset(overall_dataset_name, run_names=run_names,
                                                             distance_measure=distance_measure, data_type=data_type,
                                                             data_dir=data_dir, results_dir=results_dir,
                                                             internal_measures=internal_measures)
