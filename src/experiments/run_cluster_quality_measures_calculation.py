import os

import pandas as pd

from src.evaluation.describe_bad_partitions import DescribeBadPartitions
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, \
    get_internal_measures_summary_file_name, internal_measure_calculation_dir_for, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data_and_labels_for_bad_partitions


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


def run_internal_measure_calculation_for_dataset(overall_ds_name: str, data_dict: [str],
                                                 gt_labels_dict: [str], partitions_dict: [str], distance_measure: str,
                                                 data_type: str, data_dir: str, results_dir: str,
                                                 internal_measures: [str]):
    """
    Calculates the internal measure assessment for all ds in the csv files of the generated runs
    :param overall_ds_name: a name for the dataset we're using e.g. n30 or n2
    :param data_dict: a dictionary with key run_id and value data for that run id
    :param gt_labels_dict: a dictionary with key run_id and value gt labels for that run id
    :param partitions_dict: a dictionary with key run_id and value partitions dict for that run id
    :param run_names: list of names for the runs to calculate the clustering quality measures for
    :param distance_measure: name of distance measure to run assessment for
    :param data_type: which datatype to use see SyntheticDataType
    :param data_dir: where to read the data from
    :param results_dir: directory where to store the results, it will use a subdirectory based on the distance measure,
    and the data type
    :param internal_measures: list of internal measures to assess
    """
    assert len(data_dict) == len(gt_labels_dict) == len(partitions_dict), "Inconsistent run ids for data, labels and partitions"
    store_results_in = internal_measure_calculation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type=data_type,
        results_dir=results_dir,
        data_dir=data_dir,
        distance_measure=distance_measure)
    for run_id in data_dict.keys():
        print(run_id)
        # give preloaded data to only load once per data variant for each distance measure
        sum_df = DescribeBadPartitions(ds_name=run_id,
                                       distance_measure=distance_measure,
                                       data_type=data_type,
                                       internal_measures=internal_measures,
                                       data_dir=data_dir,
                                       data=data_dict[run_id],
                                       gt_label=gt_labels_dict[run_id],
                                       partitions=partitions_dict[run_id]).summary_df.copy()
        file_name = get_internal_measures_summary_file_name(run_id)
        sum_df.to_csv(str(os.path.join(store_results_in, file_name)))


def load_all_clustering_data_for_subjects_and_data_type(run_ids: str, data_type: str, data_dir: str):
    """ Load all data for each data variant only ones to save time """
    data_dict = {}
    gt_labels_dict = {}
    partitions_dict = {}
    for run_id in run_ids:
        # load data, ground truth labels and all bad clusterings for given ds_name
        data, gt_label, partitions = load_synthetic_data_and_labels_for_bad_partitions(run_id,
                                                                                       data_type=data_type,
                                                                                       data_dir=data_dir)
        data_dict[run_id] = data
        gt_labels_dict[run_id] = gt_label
        partitions_dict[run_id] = partitions

    return data_dict, gt_labels_dict, partitions_dict


if __name__ == "__main__":
    overall_dataset_name = "n30"
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    # run on 15th May 25
    distance_measures = [DistanceMeasures.l1_cor_dist,
                         DistanceMeasures.l1_with_ref,
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.l5_with_ref,
                         DistanceMeasures.linf_cor_dist,
                         DistanceMeasures.dot_transform_linf,
                         DistanceMeasures.log_frob_cor_dist,
                         DistanceMeasures.foerstner_cor_dist
                         ]
    # distance_measures = [DistanceMeasures.dot_transform_linf,
    #                      DistanceMeasures.log_frob_cor_dist,
    #                      DistanceMeasures.foerstner_cor_dist]
    # data_types = [SyntheticDataType.rs_1min]

    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi]
    data_types = [SyntheticDataType.raw, SyntheticDataType.normal_correlated,
                  SyntheticDataType.non_normal_correlated, SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR]
    # Config for L2 only ran for downsampled, complete data
    # distance_measures = [DistanceMeasures.l2_cor_dist]
    # data_types = [SyntheticDataType.rs_1min]
    # data_dirs = [SYNTHETIC_DATA_DIR]
    # Config for L3 only ran for downsampled, complete data
    # distance_measures = [DistanceMeasures.l3_cor_dist]
    # data_types = [SyntheticDataType.rs_1min]
    # data_dirs = [SYNTHETIC_DATA_DIR]
    results_dir = ROOT_RESULTS_DIR

    for data_dir in data_dirs:
        for data_type in data_types:
            print(f"Load all data for data type {data_type} and data dir {data_dir}")
            data_dict, gt_labels_dict, partitions_dict = load_all_clustering_data_for_subjects_and_data_type(
                run_ids=run_names, data_type=data_type, data_dir=data_dir)
            for distance_measure in distance_measures:
                print(f"Calculate Clustering Quality for distance measure {distance_measure}")
                run_internal_measure_calculation_for_dataset(overall_dataset_name, data_dict=data_dict,
                                                             gt_labels_dict=gt_labels_dict,
                                                             partitions_dict=partitions_dict,
                                                             distance_measure=distance_measure, data_type=data_type,
                                                             data_dir=data_dir, results_dir=results_dir,
                                                             internal_measures=internal_measures)
