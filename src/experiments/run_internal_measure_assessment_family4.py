import os

import pandas as pd

from src.evaluation.internal_measure_assessment import IAResultsCSV, InternalMeasureAssessment, InternalMeasureCols, \
    get_full_filename_for_results_csv
from src.experiments.run_cluster_quality_measures_calculation import read_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, internal_measure_evaluation_dir_for, \
    SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, get_data_completeness_from
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.stats import calculate_wilcox_signed_rank, StatsCols
from src.visualisation.run_average_rank_visualisations import data_variant_description


def run_wilcox_signed_rank_for(overall_ds_name: str, run_names: [str], distance_measure: str,
                               data_type: str, data_dir: str, results_dir: str, internal_measure1: str,
                               internal_measure2: str, alternative: str, non_zero: float, alpha: float,
                               bonferroni_adjust: int):
    """ Runs the internal measure assessment on all ds in the csv files of the generated runs
    :param overall_ds_name: a name for the dataset we're using e.g. n30 or n2
    :param run_names: list of run_names to load (subjects)
    :param distance_measure: name of distance measure to run assessment for
    :param data_type: which datatype to use see SyntheticDataType
    :param data_dir: where to read the data from
    :param results_dir: directory where to store the results, it will use a subdirectory based on the distance measure,
    and the data type
    :param internal_measure1: measure 1
    :param internal_measure2: measure 2
    :param alternative: which alternative to use to assess the measure
    :param non_zero: non-zero value to control what differences are considered 0
    :return wilcox result df
    """
    # load all the internal measure calculation summaries
    partitions = read_clustering_quality_measures(overall_ds_name=overall_ds_name, data_type=data_type,
                                                  root_results_dir=results_dir, data_dir=data_dir,
                                                  distance_measure=distance_measure, run_names=run_names)
    ia = InternalMeasureAssessment(distance_measure=distance_measure, dataset_results=partitions,
                                   internal_measures=[internal_measure1, internal_measure2], )

    # Wilcox signed rank
    df = ia.wilcoxon_signed_rank_tests_correlation_coefficients(alpha=alpha, alternative=alternative,
                                                                bonferroni_adjust=bonferroni_adjust, non_zero=non_zero)
    df = df.T  # rows are the two measures, columns are the stats
    variant_desc = data_variant_description[(get_data_completeness_from(data_dir), data_type)]
    df.insert(0, "Data Variant", variant_desc)

    return df


if __name__ == "__main__":
    overall_ds_name = "n30"
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]

    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    alternative = "two-sided"
    alpha = 0.05
    non_zero = 0.001
    bonferroni_adjust = 1

    distance_measure = DistanceMeasures.l5_cor_dist

    internal_measure1 = ClusteringQualityMeasures.silhouette_score
    internal_measure2 = ClusteringQualityMeasures.dbi

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    results = []

    # calculate for each data variant
    for data_dir in data_dirs:
        for data_type in dataset_types:
            print("Dataset type: " + data_type + ", Compactness: " + data_dir)
            df_for_variant = run_wilcox_signed_rank_for(overall_ds_name="n30", run_names=run_names,
                                                        distance_measure=distance_measure, data_type=data_type,
                                                        data_dir=data_dir, results_dir=root_result_dir,
                                                        internal_measure1=internal_measure1,
                                                        internal_measure2=internal_measure2,
                                                        alternative=alternative,
                                                        non_zero=non_zero,
                                                        alpha=alpha,
                                                        bonferroni_adjust=bonferroni_adjust)
            results.append(df_for_variant)

    # assemble results
    overall_df = pd.concat(results)
    overall_df = overall_df.reset_index(drop=True)
    overall_df.insert(1, 'Compared', internal_measure1 + " vs " + internal_measure2)

    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type='',  # all datatypes as rows
        results_dir=root_result_dir,
        data_dir='',  # all data data comp as rows
        distance_measure='')  # fixed to one

    # save stats results
    overall_df.to_csv(str(os.path.join(store_results_in, distance_measure + "_" + IAResultsCSV.family_4)))
