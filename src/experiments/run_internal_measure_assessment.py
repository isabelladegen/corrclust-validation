import pandas as pd

from src.evaluation.internal_measure_assessment import get_full_filename_for_results_csv, IAResultsCSV, \
    InternalMeasureAssessment
from src.experiments.run_cluster_quality_measures_calculation import read_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, internal_measure_evaluation_dir_for, \
    SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType


def run_internal_measure_assessment_datasets(overall_ds_name: str, run_names: [str], distance_measure: str,
                                             data_type: str, data_dir: str, results_dir: str, internal_measures: [str]):
    """ Runs the internal measure assessment on all ds in the csv files of the generated runs
    :param overall_ds_name: a name for the dataset we're using e.g. n30 or n2
    :param run_names: list of run_names to load (subjects)
    :param distance_measure: name of distance measure to run assessment for
    :param data_type: which datatype to use see SyntheticDataType
    :param data_dir: where to read the data from
    :param results_dir: directory where to store the results, it will use a subdirectory based on the distance measure,
    and the data type
    :param internal_measures: list of internal measures to assess
    on all the cluster
    using all segments
    """
    # load all the internal measure calculation summaries
    partitions = read_clustering_quality_measures(overall_ds_name=overall_ds_name, data_type=data_type,
                                                  root_results_dir=results_dir, data_dir=data_dir,
                                                  distance_measure=distance_measure, run_names=run_names)
    ia = InternalMeasureAssessment(distance_measure=distance_measure, dataset_results=partitions,
                                   internal_measures=internal_measures)
    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type=data_type,
        results_dir=results_dir, data_dir=data_dir,
        distance_measure=distance_measure)

    # correlation summary
    ia.correlation_summary.to_csv(
        get_full_filename_for_results_csv(store_results_in, IAResultsCSV.correlation_summary))

    # effect size between difference of mean correlation of worst and gt
    ia.differences_between_worst_and_best_partition().to_csv(
        get_full_filename_for_results_csv(store_results_in, IAResultsCSV.effect_size_difference_worst_best))

    # descriptive statistics
    ia.descriptive_statistics_for_internal_measures_correlation().to_csv(
        get_full_filename_for_results_csv(store_results_in, IAResultsCSV.descriptive_statistics_measure_summary))

    # 95% CI of differences in mean correlation between internal measures
    ia.ci_of_differences_between_internal_measure_correlations().to_csv(
        get_full_filename_for_results_csv(store_results_in, IAResultsCSV.ci_of_differences_between_measures))

    # paired samples t test on fisher transformed correlation coefficients
    df = ia.paired_samples_t_test_on_fisher_transformed_correlation_coefficients(alpha=0.05, alternative='two-sided')
    df.to_csv(get_full_filename_for_results_csv(store_results_in, IAResultsCSV.paired_t_test))


if __name__ == "__main__":
    overall_ds_name = "n30"
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.raw,
                     SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    distance_measures = [DistanceMeasures.l1_cor_dist,
                         DistanceMeasures.l1_with_ref,
                         DistanceMeasures.foerstner_cor_dist]

    # Config for L2 only ran for downsampled, complete data
    # distance_measures = [DistanceMeasures.l2_cor_dist]
    # dataset_types = [SyntheticDataType.rs_1min]
    # data_dirs = [SYNTHETIC_DATA_DIR]

    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.dbi, ClusteringQualityMeasures.vrc]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    for distance_measure in distance_measures:
        for data_dir in data_dirs:
            for data_type in dataset_types:
                print(
                    "Distance measure: " + distance_measure + " , Dataset type: " + data_type + ", Compactness: " + data_dir)
                run_internal_measure_assessment_datasets(overall_ds_name="n30", run_names=run_names,
                                                         distance_measure=distance_measure, data_type=data_type,
                                                         data_dir=data_dir, results_dir=root_result_dir,
                                                         internal_measures=internal_measures)
