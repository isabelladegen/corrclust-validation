import pandas as pd

from src.evaluation.internal_measure_assessment import get_full_filename_for_results_csv, IAResultsCSV, \
    InternalMeasureAssessment
from src.evaluation.run_cluster_quality_measures_calculation import read_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, internal_measure_evaluation_dir_for, \
    SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType


def assess_internal_measures(overall_dataset_name: str, run_names: [str], data_type: str,
                             root_results_dir: str, data_dir: str,
                             distance_measure: str,
                             internal_measures: [str], n_clusters=0, n_segments=0):
    # load all the internal measure calculation summaries
    partitions = read_clustering_quality_measures(overall_ds_name=overall_dataset_name,
                                                  data_type=data_type,
                                                  root_results_dir=root_results_dir,
                                                  data_dir=data_dir,
                                                  distance_measure=distance_measure,
                                                  n_dropped_clusters=n_clusters,
                                                  n_dropped_segments=n_segments,
                                                  run_names=run_names)

    ia = InternalMeasureAssessment(distance_measure=distance_measure, dataset_results=partitions,
                                   internal_measures=internal_measures)

    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_dataset_name,
        data_type=data_type,
        results_dir=root_results_dir, data_dir=data_dir,
        distance_measure=distance_measure,
        drop_segments=n_segments, drop_clusters=n_clusters)

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


def run_internal_measure_assessment_datasets(overall_ds_name: str,
                                             run_names: [str],
                                             distance_measure: str = DistanceMeasures.l1_cor_dist,
                                             data_type: str = SyntheticDataType.non_normal_correlated,
                                             data_dir: str = SYNTHETIC_DATA_DIR,
                                             results_dir: str = ROOT_RESULTS_DIR,
                                             internal_measures: [str] = [ClusteringQualityMeasures.silhouette_score,
                                                                         ClusteringQualityMeasures.pmb],
                                             n_dropped_clusters: [int] = [],
                                             n_dropped_segments: [int] = [],
                                             ):
    """ Runs the internal measure assessment on all ds in the csv files of the generated runs
    :param overall_ds_name: a name for the dataset we're using e.g. n30 or n2
    :param run_names: list of run_names to load (subjects)
    :param distance_measure: name of distance measure to run assessment for
    :param data_type: which datatype to use see SyntheticDataType
    :param data_dir: where to read the data from
    :param results_dir: directory where to store the results, it will use a subdirectory based on the distance measure,
    and the data type
    :param internal_measures: list of internal measures to assess
    :param n_dropped_clusters: list of the number of clusters to drop in each run, if empty then we run the assessment
    on all the cluster
    :param n_dropped_segments: list of the number of segments to drop in each run, if empty then we run the assessment
    using all segments
    """
    # decide which assessment to run
    if len(n_dropped_clusters) == 0 and len(n_dropped_segments) == 0:

        assess_internal_measures(overall_dataset_name=overall_ds_name, run_names=run_names,
                                 data_type=data_type, root_results_dir=results_dir, data_dir=data_dir,
                                 distance_measure=distance_measure,
                                 internal_measures=internal_measures)
    else:
        # run evaluation for all dropped clusters and for all dropped segments separately
        # for this we just do clusters first
        for n_clus in n_dropped_clusters:
            assess_internal_measures(overall_dataset_name=overall_ds_name, run_names=run_names,
                                     data_type=data_type, root_results_dir=results_dir, data_dir=data_dir,
                                     distance_measure=distance_measure,
                                     internal_measures=internal_measures, n_clusters=n_clus)
        # and second we do segments
        for n_seg in n_dropped_segments:
            assess_internal_measures(overall_dataset_name=overall_ds_name, run_names=run_names,
                                     data_type=data_type, root_results_dir=results_dir, data_dir=data_dir,
                                     distance_measure=distance_measure,
                                     internal_measures=internal_measures, n_segments=n_seg)


if __name__ == "__main__":
    overall_ds_name = "n2"
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    run_internal_measure_assessment_datasets(overall_ds_name, run_names=run_names)
