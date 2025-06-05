import pandas as pd

from src.evaluation.internal_measure_assessment import InternalMeasureAssessment, get_full_filename_for_results_csv, \
    IAResultsCSV
from src.experiments.run_cluster_quality_measures_calculation import read_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, \
    GENERATED_DATASETS_FILE_PATH, ROOT_RESULTS_DIR, internal_measure_evaluation_dir_for
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    overall_dataset_name = "n30"
    root_result_dir = ROOT_RESULTS_DIR
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
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

    min_corr_required = 0.5

    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi]
    data_types = [SyntheticDataType.normal_correlated,
                  SyntheticDataType.non_normal_correlated, SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR]

    # list of distance measure for each internal measure that has correlation > 0.5 for all data variants
    correlation_results = {i: [] for i in internal_measures}

    for distance_measure in distance_measures:
        # list of true or false whether all correlation are > 0.5 for internal measure for data variant
        correlation_significant = correlation_results.copy()
        for data_dir in data_dirs:
            for data_type in data_types:
                # load data
                print(
                    "Distance measure: " + distance_measure + " , Dataset type: " + data_type + ", Compactness: " + data_dir)

                partitions = read_clustering_quality_measures(overall_ds_name=overall_dataset_name, data_type=data_type,
                                                              root_results_dir=root_result_dir, data_dir=data_dir,
                                                              distance_measure=distance_measure, run_names=run_names)
                ia = InternalMeasureAssessment(distance_measure=distance_measure, dataset_results=partitions,
                                               internal_measures=internal_measures)
                store_results_in = internal_measure_evaluation_dir_for(
                    overall_dataset_name=overall_dataset_name,
                    data_type=data_type,
                    results_dir=root_result_dir, data_dir=data_dir,
                    distance_measure=distance_measure)

                # correlation summary
                summary = ia.correlation_summary

                # evaluate if result is significant
                sig_cor = (summary[ia.measures_corr_col_names] > min_corr_required).all().tolist()
                for idx, col in enumerate(ia.measures_corr_col_names):
                    measure_name = col.split(', ')[0].replace('r ', '')
                    correlation_significant[measure_name].append(sig_cor[idx])

                # save result
                summary.to_csv(
                    get_full_filename_for_results_csv(store_results_in, IAResultsCSV.correlation_summary))

        # now check if all data variant were significant and save in overall result
        for measure in internal_measures:
            if all(correlation_significant[measure]):
                # only add the distance measures that were significant across all data variants
                correlation_results[measure].append(distance_measure)


    print("Filtered results:")
    print(correlation_results)