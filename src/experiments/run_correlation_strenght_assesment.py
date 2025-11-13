import pandas as pd

from src.evaluation.internal_measure_assessment import InternalMeasureAssessment, get_full_filename_for_results_csv, \
    IAResultsCSV
from src.experiments.run_cluster_quality_measures_calculation import read_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, \
    GENERATED_DATASETS_FILE_PATH, ROOT_RESULTS_DIR, internal_measure_evaluation_dir_for, get_data_completeness_from
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.visualisation.run_average_rank_visualisations import data_variant_description

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

    # distance_measures = [
    #     DistanceMeasures.l2_with_ref,
    #     DistanceMeasures.l3_with_ref,
    #     DistanceMeasures.linf_with_ref,
    #     DistanceMeasures.dot_transform_l1,  # dot transform + lp norms
    #     DistanceMeasures.dot_transform_l2,
    # ]

    min_corr_required = 0.5

    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi]
    data_types = [SyntheticDataType.normal_correlated,
                  SyntheticDataType.non_normal_correlated, SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR]

    data_variants = []
    dms = []
    corr_strengths = {"r " +im: []for im in internal_measures}
    passes = {im: []for im in internal_measures}

    for distance_measure in distance_measures:
        # list of true or false whether all correlation are > 0.5 for internal measure for data variant
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

                # mean values
                mean_series = ia.descriptive_statistics_for_internal_measures_correlation().loc["mean"]

                # results
                data_variants.append(data_variant_description[(get_data_completeness_from(data_dir), data_type)])
                dms.append(distance_measure)
                for im_long in mean_series.index:
                    means = mean_series[im_long]
                    im = im_long.split(', ')[0].replace('r ', '')
                    corr_strengths["r " + im].append(means)
                    # deal with DBI being negative
                    if im == ClusteringQualityMeasures.dbi:
                        passes[im].append((means * -1) > min_corr_required)
                    else:
                        passes[im].append(means > min_corr_required)

                # save result
                summary.to_csv(
                    get_full_filename_for_results_csv(store_results_in, IAResultsCSV.correlation_summary))

    # save significant result
    df_data = {}
    df_data['Data variant'] = data_variants
    df_data['Distance Measure'] = dms
    df_data.update(corr_strengths)
    df_data.update(passes)
    df = pd.DataFrame(df_data)

    store_overall_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_dataset_name,
        data_type="",
        results_dir=root_result_dir, data_dir="",
        distance_measure="")
    df.to_csv(get_full_filename_for_results_csv(store_overall_in, IAResultsCSV.passes_min_correlation))