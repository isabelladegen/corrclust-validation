from collections import defaultdict
from os import path

import pandas as pd

from src.evaluation.describe_clustering_quality_for_data_variant import DescribeClusteringQualityForDataVariant, \
    IntSummaryCols
from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, get_clustering_quality_multiple_data_variants_result_folder, \
    ResultsType
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_multiple_data_variants import get_row_name_from


def summarise_clustering_quality(data_dirs: [str], data_types: [str], run_file: str, root_results_dir: str,
                                 distance_measures: [str], clustering_quality_measures: [str], overall_ds_name: str):
    all_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi,
                    ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.pmb]
    all_mean_corr_dfs = []
    all_paired_t_test_dfs = []
    all_gt_worst_dfs = []
    for data_type in data_types:
        for data_dir in data_dirs:
            # turn row data into dictionary grouping all distance measures into same row
            mean_cor_row_data = {}
            paired_t_row_data = {}
            gt_worst_row_data = {}
            for distance_measure in distance_measures:
                describe = DescribeClusteringQualityForDataVariant(wandb_run_file=run_file,
                                                                   overall_ds_name=overall_ds_name,
                                                                   data_type=data_type, data_dir=data_dir,
                                                                   results_root_dir=root_results_dir,
                                                                   distance_measure=distance_measure)
                df = describe.mean_sd_correlation_for(quality_measures=clustering_quality_measures)
                # Combine distance measure results for correlation into one row
                mean_cor_row_data[IntSummaryCols.data_stage] = df[IntSummaryCols.data_stage].iloc[0]
                mean_cor_row_data[IntSummaryCols.data_completeness] = df[IntSummaryCols.data_completeness].iloc[0]
                for quality_measure in clustering_quality_measures:
                    mean_cor_row_data[(quality_measure, distance_measure)] = df[quality_measure].iloc[0]

                # Combine distance measure results for gt and worst value
                gtwdf = describe.mean_sd_measure_values_for_ground_truth_and_lowest_jaccard_index(
                    quality_measures=all_measures)
                gt_worst_row_data[IntSummaryCols.data_stage] = gtwdf[IntSummaryCols.data_stage].iloc[0]
                gt_worst_row_data[IntSummaryCols.data_completeness] = gtwdf[IntSummaryCols.data_completeness].iloc[0]
                for quality_measure in all_measures:
                    # gt
                    gt_worst_row_data[(quality_measure, IntSummaryCols.gt, distance_measure)] = \
                        gtwdf[(quality_measure, IntSummaryCols.gt)].iloc[0]
                    # worst
                    gt_worst_row_data[(quality_measure, IntSummaryCols.worst, distance_measure)] = \
                        gtwdf[(quality_measure, IntSummaryCols.worst)].iloc[0]

                # Combine distance measure results for paired t-test into one row
                ptdf = describe.p_value_and_effect_size_of_correlation_for(quality_measures=all_measures)
                paired_t_row_data[IntSummaryCols.data_stage] = ptdf[IntSummaryCols.data_stage].iloc[0]
                paired_t_row_data[IntSummaryCols.data_completeness] = ptdf[IntSummaryCols.data_completeness].iloc[0]
                cols = list(ptdf.columns)
                cols.remove(IntSummaryCols.data_stage)
                cols.remove(IntSummaryCols.data_completeness)
                for column in cols:
                    paired_t_row_data[(column, distance_measure)] = ptdf[column].iloc[0]

            all_mean_corr_dfs.append(mean_cor_row_data)
            all_gt_worst_dfs.append(gt_worst_row_data)
            all_paired_t_test_dfs.append(paired_t_row_data)

    mean_corr_df = pd.DataFrame(all_mean_corr_dfs)
    paired_t_test_df = pd.DataFrame(all_paired_t_test_dfs)
    gt_worst_df = pd.DataFrame(all_gt_worst_dfs) # raw values df

    # sort the rows
    data_stage_order = [
        SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.non_normal_correlated),
        SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.normal_correlated),
        SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.rs_1min),
        SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.raw)
    ]

    completeness_order = [
        get_row_name_from(IRREGULAR_P30_DATA_DIR),
        get_row_name_from(IRREGULAR_P90_DATA_DIR),
        get_row_name_from(SYNTHETIC_DATA_DIR)
    ]

    prefix = ""
    if len(distance_measures) == 1:
        prefix = distance_measures[0] + '_'

    mean_corr_df = mean_corr_df.sort_values(
        by=[IntSummaryCols.data_stage, IntSummaryCols.data_completeness],
        ascending=[True, True],
        key=lambda x: pd.Categorical(x,
                                     categories=data_stage_order if x.name == IntSummaryCols.data_stage else completeness_order)
    )
    folder = get_clustering_quality_multiple_data_variants_result_folder(
        results_type=ResultsType.internal_measure_evaluation,
        overall_dataset_name=overall_ds_name,
        results_dir=root_results_dir,
        distance_measure="")
    file_name = path.join(folder, prefix + IAResultsCSV.mean_correlation_data_variant)
    mean_corr_df.to_csv(str(file_name))

    gt_worst_df = gt_worst_df.sort_values(
        by=[IntSummaryCols.data_stage, IntSummaryCols.data_completeness],
        ascending=[True, True],
        key=lambda x: pd.Categorical(x,
                                     categories=data_stage_order if x.name == IntSummaryCols.data_stage else completeness_order)
    )
    gt_worst_file_name = path.join(folder, prefix + IAResultsCSV.gt_worst_measure_data_variants)
    gt_worst_df.to_csv(str(gt_worst_file_name))

    paired_t_test_df = paired_t_test_df.sort_values(
        by=[IntSummaryCols.data_stage, IntSummaryCols.data_completeness],
        ascending=[True, True],
        key=lambda x: pd.Categorical(x,
                                     categories=data_stage_order if x.name == IntSummaryCols.data_stage else completeness_order)
    )
    paired_t_file_name = path.join(folder, prefix + IAResultsCSV.paired_t_test_data_variant)
    paired_t_test_df.to_csv(str(paired_t_file_name))


if __name__ == "__main__":
    # creates summary table for all data variants of various results for different distance measures
    # backend = Backends.visible_tests.value
    backend = Backends.none.value
    save_fig = True
    overall_ds_name = "n30"
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    # distance_measures = [DistanceMeasures.l1_cor_dist,
    #                      DistanceMeasures.l1_with_ref,
    #                      DistanceMeasures.foerstner_cor_dist]

    distance_measures = [DistanceMeasures.l1_cor_dist,
                         DistanceMeasures.l1_with_ref,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.l5_with_ref,
                         DistanceMeasures.linf_cor_dist]

    # Config for L2 only ran for downsampled, complete data
    # distance_measures = [DistanceMeasures.l2_cor_dist]
    # dataset_types = [SyntheticDataType.rs_1min]
    # data_dirs = [SYNTHETIC_DATA_DIR]

    clustering_quality_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi,
                                   ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.pmb]

    run_file = GENERATED_DATASETS_FILE_PATH

    summarise_clustering_quality(data_dirs=data_dirs, data_types=dataset_types, run_file=run_file,
                                 root_results_dir=root_result_dir, distance_measures=distance_measures,
                                 clustering_quality_measures=clustering_quality_measures,
                                 overall_ds_name=overall_ds_name)
