import os

import pandas as pd

from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.experiments.run_cluster_quality_measures_calculation import read_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, ResultsType, ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, \
    IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, get_data_dir, \
    get_root_folder_for_reduced_cluster, DataCompleteness, get_root_folder_for_reduced_segments, \
    ROOT_REDUCED_RESULTS_DIR, number_for_completeness
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

index_thresholds = {
    "optimal": {
        ClusteringQualityMeasures.silhouette_score: lambda x: x > 0.9,
        ClusteringQualityMeasures.dbi: lambda x: x < 0.15
    },
    "bad": {
        ClusteringQualityMeasures.silhouette_score: lambda x: x < 0,
        ClusteringQualityMeasures.dbi: lambda x: x > 2,
    }
}


def calculate_mean_sd(distance_measures, internal_indices, run_names, data_type, synthetic_data_dir, root_results_dir,
                      round_to=3):
    # empty dict for results
    results = {
        "d": []
    }
    means_opt = {"mean opt " + idx: [] for idx in internal_indices}
    means_worst = {"mean bad " + idx: [] for idx in internal_indices}
    optimal = {"optimal " + idx: [] for idx in internal_indices}
    worst = {"bad " + idx: [] for idx in internal_indices}
    results.update(means_opt)
    results.update(means_worst)
    results.update(optimal)
    results.update(worst)

    # create a row per dm index combination
    for dm in distance_measures:
        # list of dfs with results summaries for each run_name
        result_summaries = read_clustering_quality_measures(overall_ds_name="n30", data_type=data_type,
                                                            root_results_dir=root_results_dir,
                                                            data_dir=synthetic_data_dir,
                                                            distance_measure=dm, run_names=run_names)
        rows_gt = []
        rows_highest_mae = []
        gt_optimal_results = {idx: [] for idx in internal_indices}
        highest_mae_worst_results = {idx: [] for idx in internal_indices}
        for summary_df in result_summaries:
            gt_row = summary_df[(summary_df[DescribeBadPartCols.n_wrong_clusters] == 0) &
                                (summary_df[DescribeBadPartCols.n_obs_shifted] == 0)]
            assert len(
                gt_row) == 1, f"Run {summary_df.iloc[0]['file name']} for d {dm} has {len(gt_row)} rows with wrong cluster and obs shifted both 0, expected 1"
            rows_gt.append(gt_row)

            max_value = summary_df[DescribeBadPartCols.errors].max()
            worst_row = summary_df[summary_df[DescribeBadPartCols.errors] == max_value]
            if len(worst_row) > 1:
                print(
                    f"Run {summary_df.iloc[0]['file name']} for d {dm} has {len(worst_row)} rows with same high mae, expected 1")
                worst_row = worst_row[worst_row[ClusteringQualityMeasures.silhouette_score] == worst_row[
                    ClusteringQualityMeasures.silhouette_score].min()]

            rows_highest_mae.append(worst_row)

            # assert these are optimal and worst index values for this run
            for idx in internal_indices:
                if idx == ClusteringQualityMeasures.dbi:
                    # inverted logic since lower values are better
                    gt_optimal_results[idx].append(summary_df[idx].min() == gt_row[idx].values[0])
                    highest_mae_worst_results[idx].append(summary_df[idx].max() == worst_row[idx].values[0])
                else:
                    # logic for higher index is better
                    gt_optimal_results[idx].append(summary_df[idx].max() == gt_row[idx].values[0])
                    highest_mae_worst_results[idx].append(summary_df[idx].min() == worst_row[idx].values[0])

        # Combine into single df across all runs
        gt_results = pd.concat(rows_gt, ignore_index=True)
        highest_mae_results = pd.concat(rows_highest_mae, ignore_index=True)

        # store results in dictionary
        results["d"].append(dm)
        for idx in internal_indices:
            opt_mean = round(gt_results[idx].mean(), round_to)
            add_star = index_thresholds["optimal"].get(idx, lambda x: False)(opt_mean)
            star = "*" if add_star else ""
            opt_sd = round(gt_results[idx].std(), round_to)
            results["mean opt " + idx].append(f"{opt_mean} (SD {opt_sd}){star}")

            worst_mean = round(highest_mae_results[idx].mean(), round_to)
            add_star_worst = index_thresholds["bad"].get(idx, lambda x: False)(worst_mean)
            star_worst = "*" if add_star_worst else ""
            worst_sd = round(highest_mae_results[idx].std(), round_to)
            results["mean bad " + idx].append(f"{worst_mean} (SD {worst_sd}){star_worst}")

            results["optimal " + idx].append(all(gt_optimal_results[idx]))
            results["bad " + idx].append(all(highest_mae_worst_results[idx]))

    # create df from dict
    results_df = pd.DataFrame(results)
    return results_df

def construct_test_3(distance_measures, internal_measures, dropped_clusters, data_type, data_completeness,
                     root_result_dir, additional_filename: str = ''):
    dir_for_cluster = get_root_folder_for_reduced_cluster(root_result_dir, dropped_clusters)
    results_dir = get_root_folder_for_reduced_cluster(root_result_dir, dropped_clusters)
    data_dir = get_data_dir(dir_for_cluster, data_completeness)
    df = calculate_mean_sd(distance_measures, internal_measures, run_names,
                                                         data_type, data_dir, results_dir)
    comp = number_for_completeness(data_completeness)
    n_clusters = 23 - dropped_clusters
    filename = f'{additional_filename}construct-3-mean_sd_{data_type}_{comp}_cluster_{n_clusters}.csv'
    df.to_csv(os.path.join(save_to_folder, filename))

def construct_test_4(distance_measures, internal_measures, dropped_segments, data_type, data_completeness,
                     root_result_dir, additional_filename: str = ''):
    dir_for_segments = get_root_folder_for_reduced_segments(root_reduced_dir, dropped_segments)
    results_dir = get_root_folder_for_reduced_segments(root_result_dir, dropped_segments)
    data_dir = get_data_dir(dir_for_segments, data_completeness)
    df = calculate_mean_sd(distance_measures, internal_measures, run_names,
                                                         data_type, data_dir, results_dir)
    comp = number_for_completeness(data_completeness)
    n_segments = 100 - dropped_segments
    filename = f'{additional_filename}construct-4-mean_sd_{data_type}_{comp}_segments_{n_segments}.csv'
    df.to_csv(os.path.join(save_to_folder, filename))

if __name__ == "__main__":
    main_result_dir = ROOT_RESULTS_DIR

    # this is an extensive list
    distance_measures = [DistanceMeasures.l1_cor_dist,  # lp norms
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.dot_transform_l1,  # dot transform + lp norms
                         DistanceMeasures.dot_transform_l2]

    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.pmb]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    save_to_folder = os.path.join(main_result_dir, ResultsType.internal_measure_evaluation, 'validity-outcomes')
    os.makedirs(save_to_folder, exist_ok=True)

    # calculate mean and sd for each criterion for valid dm and IVCI combinations
    # 1. Structural pass 4 tests for Normal 100%
    mean_sd_df_normal_100 = calculate_mean_sd(distance_measures, internal_measures, run_names,
                                              SyntheticDataType.normal_correlated,
                                              SYNTHETIC_DATA_DIR, main_result_dir)
    mean_sd_df_normal_100.to_csv(os.path.join(save_to_folder, 'construct-1-2-mean_sd_normal_100.csv'))

    # 2.1 Discriminant fail 4 structural test for raw 100%
    mean_sd_df_raw_100 = calculate_mean_sd(distance_measures, internal_measures, run_names,
                                           SyntheticDataType.raw,
                                           SYNTHETIC_DATA_DIR, main_result_dir)
    mean_sd_df_raw_100.to_csv(os.path.join(save_to_folder, 'discriminant-mean_sd_raw_100.csv'))

    # 2.2 Discriminant degrade 4 structural tests for ds 100%
    mean_sd_df_ds_100 = calculate_mean_sd(distance_measures, internal_measures, run_names,
                                          SyntheticDataType.rs_1min,
                                          SYNTHETIC_DATA_DIR, main_result_dir)
    mean_sd_df_ds_100.to_csv(os.path.join(save_to_folder, 'discriminant-mean_sd_ds_100.csv'))

    # 3. External Validity pass 4 structural for Normal 70% and 10% and NN 100% and 10%
    mean_sd_df_normal_70 = calculate_mean_sd(distance_measures, internal_measures, run_names,
                                             SyntheticDataType.normal_correlated,
                                             IRREGULAR_P30_DATA_DIR, main_result_dir)
    mean_sd_df_normal_70.to_csv(os.path.join(save_to_folder, 'external-construct-1-2-mean_sd_normal_70.csv'))

    mean_sd_df_normal_10 = calculate_mean_sd(distance_measures, internal_measures, run_names,
                                             SyntheticDataType.normal_correlated,
                                             IRREGULAR_P90_DATA_DIR, main_result_dir)
    mean_sd_df_normal_10.to_csv(os.path.join(save_to_folder, 'external-construct-1-2-mean_sd_normal_10.csv'))

    mean_sd_df_non_normal_100 = calculate_mean_sd(distance_measures, internal_measures, run_names,
                                                  SyntheticDataType.non_normal_correlated,
                                                  SYNTHETIC_DATA_DIR, main_result_dir)
    mean_sd_df_non_normal_100.to_csv(os.path.join(save_to_folder, 'external-construct-1-2-mean_sd_non_normal_100.csv'))

    mean_sd_df_non_normal_10 = calculate_mean_sd(distance_measures, internal_measures, run_names,
                                                 SyntheticDataType.non_normal_correlated,
                                                 IRREGULAR_P90_DATA_DIR, main_result_dir)
    mean_sd_df_non_normal_10.to_csv(os.path.join(save_to_folder, 'external-construct-1-2-mean_sd_non_normal_10.csv'))

    root_reduced_dir = ROOT_REDUCED_RESULTS_DIR

    # 4. Construct Structural Test 3 (different number of clusters) for Normal 100%
    construct_test_3(distance_measures, internal_measures, 12, SyntheticDataType.normal_correlated,
                     DataCompleteness.complete, root_reduced_dir)
    construct_test_3(distance_measures, internal_measures, 17, SyntheticDataType.normal_correlated,
                     DataCompleteness.complete, root_reduced_dir)

    # 4. External structural test 3 Normal 70%, 10% and
    construct_test_3(distance_measures, internal_measures, 12, SyntheticDataType.normal_correlated,
                     DataCompleteness.irregular_p30, root_reduced_dir, 'external_')
    construct_test_3(distance_measures, internal_measures, 17, SyntheticDataType.normal_correlated,
                     DataCompleteness.irregular_p30, root_reduced_dir, 'external_')
    construct_test_3(distance_measures, internal_measures, 12, SyntheticDataType.normal_correlated,
                     DataCompleteness.irregular_p90, root_reduced_dir, 'external_')
    construct_test_3(distance_measures, internal_measures, 17, SyntheticDataType.normal_correlated,
                     DataCompleteness.irregular_p90, root_reduced_dir, 'external_')

    #4. External structural test 3 Non-normal 100% and 10%
    construct_test_3(distance_measures, internal_measures, 12, SyntheticDataType.non_normal_correlated,
                     DataCompleteness.complete, root_reduced_dir, 'external_')
    construct_test_3(distance_measures, internal_measures, 17, SyntheticDataType.non_normal_correlated,
                     DataCompleteness.complete, root_reduced_dir, 'external_')
    construct_test_3(distance_measures, internal_measures, 12, SyntheticDataType.non_normal_correlated,
                     DataCompleteness.irregular_p90, root_reduced_dir, 'external_')
    construct_test_3(distance_measures, internal_measures, 17, SyntheticDataType.non_normal_correlated,
                     DataCompleteness.irregular_p90, root_reduced_dir, 'external_')

    # 5. Construct Structural Test 5 (different number of clusters) for Normal 100%
    construct_test_4(distance_measures, internal_measures, 50, SyntheticDataType.normal_correlated,
                     DataCompleteness.complete, root_reduced_dir)
    construct_test_4(distance_measures, internal_measures, 75, SyntheticDataType.normal_correlated,
                     DataCompleteness.complete, root_reduced_dir)

    # 5. External structural test 4 Normal 70%, 10% and
    construct_test_4(distance_measures, internal_measures, 50, SyntheticDataType.normal_correlated,
                     DataCompleteness.irregular_p30, root_reduced_dir, 'external_')
    construct_test_4(distance_measures, internal_measures, 75, SyntheticDataType.normal_correlated,
                     DataCompleteness.irregular_p30, root_reduced_dir, 'external_')
    construct_test_4(distance_measures, internal_measures, 50, SyntheticDataType.normal_correlated,
                     DataCompleteness.irregular_p90, root_reduced_dir, 'external_')
    construct_test_4(distance_measures, internal_measures, 75, SyntheticDataType.normal_correlated,
                     DataCompleteness.irregular_p90, root_reduced_dir, 'external_')

    # 5. External structural test 4 Non-normal 100% and 10%
    construct_test_4(distance_measures, internal_measures, 50, SyntheticDataType.non_normal_correlated,
                   DataCompleteness.complete, root_reduced_dir, 'external_')
    construct_test_4(distance_measures, internal_measures, 75, SyntheticDataType.non_normal_correlated,
                   DataCompleteness.complete, root_reduced_dir, 'external_')
    construct_test_4(distance_measures, internal_measures, 50, SyntheticDataType.non_normal_correlated,
                   DataCompleteness.irregular_p90, root_reduced_dir, 'external_')
    construct_test_4(distance_measures, internal_measures, 75, SyntheticDataType.non_normal_correlated,
                     DataCompleteness.irregular_p90, root_reduced_dir, 'external_')
