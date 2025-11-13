import os
from os import path

import pandas as pd

from src.evaluation.internal_measure_assessment import read_internal_assessment_result_for, IAResultsCSV
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, SYNTHETIC_DATA_DIR, ResultsType
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    overall_dataset_name = "n30"
    root_result_dir = ROOT_RESULTS_DIR

    # this is an extensive list
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
                         DistanceMeasures.log_frob_cor_dist,  # correlation metrics
                         DistanceMeasures.foerstner_cor_dist]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi]

    corr_threshold = 0.5

    # read correlation_summary.csv for each Internal Measure and df combination
    correlation_summaries = {}
    for dist_function in distance_measures:
        summary = read_internal_assessment_result_for(
            result_type=IAResultsCSV.correlation_summary,
            overall_dataset_name=overall_dataset_name,
            results_dir=root_result_dir,
            data_type=SyntheticDataType.normal_correlated,
            data_dir=SYNTHETIC_DATA_DIR,
            distance_measure=dist_function)
        correlation_summaries[dist_function] = summary

    results = {}
    # calculate mean correlation and sd
    for dist_func, df in correlation_summaries.items():
        row_data = {}
        for icvi in internal_measures:
            col_name = f"r {icvi}, Jaccard"
            mean = df[col_name].mean()
            sd = df[col_name].std()
            star = '*' if abs(mean) > corr_threshold else ''
            row_data[icvi] = f"{mean:.2f} (SD {sd:.2f}){star}"
        results[dist_func] = row_data

    result_df = pd.DataFrame.from_dict(results, orient='index')

    save_to_folder = path.join(root_result_dir, ResultsType.internal_measure_evaluation, 'validity-outcomes')
    os.makedirs(save_to_folder, exist_ok=True)

    result_df.to_csv(path.join(save_to_folder, 'icvi-criterion-mean_sd_normal_100.csv'))


