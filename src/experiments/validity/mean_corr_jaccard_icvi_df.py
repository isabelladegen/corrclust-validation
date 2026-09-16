import os
from os import path

import pandas as pd

from src.evaluation.internal_measure_assessment import read_internal_assessment_result_for, IAResultsCSV
from src.experiments.validity.icvi_validity import ICVIValCriteria
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, SYNTHETIC_DATA_DIR, ResultsType, \
    number_for_completeness, DataCompleteness, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, \
    internal_measure_evaluation_dir_for, get_latex_results_path, ICVI_MEAN_RESULTS_LATEX_FILE
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType, DataVariant


def calculate_mean_sd_for(distance_measures, internal_measures, data_type, data_dir, completeness, threshold,
                          root_result_dir,
                          filname_addition: str = ''):
    # read correlation_summary.csv for each Internal Measure and df combination
    correlation_summaries = {}
    for dist_function in distance_measures:
        summary = read_internal_assessment_result_for(
            result_type=IAResultsCSV.correlation_summary,
            overall_dataset_name="n30",
            results_dir=root_result_dir,
            data_type=data_type,
            data_dir=data_dir,
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
            star = '*' if abs(mean) > threshold else ''
            row_data[icvi] = f"{mean:.2f} (SD {sd:.2f}){star}"
        results[dist_func] = row_data

    result_df = pd.DataFrame.from_dict(results, orient='index')

    save_to_folder = path.join(root_result_dir, ResultsType.internal_measure_evaluation, 'validity-outcomes')
    os.makedirs(save_to_folder, exist_ok=True)

    comp = number_for_completeness(completeness)
    result_df.to_csv(path.join(save_to_folder, f'icvi-criterion-mean_sd_{data_type}_{comp}{filname_addition}.csv'))
    return result_df


def generate_latex_table_for_icvi_mean_correlation(results_by_condition: dict, internal_measures: list,
                                                   distance_measures: list, threshold: float) -> str:
    """
    results_by_condition: dict mapping condition label (e.g. "Normal 100\\%") to the result_df
    returned by calculate_mean_sd_for, in the order the conditions should appear in the table.
    Returns the LaTeX table as a string.
    """
    ordered_internal_measures = ClusteringQualityMeasures.order_measures(internal_measures)
    ordered_distance_measures = DistanceMeasures.order_measures(distance_measures)

    header = ' & '.join(r'\textbf{' + ClusteringQualityMeasures.get_display_name_for_measure(im) + '}'
                        for im in ordered_internal_measures)
    col_spec = 'l ' + ' '.join(['c'] * len(ordered_internal_measures))

    lines = [
        r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}' + col_spec + '}',
        r'\toprule',
        r'$d$ & ' + header + r' \\',
        r'\midrule',
    ]

    conditions = list(results_by_condition.items())
    for i, (condition_label, result_df) in enumerate(conditions):
        lines.append(r'\textbf{' + condition_label + r'} & ' +
                     ' & '.join([''] * len(ordered_internal_measures)) + r' \\')
        for dm in ordered_distance_measures:
            if dm not in result_df.index:
                raise ValueError(f"Distance measure '{dm}' missing from results for condition '{condition_label}'")
            row_values = [result_df.loc[dm, im] for im in ordered_internal_measures]
            lines.append('$' + DistanceMeasures.get_latex_for_measure(dm) + '$ & ' +
                         ' & '.join(row_values) + r' \\')
        lines.append(r'\bottomrule' if i == len(conditions) - 1 else r'\midrule')

    n_cols = len(ordered_internal_measures) + 1
    lines.append(r'\multicolumn{' + str(n_cols) + r'}{l}{$^*$ Passed validity threshold of $|r|>' +
                 str(threshold) + r'$.} \\')
    lines.append(r'\end{tabular*}')

    return '\n'.join(lines)


if __name__ == "__main__":
    """This requires correlation_summary.csv --> internal_measure_assessment, 
    which needs run_cluster_quality_measures_calculation -> which needs to run describe_bad_partitions.py"""
    write_latex_table = True
    ds_name = "n30"
    distance_measures = [DistanceMeasures.l1_cor_dist,  # lp norms
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.l1_with_ref,
                         DistanceMeasures.dot_transform_l1,  # dot transform + lp norms
                         DistanceMeasures.dot_transform_l2]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.pmb,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.dbi]

    corr_threshold = ICVIValCriteria.criterion_threshold()

    data_variants = [
        DataVariant(SyntheticDataType.normal_correlated, SYNTHETIC_DATA_DIR),  # normal 100%
        DataVariant(SyntheticDataType.normal_correlated, IRREGULAR_P30_DATA_DIR),  # normal 70%
        DataVariant(SyntheticDataType.normal_correlated, IRREGULAR_P90_DATA_DIR),  # normal 10%
        DataVariant(SyntheticDataType.non_normal_correlated, SYNTHETIC_DATA_DIR),  # non-normal 100%
        DataVariant(SyntheticDataType.non_normal_correlated, IRREGULAR_P90_DATA_DIR),  # non-normal 10%
    ]

    results_by_condition = {}
    for dv in data_variants:
        results_by_condition[dv.label()] = calculate_mean_sd_for(
            distance_measures, internal_measures, dv.data_type, dv.data_dir, dv.completeness, corr_threshold,
            ROOT_RESULTS_DIR)

    if write_latex_table:
        latex_table = generate_latex_table_for_icvi_mean_correlation(
            results_by_condition=results_by_condition,
            internal_measures=internal_measures,
            distance_measures=distance_measures,
            threshold=corr_threshold)

        latex_output_path = get_latex_results_path(
            internal_measure_evaluation_dir_for(overall_dataset_name=ds_name, data_type="", results_dir=ROOT_RESULTS_DIR,
                                                data_dir="", distance_measure=""), ICVI_MEAN_RESULTS_LATEX_FILE)

        with open(latex_output_path, 'w') as f:
            f.write(latex_table)
