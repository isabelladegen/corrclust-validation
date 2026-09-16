import os

import pandas as pd

from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.evaluation.internal_measure_assessment import read_internal_assessment_result_for, IAResultsCSV
from src.experiments.run_cluster_quality_measures_calculation import read_clustering_quality_measures
from src.experiments.validity.icvi_validity import ICVIValidity, ICVIValCriteria, CriteriaForVariant
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, ResultsType, ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, \
    IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, get_data_dir, \
    get_root_folder_for_reduced_cluster, DataCompleteness, get_root_folder_for_reduced_segments, \
    ROOT_REDUCED_RESULTS_DIR, Aggregators, ICVI_CONSTRUCT_TEST3_SWC_LATEX_FILE, \
    ICVI_CONSTRUCT_TEST3_VRC_LATEX_FILE, ICVI_CONSTRUCT_TEST4_SWC_LATEX_FILE, ICVI_CONSTRUCT_TEST4_VRC_LATEX_FILE, \
    ICVI_MEAN_RESULTS_LATEX_FILE, ICVI_CONSTRUCT_TEST1_LATEX_FILE, ICVI_CONSTRUCT_TEST2_LATEX_FILE, icvi_latex_path
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


def _to_multiindex_stats_df(distance_measures: list, internal_measures: list, stats: dict) -> pd.DataFrame:
    """stats: {measure: [(mean, sd), ...]} in distance_measures order. Returns a MultiIndex-column
    df: level 0 = internal measure, level 1 = Aggregators stat, index = distance_measure."""
    data = {}
    for idx in internal_measures:
        data[(idx, Aggregators.mean)] = [v[0] for v in stats[idx]]
        data[(idx, Aggregators.std)] = [v[1] for v in stats[idx]]
    df = pd.DataFrame(data, index=distance_measures)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def calculate_criterion_stats(distance_measures: list, internal_measures: list, data_type: str, data_dir: str,
                              root_result_dir: str) -> pd.DataFrame:
    """Raw mean/sd of correlation with Jaccard, per measure, per distance measure, for one data
    variant. Reads correlation_summary.csv (produced by InternalMeasureAssessment)."""
    stats = {idx: [] for idx in internal_measures}
    for dm in distance_measures:
        summary = read_internal_assessment_result_for(IAResultsCSV.correlation_summary, "n30", root_result_dir,
                                                      data_type, data_dir, dm)
        for measure in internal_measures:
            col = f"r {measure}, Jaccard"
            stats[measure].append((summary[col].mean(), summary[col].std()))
    return _to_multiindex_stats_df(distance_measures, internal_measures, stats)


def select_ground_truth_row(subject_df: pd.DataFrame) -> pd.DataFrame:
    return subject_df[(subject_df[DescribeBadPartCols.n_wrong_clusters] == 0) &
                      (subject_df[DescribeBadPartCols.n_obs_shifted] == 0)]


def select_worst_row(subject_df: pd.DataFrame) -> pd.DataFrame:
    """Worst engineered partition"""
    return \
        subject_df.sort_values(by=[DescribeBadPartCols.n_wrong_clusters, DescribeBadPartCols.errors],
                               ascending=False).iloc[
            [0]]


def calculate_stats_for_selection(distance_measures, internal_measures, run_names, data_type, data_dir,
                                  root_results_dir, select_row):
    stats = {idx: [] for idx in internal_measures}
    for dm in distance_measures:
        subject_dfs = read_clustering_quality_measures(overall_ds_name="n30", data_type=data_type,
                                                       root_results_dir=root_results_dir, data_dir=data_dir,
                                                       distance_measure=dm, run_names=run_names)
        rows = []
        for subject_df in subject_dfs:
            row = select_row(subject_df)
            assert len(row) == 1, f"Expected exactly one row, got {len(row)}"
            rows.append(row)
        selected = pd.concat(rows, ignore_index=True)
        for idx in internal_measures:
            stats[idx].append((selected[idx].mean(), selected[idx].std()))
    return _to_multiindex_stats_df(distance_measures, internal_measures, stats)


def build_stats_for_data_variant(variant, distance_measures, internal_measures, run_names, data_type, data_dir,
                                 root_results_dir,
                                 root_reduced_dir, completeness):
    criteria = CriteriaForVariant.criteria_for(variant)
    result = {}
    for crit in criteria:
        if crit == ICVIValCriteria.criterion:
            result[crit] = calculate_criterion_stats(distance_measures, internal_measures, data_type, data_dir,
                                                     root_results_dir)
        if crit == ICVIValCriteria.structural_1:
            result[crit] = calculate_stats_for_selection(distance_measures, internal_measures, run_names, data_type,
                                                         data_dir, root_results_dir, select_ground_truth_row)
        if crit == ICVIValCriteria.structural_2:
            result[crit] = calculate_stats_for_selection(distance_measures, internal_measures, run_names, data_type,
                                                         data_dir, root_results_dir, select_worst_row)
        if crit == ICVIValCriteria.structural_3:
            result[crit] = [calculate_stats_for_selection(distance_measures, internal_measures, run_names, data_type,
                                                          get_data_dir(
                                                              get_root_folder_for_reduced_cluster(root_reduced_dir, n),
                                                              completeness),
                                                          get_root_folder_for_reduced_cluster(root_reduced_dir, n),
                                                          select_ground_truth_row) for n in (12, 17)]
        if crit == ICVIValCriteria.structural_4:
            result[crit] = [calculate_stats_for_selection(distance_measures, internal_measures, run_names, data_type,
                                                          get_data_dir(
                                                              get_root_folder_for_reduced_segments(root_reduced_dir, n),
                                                              completeness),
                                                          get_root_folder_for_reduced_segments(root_reduced_dir, n),
                                                          select_ground_truth_row) for n in (50, 75)]

    return result


def generate_latex_table_for_icvi_mean_correlation(results_by_condition: dict, internal_measures: list,
                                                   distance_measures: list, footnote: str) -> str:
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
            row_values = [result_df.loc[dm, ClusteringQualityMeasures.get_display_name_for_measure(im)]
                          for im in ordered_internal_measures]
            lines.append('$' + DistanceMeasures.get_latex_for_measure(dm) + '$ & ' +
                         ' & '.join(row_values) + r' \\')
        lines.append(r'\bottomrule' if i == len(conditions) - 1 else r'\midrule')

    n_cols = len(ordered_internal_measures) + 1
    lines.append(r'\multicolumn{' + str(n_cols) + r'}{l}{$^*$ Passed validity threshold of ' +
                 footnote + r'.} \\')
    lines.append(r'\end{tabular*}')
    return '\n'.join(lines)


def generate_latex_table_for_icvi_subcriteria(results_by_condition: dict, measure_pair: list,
                                              distance_measures: list, sub_labels: tuple,
                                              criteria: str) -> str:
    ordered_measures = ClusteringQualityMeasures.order_measures(measure_pair)
    ordered_distance_measures = DistanceMeasures.order_measures(distance_measures)
    display_names = [ClusteringQualityMeasures.get_display_name_for_measure(m) for m in ordered_measures]

    top_header = ' & ' + ' & '.join(r'\multicolumn{2}{c}{\textbf{' + n + '}}' for n in display_names) + r' \\'
    cmidrules = ' '.join(fr'\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(display_names)))
    sub_header = '$d$ & ' + ' & '.join(sub_labels[j] for _ in display_names for j in range(2)) + r' \\'
    col_spec = 'l ' + ' '.join(['c'] * (2 * len(display_names)))

    lines = [
        r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}' + col_spec + '}',
        r'\toprule', top_header, cmidrules, sub_header, r'\midrule',
    ]

    conditions = list(results_by_condition.items())
    for i, (condition_label, result_df) in enumerate(conditions):
        lines.append(r'\textbf{' + condition_label + r'} & ' +
                     ' & '.join([''] * (2 * len(display_names))) + r' \\')
        for dm in ordered_distance_measures:
            if dm not in result_df.index:
                raise ValueError(f"Distance measure '{dm}' missing from results for condition '{condition_label}'")
            row_values = [result_df.loc[dm, (name, lbl)] for name in display_names for lbl in sub_labels]
            lines.append('$' + DistanceMeasures.get_latex_for_measure(dm) + '$ & ' +
                         ' & '.join(row_values) + r' \\')
        lines.append(r'\bottomrule' if i == len(conditions) - 1 else r'\midrule')

    footnote = ICVIValCriteria.footnote_text_for(criteria, ordered_measures)
    n_cols = 2 * len(ordered_measures) + 1
    lines.append(r'\multicolumn{' + str(n_cols) + r'}{l}{$^*$ Passed validity threshold of ' +
                 footnote + r'.} \\')
    lines.append(r'\end{tabular*}')
    return '\n'.join(lines)


if __name__ == "__main__":
    """Before this can run the internal measures have to be calculated (run_cluster_quality_internal_measures_calculation.py) 
    and the correlations for criterion validity needs to be calculated (run_correlation_measures_calculation.py) 
    """
    main_result_dir = ROOT_RESULTS_DIR
    ds_name = "n30"

    # this is an extensive list
    distance_measures = [DistanceMeasures.l1_cor_dist,  # lp norms
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.l1_with_ref,
                         DistanceMeasures.dot_transform_l1,  # dot transform + lp norms
                         DistanceMeasures.dot_transform_l2]

    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi,
                         ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.pmb]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    save_to_folder = os.path.join(main_result_dir, ResultsType.internal_measure_evaluation, 'validity-outcomes')
    os.makedirs(save_to_folder, exist_ok=True)

    variants = {
        CriteriaForVariant.normal_100: (SyntheticDataType.normal_correlated, SYNTHETIC_DATA_DIR,
                                        DataCompleteness.complete),
        CriteriaForVariant.normal_70: (SyntheticDataType.normal_correlated, IRREGULAR_P30_DATA_DIR,
                                       DataCompleteness.irregular_p30),
        CriteriaForVariant.normal_10: (SyntheticDataType.normal_correlated, IRREGULAR_P90_DATA_DIR,
                                       DataCompleteness.irregular_p90),
        CriteriaForVariant.non_normal_100: (SyntheticDataType.non_normal_correlated, SYNTHETIC_DATA_DIR,
                                            DataCompleteness.complete),
        CriteriaForVariant.non_normal_10: (SyntheticDataType.non_normal_correlated, IRREGULAR_P90_DATA_DIR,
                                           DataCompleteness.irregular_p90),
        CriteriaForVariant.raw_100: (SyntheticDataType.raw, SYNTHETIC_DATA_DIR, DataCompleteness.complete),
        CriteriaForVariant.ds_100: (SyntheticDataType.rs_1min, SYNTHETIC_DATA_DIR, DataCompleteness.complete),
    }

    stats = {}
    for variant, (data_type, data_dir, completeness) in variants.items():
        stats[variant] = build_stats_for_data_variant(variant, distance_measures, internal_measures, run_names,
                                                      data_type, data_dir,
                                                      ROOT_RESULTS_DIR, ROOT_REDUCED_RESULTS_DIR, completeness)
        print(f'Calculated stats for {variant}')

    validity = ICVIValidity(internal_measures=internal_measures, **stats)

    # # save mean (sd) * table for each variant considered
    # for name in variants:
    #     validity.mean_sd_valid_summary_table(stats[name]).to_csv(
    #         os.path.join(save_to_folder, f'mean_sd_valid_{name}.csv'))

    # save overall validity results
    validity.overall_validity().to_csv(os.path.join(save_to_folder, 'overall_validity_results.csv'))
    validity.external_validity_details().to_csv(os.path.join(save_to_folder, 'external_validity_results.csv'))
    validity.discriminant_validity_details().to_csv(os.path.join(save_to_folder, 'discriminant_validity_results.csv'))

    # create latex tables for the numerical results
    construct_conditions = [
        CriteriaForVariant.normal_100,
        CriteriaForVariant.normal_70,
        CriteriaForVariant.normal_10,
        CriteriaForVariant.non_normal_100,
        CriteriaForVariant.non_normal_10,
    ]

    # Criterion
    criterion_results = {CriteriaForVariant.display_name_for(variant): validity.mean_sd_valid_summary_table(
        stats[variant][ICVIValCriteria.criterion], ICVIValCriteria.criterion)
        for variant in construct_conditions}
    with open(icvi_latex_path(ICVI_MEAN_RESULTS_LATEX_FILE, ds_name, main_result_dir), 'w') as f:
        f.write(generate_latex_table_for_icvi_mean_correlation(
            criterion_results, internal_measures, distance_measures,
            f'$|r|>{ICVIValCriteria.criterion_threshold()}$'))

    # Structural 1 and 2
    for criteria, filename in [(ICVIValCriteria.structural_1, ICVI_CONSTRUCT_TEST1_LATEX_FILE),
                               (ICVIValCriteria.structural_2, ICVI_CONSTRUCT_TEST2_LATEX_FILE)]:
        results = {CriteriaForVariant.display_name_for(variant): validity.mean_sd_valid_summary_table(stats[variant][criteria], criteria)
                   for variant in construct_conditions}
        with open(icvi_latex_path(filename, ds_name, main_result_dir), 'w') as f:
            f.write(generate_latex_table_for_icvi_mean_correlation(
                results, internal_measures, distance_measures,
                ICVIValCriteria.footnote_text_for(criteria, internal_measures)))

    # Structural 3 and 4
    swc_dbi = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi]
    vrc_pbm = [ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.pmb]
    sub_criteria_specs = [
        (ICVIValCriteria.structural_3, ('11 clusters', '6 clusters'),
         ICVI_CONSTRUCT_TEST3_SWC_LATEX_FILE, ICVI_CONSTRUCT_TEST3_VRC_LATEX_FILE),
        (ICVIValCriteria.structural_4, ('50 segments', '25 segments'),
         ICVI_CONSTRUCT_TEST4_SWC_LATEX_FILE, ICVI_CONSTRUCT_TEST4_VRC_LATEX_FILE),
    ]
    for criteria, sub_labels, swc_dbi_file, vrc_pbm_file in sub_criteria_specs:
        for measure_pair, filename in [(swc_dbi, swc_dbi_file), (vrc_pbm, vrc_pbm_file)]:
            results = {CriteriaForVariant.display_name_for(variant): validity.mean_sd_valid_summary_table_for_subcriteria(
                stats[variant][criteria], criteria, sub_labels, measures=measure_pair)
                for variant in construct_conditions}
            with open(icvi_latex_path(filename, ds_name, main_result_dir), 'w') as f:
                f.write(generate_latex_table_for_icvi_subcriteria(
                    results, measure_pair, distance_measures, sub_labels, criteria))
