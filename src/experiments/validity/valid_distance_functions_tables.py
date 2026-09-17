import os
from dataclasses import dataclass
from os import path
from typing import ClassVar

import numpy as np
import pandas as pd

from src.evaluation.distance_metric_evaluation import read_csv_of_raw_values_for_all_criteria, criteria_short_names, \
    EvaluationCriteria, DistanceMeasureCols
from src.experiments.validity.distance_measure_validity import DistanceMeasureValidity, DM_THRESHOLDS, REVIEWED_RULES, \
    Comparison
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, ResultsType, Aggregators, DF_CONSTRUCT_NORMAL_LATEX_FILE, \
    DF_EXTERNAL_NORMAL_70_LATEX_FILE, DF_EXTERNAL_NORMAL_10_LATEX_FILE, DF_EXTERNAL_NON_NORMAL_100_LATEX_FILE, \
    DF_EXTERNAL_NON_NORMAL_10_LATEX_FILE, DF_DISCRIMINANT_RAW_LATEX_FILE, DF_DISCRIMINANT_DOWNSAMPLED_LATEX_FILE, \
    distance_measure_evaluation_results_dir_for, STRUCTURAL_2_CI_FILENAME, get_latex_results_path
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType


def read_average_level_set_rate_of_increase(run_names: list, data_type: str, data_dir: str,
                                            root_results_dir: str) -> pd.DataFrame:
    """Per distance measure, per adjacent level-set pair: mean of the clustered mean difference
    (Mean diff, earlier level set minus later) across all 30 subjects. Signed, not abs'd: valid
    pairs are expected negative, a positive value means that pair decreased instead of increasing.
    Returns long-form columns=[DistanceMeasureCols.type, DistanceMeasureCols.compared,
    Aggregators.mean], one row per (distance_measure, pair)."""
    all_rows = []
    for run_name in run_names:
        result_dir = distance_measure_evaluation_results_dir_for(run_name=run_name, data_type=data_type,
                                                                  base_results_dir=root_results_dir, data_dir=data_dir)
        df = pd.read_csv(path.join(result_dir, STRUCTURAL_2_CI_FILENAME))
        all_rows.append(df[[DistanceMeasureCols.type, DistanceMeasureCols.compared, DistanceMeasureCols.mean_diff]])

    combined = pd.concat(all_rows, ignore_index=True)
    grouped = combined.groupby([DistanceMeasureCols.type, DistanceMeasureCols.compared])[DistanceMeasureCols.mean_diff]
    return grouped.mean().rename(Aggregators.mean).reset_index()


def _select_weakest_pair_idx(per_pair: pd.DataFrame) -> pd.Series:
    """Row label into per_pair of the weakest pair per distance measure: smallest positive mean
    if any pair decreased, else largest (least negative) mean. Two plain SeriesGroupBy
    reductions, not DataFrameGroupBy.apply on the whole group frame (which pandas deprecates for
    operating on the grouping column)."""
    decreased = per_pair[per_pair[Aggregators.mean] > 0]
    decreased_idx = decreased.groupby(DistanceMeasureCols.type)[Aggregators.mean].idxmin()
    fallback_idx = per_pair.groupby(DistanceMeasureCols.type)[Aggregators.mean].idxmax()
    return decreased_idx.combine_first(fallback_idx).astype(int)


def format_structural_2_column(per_pair: pd.DataFrame) -> pd.Series:
    weakest_idx = _select_weakest_pair_idx(per_pair)
    weakest = per_pair.loc[weakest_idx.values].copy()
    weakest.index = weakest_idx.index
    pair = weakest[DistanceMeasureCols.compared].astype(str).str.replace(', ', ',', regex=False)
    mean_str = weakest[Aggregators.mean].map(lambda x: f'{x:.2f}')
    return mean_str + ' ' + pair


def calculate_mean_sd_min_max(measures, run_names, data_type, data_dir, root_results_dir):
    """Mean, SD, min, max across subjects, for one data variant. Returns a MultiIndex-column
        df: level 0 = EvaluationCriteria, level 1 = Aggregators stat, index = distance_measure."""
    # Load all raw_criteria_data for this data variant
    measures = measures

    raw_dfs = []
    for run_name in run_names:
        raw_criteria_df = read_csv_of_raw_values_for_all_criteria(run_name=run_name, data_type=data_type,
                                                                  data_dir=data_dir,
                                                                  base_results_dir=root_results_dir)
        # filter measures and criteria
        raw_dfs.append(raw_criteria_df.loc[list(DM_THRESHOLDS.keys()), measures])

    # Stack all DataFrames along a new axis
    stacked_data = np.stack([df.values for df in raw_dfs]).astype(float)  # (subjects, criteria, measures)

    stat_values = {
        Aggregators.mean: np.mean(stacked_data, axis=0),
        Aggregators.std: np.std(stacked_data, axis=0, ddof=1),
        Aggregators.min: np.min(stacked_data, axis=0),
        Aggregators.max: np.max(stacked_data, axis=0),
    }

    # calculate all stats
    stat_dfs = []
    for stat, values in stat_values.items():
        df = pd.DataFrame(values, columns=raw_dfs[0].columns, index=raw_dfs[0].index).T
        df.columns = pd.MultiIndex.from_product([df.columns, [stat]])
        stat_dfs.append(df)

    return pd.concat(stat_dfs, axis=1).sort_index(axis=1, level=0)


def generate_latex_table_for_distance_measures(result_df: pd.DataFrame, distance_measures: list,
                                               criteria_order: list, footnote: str) -> str:
    ordered_distance_measures = DistanceMeasures.order_measures(distance_measures)
    columns = [criteria_short_names[c] for c in criteria_order]
    header = ' & '.join(EvaluationCriteriaLatex.header_for(c) for c in criteria_order)
    col_spec = 'l ' + ' '.join([r'>{\centering\arraybackslash}X'] * len(criteria_order))   # was: ' '.join(['c'] * len(criteria_order))

    lines = [r'\begin{tabularx}{\columnwidth}{' + col_spec + '}', r'\toprule',              # was: \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}' + col_spec + '}'
             '& ' + header + r' \\', r'\midrule']
    for dm in ordered_distance_measures:
        if dm not in result_df.index:
            raise ValueError(f"Distance measure '{dm}' missing from results")
        row_values = [result_df.loc[dm, col] for col in columns]
        lines.append('$' + DistanceMeasures.get_latex_for_measure(dm) + '$ & ' + ' & '.join(row_values) + r' \\')
    lines.append(r'\bottomrule')

    n_cols = len(criteria_order) + 1
    lines.append(r'\multicolumn{' + str(n_cols) + r'}{l}{$^*$ ' + footnote + r'.} \\')
    if EvaluationCriteria.inter_ii in criteria_order:
        lines.append(r'\multicolumn{' + str(n_cols) + r'}{l}{' + EvaluationCriteriaLatex.STRUCTURAL_2_FOOTNOTE_MARK +
                     ' ' + EvaluationCriteriaLatex.STRUCTURAL_2_FOOTNOTE_TEXT + r'.} \\')
    lines.append(r'\end{tabularx}')                                                          # was: \end{tabular*}
    return '\n'.join(lines)


@dataclass
class LatexOperator:
    """Token -> LaTeX rendering. The only place an operator token becomes a LaTeX command."""
    _latex: ClassVar[dict] = {
        Comparison.le: r'\leq', Comparison.ge: r'\geq', Comparison.eq: '=',
        Comparison.neq: r'\neq', Comparison.lt: '<', Comparison.gt: '>',
    }

    @staticmethod
    def latex_for(operator: str) -> str:
        return LatexOperator._latex[operator]


@dataclass
class EvaluationCriteriaLatex:
    """LaTeX macros from Table 1's threshold column. scale_free_inter_i/iii share \\cliffsDelta
    but are told apart by test label, not python field name. inter_ii (Structural 2) has no
    macro here: Table 1 gives it 'Pass/Fail', not a $symbol op value$ expression, so it's handled
    by its own footnote mark/text below instead of _macros/threshold_text_for."""
    _labels: ClassVar[dict] = {
        EvaluationCriteria.scale_free_inter_i: "Structural 1",
        EvaluationCriteria.inter_ii: "Structural 2",
        EvaluationCriteria.scale_free_inter_iii: "Structural 3",
        EvaluationCriteria.disc_iii: "Criterion",
    }
    _macros: ClassVar[dict] = {
        EvaluationCriteria.scale_free_inter_i: r"\cliffsDelta",
        EvaluationCriteria.scale_free_inter_iii: r"\cliffsDelta",
        EvaluationCriteria.disc_iii: "F_1",
    }
    STRUCTURAL_2_CONDITION: ClassVar[str] = r"$\avgLevelSetDistance[i] <^* \avgLevelSetDistance[j]$"
    STRUCTURAL_2_FOOTNOTE_MARK: ClassVar[str] = r"$^{**}$"
    STRUCTURAL_2_FOOTNOTE_TEXT: ClassVar[str] = (
        r"Mean $\min(\avgLevelSetDistance[i]-\avgLevelSetDistance[j])$ if $\geq 0$ or mean "
        r"$\max(\avgLevelSetDistance[i]-\avgLevelSetDistance[j])$ if $< 0$ "
        r"and indices $(\levelSetIndex_i, \levelSetIndex_j)$ for adjacent level-set pair closest to zero"
    )
    @staticmethod
    def header_for(criterion: str) -> str:
        if criterion == EvaluationCriteria.inter_ii:
            return f"{EvaluationCriteriaLatex._labels[criterion]}{EvaluationCriteriaLatex.STRUCTURAL_2_FOOTNOTE_MARK}"
        return f"{EvaluationCriteriaLatex._labels[criterion]} (${EvaluationCriteriaLatex._macros[criterion]}$)"

    @staticmethod
    def threshold_text_for(criterion: str, data_type: str = None) -> str:
        """The '$^*$' footnote entry: each criterion's own pass condition, stated once, not
        re-explained. Structural 2's condition has no operator/threshold pair to build from
        DM_THRESHOLDS, so its entry is the fixed STRUCTURAL_2_CONDITION string instead.
        scale_free_inter_iii's threshold is |Cliff's delta| > 0.4 (Table 1), so it's the only
        criterion wrapped in absolute-value bars here."""
        if criterion == EvaluationCriteria.inter_ii:
            return EvaluationCriteriaLatex.STRUCTURAL_2_CONDITION
        macro = EvaluationCriteriaLatex._macros[criterion]
        quantity = f"|{macro}|" if criterion in DistanceMeasureValidity.absolute_value_criteria else macro
        operator = DistanceMeasureValidity.comparison_symbol_for(criterion, data_type)
        return f"${quantity} {LatexOperator.latex_for(operator)} {DM_THRESHOLDS[criterion]}$"


def footnote_text_for(criteria: list, data_type: str = None) -> str:
    """'$^*$' footnote: each criterion's own pass condition, Structural 2 included on equal
    footing with the others (it has a condition, just not a numeric threshold)."""
    if data_type == SyntheticDataType.rs_1min:
        return r'Passed validity by degrading from Normal 100\%'
    thresholds = ', '.join(EvaluationCriteriaLatex.threshold_text_for(c, data_type) for c in criteria)
    return f'Passed validity threshold of {thresholds}'


if __name__ == "__main__":
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

    save_to_folder = path.join(root_result_dir, ResultsType.distance_measure_evaluation, 'validity-outcomes')
    os.makedirs(save_to_folder, exist_ok=True)

    variants = {
        'normal_100': (SyntheticDataType.normal_correlated, SYNTHETIC_DATA_DIR, DF_CONSTRUCT_NORMAL_LATEX_FILE),
        'normal_70': (SyntheticDataType.normal_correlated, IRREGULAR_P30_DATA_DIR, DF_EXTERNAL_NORMAL_70_LATEX_FILE),
        'normal_10': (SyntheticDataType.normal_correlated, IRREGULAR_P90_DATA_DIR, DF_EXTERNAL_NORMAL_10_LATEX_FILE),
        'non_normal_100': (SyntheticDataType.non_normal_correlated, SYNTHETIC_DATA_DIR,
                           DF_EXTERNAL_NON_NORMAL_100_LATEX_FILE),
        'non_normal_10': (SyntheticDataType.non_normal_correlated, IRREGULAR_P90_DATA_DIR,
                          DF_EXTERNAL_NON_NORMAL_10_LATEX_FILE),
        'raw_100': (SyntheticDataType.raw, SYNTHETIC_DATA_DIR, DF_DISCRIMINANT_RAW_LATEX_FILE),
        'downsampled_100': (SyntheticDataType.rs_1min, SYNTHETIC_DATA_DIR, DF_DISCRIMINANT_DOWNSAMPLED_LATEX_FILE),
    }

    # calculate and save stats df
    stats = {}
    for name, (data_type, data_dir, latex_file) in variants.items():
        stats[name] = calculate_mean_sd_min_max(distance_measures, run_names, data_type, data_dir, root_result_dir)
        stats[name].to_csv(path.join(save_to_folder, f'summary_statistics_{name}.csv'))

    # create validity assessment class
    validity = DistanceMeasureValidity(validity_rule=REVIEWED_RULES,
                                       normal_100=stats['normal_100'],
                                       normal_70=stats['normal_70'],
                                       normal_10=stats['normal_10'], non_normal_100=stats['non_normal_100'],
                                       non_normal_10=stats['non_normal_10'], raw_100=stats['raw_100'],
                                       downsampled_100=stats['downsampled_100'])

    # # save mean (sd) * table for each variant considered
    # for name in variants:
    #     validity.mean_sd_valid_summary_table(stats[name]).to_csv(
    #         path.join(save_to_folder, f'mean_sd_valid_{name}.csv'))

    # overall validity results
    validity.overall_validity().to_csv(path.join(save_to_folder, 'overall_validity_results.csv'))
    validity.external_validity_details().to_csv(path.join(save_to_folder, 'external_validity_results.csv'))
    validity.discriminant_validity_details().to_csv(path.join(save_to_folder, 'discriminant_validity_results.csv'))

    # latex result tables
    criteria_order = validity.default_criteria_order()
    for name, (data_type, data_dir, latex_file) in variants.items():
        pass_data_type = data_type if data_type in (SyntheticDataType.raw, SyntheticDataType.rs_1min) else None
        reference_df = stats['normal_100'] if pass_data_type == SyntheticDataType.rs_1min else None

        table = validity.mean_sd_valid_summary_table(stats[name], criteria_order, pass_data_type, reference_df)

        per_pair = read_average_level_set_rate_of_increase(run_names, data_type, data_dir, root_result_dir)
        missing = set(distance_measures) - set(per_pair[DistanceMeasureCols.type])
        if missing:
            raise ValueError(
                f"No Structural 2 rate-of-increase data found for distance measures: {missing} and data type: {data_type} and data directory: {data_dir}")
        inter_ii_star = validity.passes(stats[name], EvaluationCriteria.inter_ii, pass_data_type, reference_df) \
            .map({True: '*', False: ''})
        table[criteria_short_names[EvaluationCriteria.inter_ii]] = format_structural_2_column(per_pair) + inter_ii_star

        table.to_csv(path.join(save_to_folder, f'mean_sd_valid_{name}.csv'))

        footnote = footnote_text_for(criteria_order, pass_data_type)
        latex = generate_latex_table_for_distance_measures(table, distance_measures, criteria_order, footnote)
        with open(get_latex_results_path(save_to_folder, latex_file), 'w') as f:
            f.write(latex)
