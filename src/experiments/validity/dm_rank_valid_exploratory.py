import os
from os import path

import pandas as pd

from src.evaluation.distance_metric_ranking import read_csv_of_overall_rank_per_dataset
from src.experiments.run_distance_distance_metric_ranking import run_ranking_for
from src.experiments.run_distance_evaluation_raw_criteria import run_distance_evaluation_raw_criteria_for_ds
from src.utils.configurations import SYNTHETIC_DATA_DIR, \
    IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, VALID_ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, \
    get_latex_results_path, ResultsType, DataCompleteness
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType


def create_avg_rank_latex_table(data_dirs: list, dataset_types: list, distance_measures: list,
                                root_result_dir: str, overall_ds_name: str) -> str:
    """
    Builds the average-rank LaTeX table (mean rank across subjects/runs, +- SD, per distance
    measure) for the paper. data_dirs and dataset_types are used in the order given, no
    sorting, so columns/groups appear left-to-right in that order.

    Row order and LaTeX symbols come from DistanceMeasures.order_measures / get_latex_for_measure.
    Column and group headers come from SyntheticDataType.get_display_name_for_data_type and
    DataCompleteness.number_for_data_dir applied to the exact same data_type/data_dir
    pairs used to compute the cells, so headers and values can't drift apart.

    Significance stars are NOT calculated here, they must be added manually afterwards by
    cross-referencing stats_validation_average_ranks.csv (see NOTE in the output).
    """
    ordered_measures = DistanceMeasures.order_measures(distance_measures)

    # (data_type, data_dir) -> (mean_series, std_series), each indexed by distance_measures
    stats = {}
    for data_type in dataset_types:
        for data_dir in data_dirs:
            df = read_csv_of_overall_rank_per_dataset(overall_run_name=overall_ds_name, data_type=data_type,
                                                       data_dir=data_dir, base_results_dir=root_result_dir)
            df = df[ordered_measures]
            stats[(data_type, data_dir)] = (df.mean(axis=0), df.std(axis=0))

    def fmt(measure: str, data_type: str, data_dir: str) -> str:
        mean_series, std_series = stats[(data_type, data_dir)]
        return f"{mean_series[measure]:.2f} (SD {std_series[measure]:.2f})"

    n_per_group = len(data_dirs)
    n_cols = n_per_group * len(dataset_types)
    col_spec = "l " + " ".join(["c"] * n_cols)

    group_header = " & ".join(
        r"\multicolumn{%d}{c}{\textbf{%s}}" % (n_per_group, SyntheticDataType.get_display_name_for_data_type(data_type))
        for data_type in dataset_types)

    cmidrules = " ".join(
        r"\cmidrule(lr){%d-%d}" % (2 + i * n_per_group, 1 + (i + 1) * n_per_group)
        for i in range(len(dataset_types)))

    percent_cells = [r"\textbf{%d\%%}" % DataCompleteness.number_for_data_dir(data_dir) for data_dir in data_dirs]
    percent_header = " & ".join(percent_cells * len(dataset_types))

    rows = []
    for measure in ordered_measures:
        label = f"${DistanceMeasures.get_latex_for_measure(measure)}$"
        cells = [fmt(measure, data_type, data_dir) for data_type in dataset_types for data_dir in data_dirs]
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    rows_str = "\n".join(rows)

    footnote_measure = f"${DistanceMeasures.get_latex_for_measure(DistanceMeasures.l1_cor_dist)}$"

    return (
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}" + col_spec + "}\n"
        r"\toprule" + "\n"
        "& " + group_header + r" \\" + "\n"
        + cmidrules + "\n"
        "& " + percent_header + r" \\" + "\n"
        r"\midrule" + "\n"
        + rows_str + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular*}" + "\n"
    )


if __name__ == "__main__":
    # Calculate raw criteria and rankings only for valid DM
    overall_dataset_name = "n30"
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    # all variants but raw
    data_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated]
    data_dirs = [SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR]

    root_results_dir = VALID_ROOT_RESULTS_DIR

    # all distance measures
    # valid list of distance measures
    distance_measures = [DistanceMeasures.l1_cor_dist,  # lp norms
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.l1_with_ref,  # newly valid since reviewed tests
                         DistanceMeasures.dot_transform_l1,  # dot transform + lp norms
                         DistanceMeasures.dot_transform_l2,
                         ]

    # # 1. Calculate raw criteria for valid distance measures
    # # Recalculation would not be required but given the root_results dir is where we read and safe to this is simpler
    # # THIS SAVES TO VALIDITY_RESULTS HENCE WHY RECALCULATING - SIGH
    # run_distance_evaluation_raw_criteria_for_ds(data_dirs=data_dirs, dataset_types=data_types, run_names=run_names,
    #                                             root_result_dir=root_results_dir, distance_measures=distance_measures)
    #
    # # 2. Rank distance measures
    # run_ranking_for(data_dirs=data_dirs, dataset_types=data_types, run_names=run_names,
    #                 root_result_dir=root_results_dir, distance_measures=distance_measures,
    #                 overall_ds_name=overall_dataset_name)

    # 3. Create LaTeX table of average ranks for the paper
    latex = create_avg_rank_latex_table(data_dirs=data_dirs, dataset_types=data_types,
                                        distance_measures=distance_measures, root_result_dir=root_results_dir,
                                        overall_ds_name=overall_dataset_name)

    save_to_folder = path.join(root_results_dir, ResultsType.distance_measure_evaluation)
    os.makedirs(save_to_folder, exist_ok=True)
    latex_path = get_latex_results_path(results_dir=save_to_folder, filename="df-avg-ranks.tex")
    with open(latex_path, "w") as f:
        f.write(latex)

