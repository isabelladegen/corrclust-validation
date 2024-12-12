from dataclasses import dataclass
from os import path
from pathlib import Path

import pandas as pd
import numpy as np
import wandb
import string
import random
from numpy.linalg import LinAlgError
from scipy.stats import spearmanr

from matplotlib import pyplot as plt

from src.ticc.ticc_evaluation import TICCEvaluation
from experiments.evaluate.evaluate_against_ground_truth import EvaluateAgainstGroundTruth
from src.ticc.ticc_result import TICCResult
from src.stats import Stats, ConfidenceIntervalCols
from src.evaluation.clustering_result import SegmentValueClusterResult, segment_col, cluster_col
from src.utils.timeseries_utils import DailyTimeseries, Segmentation, TimeColumns
from src.utils.configurations import Hourly, Aggregators
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, fontsize, display_legend, \
    set_axis_label_font_size, up_block_size


@dataclass
class CommonLogKeys:
    """
    Keys Generic for different algorithms
    """
    number_of_segments = "Segments"
    number_of_observations_per_segment = "Observations per segment"
    min_segment_length = "Min segment length"
    max_segment_length = "Max segment length"
    mean_segment_length = "Mean segment length"
    number_of_times_each_cluster_is_used = "Times each cluster is used"
    number_of_observations = "Observations"

    # External Validity Index
    jaccard_index = "Jaccard iIndex"

    # Segment
    segment_table = "Segments Characteristics"
    segment_values_table = "Segment Values"

    # Correlation stuff
    overall_correlations = "Overall Correlations"

    # Distribution
    distribution_table = "Distribution"

    # Statistical tests
    overall_wilcoxon_table = "Overall Wilcoxon test"
    time_gaps_table = "Time Gaps"

    # Correlation stuff
    spearman_df = "Spearman Correlation"


def log_ticc_and_general_metrics(result: TICCResult, evaluation: TICCEvaluation,
                                 config,
                                 ground_truth_evaluation: EvaluateAgainstGroundTruth, keys: CommonLogKeys):
    wandb.log({
        keys.number_of_times_each_cluster_is_used: str(result.number_of_times_each_cluster_is_used()),
        keys.number_of_observations: len(result.cluster_assignment),
        keys.number_of_segments: result.number_of_segments(),
        keys.number_of_observations_per_segment: str(result.number_of_observations_per_segment()),
        keys.min_segment_length: result.min_segment_length(),
        keys.max_segment_length: result.max_segment_length(),
        keys.mean_segment_length: result.mean_segment_length(),
    })
    log_general_metrics(evaluation, config, ground_truth_evaluation, keys)


def log_general_metrics(evaluation: TICCEvaluation, config, ground_truth_evaluation: EvaluateAgainstGroundTruth,
                        keys: CommonLogKeys, patterns: {} = None):
    """ Log metrics applicable to multiple algorithms
    If patterns provided the overall correlations will compare if the cluster stayed within the overall pattern
    """

    print("IN LOGGING METRICS: EVALUATING EVALUATION METRICS")
    no_seg = evaluation.no_segments

    print("... log overall correlations")
    modified_patterns = patterns
    if patterns is not None:
        first_val = patterns[next(iter(patterns))]
        if isinstance(first_val, tuple):
            modified_patterns = {k: v[0] for k, v in patterns.items()}

    overall_correlations = evaluation.spearman_correlation_df(modified_patterns)
    overall_correlations_table = wandb.Table(dataframe=overall_correlations, allow_mixed_types=True)
    wandb.log({keys.overall_correlations: overall_correlations_table})

    if config.do_ground_truth_analysis:
        print("... Ground truth analysis")
        wandb.log({
            keys.jaccard_index: ground_truth_evaluation.jaccard_index,
        })


def log_ticc_and_general_figures_locally(config, evaluation: TICCEvaluation,
                                         ground_truth_evaluation: EvaluateAgainstGroundTruth):
    if not config.log_figures_local:
        return
    log_general_figures_locally(config, evaluation, ground_truth_evaluation)


def get_folder_name(config):
    run_name = wandb.run.name
    if run_name is None:
        run_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
    folder_name = path.join(config.local_image_folder_name, run_name)
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    return folder_name


def log_general_figures_locally(config, evaluation: TICCEvaluation,
                                ground_truth_evaluation: EvaluateAgainstGroundTruth):
    if not config.log_figures_local:
        return

    # Current figures I have don't log well on wandb, so instead logging locally with the run name
    folder_name = get_folder_name(config)

    if config.do_ground_truth_analysis:
        variate = config.columns[-1]  # pick a random column to visualise
        title = evaluation.alg_name
        if config.min_max_scaled:
            title = title + ": min-max scaled" + str(config.value_range)
        up_block_size(block_size=10000)
        gt_fig = ground_truth_evaluation.plot_ground_truth_against_algorithm_for(variate, title)
        gt_fig.savefig(path.join(folder_name, "ground_truth_segmentation.png"))
        plt.close(gt_fig)

    # plot euclidian silhouette analysis
    if config.do_euclidian_silhouette_analysis:
        try:
            euclidian_sil_analysis_fig = evaluation.plot_euclidean_silhouette_analysis()
            euclidian_sil_analysis_fig.savefig(path.join(folder_name, "euclidean_silhouette_analysis.png"))
            plt.close(euclidian_sil_analysis_fig)
        except (LinAlgError, ValueError) as e:
            print("Cannot plot Euclidean silhouette")

    # plot foerstner silhouette analysis
    if config.calculate_foerstner_distance:
        try:
            foerstner_sil_analysis_fig = evaluation.plot_foerstner_silhouette_analysis()
            if foerstner_sil_analysis_fig is not None:
                foerstner_sil_analysis_fig.savefig(path.join(folder_name, "foerstner_silhouette_analysis.png"))
                plt.close(foerstner_sil_analysis_fig)
        except (LinAlgError, ValueError) as e:
            print("Cannot plot foerstner silhouette")

    # plot log(cov) frobenius silhouette analysis
    if config.do_logcovfrob_silhouette_analysis:
        try:
            logcovfrob_sil_analysis_fig = evaluation.plot_log_cov_frobenius_silhouette_analysis()
            if logcovfrob_sil_analysis_fig is not None:
                logcovfrob_sil_analysis_fig.savefig(path.join(folder_name, "logcovfrob_silhouette_analysis.png"))
                plt.close(logcovfrob_sil_analysis_fig)
        except (LinAlgError, ValueError) as e:
            print("Cannot plot logcovfrob silhouette")

    # plot log(corr) frobenius silhouette analysis
    if config.do_logcorrfrob_silhouette_analysis:
        try:
            logcorrfrob_sil_analysis_fig = evaluation.plot_log_corr_frobenius_silhouette_analysis()  # todo
            if logcorrfrob_sil_analysis_fig is not None:
                logcorrfrob_sil_analysis_fig.savefig(path.join(folder_name, "logcorrfrob_silhouette_analysis.png"))
                plt.close(logcorrfrob_sil_analysis_fig)
        except (LinAlgError, ValueError) as e:
            print("Cannot plot logcorrfrob silhouette")

    # plot segment length box & violin graphs
    plot_segment_lengths_box_and_violin_for_all_clusters(evaluation, folder_name)

    # plot values box plots & violin
    plot_segment_values_box_and_violin_for_all_clusters(evaluation, folder_name)

    # original data and x-train data box & violin plots
    original_box_plot = evaluation.plot_box_or_violin_plot_for_unclustered_data(original=True, box_plot=True)
    original_box_plot.savefig(path.join(folder_name, "original_boxplot.png"))
    plt.close(original_box_plot)
    original_violin_plot = evaluation.plot_box_or_violin_plot_for_unclustered_data(original=True, box_plot=False)
    original_violin_plot.savefig(path.join(folder_name, "original_violinplot.png"))
    plt.close(original_violin_plot)
    xtrain_box_plot = evaluation.plot_box_or_violin_plot_for_unclustered_data(original=False, box_plot=True)
    xtrain_box_plot.savefig(path.join(folder_name, "xtrain_boxplot.png"))
    plt.close(xtrain_box_plot)
    xtrain_violin_plot = evaluation.plot_box_or_violin_plot_for_unclustered_data(original=False, box_plot=False)
    xtrain_violin_plot.savefig(path.join(folder_name, "xtrain_violinplot.png"))
    plt.close(xtrain_violin_plot)

    if config.has_datetime_index & config.plot_agp_like_graphs:
        plot_agp_like_graphs(evaluation, folder_name, config)

    # plot distribution for segments
    if config.plot_distribution_profile_of_segment:
        for cluster_id in evaluation.cluster_ids:
            columns = config.columns

            for cl in columns:
                tmp_fig = evaluation.plot_distribution_profile_for(cluster_id, cl)
                tmp_fig.savefig(
                    path.join(folder_name,
                              "distribution_profile_for_cluster_" + str(cluster_id) + "_" + str(cl) + ".png"))
                plt.close(tmp_fig)

    if config.has_datetime_index & config.plot_time_heatmaps:
        # only makes sense if a datatime is provided
        # plot cluster assignment heatmap over time
        cluster_as_heatmap = evaluation.plot_cluster_mode_heatmap_over_days_and_months()
        cluster_as_heatmap.savefig(path.join(folder_name, "cluster_assignment_mode_over_time.png"))
        plt.close(cluster_as_heatmap)

        # plot cluster assignments for each of the clusters
        for cluster_id in evaluation.cluster_ids:
            file_name = "cluster_assignment_for_cluster_" + str(cluster_id) + "_over_time.png"
            cluster_mode_heatmap = evaluation.plot_cluster_mode_heatmap_over_days_and_months(cluster=cluster_id)
            cluster_mode_heatmap.savefig(path.join(folder_name, file_name))
            plt.close(cluster_mode_heatmap)

        # plot segment time gap histograms
        tg_histograms = evaluation.plot_histograms_for_time_gaps()
        tg_histograms.savefig(path.join(folder_name, "time_gap_histograms.png"))
        plt.close(tg_histograms)

    # plot segments ts
    if config.plot_resulting_segments:
        plot_segments_graphs_for_all_clusters(evaluation, folder_name)

    # plot confidence intervalls
    # create stats results
    stats_dict = {}
    seg_values = evaluation.segment_value_results
    for k in seg_values.cluster_ids():
        cluster_df = seg_values.df[seg_values.df[cluster_col] == k]
        hours = TimeColumns.hour
        stats = Stats(cluster_df, config.sampling, DailyTimeseries(), [hours], seg_values.scaled_columns)
        stats_df = stats.stats_per_time[hours][
            [ConfidenceIntervalCols.ci_96lo, ConfidenceIntervalCols.ci_96hi, Aggregators.mean, Aggregators.count]]
        stats_dict[k] = stats_df
    stats_results = pd.concat(stats_dict.values(), axis=1, keys=stats_dict.keys())
    ci_figure = plot_ci_intervals_of_mean_for_variates_with_clusters_as_rows(seg_values,
                                                                             stats_results,
                                                                             plot_columns=config.plot_columns,
                                                                             plot_segmentation=DailyTimeseries(),
                                                                             title="CI Intervals",
                                                                             show_title=True,
                                                                             backend=config.backend)
    ci_figure.savefig(path.join(folder_name, "ci_over_day.png"))
    plt.close(ci_figure)


def plot_segment_lengths_box_and_violin_for_all_clusters(evaluation, folder_name):
    seg_lengths_fig = evaluation.plot_box_or_violin_plots_for_segment_length()
    seg_lengths_fig.savefig(path.join(folder_name, "segment_lengths_boxplot.png"))
    plt.close(seg_lengths_fig)
    seg_lengths_violin_fig = evaluation.plot_box_or_violin_plots_for_segment_length(box_plot=False)
    seg_lengths_violin_fig.savefig(path.join(folder_name, "segment_lengths_violinplot.png"))
    plt.close(seg_lengths_violin_fig)


def plot_segment_values_box_and_violin_for_all_clusters(evaluation, folder_name):
    values_box_plot_fig = evaluation.plot_box_or_violin_plots_values_over_clusters()
    values_box_plot_fig.savefig(path.join(folder_name, "values_clusters_boxplot.png"))
    plt.close(values_box_plot_fig)
    values_violin_plot_fig = evaluation.plot_box_or_violin_plots_values_over_clusters(box_plot=False)
    values_violin_plot_fig.savefig(path.join(folder_name, "values_clusters_violinplot.png"))
    plt.close(values_violin_plot_fig)


def plot_segments_graphs_for_all_clusters(evaluation, folder):
    for cluster_id in evaluation.cluster_ids:
        tmp_fig = evaluation.plot_segments_for_cluster(cluster_id)
        if tmp_fig:
            tmp_fig.savefig(path.join(folder, "segments_for_cluster_" + str(cluster_id) + ".png"))
            plt.close(tmp_fig)


def plot_agp_like_graphs(evaluation, folder, config, plot_scaled_data: bool = False, plot_clusters_alone: bool = False):
    # agp like graph for hours of the day and months
    if evaluation.no_clusters > 3 or plot_clusters_alone:
        # paint by cluster
        for cluster_id in evaluation.cluster_ids:
            name_hours = "aib_acb_agp_hours_cluster_" + str(cluster_id) + ".png"
            name_months = "aib_acb_agp_months_cluster_" + str(cluster_id) + ".png"
            name_years = "aib_acb_agp_years_cluster_" + str(cluster_id) + ".png"

            profiles_hours = evaluation.plot_aip_acp_agp(cluster_id=cluster_id, norm_y_axis=True,
                                                         plot_scaled_data=plot_scaled_data)
            profiles_hours.savefig(path.join(folder, name_hours))
            plt.close(profiles_hours)

            profiles_months = evaluation.plot_aip_acp_agp(cluster_id=cluster_id, time_col=TimeColumns.month,
                                                          norm_y_axis=True, plot_scaled_data=plot_scaled_data)
            profiles_months.savefig(path.join(folder, name_months))
            plt.close(profiles_months)

            profiles_year = evaluation.plot_aip_acp_agp(cluster_id=cluster_id, time_col=TimeColumns.year,
                                                        norm_y_axis=True, plot_scaled_data=plot_scaled_data)
            profiles_year.savefig(path.join(folder, name_years))
            plt.close(profiles_year)

    else:
        profiles_hours = evaluation.plot_aip_acp_agp(norm_y_axis=True, plot_scaled_data=plot_scaled_data)
        profiles_hours.savefig(path.join(folder, "aib_acb_agp_hours.png"))
        plt.close(profiles_hours)

        profiles_months = evaluation.plot_aip_acp_agp(time_col=TimeColumns.month, norm_y_axis=True,
                                                      plot_scaled_data=plot_scaled_data)
        profiles_months.savefig(path.join(folder, "aib_acb_agp_months.png"))
        plt.close(profiles_months)

        profiles_year = evaluation.plot_aip_acp_agp(time_col=TimeColumns.year, norm_y_axis=True,
                                                    plot_scaled_data=plot_scaled_data)
        profiles_year.savefig(path.join(folder, "aib_acb_agp_years.png"))
        plt.close(profiles_year)


def plot_ci_intervals_of_mean_for_variates_with_clusters_as_rows(segment_value_results: SegmentValueClusterResult,
                                                                 stats_results: pd.DataFrame,
                                                                 plot_columns: [],
                                                                 backend: Backends,
                                                                 plot_segmentation: Segmentation = DailyTimeseries(),
                                                                 title: str = "",
                                                                 show_title: bool = True,
                                                                 use_actual_cluster_names: bool = False,
                                                                 show_ts_count: bool = True,
                                                                 plot_count: bool = True,
                                                                 y_lim: int = 5,
                                                                 fig_size: () = ()):
    clusters = list(stats_results.columns.get_level_values(0).unique())
    no_clusters = len(clusters)
    scaled_cols = segment_value_results.scaled_columns

    # setup figure
    reset_matplotlib(backend)
    if len(fig_size) == 0:
        fig_size = fig_size = (15, no_clusters * 4)
    fig, axs = plt.subplots(nrows=no_clusters,
                            ncols=1,
                            sharey=True,
                            sharex=True,
                            figsize=fig_size, squeeze=0)
    if show_title:
        fig.suptitle(title, fontsize=fontsize)

    # clusters are on the rows
    for row_idx, cluster in enumerate(clusters):
        df = stats_results[cluster]
        ax = axs[row_idx, 0]
        ts_in_cluster = segment_value_results.data_for_cluster(cluster)[segment_col].unique()

        for vidx, variate in enumerate(plot_columns):
            # plot barycenter for cluster
            line, = ax.plot(df[Aggregators.mean][scaled_cols[vidx]], "-", marker='.', label=variate)
            ci_lo = df[ConfidenceIntervalCols.ci_96lo][scaled_cols[vidx]]
            ci_hi = df[ConfidenceIntervalCols.ci_96hi][scaled_cols[vidx]]
            # use only the hours we have
            x = list(ci_lo.index)
            ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=line.get_color())

        ax.set_xticks(plot_segmentation.x_ticks)
        # TODO this should be the actual scale
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_ylim(0, y_lim)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.grid(which='major', alpha=0.2, color='grey')
        ax.zorder = 1
        ax.patch.set_visible(False)

        # add count with different y-axis scales
        count_ax = ax.twinx()
        count_ax.zorder = 0
        count_ax.set_ylim(0, 550)

        count_data = list(df[Aggregators.count][scaled_cols[0]])
        x = range(0, len(count_data))
        count_ax.plot(x, count_data, marker='', linewidth=1, drawstyle="steps", color='grey')
        count_ax.set_ylabel("Count", fontsize=fontsize, color="grey")
        count_ax.tick_params(axis='y', labelcolor="grey")
        set_axis_label_font_size(count_ax)

        # set y label for row with cluster information, use idx to have nice cluster names from
        cluster_name = str(cluster) if use_actual_cluster_names else str(row_idx + 1)
        if show_ts_count:
            ax.set_ylabel('Cluster ' + cluster_name + '\n TS = ' + str(len(ts_in_cluster)), fontsize=fontsize)
        else:
            ax.set_ylabel('Cluster ' + cluster_name, fontsize=fontsize)

    if not plot_count:
        display_legend(axs[0, 0], fig)

    plt.subplots_adjust(top=.9)
    # add overall x, y text
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(plot_segmentation.description, labelpad=fontsize, fontsize=fontsize)
    plt.show()
    return fig


def log_general_tables(evaluation: TICCEvaluation, keys: CommonLogKeys, config):
    """ Log algorithm independent tables"""
    # log segment values table - this will allow to rerun
    if config.log_segment_values_table:
        segments_df = evaluation.segment_value_results.df.copy()
        if config.has_datetime_index:
            segments_df.reset_index(inplace=True)  # to ensure daytime gets saved as column
            segments_df['datetime'] = segments_df['datetime'].dt.strftime('%Y/%m/%d %H:%M:%S%z')
        segment_values_table = wandb.Table(dataframe=segments_df, allow_mixed_types=True)
        wandb.run.log({keys.segment_values_table: segment_values_table})

    # log segment df
    segment_table = wandb.Table(dataframe=evaluation.segment_df)
    wandb.run.log({keys.segment_table: segment_table})

    # log distribution df
    distribution_table = wandb.Table(dataframe=evaluation.distribution_table_df(), allow_mixed_types=True)
    wandb.run.log({keys.distribution_table: distribution_table})

    # log spearman correlation df
    indices = np.triu_indices(len(config.columns), k=1)
    seg_value = evaluation.segment_value_results
    clusters = []
    correlations = []
    p_values = []
    for k in evaluation.cluster_ids:
        data_for_k = seg_value.data_for_cluster(k)[seg_value.scaled_columns]
        result = spearmanr(data_for_k.to_numpy())
        clusters.append(k)
        if np.isnan(result.statistic).any():
            cor = None
            p = None
        else:
            cor = result.statistic[indices]
            p = result.pvalue[indices]
        correlations.append(cor)
        p_values.append(p)
    spearmancor_df = pd.DataFrame({cluster_col: clusters, 'correlation': correlations, "p-values": p_values})
    spearman_table = wandb.Table(dataframe=spearmancor_df, allow_mixed_types=True)
    wandb.run.log({keys.spearman_df: spearman_table})

    # log overall statistical differences between clusters df
    overall_wilcoxon_table = wandb.Table(
        dataframe=evaluation.wilcoxon_test_for_cluster_and_variate_df(), allow_mixed_types=True)
    wandb.run.log({keys.overall_wilcoxon_table: overall_wilcoxon_table})

    if config.has_datetime_index:
        # log time gaps overall and in segments
        temp_df = evaluation.time_gap_df()
        temp_df["gaps"] = temp_df["gaps"].astype(str)  # required for JASON serialisation
        time_gaps_table = wandb.Table(dataframe=temp_df, allow_mixed_types=True)
        wandb.run.log({keys.time_gaps_table: time_gaps_table})


def log_network_centrality_metrics_for_t0(keys, result):
    node_columns = result.times_series_names
    betweenness = result.betweenness_centrality_for_all_cluster()
    closeness = result.closeness_centrality_for_all_clusters()
    degree = result.degree_centrality_for_all_clusters()
    betweenness_wandb = wandb.Table(data=betweenness, columns=node_columns)
    closeness_wandb = wandb.Table(data=closeness, columns=node_columns)
    degree_wandb = wandb.Table(data=degree, columns=node_columns)
    wandb.run.log({keys.betweenness_centrality: betweenness_wandb})
    wandb.run.log({keys.closeness_centrality: closeness_wandb})
    wandb.run.log({keys.degree_centrality: degree_wandb})
