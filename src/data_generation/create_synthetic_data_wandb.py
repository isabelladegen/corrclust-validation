import gc
import itertools
import traceback
from dataclasses import dataclass, field, asdict
from os import path

import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import genextreme, nbinom, lognorm

import wandb
from experiments.evaluate.evaluate_against_ground_truth import EvaluateAgainstGroundTruth
from scipy.stats._distn_infrastructure import rv_generic

from src.utils.wand_db_common import CommonLogKeys, get_folder_name, log_general_metrics, log_general_figures_locally
from experiments.run_ticc_on_openaps_data_log_to_wandb import WhatData
from src.utils.configurations import Resampling, Hourly, Configuration, GeneralisedCols, ROOT_DIR, WandbConfiguration, \
    SyntheticDataVariates, SYNTHETIC_DATA_DIR
from src.utils.plots.matplotlib_helper_functions import Backends
from src.evaluation.distribution_fit import DistributionFit
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticSegmentedData, SyntheticDataSegmentCols, \
    min_max_scaled_df
from src.evaluation.clustering_result import SegmentValueClusterResult, cluster_col
from src.ticc.ticc_evaluation import TICCEvaluation


@dataclass
class SyntheticDataConfig:
    wandb_project_name: str = WandbConfiguration.wandb_project_name
    wandb_entity: str = WandbConfiguration.wandb_entity
    wandb_mode: str = 'online'
    wandb_notes = "creates synthetic data, uncorrelated, normal and correlated"
    tags = ['Synthetic']

    correlation_model = "loadings"

    # DATA
    min_max_scaled: bool = False
    value_range: (float, float) = (0., 10.)
    columns: [str] = field(default_factory=lambda: SyntheticDataVariates.columns())
    plot_columns: [str] = field(default_factory=lambda: SyntheticDataVariates.plot_columns())

    # how to load data (standard or from a segment_value_df
    load_data_method: str = WhatData.original.value  # original loads data as usually

    # preloaded data - this helps keep a record of what files were loaded in wandb
    dataset_to_load: str = ""
    data_type: str = ""

    # data generation config
    number_of_variates: int = 3
    number_of_segments: int = 100
    max_repetitions: int = 1000
    downsampling_rule: str = "1min"
    short_segment_durations: [int] = field(
        default_factory=lambda: [15 * 60, 20 * 60, 30 * 60, 60 * 60, 120 * 60, 180 * 60, 180 * 60, 180 * 60, 180 * 60,
                                 240 * 60])  # * 60 for seconds
    long_segment_durations: [int] = field(
        default_factory=lambda: [360 * 60, 480 * 60, 600 * 60, 720 * 60])  # * 60 for seconds

    distributions_for_variates: [rv_generic] = field(default_factory=lambda: [genextreme, nbinom, genextreme])
    # iob distribution parameters
    c_iob: float = -0.22
    loc_iob: float = 0.5
    scale_iob: float = 1.52
    # cob distribution parameters
    n_cob: int = 1  # number of successes
    p_cob: float = 0.05  # likelihood of success
    # ig distribution parameters
    c_ig: float = 0.04
    loc_ig: float = 119.27
    scale_ig: float = 39.40

    # EVALUATION
    local_image_folder_name: str = "wandb-images/synthetic-data"
    log_figures_local: bool = True  # if false no figures are logged
    max_segment: int = 2500  # after this the covariance based sil score is no longer evaluated
    backend: str = Backends.none.value
    do_distribution_fit: bool = True
    plot_resulting_segments: bool = True
    plot_distribution_profile_of_segment: bool = True
    log_segment_values_table: bool = True
    has_datetime_index: bool = True  # this will decide if datetime analysis is run

    def as_dict(self):
        return asdict(self)


@dataclass
class SyntheticGroundTruthFiles:
    uncorrelated_splendid_sunset: str = "splendid-sunset-12-gt-uncorrelated"
    uncorrelated_splendid_sunset_1min: str = "1min-splendid-sunset-12-gt-uncorrelated"


@dataclass
class SyntheticDataLogKeys(CommonLogKeys):
    """
    Keys specific to generating synthetic data
    """
    generated_segment_table: str = "generated segment"
    generated_data_table: str = "generated data"
    resampled_segment_table: str = "resampled segment"
    resampled_data_table: str = "resampled data"
    n_cor_success: str = "n cor success"
    n_cor_failure: str = "n cor failure"
    all_cor_successful: str = "all cor successful"
    n_cor_success_resampled: str = "n cor success resampled"
    n_cor_failure_resampled: str = "n cor failure resampled"
    all_cor_successful_resampled: str = "all cor successful resampled"
    n_observations_downsampled: str = "Observations resampled"
    distribution_fit: str = "Distribution fit "
    start_date: str = "Start date"
    end_date: str = "End date"
    clustering_jaccard_coef: str = "Clustering Jaccard coeff"


def one_synthetic_creation_run(config: SyntheticDataConfig, data_df: pd.DataFrame = None,
                               labels_df: pd.DataFrame = None):
    """
    Wandb synthetic data creation run
    :param config: wandb config
    :param data_df: default none, if provided no data is generated analysis is run on the data provided
    :param labels_df: default none, only required if data_df is provided
    :return: evaluation class
    """
    try:
        wandb.init(project=config.wandb_project_name,
                   entity=config.wandb_entity,
                   mode=config.wandb_mode,
                   notes=config.wandb_notes,
                   tags=config.tags,
                   config=config.as_dict())

        # Save mode inputs and hyperparameters (this allows for sweep)
        # skip as not doing sweeps so config won't be rehydrated

        args_iob: tuple = (config.c_iob,)
        kwargs_iob = {'loc': config.loc_iob, 'scale': config.scale_iob}
        args_cob = (config.n_cob, config.p_cob)
        kwargs_cob = {}  # none, loc will be 0
        args_ig = (config.c_ig,)
        kwargs_ig = {'loc': config.loc_ig, 'scale': config.scale_ig}
        distributions_args = [args_iob, args_cob, args_ig]
        distributions_kwargs = [kwargs_iob, kwargs_cob, kwargs_ig]

        keys = SyntheticDataLogKeys()
        if config.correlation_model == "cholesky":
            patterns = ModelCorrelationPatterns().patterns_to_model()
        elif config.correlation_model == "loadings":
            patterns = ModelCorrelationPatterns().ideal_correlations()
        else:
            assert False, "Unknown correlation model method {}".format(config.correlation_model)
        exit_code = 0
        generator = None

        run_name = wandb.run.name
        if run_name == "":
            run_name = 'test-run'

        # Generate data
        if data_df is None:
            print("1. GENERATING DATA")

            generator = SyntheticSegmentedData(config.number_of_segments, config.number_of_variates,
                                               config.distributions_for_variates,
                                               distributions_args, distributions_kwargs, config.short_segment_durations,
                                               config.long_segment_durations, patterns, config.columns,
                                               config.correlation_model,
                                               config.max_repetitions)
            generator.generate()

            print("2. DOWNSAMPLE")
            generator.resample(rule=config.downsampling_rule)

            print("3. LOG GENERATED DATA")
            # Original second sampled
            labels_df = generator.generated_segment_df
            generated_segment_table = wandb.Table(dataframe=labels_df, allow_mixed_types=True)
            wandb.log({keys.generated_segment_table: generated_segment_table})

            data_df = generator.generated_df.copy()
            data_df[GeneralisedCols.datetime] = data_df[GeneralisedCols.datetime].dt.strftime(
                '%Y/%m/%d %H:%M:%S%z')
            generated_df_table = wandb.Table(dataframe=data_df, allow_mixed_types=True)
            wandb.log({keys.generated_data_table: generated_df_table})

            # Downsampled data
            resampled_segment_df = generator.resampled_segment_df
            resampled_segment_table = wandb.Table(dataframe=resampled_segment_df, allow_mixed_types=True)
            wandb.log({keys.resampled_segment_table: resampled_segment_table})

            resampled_df = generator.resampled_data.copy()
            resampled_df = resampled_df.reset_index(drop=False).rename(columns={'index': GeneralisedCols.datetime})
            resampled_df[GeneralisedCols.datetime] = resampled_df[GeneralisedCols.datetime].dt.strftime(
                '%Y/%m/%d %H:%M:%S%z')
            resampled_table = wandb.Table(dataframe=resampled_df, allow_mixed_types=True)
            wandb.log({keys.resampled_data_table: resampled_table})

            # not all tables are fully logged on wandb, saving them directly to data
            print("...saving generated data to csv")

            generated_data_file = path.join(SYNTHETIC_DATA_DIR, run_name + '-data.csv')
            data_df.to_csv(generated_data_file)

            generated_normal_data_file = path.join(SYNTHETIC_DATA_DIR, run_name + '-normal-data.csv')
            generator.normal_generated_df().to_csv(generated_normal_data_file)

            generated_normal_correlated_data_file = path.join(SYNTHETIC_DATA_DIR,
                                                              run_name + '-normal-correlated-data.csv')
            generator.normal_correlated_generated_df().to_csv(generated_normal_correlated_data_file)

            generated_segment_file = path.join(SYNTHETIC_DATA_DIR, run_name + '-labels.csv')
            labels_df.to_csv(generated_segment_file)

            downsampled_generated_data_file = path.join(SYNTHETIC_DATA_DIR,
                                                        config.downsampling_rule + "-" + run_name + '-data.csv')
            resampled_df.to_csv(downsampled_generated_data_file)

            downsampled_generated_segment_file = path.join(SYNTHETIC_DATA_DIR,
                                                           config.downsampling_rule + "-" + run_name + '-labels.csv')
            resampled_segment_df.to_csv(downsampled_generated_segment_file)

            # do we need to store segment values version for future analysis

            print("4. LOG CORRELATION METRICS")
            list_of_lists = labels_df[SyntheticDataSegmentCols.actual_within_tolerance].values
            flat_result = list(itertools.chain.from_iterable(list_of_lists))
            n_cor_success = sum(flat_result)
            n_cor_failure = len(flat_result) - n_cor_success

            list_of_lists = resampled_segment_df[SyntheticDataSegmentCols.actual_within_tolerance].values
            down_sampled_flat_results = list(itertools.chain.from_iterable(list_of_lists))
            n_cor_success_resampled = sum(down_sampled_flat_results)
            n_cor_failure_resampled = len(down_sampled_flat_results) - n_cor_success_resampled
            wandb.log({
                keys.n_cor_success: n_cor_success,
                keys.n_cor_failure: n_cor_failure,
                keys.all_cor_successful: all(flat_result),
                keys.n_cor_success_resampled: n_cor_success_resampled,
                keys.n_cor_failure_resampled: n_cor_failure_resampled,
                keys.all_cor_successful_resampled: all(down_sampled_flat_results),
                keys.number_of_segments: config.number_of_segments,  # just for ease of comparing runs
                keys.number_of_observations: data_df.shape[0],
                keys.n_observations_downsampled: resampled_df.shape[0],
                keys.start_date: data_df[GeneralisedCols.datetime].min(),
                keys.end_date: data_df[GeneralisedCols.datetime].max()
            })

        # this is done always either on preloaded dfs or on generated ones
        print("5. CREATE EVALUATION CLASSES")
        if config.min_max_scaled:
            data_df = min_max_scaled_df(data_df, scale_range=config.value_range, columns=config.columns)
            # save data so we can use it in describe
            generated_scaled_data_file = path.join(SYNTHETIC_DATA_DIR, run_name + 'scaled-data.csv')
            data_df.to_csv(generated_scaled_data_file)

        data_as_numpy = data_df[config.columns].to_numpy()
        segment_value_result = SegmentValueClusterResult.create_from_segment_df(labels_df, data_df,
                                                                                data_as_numpy,
                                                                                config.columns)
        evaluation = None
        evaluation = TICCEvaluation(segment_value_result, config.cov_regularisation, max_seg=config.max_segment,
                                    backend=config.backend, algorithm_name="synthetic datagen")

        gt_evaluation = None
        if config.do_ground_truth_analysis:
            print("... create ground truth evaluation")
            # a dictionary with key being the end index of a segment and values being the label for that segment
            labels_dict = dict(zip(labels_df[SyntheticDataSegmentCols.end_idx],
                                   labels_df[SyntheticDataSegmentCols.pattern_id]))
            gt_evaluation = EvaluateAgainstGroundTruth(segment_value_result, labels_dict, backend=config.backend)

        print("6. LOG EVALUATION METRICS")
        log_general_metrics(evaluation, config, gt_evaluation, keys, patterns=patterns)

        if config.calculate_clustering_jaccard_coefficient:
            print("... Calculate clustering jaccard coefficient")
            # create segment value for ground truth labels
            if ground_truth_segment_df is None:
                gt_segment_value_result = segment_value_result  # ground truth is the same as the partition calculated
            else:
                gt_segment_value_result = SegmentValueClusterResult.create_from_segment_df(ground_truth_segment_df,
                                                                                           data_df,
                                                                                           data_as_numpy,
                                                                                           config.columns)

            gt_clusters = gt_segment_value_result.df[cluster_col]
            clustering_jaccard_coefficient = evaluation.clustering_jaccard_coeff(list(gt_clusters))
            wandb.log({
                keys.clustering_jaccard_coef: clustering_jaccard_coefficient})

        print("7. LOG FIGURES")
        if config.log_figures_local:
            folder_name = get_folder_name(config)

            if config.do_distribution_fit:
                # plot distribution for each segment only if data has been generated in this run
                if generator is not None:
                    for segment_id in range(config.number_of_segments):
                        fig = generator.plot_distribution_for_segment(segment_id, backend=config.backend)
                        fig.savefig(path.join(folder_name, str(segment_id) + "_segment_pdf_histogram.png"))
                        plt.close(fig)

                print("... fit distribution to overall data for each variate")
                genextreme_bounds = {'c': (-1, 1), 'loc': (-10, 150), 'scale': (0, 100)}
                nbinom_bounds = {'n': (0, 50)}
                lognorm_bounds = {'s': (-1, 5), 'loc': (-1, 200), 'scale': (0, 100)}
                bounds = [genextreme_bounds, lognorm_bounds, nbinom_bounds]
                fit_distributions = [genextreme, lognorm, nbinom]

                for idx, variate_name in enumerate(config.columns):
                    dist_fit = DistributionFit(data_as_numpy[:, idx], fit_distributions, bounds)
                    dist_fit.fit()

                    dist_fit_table = wandb.Table(dataframe=dist_fit.summary_df, allow_mixed_types=True)
                    log_key = keys.distribution_fit + "_" + variate_name
                    wandb.log({log_key: dist_fit_table})

                    for dist in fit_distributions:
                        fig = dist_fit.plot_results_for(dist, backend=config.backend)
                        fig.savefig(path.join(folder_name,
                                              variate_name + "_" + dist.name + "_distribution_fit_overall_segments.png"))
                        plt.close(fig)

            # plot general figures like for TICC
            print("... general figures")
            log_general_figures_locally(config, evaluation, gt_evaluation)

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)  # try to get error into wandb log
        exit_code = 1

    wandb.finish(exit_code=exit_code)
    gc.collect()
    if exit_code == 1:
        raise
    return evaluation


def main():
    config = SyntheticDataConfig()
    # Blooming donkey: create normal uncorrelated and distribution shifted data, just one segment
    # config.number_of_segments = 1
    # config.short_segment_durations = [1226400]  # all observations in one huge segment

    # create 30 datasets run
    config.log_figures_local = False
    config.tags.append("30_ds_creation")

    for n in range(30):
        one_synthetic_creation_run(config)


if __name__ == "__main__":
    main()
