from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd
import wandb
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols

from src.use_case.algorithm_evaluation import AlgorithmEvaluation
from src.use_case.ticc.TICC_solver import TICC
from src.use_case.wandb_run_ticc import TICCWandbUseCaseConfig
from src.utils.configurations import WandbConfiguration, SyntheticDataVariates, SYNTHETIC_DATA_DIR, \
    GENERATED_DATASETS_FILE_PATH, DataCompleteness, get_data_dir, get_algorithm_use_case_result_dir, \
    ROOT_RESULTS_DIR
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data, SyntheticFileTypes
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.run_average_rank_visualisations import data_variant_description
from tests.use_case.ticc.test_ticc_runs_on_original_test_data import TICCSettings

def one_run_ticc(original_config: TICCWandbUseCaseConfig, subject: str, save_results: bool = False):
    """
    One run of ticc for given config and subject
    :param original_config: TICCWandbUseCaseConfig that configures which data variant and subjects TICC is trained and run on
    :param subject: run_name of the subject to use
    :return: evaluates dict with key subject and value AlgorithmEvaluation, wandb summaries with key subject and value
     wandb dict with the keys and values logged to wandb
    """
    # pre wandb setup
    evaluates = {}  # per subject, per run evaluate objects
    wandb_summaries = {}  # per subject wandb run summary

    # string description of data variant
    variant_description = data_variant_description[(original_config.completeness_level, original_config.data_type)]
    # add data variant to wandb run name
    run_name = subject + ":" + variant_description

    wandb.init(project=original_config.wandb_project_name,
               entity=original_config.wandb_entity,
               name=run_name,
               mode=original_config.wandb_mode,
               notes=original_config.wandb_notes,
               tags=original_config.tags,
               config=original_config.as_dict())

    # Rehydrate config (this allows for sweep)
    config = TICCWandbUseCaseConfig(**wandb.config)
    tags = original_config.tags  # don't translate for sweeps so doing it like this
    # store the dataclasses back from the original config
    config.tags = tags

    ticc = TICC(window_size=config.window_size,
                number_of_clusters=config.number_of_clusters,
                lambda_parameter=config.lambda_var,
                beta=config.switch_penalty,
                max_iters=config.max_iter,
                threshold=config.threshold,
                biased=config.biased,
                allow_zero_cluster_inbetween=config.allow_zero_cluster_inbetween,
                do_training_split=config.do_training_split,
                cluster_reassignment=config.cluster_reassignment,
                keep_track_of_assignments=config.keep_track_of_assignments,
                backend=config.backend)

    print("DATA VARIANT: " + variant_description)
    print("LOAD GROUND TRUTH DATA")
    data_dir = get_data_dir(root_data_dir=config.root_data_dir, extension_type=config.completeness_level)
    data_df, gt_labels_df = load_synthetic_data(subject, config.data_type, data_dir)

    data_np = data_df[config.data_cols].to_numpy()


    print("TRAIN TICC ON SUBJECT: " + subject)
    result = ticc.fit(data=data_np, use_gmm_initialisation=config.use_gmm_initialisation,
                      reassign_points_to_zero_clusters=config.reassign_points_to_zero_clusters)

    result_labels_df = result.to_labels_df(subject)

    # log results df
    results_labels_table = wandb.Table(dataframe=result_labels_df, allow_mixed_types=True)
    wandb.log({"Results Table": results_labels_table})

    wandb.log({
        "Has converged": result.has_converged,
    })

    print("EVALUATE")
    evaluate = AlgorithmEvaluation(result_labels_df, gt_labels_df, data_df, subject, data_dir, config.data_type)

    # log cluster map
    map_df = evaluate.map_clusters().copy()
    # convert all arrays to lists from np arrays
    for col in map_df.columns:
        if map_df[col].dtype == 'object':
            map_df[col] = map_df[col].apply(lambda x:
                                            # If it's a list of arrays, convert each array to a list
                                            [arr.tolist() if hasattr(arr, 'tolist') else arr for arr in x]
                                            if isinstance(x, list) else
                                            # If it's a single array, convert it to a list
                                            x.tolist() if hasattr(x, 'tolist') else x
                                            )
    map_clusters_table = wandb.Table(dataframe=map_df, allow_mixed_types=True)
    wandb.log({"Map Clusters": map_clusters_table})

    # save results to disk
    if save_results:
        results_data_dir = get_data_dir(root_data_dir=config.root_results_dir,
                                        extension_type=config.completeness_level)
        # create if it doesn't exist
        Path(results_data_dir).mkdir(parents=True, exist_ok=True)
        # save resulting labels_df
        # name the same as the synthetic file so it can be loaded the same way
        labels_file_name = Path(results_data_dir, subject + SyntheticFileTypes.labels)
        result_labels_df.to_parquet(labels_file_name, index=False, engine="pyarrow")

        # save resulting cluster map_df
        map_file_name = Path(results_data_dir, subject + "-TICC-cluster-map.csv")
        map_df.to_csv(map_file_name)

    # log numerical results
    wandb.log({
        "Jaccard Index": evaluate.jaccard_index(),
        "SWC": evaluate.silhouette_score(),
        "DBI": evaluate.dbi(),
        "Pattern Discovery ": evaluate.pattern_discovery_percentage(),
        "Pattern Specificity": evaluate.pattern_specificity_percentage(),
        "Segmentation Ratio": evaluate.segmentation_ratio(),
        "Segment Length Ratio": evaluate.segmentation_length_ratio(),
        "Undiscovered Patterns": evaluate.pattern_not_discovered(),
        "mean MAE TICC result - relaxed pattern": evaluate.mae_stats_mapped_resulting_patterns_relaxed()['mean'],
        "mean MAE ground truth - relaxed pattern": evaluate.mae_stats_mapped_gt_patterns_relaxed()['mean'],
        "n Clusters TICC": len(result_labels_df[SyntheticDataSegmentCols.pattern_id].unique()),
        "n Clusters ground truth": len(gt_labels_df[SyntheticDataSegmentCols.pattern_id].unique()),
    })

    # build return results
    evaluates[subject] = evaluate
    wandb_summaries[subject] = dict(wandb.run.summary)
    wandb.finish()
    return evaluates, wandb_summaries


def get_sweep_config(sweep_id: str):
    """Intention is to use the same configuration for all benchmark with each of them perhaps being more or
    less good for a DB
    """
    sweep_config = {
        'name': 'TICC CSTS Benchmark Sweep (' + sweep_id + ')',
        'method': 'grid',
        'parameters': {
            'window_size': {
                'values': [1, 2, 3]
            },
            'number_of_clusters': {
                'values': [23]
            },
            'switch_penalty': {
                'values': [1.9, 20, 110, 150]
            },
            'lambda_var': {
                'values': [0.11, 0.01, 0.001]
            },
        }
    }
    return sweep_config


if __name__ == "__main__":
    """ 
    Main function to do run TICC on multiple data variants. For each run we log the configuration and 
    evaluation results. These can be downloaded from wandb for analysis
    """
    config = TICCWandbUseCaseConfig()
    config.wandb_project_name =WandbConfiguration.wandb_ticc_tuning_project_name
    config.wandb_entity = WandbConfiguration.wandb_entity
    config.wandb_mode= 'online'
    config.wandb_notes= "tuning TICC"
    tags = ['TICC tuning', 'normal', 'complete']
    config.root_data_dir = SYNTHETIC_DATA_DIR
    # we won't save the results
    config.root_results_dir = get_algorithm_use_case_result_dir(root_results_dir=ROOT_RESULTS_DIR,
                                                                algorithm_id='ticc-grid-sweep-normal-complete')

    config.data_type = SyntheticDataType.normal_correlated
    config.completeness_level = DataCompleteness.complete
    config.max_iter = 10

    # run on all 30 subjects for each data variant
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    train_on_subject = run_names[4]  # data to use for training

    sweep_config_grid = get_sweep_config('TICC Grid Normal 1')
    sweep_id = wandb.sweep(sweep_config_grid, project=config.wandb_project_name)
    wandb.agent(sweep_id, function=lambda: one_run_ticc(config, subject=train_on_subject, save_results=False))
