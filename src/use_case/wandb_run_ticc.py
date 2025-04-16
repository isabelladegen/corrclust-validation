from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd
import wandb
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols

from src.use_case.algorithm_evaluation import AlgorithmEvaluation
from src.use_case.ticc.TICC_solver import TICC
from src.utils.configurations import WandbConfiguration, SyntheticDataVariates, SYNTHETIC_DATA_DIR, \
    GENERATED_DATASETS_FILE_PATH, DataCompleteness, get_data_dir, get_algorithm_use_case_result_dir, \
    ROOT_RESULTS_DIR
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data, SyntheticFileTypes
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.run_average_rank_visualisations import data_variant_description
from tests.test_utils.configurations_for_testing import TEST_ROOT_RESULTS_DIR
from tests.use_case.ticc.test_ticc_runs_on_original_test_data import TICCSettings


@dataclass
class TICCDefaultSettings(TICCSettings):
    window_size: int = 5
    number_of_clusters: int = 23  # from ground truth
    switch_penalty: float = 400
    lambda_var: float = 11e-2
    max_iter: int = 10 # this is usually 100 but takes a long time
    threshold: float = 2e-5
    allow_zero_cluster_inbetween: bool = False
    use_gmm_initialisation: bool = True
    reassign_points_to_zero_clusters: bool = True
    biased: bool = True
    do_training_split: bool = False
    keep_track_of_assignments: bool = False
    cluster_reassignment: int = 30  # min segment length
    backend = Backends.none.value


@dataclass
class TICCWandbUseCaseConfig(TICCDefaultSettings):
    # WANDB CONFIG
    wandb_project_name: str = WandbConfiguration.wandb_use_case_project_name
    wandb_entity: str = WandbConfiguration.wandb_entity
    wandb_mode: str = 'online'
    wandb_notes = "use case, TICC"
    tags = ['TICC Use Case']

    # DATA LOADING CONFIG
    # This decides which dataset to use set to SYNTHETIC_DATA_DIR for exploratory, or CONFIRMATORY_SYNTHETIC_DATA_DIR
    # for confirmatory dataset
    root_data_dir: str = ''
    # completeness level of run
    completeness_level: str = ''
    # Data type for run
    data_type: str = ''
    # data cols to use - these are the data columns to load
    data_cols: [str] = field(default_factory=lambda: SyntheticDataVariates.columns())
    # indicates if training or predicting
    is_training_run: bool = True

    # SAVE RESULTS LOCATION
    # results are saved in the same structure as generated data to distinguish between variants
    root_results_dir: str = ''

    # allows for sweeping if done in the future
    def as_dict(self):
        return asdict(self)


def run_ticc_on_a_data_variant(config: TICCWandbUseCaseConfig, run_names: [str], training_subject_name: str,
                               save_results: bool = False):
    """
    Wandb TICC use case run. Trains on one individual specified in the config, predicts on the rest of the individuals
    there's a new run for each subject in each data variant
    :param config: TICCWandbUseCaseConfig that configures which data variant and subjects TICC is trained and run on
    :param run_names: list of subjects (run names from generation run) to run TICC on
    :param training_subject_name: name of subject for which TICC is trained
    :param save_results: if TRUE the resulting labels df and cluster mapping df are saved to disk as well as to wandb
    ensure you set results dir in config appropriately
    :return: evaluates dict with key subject and value AlgorithmEvaluation, wandb summaries with key subject and value
     wandb dict with the keys and values logged to wandb
    """
    # pre wandb setup
    evaluates = {}  # per subject, per run evaluate objects
    wandb_summaries = {}  # per subject wandb run summary

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
    subjects = run_names.copy()  # all subjects

    # ensure training subject comes first
    subjects.remove(training_subject_name)
    subjects.insert(0, training_subject_name)

    for subject in subjects:
        # string description of data variant
        variant_description = data_variant_description[(config.completeness_level, config.data_type)]
        # add data variant to wandb run name
        run_name = subject + ":" + variant_description

        is_training_run = (subject == training_subject_name)
        config.is_training_run = is_training_run

        wandb.init(project=config.wandb_project_name,
                   entity=config.wandb_entity,
                   name=run_name,
                   mode=config.wandb_mode,
                   notes=config.wandb_notes,
                   tags=config.tags,
                   config=config.as_dict())

        print("DATA VARIANT: " + variant_description)
        print("LOAD GROUND TRUTH DATA")
        data_dir = get_data_dir(root_data_dir=config.root_data_dir, extension_type=config.completeness_level)
        data_df, gt_labels_df = load_synthetic_data(subject, config.data_type, data_dir)

        data_np = data_df[config.data_cols].to_numpy()

        if config.is_training_run:
            print("TRAIN TICC ON SUBJECT: " + subject)
            result = ticc.fit(data=data_np, use_gmm_initialisation=config.use_gmm_initialisation,
                              reassign_points_to_zero_clusters=config.reassign_points_to_zero_clusters)
        else:
            print("PREDICT TICC ON SUBJECT: " + subject)
            result = ticc.predict_clusters(data_np)

        result_labels_df = result.to_labels_df()

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
            result_labels_df.to_csv(labels_file_name)

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


if __name__ == "__main__":
    """ 
    Main function to do run TICC on multiple data variants. For each run we log the configuration and 
    evaluation results. These can be downloaded from wandb for analysis
    """
    # run on normal and non-normal data variant
    dataset_types = [SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated]
    # use exploratory data: run for all three completeness level
    completeness_levels = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]

    config = TICCWandbUseCaseConfig()
    config.root_data_dir = SYNTHETIC_DATA_DIR
    config.root_results_dir = get_algorithm_use_case_result_dir(root_results_dir=ROOT_RESULTS_DIR,
                                                                algorithm_id='ticc-untuned')

    # run on all 30 subjects for each data variant
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    train_on_subject = run_names[4]  # data to use for training

    for data_type in dataset_types:
        for level in completeness_levels:
            config.data_type = data_type
            config.completeness_level = level
            run_ticc_on_a_data_variant(config, run_names=run_names, training_subject_name=train_on_subject,
                                       save_results=True)
