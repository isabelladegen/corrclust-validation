import gc
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.stats import genextreme, nbinom, lognorm

import wandb
from scipy.stats._distn_infrastructure import rv_generic

from src.evaluation.describe_synthetic_dataset import DescribeSyntheticDataset
from src.utils.load_synthetic_data import SyntheticFileTypes, SyntheticDataType
from src.utils.configurations import WandbConfiguration, SyntheticDataVariates, SYNTHETIC_DATA_DIR, \
    dir_for_data_type
from src.utils.plots.matplotlib_helper_functions import Backends
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticSegmentedData


@dataclass
class SyntheticDataConfig:
    wandb_project_name: str = WandbConfiguration.wandb_project_name
    wandb_entity: str = WandbConfiguration.wandb_entity
    wandb_mode: str = 'online'
    wandb_notes = "creates synthetic data, uncorrelated, correlated normal and non-normal"
    tags = ['Synthetic']

    correlation_model = "loadings"

    # Store results
    data_dir = SYNTHETIC_DATA_DIR

    # DATA
    columns: [str] = field(default_factory=lambda: SyntheticDataVariates.columns())

    # data generation config
    number_of_variates: int = 3
    number_of_segments: int = 100
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
    backend: str = Backends.none.value
    has_datetime_index: bool = True  # this will decide if datetime analysis is run

    def as_dict(self):
        return asdict(self)


@dataclass
class SyntheticDataLogKeys:
    """
    Keys specific to generating synthetic data
    """
    nn_labels_table: str = "non normal labels"
    nc_labels_table: str = "normal labels"
    raw_labels_table: str = "raw labels"
    generated_data_table: str = "generated data"
    downsampled_labels_table: str = "downsampled labels"
    dataset_seed: str = "dataset seed"


def save_data_labels_to_file(data_dir, data_type, raw_data_df, raw_labels_df, run_name):
    file_dir = dir_for_data_type(data_type, data_dir)
    Path(file_dir).mkdir(parents=True, exist_ok=True)
    data_file_name = Path(file_dir, run_name + SyntheticFileTypes.data)
    labels_file_name = Path(file_dir, run_name + SyntheticFileTypes.labels)
    raw_data_df.to_csv(data_file_name)
    raw_labels_df.to_csv(labels_file_name)


def one_synthetic_creation_run(config: SyntheticDataConfig, seed: int = 66666):
    """
    Wandb generate synthetic data according to the config provided
    :param config: SyntheticDataConfig that configures the creation
    :param seed: base seed to use for random, the dataset will be using a random int
    :return: DescribeSyntheticData class
    """
    raw_desc, nc_desc, nn_desc, downsampled_desc = None, None, None, None
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
        wandb.log({keys.dataset_seed: seed})

        if config.correlation_model == "cholesky":
            patterns = ModelCorrelationPatterns().patterns_to_model()
        elif config.correlation_model == "loadings":
            patterns = ModelCorrelationPatterns().ideal_correlations()
        else:
            assert False, "Unknown correlation model method {}".format(config.correlation_model)
        exit_code = 0

        run_name = wandb.run.name
        if run_name == "":
            run_name = 'test-run'

        # Generate data

        print("1. GENERATING DATA")
        generator = SyntheticSegmentedData(config.number_of_segments, config.number_of_variates,
                                           config.distributions_for_variates,
                                           distributions_args, distributions_kwargs, config.short_segment_durations,
                                           config.long_segment_durations, patterns, config.columns,
                                           config.correlation_model)
        generator.generate(seed=seed)

        print("2. DOWNSAMPLE")
        generator.resample(rule=config.downsampling_rule)

        print("3. SAVE LABELS DF ON WANDB")
        # get dataframes
        raw_data_df, raw_labels_df = generator.raw_generated_data_labels_df()
        nc_data_df, nc_labels_df = generator.normal_correlated_generated_data_labels_df()
        nn_data_df = generator.non_normal_data_df
        nn_labels_df = generator.non_normal_labels_df
        # reset index required to match other dfs as datetime was set as index for the resampling
        downsampled_data_df = generator.resampled_data.reset_index()
        downsampled_labels_df = generator.resampled_labels_df

        # data tables are too big to be logged on wandb, saving them directly to data_dir
        print("...saving generated data to local file storage")
        data_dir = config.data_dir
        # raw
        save_data_labels_to_file(data_dir, SyntheticDataType.raw, raw_data_df, raw_labels_df, run_name)
        # nc
        save_data_labels_to_file(data_dir, SyntheticDataType.normal_correlated, nc_data_df, nc_labels_df, run_name)
        # nn
        save_data_labels_to_file(data_dir, SyntheticDataType.non_normal_correlated, nn_data_df, nn_labels_df, run_name)
        # downsampled
        save_data_labels_to_file(data_dir, SyntheticDataType.downsampled_1min, downsampled_data_df,
                                 downsampled_labels_df, run_name)

        print("...saving labels to wandb")
        # Non Normal labels
        nn_labels_table = wandb.Table(dataframe=nn_labels_df, allow_mixed_types=True)
        wandb.log({keys.nn_labels_table: nn_labels_table})

        # Normal labels
        nc_labels_table = wandb.Table(dataframe=nc_labels_df, allow_mixed_types=True)
        wandb.log({keys.nc_labels_table: nc_labels_table})

        # raw labels
        raw_labels_table = wandb.Table(dataframe=raw_labels_df, allow_mixed_types=True)
        wandb.log({keys.raw_labels_table: raw_labels_table})

        # Downsampled data
        downsampled_labels_table = wandb.Table(dataframe=downsampled_labels_df, allow_mixed_types=True)
        wandb.log({keys.downsampled_labels_table: downsampled_labels_table})

        print("4. LOG RAW DESCRIPTION")
        raw_desc = DescribeSyntheticDataset(run_name, data_type=SyntheticDataType.raw, data_dir=config.data_dir)
        log_dataset_description(raw_desc, "Raw")

        print("5. LOG NORMAL CORRELATED DESCRIPTION")
        nc_desc = DescribeSyntheticDataset(run_name, data_type=SyntheticDataType.normal_correlated,
                                           data_dir=config.data_dir)
        log_dataset_description(raw_desc, "NC")

        print("6. LOG NON-NORMAL CORRELATED DESCRIPTION")
        nn_desc = DescribeSyntheticDataset(run_name, data_type=SyntheticDataType.non_normal_correlated,
                                           data_dir=config.data_dir)
        log_dataset_description(raw_desc, "NN")

        print("7. LOG DOWNSAMPLED DESCRIPTION")
        downsampled_desc = DescribeSyntheticDataset(run_name, data_type=SyntheticDataType.downsampled_1min,
                                                    data_dir=config.data_dir)
        log_dataset_description(raw_desc, "DS")

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)  # try to get error into wandb log
        exit_code = 1

    wandb.finish(exit_code=exit_code)
    gc.collect()
    if exit_code == 1:
        raise
    return raw_desc, nc_desc, nn_desc, downsampled_desc


def log_dataset_description(describe: DescribeSyntheticDataset, key_id: str):
    """"
    Logs information about the dataset using key_id as a prefix for each key
    """
    wandb.log({
        key_id + " n observations": describe.number_of_observations,
        key_id + " n segments": describe.number_of_segments,
        key_id + " n patterns": describe.n_patterns,
        key_id + " n segments within tolerance": describe.n_segment_within_tolerance,
        key_id + " n segments outside tolerance": describe.n_segment_outside_tolerance,
        key_id + " mean MAE": describe.mae_stats['mean'],
        key_id + " std MAE": describe.mae_stats['std'],
        key_id + " median MAE": describe.mae_stats['50%'],
        key_id + " min MAE": describe.mae_stats['min'],
        key_id + " max MAE": describe.mae_stats['max'],
        key_id + " mean pattern frequency": describe.patterns_stats['mean'],
        key_id + " std pattern frequency": describe.patterns_stats['std'],
        key_id + " median pattern frequency": describe.patterns_stats['50%'],
        key_id + " min pattern frequency": describe.patterns_stats['min'],
        key_id + " max pattern frequency": describe.patterns_stats['max'],
        key_id + " mean segment length": describe.segment_length_stats['mean'],
        key_id + " std segment length": describe.segment_length_stats['std'],
        key_id + " median segment length": describe.segment_length_stats['50%'],
        key_id + " min segment length": describe.segment_length_stats['min'],
        key_id + " max segment length": describe.segment_length_stats['max'],
        key_id + " frequency": describe.frequency,
        key_id + " duration in days": describe.duration.days,
        key_id + " start date": describe.start_date.isoformat(),
        key_id + " end date": describe.end_date.isoformat(),
    })


def create_datasets(n: int = 2, tag: str = 'synthetic_creation'):
    """"
    Create n datasets
    """
    config = SyntheticDataConfig()
    config.tags.append(tag)

    for n in range(n):
        np.random.seed(66 + n)
        dataset_seed = np.random.randint(low=100, high=1000000)
        one_synthetic_creation_run(config, seed=dataset_seed)


if __name__ == "__main__":
    create_datasets(2, '2_ds_creation')
