import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.stats import genextreme, nbinom

import wandb
from scipy.stats._distn_infrastructure import rv_generic

from src.data_generation.model_distribution_params import ModelDistributionParams, DistParamsCols
from src.evaluation.describe_synthetic_dataset import DescribeSyntheticDataset
from src.utils.load_synthetic_data import SyntheticFileTypes, SyntheticDataType
from src.utils.configurations import WandbConfiguration, SyntheticDataVariates, SYNTHETIC_DATA_DIR, \
    dir_for_data_type
from src.utils.plots.matplotlib_helper_functions import Backends
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticSegmentedData
from tests.test_utils.configurations_for_testing import TEST_DATA_DIR


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
    resample_rule: str = "1min"
    # short events: minutes * 60 for seconds, think T1D events/activities duration
    segment_durations_short: [int] = field(
        default_factory=lambda: [15 * 60, 20 * 60, 30 * 60, 60 * 60, 120 * 60, 180 * 60, 240 * 60, 300 * 60])
    segment_durations_long: [int] = field(
        default_factory=lambda: [360 * 60, 480 * 60, 600 * 60])  # minutes * 60 for seconds

    # NN Distribution settings
    distributions_for_variates: [rv_generic] = field(default_factory=lambda: [genextreme, nbinom, genextreme])
    # iob args (give in same order as the distribution methods will take
    distributions_args_iob: () = field(default_factory=lambda: (-0.22,))
    distributions_kwargs_iob: {str: float} = field(default_factory=lambda: {'loc': 0.5, 'scale': 1.52})
    distributions_args_cob: () = field(default_factory=lambda: (1, 0.05))
    distributions_kwargs_cob: {str: float} = field(default_factory=lambda: {})  # none, loc will be 0
    distributions_args_ig: () = field(default_factory=lambda: (0.04,))
    distributions_kwargs_ig: {str: float} = field(default_factory=lambda: {'loc': 119.27, 'scale': 39.40})

    # EVALUATION
    backend: str = Backends.none.value

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
    rs_labels_table: str = "rs labels"
    dataset_seed: str = "dataset seed"


def save_data_labels_to_file(data_dir, data_type, raw_data_df, raw_labels_df, run_name):
    file_dir = dir_for_data_type(data_type, data_dir)
    Path(file_dir).mkdir(parents=True, exist_ok=True)
    data_file_name = Path(file_dir, run_name + SyntheticFileTypes.data)
    labels_file_name = Path(file_dir, run_name + SyntheticFileTypes.labels)
    raw_data_df.to_csv(data_file_name)
    raw_labels_df.to_csv(labels_file_name)


def one_synthetic_creation_run(config: SyntheticDataConfig, seed: int = 6666):
    """
    Wandb generate synthetic data according to the config provided
    :param config: SyntheticDataConfig that configures the creation
    :param seed: base seed to use for random, the dataset will be using a random int
    :return: dictionary of DescribeSyntheticData class for all data variations as well as wandb log summary dic
    """
    raw_desc, nc_desc, nn_desc, rs_desc = None, None, None, None
    try:
        wandb.init(project=config.wandb_project_name,
                   entity=config.wandb_entity,
                   mode=config.wandb_mode,
                   notes=config.wandb_notes,
                   tags=config.tags,
                   config=config.as_dict())

        # This won't work for sweeps as config won't be rehydrated but we're not doing sweps
        distributions_args = [config.distributions_args_iob, config.distributions_args_cob,
                              config.distributions_args_ig]
        distributions_kwargs = [config.distributions_kwargs_iob, config.distributions_kwargs_cob,
                                config.distributions_kwargs_ig]

        keys = SyntheticDataLogKeys()
        wandb.log({keys.dataset_seed: seed})

        if config.correlation_model == "cholesky":
            patterns = ModelCorrelationPatterns().relaxed_patterns_and_regularisation_term()
        elif config.correlation_model == "loadings":
            patterns = ModelCorrelationPatterns().canonical_patterns()
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
                                           distributions_args, distributions_kwargs, config.segment_durations_short,
                                           config.segment_durations_long, patterns, config.columns,
                                           config.correlation_model)
        generator.generate(seed=seed)

        print("2. RESAMPLE")
        generator.resample(rule=config.resample_rule)

        print("3. SAVE LABELS DF ON WANDB")
        # get dataframes
        raw_data_df, raw_labels_df = generator.raw_generated_data_labels_df()
        nc_data_df, nc_labels_df = generator.normal_correlated_generated_data_labels_df()
        nn_data_df = generator.non_normal_data_df
        nn_labels_df = generator.non_normal_labels_df
        # reset index required to match other dfs as datetime was set as index for the resampling
        rs_data_df = generator.resampled_data
        rs_labels_df = generator.resampled_labels_df

        # data tables are too big to be logged on wandb, saving them directly to data_dir
        print("...saving generated data to local file storage")
        data_dir = config.data_dir
        # raw
        save_data_labels_to_file(data_dir, SyntheticDataType.raw, raw_data_df, raw_labels_df, run_name)
        # nc
        save_data_labels_to_file(data_dir, SyntheticDataType.normal_correlated, nc_data_df, nc_labels_df, run_name)
        # nn
        save_data_labels_to_file(data_dir, SyntheticDataType.non_normal_correlated, nn_data_df, nn_labels_df, run_name)
        # resampled
        save_data_labels_to_file(data_dir, SyntheticDataType().resample(config.resample_rule), rs_data_df, rs_labels_df,
                                 run_name)

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

        # Resampled data
        resampled_labels_table = wandb.Table(dataframe=rs_labels_df, allow_mixed_types=True)
        wandb.log({keys.rs_labels_table: resampled_labels_table})

        print("4. LOG RAW DESCRIPTION")
        raw_desc = DescribeSyntheticDataset(run_name, data_type=SyntheticDataType.raw, data_dir=config.data_dir)
        log_dataset_description(raw_desc, "RAW")

        print("5. LOG NORMAL CORRELATED DESCRIPTION")
        nc_desc = DescribeSyntheticDataset(run_name, data_type=SyntheticDataType.normal_correlated,
                                           data_dir=config.data_dir)
        log_dataset_description(nc_desc, "NC")

        print("6. LOG NON-NORMAL CORRELATED DESCRIPTION")
        nn_desc = DescribeSyntheticDataset(run_name, data_type=SyntheticDataType.non_normal_correlated,
                                           data_dir=config.data_dir)
        log_dataset_description(nn_desc, "NN")

        print("7. LOG RESAMPLED DESCRIPTION")
        rs_desc = DescribeSyntheticDataset(run_name, data_type=SyntheticDataType().resample(config.resample_rule),
                                           data_dir=config.data_dir)
        log_dataset_description(rs_desc, "RS")

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)  # try to get error into wandb log
        exit_code = 1

    summary = dict(wandb.run.summary)
    wandb.finish(exit_code=exit_code)
    if exit_code == 1:
        raise
    return {"raw": raw_desc, "nc": nc_desc, "nn": nn_desc, "rs": rs_desc}, summary


def log_dataset_description(describe: DescribeSyntheticDataset, key_id: str):
    """"
    Logs information about the dataset using key_id as a prefix for each key
    """
    wandb.log({
        "n observations " + key_id: describe.number_of_observations,
        "n segments " + key_id: describe.number_of_segments,
        "n patterns " + key_id: describe.n_patterns,
        "n segments within tolerance " + key_id: describe.n_segment_within_tolerance,
        "n segments outside tolerance " + key_id: describe.n_segment_outside_tolerance,
        "mean MAE " + key_id: describe.mae_stats['mean'],
        "std MAE " + key_id: describe.mae_stats['std'],
        "median MAE " + key_id: describe.mae_stats['50%'],
        "min MAE " + key_id: describe.mae_stats['min'],
        "max MAE " + key_id: describe.mae_stats['max'],
        "mean pattern frequency " + key_id: describe.patterns_stats['mean'],
        "std pattern frequency " + key_id: describe.patterns_stats['std'],
        "median pattern frequency " + key_id: describe.patterns_stats['50%'],
        "min pattern frequency " + key_id: describe.patterns_stats['min'],
        "max pattern frequency " + key_id: describe.patterns_stats['max'],
        "mean segment length " + key_id: describe.segment_length_stats['mean'],
        "std segment length " + key_id: describe.segment_length_stats['std'],
        "median segment length " + key_id: describe.segment_length_stats['50%'],
        "min segment length " + key_id: describe.segment_length_stats['min'],
        "max segment length " + key_id: describe.segment_length_stats['max'],
        "frequency " + key_id: describe.frequency,
        "duration in days " + key_id: describe.duration.days,
        "start date " + key_id: describe.start_date.isoformat(),
        "end date " + key_id: describe.end_date.isoformat(),
    })


def create_datasets(n: int = 2, tag: str = 'synthetic_creation'):
    """"
    Create n datasets
    """
    config = SyntheticDataConfig()
    config.tags.append(tag)
    # config.data_dir = TEST_DATA_DIR  # store this trial in test

    # load distribution parameters
    mpam = ModelDistributionParams()
    c_iobs = mpam.get_params_for(DistParamsCols.c_iob)
    loc_iobs = mpam.get_params_for(DistParamsCols.loc_iob)
    scale_iobs = mpam.get_params_for(DistParamsCols.scale_iob)
    n_cobs = mpam.get_params_for(DistParamsCols.n_cob)
    p_cobs = mpam.get_params_for(DistParamsCols.p_cob)
    c_igs = mpam.get_params_for(DistParamsCols.c_ig)
    loc_igs = mpam.get_params_for(DistParamsCols.loc_ig)
    scale_igs = mpam.get_params_for(DistParamsCols.scale_ig)

    for n in range(n):
        np.random.seed(666 + n)
        dataset_seed = np.random.randint(low=100, high=1000000)
        # configure distribution params
        index = n % len(c_iobs)
        config.distributions_args_iob = (c_iobs[index],)
        config.distributions_kwargs_iob = {'loc': loc_iobs[index], 'scale': scale_iobs[index]}
        config.distributions_args_cob = (n_cobs[index], p_cobs[index])
        config.distributions_kwargs_cob = {}  # none, loc will be 0
        config.distributions_args_ig = (c_igs[index],)
        config.distributions_kwargs_ig = {'loc': loc_igs[index], 'scale': scale_igs[index]}

        one_synthetic_creation_run(config, seed=dataset_seed)


if __name__ == "__main__":
    create_datasets(30, '30_ds_creation')
