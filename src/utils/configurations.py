from dataclasses import dataclass, field

import yaml
from os import path

ROOT_DIR = path.realpath(path.join(path.dirname(__file__), '../..'))
PATTERNS_TO_MODEL_PATH = path.join(ROOT_DIR, 'src/data_generation/config/correlation_patterns_to_model.csv')
DISTRIBUTION_PARAMS_TO_MODEL_PATH = path.join(ROOT_DIR,
                                              'src/data_generation/config/n30_genextreme_nbinom_genextreme_params.csv')
SYNTHETIC_DATA_DIR = path.join(ROOT_DIR, 'data/synthetic_data')
GENERATED_DATASETS_FILE_PATH = path.join(ROOT_DIR, 'src/data_generation/config/n30_generated_datasets.csv')


def dir_for_data_type(data_type: str, data_dir: str = SYNTHETIC_DATA_DIR):
    """ Returns the path to the data for the given type"""
    return path.join(data_dir, data_type)


def bad_partition_dir_for_data_type(data_type: str, data_dir: str = SYNTHETIC_DATA_DIR):
    """ Returns the path to the bad partitions labels files for the given data type"""
    main_dir = path.join(data_dir, data_type)
    return path.join(main_dir, 'bad_partitions')


def load_private_yaml():
    private_file = path.join(ROOT_DIR, 'private.yaml')
    assert (path.exists(private_file))
    with open(private_file, "r") as f:
        config = yaml.safe_load(f)
    return config


@dataclass
class SyntheticDataVariates:
    iob: str = 'iob'  # insulin on board
    cob: str = 'cob'  # carbs on board
    ig: str = 'ig'  # interstitial glucose

    @staticmethod
    def columns():
        return [SyntheticDataVariates.iob, SyntheticDataVariates.cob, SyntheticDataVariates.ig]

    @staticmethod
    def plot_columns():
        return [col.upper() for col in SyntheticDataVariates.columns()]


@dataclass
class Aggregators:
    # colum name and name of aggregation function
    min = 'min'
    max = 'max'
    mean = 'mean'
    std = 'std'
    count = 'count'
    diff = 'diff'
    se = "se"


@dataclass
class GeneralisedCols:
    # generalised configs across systems
    iob = 'iob'
    cob = 'cob'
    bg = 'ig'  # todo rename to ig
    id = 'id'
    mean_iob = iob + ' ' + Aggregators.mean
    mean_cob = cob + ' ' + Aggregators.mean
    mean_bg = bg + ' ' + Aggregators.mean
    min_iob = iob + ' ' + Aggregators.min
    min_cob = cob + ' ' + Aggregators.min
    min_bg = bg + ' ' + Aggregators.min
    max_iob = iob + ' ' + Aggregators.max
    max_cob = cob + ' ' + Aggregators.max
    max_bg = bg + ' ' + Aggregators.max
    std_iob = iob + ' ' + Aggregators.std
    std_cob = cob + ' ' + Aggregators.std
    std_bg = bg + ' ' + Aggregators.std
    count_iob = iob + ' ' + Aggregators.count
    count_cob = cob + ' ' + Aggregators.count
    count_bg = bg + ' ' + Aggregators.count
    datetime = 'datetime'
    system = 'system'


@dataclass
class Resampling:
    max_gap_in_min: int = None
    # how big the gap between two datetime stamps can be
    sample_rule: str = None
    # the frequency of the regular time series after resampling: 1H a reading every hour, 1D a reading every day

    description: str = 'None'
    agg_cols: [str] = field(
        default_factory=lambda: [Aggregators.min, Aggregators.max, Aggregators.mean,
                                 Aggregators.std, Aggregators.count])

    general_agg_cols_dictionary: {str: str} = field(default_factory=lambda: {GeneralisedCols.id: 'first',
                                                                             GeneralisedCols.system: 'first',
                                                                             })

    @staticmethod
    def csv_file_name():
        return ''


@dataclass
class Irregular(Resampling):
    description: str = 'None'

    @staticmethod
    def csv_file_name():
        return 'irregular_iob_cob_bg.csv'


@dataclass
class Hourly(Resampling):
    max_gap_in_min: int = 60
    # there needs to be a reading at least every hour for the data points to be resampled for that hour
    sample_rule: str = '1H'
    needs_max_gap_checking: bool = False
    description: str = 'Hourly'

    @staticmethod
    def csv_file_name():
        return 'hourly_iob_cob_bg.csv'


@dataclass
class Daily(Resampling):
    max_gap_in_min: int = 180
    # a reading every three hours for a daily resampling to be created
    sample_rule: str = '1D'
    needs_max_gap_checking: bool = True
    description: str = 'Daily'

    @staticmethod
    def csv_file_name():
        return 'daily_iob_cob_bg.csv'


@dataclass
class FifteenMin(Resampling):
    """
    Resamples times into 0, 15, 45 min buckets for each hour,
    if there's not at least one reading bucket for that time will be empty
    """
    max_gap_in_min: int = 15
    # a reading every 15min to ensure this is down sampling and not generating data
    sample_rule: str = '15T'
    needs_max_gap_checking: bool = False
    description: str = '15 min'

    @staticmethod
    def csv_file_name():
        return '15min_iob_cob_bg.csv'


@dataclass
class WandbConfiguration:
    config = load_private_yaml()
    # READ CONFIGURATIONS
    wandb_project_name: str = config['wandb_project_name']
    wandb_partitions_project_name: str = config['wandb_partitions_project_name']
    wandb_entity: str = config['wandb_entity']
