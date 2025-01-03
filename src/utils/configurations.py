from dataclasses import dataclass, field
from pathlib import Path

import yaml
from os import path

ROOT_DIR = path.realpath(path.join(path.dirname(__file__), '../..'))
PATTERNS_TO_MODEL_PATH = path.join(ROOT_DIR, 'src/data_generation/config/correlation_patterns_to_model.csv')
DISTRIBUTION_PARAMS_TO_MODEL_PATH = path.join(ROOT_DIR,
                                              'src/data_generation/config/n30_genextreme_nbinom_genextreme_params.csv')
SYNTHETIC_DATA_DIR = path.join(ROOT_DIR, 'data/synthetic_data')
IRREGULAR_P30 = path.join(SYNTHETIC_DATA_DIR, 'irregular_p30')
IRREGULAR_P90 = path.join(SYNTHETIC_DATA_DIR, 'irregular_p90')
ROOT_RESULTS_DIR = path.join(ROOT_DIR, 'results')
DISTANCE_MEASURE_ASSESSMENT_RESULTS_FOLDER_NAME = 'distance-measures-assessment'
IMAGES_FOLDER_NAME = 'images'
GENERATED_DATASETS_FILE_PATH = path.join(SYNTHETIC_DATA_DIR, 'synthetic-correlated-data-n30.csv')


@dataclass
class ResultsType:
    internal_measure_assessment: str = 'internal-measures-assessment'
    internal_measures: str = 'internal-measures'
    distance_measure_assessment: str = 'distance-measures-assessment'


def dir_for_data_type(data_type: str, data_dir: str = SYNTHETIC_DATA_DIR):
    """ Returns the path to the data for the given type"""
    return path.join(data_dir, data_type)


def bad_partition_dir_for_data_type(data_type: str, data_dir: str = SYNTHETIC_DATA_DIR):
    """ Returns the path to the bad partitions labels files for the given data type"""
    main_dir = path.join(data_dir, data_type)
    folder = path.join(main_dir, 'bad_partitions')
    # creates the folder if it does not exist
    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


def distance_measure_assessment_dir_for(overall_dataset_name: str, data_type: str, results_dir: str,
                                        distance_measure: str, drop_clusters: int = 0, drop_segments: int = 0):
    """
          Returns directory for distance measure assessments
          :param overall_dataset_name: a name to identify the dataset overall e.g n30 or n2
          :param data_type: the data type, see SyntheticDataType
          :param results_dir: the directory for results, this is the main directory, the internal_assessment folder will be
          added
          :param distance_measure: the distance measures used for the internal assessment, see DistanceMeasures
          :param drop_clusters: will create an additional folder for evaluating reduced numbers of clusters, if 0 and
          n_dropped_segment is 0 no additional subfolder will be added
          :param drop_segments: will create an additional folder for evaluating reduced numbers of segments
          :return: the path name to the results folder e.g. results/distance-measures-assessment/raw/L1/ or
          results/distance-measures-assessment/raw/n_dropped_clusters/L1 or results/distance-measures-assessment/raw/n_dropped_segments/L1
          or results/distance-measures-assessment/raw/n_dropped_clusters_m_dropped_segments/L1
      """
    return get_folder_name_for(results_type=ResultsType.distance_measure_assessment,
                               overall_dataset_name=overall_dataset_name, data_type=data_type,
                               distance_measure=distance_measure, results_dir=results_dir, drop_clusters=drop_clusters,
                               drop_segments=drop_segments)


def internal_measure_calculation_dir_for(overall_dataset_name: str, data_type: str, results_dir: str,
                                         distance_measure: str, drop_clusters: int = 0, drop_segments: int = 0):
    """
          Returns directory for internal measures calculation on bad partitions
          :param overall_dataset_name: a name to identify the dataset overall e.g n30 or n2
          :param data_type: the data type, see SyntheticDataType
          :param results_dir: the directory for results, this is the main directory, the internal_assessment folder will be
          added
          :param distance_measure: the distance measures used for the internal assessment, see DistanceMeasures
          :param drop_clusters: will create an additional folder for evaluating reduced numbers of clusters, if 0 and
          n_dropped_segment is 0 no additional subfolder will be added
          :param drop_segments: will create an additional folder for evaluating reduced numbers of segments
          :return: the path name to the results folder e.g. results/internal_assessment/raw/L1/ or
          results/internal_assessment/raw/n_dropped_clusters/L1 or results/internal_assessment/raw/n_dropped_segments/L1
          or results/internal_assessment/raw/n_dropped_clusters_m_dropped_segments/L1
      """
    return get_folder_name_for(results_type=ResultsType.internal_measures,
                               overall_dataset_name=overall_dataset_name, data_type=data_type,
                               distance_measure=distance_measure, results_dir=results_dir, drop_clusters=drop_clusters,
                               drop_segments=drop_segments)


def internal_measure_assessment_dir_for(overall_dataset_name: str, data_type: str, results_dir: str,
                                        distance_measure: str, drop_clusters: int = 0, drop_segments: int = 0):
    """
    Returns directory for internal measures assessment results tables for the given data type and distance measure
    :param overall_dataset_name: a name to identify the dataset overall e.g n30 or n2
    :param data_type: the data type, see SyntheticDataType
    :param results_dir: the directory for results, this is the main directory, the internal_assessment folder will be
    added
    :param distance_measure: the distance measures used for the internal assessment, see DistanceMeasures
    :param drop_clusters: will create an additional folder for evaluating reduced numbers of clusters, if 0 and
    n_dropped_segment is 0 no additional subfolder will be added
    :param drop_segments: will create an additional folder for evaluating reduced numbers of segments
    :return: the path name to the results folder e.g. results/internal_assessment/raw/L1/ or
    results/internal_assessment/raw/n_dropped_clusters/L1 or results/internal_assessment/raw/n_dropped_segments/L1
    or results/internal_assessment/raw/n_dropped_clusters_m_dropped_segments/L1
    """
    return get_folder_name_for(results_type=ResultsType.internal_measure_assessment,
                               overall_dataset_name=overall_dataset_name, data_type=data_type,
                               distance_measure=distance_measure, results_dir=results_dir, drop_clusters=drop_clusters,
                               drop_segments=drop_segments)


def get_folder_name_for(results_type: str, overall_dataset_name: str, data_type: str, results_dir: str,
                        distance_measure: str,
                        drop_clusters: int = 0,
                        drop_segments: int = 0):
    """
        Returns directory for the given results type using
        :param overall_dataset_name: a name to identify the dataset overall e.g n30 or n2
        :param data_type: the data type, see SyntheticDataType
        :param results_dir: the directory for results, this is the main directory, the internal_assessment folder will be
        added
        :param distance_measure: the distance measures used for the internal assessment, see DistanceMeasures
        :param drop_clusters: will create an additional folder for evaluating reduced numbers of clusters, if 0 and
        n_dropped_segment is 0 no additional subfolder will be added
        :param drop_segments: will create an additional folder for evaluating reduced numbers of segments
        :return: the path name to the results folder e.g. results/internal_assessment/raw/L1/ or
        results/internal_assessment/raw/n_dropped_clusters/L1 or results/internal_assessment/raw/n_dropped_segments/L1
        or results/internal_assessment/raw/n_dropped_clusters_m_dropped_segments/L1
    """
    # put in folder internal_assessment
    results_folder = path.join(results_dir, results_type)
    # put in sub folder e.g. raw
    results_folder = path.join(results_folder, data_type)
    # put in sub folder e.g. n30
    results_folder = path.join(results_folder, overall_dataset_name)
    # put in sub folder e.g n_dropped_cluster if not evaluating all cluster
    dropped = get_folder_name_for_dropped_clusters_and_segments(drop_clusters, drop_segments)
    if dropped != "":
        results_folder = path.join(results_folder, dropped)
    # put in sub folder e.g. for distance measure
    results_folder = path.join(results_folder, folder_name_for_distance_measure(distance_measure))
    # creates the folder if it does not exist
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    return results_folder


def get_folder_name_for_dropped_clusters_and_segments(drop_clusters: int = 0, drop_segments: int = 0):
    """ Returns a folder name including the number of dropped clusters and segments """
    if drop_clusters == 0 and drop_segments == 0:
        return ""
    if drop_clusters > 0 and drop_segments == 0:
        return str(drop_clusters) + "_dropped_clusters"
    if drop_clusters == 0 and drop_segments > 0:
        return str(drop_segments) + "_dropped_segments"
    # both are not zero
    return str(drop_clusters) + "_dropped_clusters_" + str(drop_segments) + "_dropped_segments"


def get_image_results_path(results_dir: str, filename: str):
    """ Return the path to where to safe the images. This will create the folder image in the result dir and
    will return the full path the image will be saved to
    :param results_dir: the results folder where to add the images folder to
    :param filename: the name of the image file
    :return: the full file name including path to save an image
    """
    folder = path.join(results_dir, IMAGES_FOLDER_NAME)
    # creates the folder if it does not exist
    Path(folder).mkdir(parents=True, exist_ok=True)
    return path.join(folder, filename)


def folder_name_for_distance_measure(distance_measure_name: str):
    """Returns a space free version of the distance measure name that is used for folders
    :param distance_measure_name: name of the distance measure, see DistanceMeasures
    """
    return distance_measure_name.replace(" ", "_")


def get_internal_measures_summary_file_name(ds_name: str):
    """Returns the result name for the internal measure assessment for the given ds_nambe
    :param ds_name: the name of the dataset
    :return: the file name for the results csv
    """
    return ds_name + '_measures_summary.csv'


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
    wandb_irregular_project_name: str = config['wandb_irregular_project_name']
    wandb_entity: str = config['wandb_entity']
