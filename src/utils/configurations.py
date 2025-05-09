import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from os import path

from src.utils.distance_measures import short_distance_measure_names


@dataclass
class DataCompleteness:
    complete: str = ''  # 100% of data
    irregular_p30: str = 'irregular_p30'  # 70% of data
    irregular_p90: str = 'irregular_p90'  # 10% of data


ROOT_DIR = path.realpath(path.join(path.dirname(__file__), '../..'))
PATTERNS_TO_MODEL_PATH = path.join(ROOT_DIR, 'src/data_generation/config/correlation_patterns_to_model.csv')
DISTRIBUTION_PARAMS_TO_MODEL_PATH = path.join(ROOT_DIR,
                                              'src/data_generation/config/n30_genextreme_nbinom_genextreme_params.csv')
CONFIRMATORY_SYNTHETIC_DATA_DIR = path.join(ROOT_DIR, 'parquet_data/confirmatory')
SYNTHETIC_DATA_DIR = path.join(ROOT_DIR, 'parquet_data/exploratory')
ROOT_REDUCED_SYNTHETIC_DATA_DIR = path.join(SYNTHETIC_DATA_DIR, "reduced-data")
CONFIRMATORY_ROOT_REDUCED_SYNTHETIC_DATA_DIR = path.join(CONFIRMATORY_SYNTHETIC_DATA_DIR, "reduced-data")
CONF_IRREGULAR_P30_DATA_DIR = path.join(CONFIRMATORY_SYNTHETIC_DATA_DIR, DataCompleteness.irregular_p30)
CONF_IRREGULAR_P90_DATA_DIR = path.join(CONFIRMATORY_SYNTHETIC_DATA_DIR, DataCompleteness.irregular_p90)
IRREGULAR_P30_DATA_DIR = path.join(SYNTHETIC_DATA_DIR, DataCompleteness.irregular_p30)
IRREGULAR_P90_DATA_DIR = path.join(SYNTHETIC_DATA_DIR, DataCompleteness.irregular_p90)
ROOT_RESULTS_DIR = path.join(ROOT_DIR, 'results')
ROOT_REDUCED_RESULTS_DIR = path.join(ROOT_RESULTS_DIR, "reduced-data")
DISTANCE_MEASURE_ASSESSMENT_RESULTS_FOLDER_NAME = 'distance-measures-assessment'
IMAGES_FOLDER_NAME = 'images'
GENERATED_DATASETS_FILE_PATH = path.join(SYNTHETIC_DATA_DIR, 'synthetic-correlated-data-n30.csv')
CONFIRMATORY_DATASETS_FILE_PATH = path.join(CONFIRMATORY_SYNTHETIC_DATA_DIR,
                                            'confirmatory-synthetic-correlated-data-n30.csv')

MULTIPLE_DS_SUMMARY_FILE = 'multiple-datasets-summary.csv'
OVERALL_SEGMENT_LENGTH_IMAGE = 'overall_segment_length_distributions.png'
OVERALL_MAE_IMAGE = 'overall_mae_distributions.png'
OVERALL_CLUSTERING_QUALITY_DISTRIBUTION = '_distributions.png'
OVERALL_CORRELATION_COEFFICIENT_DISTRIBUTION = '_with_jaccard_correlation_coefficients_distributions.png'
OVERALL_CLUSTERING_SCATTER_PLOT = '_scatter_plot.png'
MULTI_MEASURES_SCATTER_PLOT = '_multi_measures_scatter_plot.png'
INTERVAL_HISTOGRAMS = 'interval_histograms.png'
GROUND_TRUTH_CI_PLOT = '_ground_truth_ci_plot.png'
CRITERIA_RANK_DISTRIBUTION = 'criteria_rank_distributions_across_runs.png'
AVERAGE_RANK_DISTRIBUTION = 'average_rank_distributions_across_runs.png'
PARTITIONS_QUALITY_DESCRIPTION = 'descriptive_statistics_for_segmented_clusterings.png'
HEATMAP_OF_RANKS = 'heat_map_of_ranks.png'
HEATMAP_OF_BEST_MEASURES_RAW_VALUES = 'heat_map_of_top_2_measures_raw_values.png'
GROUND_TRUTH_HEATMAP_OF_RANKS = 'ground_truth_heat_map_of_ranks.png'
GROUND_TRUTH_HEATMAP_RAW_VALUES = 'ground_truth_heat_map_raw_values.png'
OVERALL_DISTRIBUTION_IMAGE = 'overall_distributions.png'
EMPIRICAL_CORRELATION_IMAGE = 'empirical_correlation.png'
DISTANCE_MEASURE_AVG_RANK_STATS_VALIDATION = 'stats_validation_average_ranks.csv'
DISTANCE_MEASURE_EVALUATION_CRITERIA_RESULTS_FILE = 'raw_evaluation_criteria_results.csv'
DISTANCE_MEASURE_EVALUATION_CRITERIA_RANKS_RESULTS_FILE = 'rank_evaluation_criteria_results.csv'
DISTANCE_MEASURE_EVALUATION_OVERALL_RANKS_RESULTS_FILE = 'per_ds_rank_evaluation_criteria_results.csv'
DISTANCE_MEASURE_EVALUATION_AVERAGE_RANKS_PER_CRITERIA_RESULTS_FILE = 'per_criteria_avg_rank_evaluation_criteria_results.csv'
DISTANCE_MEASURE_EVALUATION_TOP_BOTTOM_MEASURES = 'top_bottom_distance_measures.csv'


@dataclass
class RunInformationCols:
    ds_name: str = 'Name'
    data_cols: str = 'columns'
    distribution: str = 'distributions_for_variates'
    distribution_args: str = 'distributions_args'
    distribution_kwargs: str = 'distributions_kwargs'

    def dist_args_for(self, variate: str) -> str:
        return self.distribution_args + "_" + str(variate)

    def dist_kwargs_for(self, variate: str) -> str:
        return self.distribution_kwargs + "_" + str(variate)


@dataclass
class ResultsType:
    internal_measure_evaluation: str = 'internal-measures-assessment'  # images, statistics
    internal_measures_calculation: str = 'internal-measures-calculation'  # the calculation of the indices
    distance_measure_assessment: str = 'distance-measures-assessment'  # per distance measure assessments
    distance_measure_evaluation: str = 'distance-measures-evaluation'  # statistical differences of distance measures
    dataset_description: str = 'dataset-description'

def get_algorithm_use_case_result_dir(root_results_dir: str, algorithm_id: str) -> str:
    folder = path.join(root_results_dir, 'use_case', algorithm_id)
    # created folder if it doesn't exit
    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder

def get_data_dir(root_data_dir: str = SYNTHETIC_DATA_DIR, extension_type: str = DataCompleteness.complete) -> str:
    return path.join(root_data_dir, extension_type)


def get_root_folder_for_reduced_cluster(root_folder: str, n_dropped: int) -> str:
    folder = path.join(root_folder, 'clusters_dropped_' + str(n_dropped))
    # created folder if it doesn't exit
    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


def get_root_folder_for_reduced_segments(root_folder: str, n_dropped: int) -> str:
    folder = path.join(root_folder, 'segments_dropped_' + str(n_dropped))
    # created folder if it doesn't exit
    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


def get_filename_for_statistical_validation_between_measures(measure1: str, measure2: str):
    m1_short = short_distance_measure_names[measure1].replace(" ", "_")
    m2_short = short_distance_measure_names[measure2].replace(" ", "_")
    return m1_short + "_VS_" + m2_short + DISTANCE_MEASURE_AVG_RANK_STATS_VALIDATION


def get_image_name_based_on_data_dir_and_data_type(image_name: str, data_dir: str, data_type: str) -> str:
    """ If the data dir is irregular_p30 it attaches an irregular_p30 to the image name, attaches data_type
    to image name"""
    result = data_type + "-" + image_name
    return get_image_name_based_on_data_dir(result, data_dir)


def get_image_name_based_on_data_dir(image_base_name: str, data_dir: str) -> str:
    """ If the data dir is irregular_p30 it attaches an irregular_p30 to the image name"""
    irr_folder_extension = get_irregular_folder_name_from(data_dir)
    if irr_folder_extension:  # avoids a leading _ if irr folder extension is ''
        irr_folder_extension = irr_folder_extension + "_"
    return irr_folder_extension + image_base_name


def get_irregular_folder_name_from(data_dir: str):
    """Returns the irregular folder name from the data directory given, '' if standard data, 'p30' if irregular
    p30, 'p90' if irregular p90"""
    data_dir_match = re.search(r'_(p\d+)$', data_dir)
    return data_dir_match.group(1) if data_dir_match else ""


def get_data_completeness_from(data_dir: str):
    """Returns the data completeness from the data dir"""
    last_folder_name = os.path.basename(os.path.normpath(data_dir))
    if 'irregular' in last_folder_name:
        return last_folder_name
    else:
        return ""


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


def base_dataset_result_folder_for_type(root_result_dir: str, result_type: str):
    """ Returns the base folder for the result type"""
    return path.join(root_result_dir, result_type)


def dataset_description_dir(overall_dataset_name: str, data_type: str, root_results_dir: str, data_dir: str):
    """
          Returns directory for dataset description results
          :param overall_dataset_name: a name to identify the dataset overall e.g n30 or n2
          :param data_type: the data type, see SyntheticDataType
          :param root_results_dir: the directory for results, this is the main directory, dataset-description will
          be added to this
          :param data_dir: the directory from which the data was read, this helps to determine if we save the results
          in an e.g. irregular_p30 folder in the dataset-description folder
          :return: the path name to the results folder e.g. results/dataset-description/raw/,
          results/dataset-description/non_normal/, results/dataset-description/irregular_p30/raw
      """
    return get_folder_name_for(results_type=ResultsType.dataset_description,
                               overall_dataset_name=overall_dataset_name, data_type=data_type,
                               distance_measure="", results_dir=root_results_dir, data_dir=data_dir, drop_clusters=0,
                               drop_segments=0)


def distance_measure_evaluation_results_dir_for(run_name: str, data_type: str, base_results_dir: str,
                                                data_dir: str):
    """
          Returns results directory for distance measure evaluation
          :param run_name: the run name, e.g. wandb run_name
          :param data_type: the data type, see SyntheticDataType
          :param base_results_dir: the directory for results, this is the main directory usually results or test results
          :param data_dir: the directory from which the data was read to be able to add the irregular folder if required
          :return: the path name to the results folder e.g.
          results/distance-measures-evaluation/normal/overall_dataset_name or
          results/distance-measures-evaluation/irregular_p30/non_normal/overall_dataset_name
      """
    return get_folder_name_for(results_type=ResultsType.distance_measure_evaluation,
                               overall_dataset_name=run_name, data_type=data_type, data_dir=data_dir,
                               distance_measure="", results_dir=base_results_dir)


def distance_measure_assessment_dir_for(overall_dataset_name: str, data_type: str, results_dir: str, data_dir: str,
                                        distance_measure: str, drop_clusters: int = 0, drop_segments: int = 0):
    """
          Returns directory for distance measure assessments
          :param overall_dataset_name: a name to identify the dataset overall e.g n30 or n2
          :param data_type: the data type, see SyntheticDataType
          :param results_dir: the directory for results, this is the main directory, the internal_assessment folder will be
          added
          :param data_dir: the directory from which the data was read to be able to add the irregular folder if required
          :param distance_measure: the distance measures used for the internal assessment, see DistanceMeasures
          :param drop_clusters: will create an additional folder for evaluating reduced numbers of clusters, if 0 and
          n_dropped_segment is 0 no additional subfolder will be added
          :param drop_segments: will create an additional folder for evaluating reduced numbers of segments
          :return: the path name to the results folder e.g. results/distance-measures-assessment/raw/L1/ or
          results/distance-measures-assessment/raw/n_dropped_clusters/L1 or results/distance-measures-assessment/raw/n_dropped_segments/L1
          or results/distance-measures-assessment/raw/n_dropped_clusters_m_dropped_segments/L1
      """
    return get_folder_name_for(results_type=ResultsType.distance_measure_assessment,
                               overall_dataset_name=overall_dataset_name, data_type=data_type, data_dir=data_dir,
                               distance_measure=distance_measure, results_dir=results_dir, drop_clusters=drop_clusters,
                               drop_segments=drop_segments)


def internal_measure_calculation_dir_for(overall_dataset_name: str, data_type: str, results_dir: str, data_dir: str,
                                         distance_measure: str, drop_clusters: int = 0, drop_segments: int = 0):
    """
          Returns directory for internal measures calculation on bad partitions
          :param overall_dataset_name: a name to identify the dataset overall e.g n30 or n2
          :param data_type: the data type, see SyntheticDataType
          :param results_dir: the directory for results, this is the main directory, the internal_assessment folder will be
          added
          :param data_dir: the directory from which the data was read to be able to add the irregular folder if required
          :param distance_measure: the distance measures used for the internal assessment, see DistanceMeasures
          :param drop_clusters: will create an additional folder for evaluating reduced numbers of clusters, if 0 and
          n_dropped_segment is 0 no additional sub folder will be added
          :param drop_segments: will create an additional folder for evaluating reduced numbers of segments
          :return: the path name to the results folder e.g. results/internal_assessment/raw/L1/ or
          results/internal_assessment/raw/n_dropped_clusters/L1 or results/internal_assessment/raw/n_dropped_segments/L1
          or results/internal_assessment/raw/n_dropped_clusters_m_dropped_segments/L1
      """
    return get_folder_name_for(results_type=ResultsType.internal_measures_calculation,
                               overall_dataset_name=overall_dataset_name, data_type=data_type,
                               distance_measure=distance_measure, results_dir=results_dir, data_dir=data_dir,
                               drop_clusters=drop_clusters, drop_segments=drop_segments)


def internal_measure_evaluation_dir_for(overall_dataset_name: str, data_type: str, results_dir: str, data_dir: str,
                                        distance_measure: str, drop_clusters: int = 0, drop_segments: int = 0):
    """
    Returns directory for internal measures assessment results tables for the given data type and distance measure
    :param overall_dataset_name: a name to identify the dataset overall e.g n30 or n2
    :param data_type: the data type, see SyntheticDataType
    :param results_dir: the directory for results, this is the main directory, the internal_assessment folder will be
    added
    :param data_dir: the directory from which the data was read to be able to add the irregular folder if required
    :param distance_measure: the distance measures used for the internal assessment, see DistanceMeasures
    :param drop_clusters: will create an additional folder for evaluating reduced numbers of clusters, if 0 and
    n_dropped_segment is 0 no additional subfolder will be added
    :param drop_segments: will create an additional folder for evaluating reduced numbers of segments
    :return: the path name to the results folder e.g. results/internal_assessment/raw/L1/ or
    results/internal_assessment/raw/n_dropped_clusters/L1 or results/internal_assessment/raw/n_dropped_segments/L1
    or results/internal_assessment/raw/n_dropped_clusters_m_dropped_segments/L1
    """
    return get_folder_name_for(results_type=ResultsType.internal_measure_evaluation,
                               overall_dataset_name=overall_dataset_name, data_type=data_type, data_dir=data_dir,
                               distance_measure=distance_measure, results_dir=results_dir, drop_clusters=drop_clusters,
                               drop_segments=drop_segments)


def get_clustering_quality_multiple_data_variants_result_folder(results_type: str, overall_dataset_name: str,
                                                                results_dir: str, distance_measure: str):
    """
        Returns directory for the given results type assuming that it summarises multiple data variants
        :param overall_dataset_name: a name to identify the dataset overall e.g n30 or n2
        :param results_dir: the directory for results, this is the main directory, the internal_assessment folder will be
        added
        :param distance_measure: the distance measures used for the internal assessment, see DistanceMeasures
        :return: the path name to the results folder e.g. results/internal_assessment/overall_ds_name/
    """
    # put in folder internal_assessment
    results_folder = path.join(results_dir, results_type)
    # put in sub folder e.g. n30
    results_folder = path.join(results_folder, overall_dataset_name)
    # put in sub folder e.g. for distance measure
    if distance_measure != "":
        results_folder = path.join(results_folder, folder_name_for_distance_measure(distance_measure))
    # creates the folder if it does not exist
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    return results_folder


def get_folder_name_for(results_type: str, overall_dataset_name: str, data_type: str, results_dir: str,
                        distance_measure: str, data_dir: str, drop_clusters: int = 0, drop_segments: int = 0):
    """
        Returns directory for the given results type using
        :param overall_dataset_name: a name to identify the dataset overall e.g n30 or n2
        :param data_type: the data type, see SyntheticDataType
        :param results_dir: the directory for results, this is the main directory, the internal_assessment folder will be
        added
        :param distance_measure: the distance measures used for the internal assessment, see DistanceMeasures
        :param data_dir: this is used to decide if creating an additional layer for e.g. irregular_p30
        :param drop_clusters: will create an additional folder for evaluating reduced numbers of clusters, if 0 and
        n_dropped_segment is 0 no additional subfolder will be added
        :param drop_segments: will create an additional folder for evaluating reduced numbers of segments
        :return: the path name to the results folder e.g. results/internal_assessment/raw/L1/ or
        results/internal_assessment/raw/n_dropped_clusters/L1 or results/internal_assessment/raw/n_dropped_segments/L1
        or results/internal_assessment/raw/n_dropped_clusters_m_dropped_segments/L1
    """
    # put in folder internal_assessment
    results_folder = path.join(results_dir, results_type)
    # check if irregular and if it is, add the irregular folder
    data_dir_irr_name = get_irregular_folder_name_from(data_dir)
    if data_dir_irr_name:  # the last folder name is an irregular one
        last_folder = os.path.basename(data_dir)
        results_folder = path.join(results_folder, last_folder)
    # put in sub folder e.g. raw
    results_folder = path.join(results_folder, data_type)
    # put in sub folder e.g. n30
    results_folder = path.join(results_folder, overall_dataset_name)
    # put in sub folder e.g n_dropped_cluster if not evaluating all cluster
    dropped = get_folder_name_for_dropped_clusters_and_segments(drop_clusters, drop_segments)
    if dropped != "":
        results_folder = path.join(results_folder, dropped)
    # put in sub folder e.g. for distance measure
    if distance_measure != "":
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
    """Returns the result name for the internal measure assessment for the given ds_name
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
    wandb_confirmatory_project_name: str = config['wandb_confirmatory_project_name']
    wandb_confirmatory_partitions_project_name: str = config['wandb_confirmatory_partitions_project_name']
    wandb_confirmatory_irregular_project_name: str = config['wandb_confirmatory_irregular_project_name']
    wandb_use_case_project_name: str = config['wandb_use_case_project_name']
