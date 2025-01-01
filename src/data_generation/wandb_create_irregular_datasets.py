import traceback
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

import wandb

from src.data_generation.create_irregular_datasets import CreateIrregularDataset
from src.data_generation.wandb_create_synthetic_data import save_data_labels_to_file, log_dataset_description
from src.evaluation.describe_synthetic_dataset import DescribeSyntheticDataset
from src.utils.configurations import WandbConfiguration, GENERATED_DATASETS_FILE_PATH, \
    SyntheticDataVariates, IRREGULAR_P30, SYNTHETIC_DATA_DIR
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends


@dataclass
class CreateIrregularDSConfig:
    wandb_project_name: str = WandbConfiguration.wandb_irregular_project_name
    wandb_entity: str = WandbConfiguration.wandb_entity
    wandb_mode: str = 'online'
    wandb_notes: str = "creates irregular ds of synthetic data"
    tags = ['Synthetic']

    # Store new data in dir
    root_result_data_dir: str = IRREGULAR_P30
    # Load data from dir
    data_dir: str = SYNTHETIC_DATA_DIR
    # Runs to create irregular versions for
    csv_of_runs: str = GENERATED_DATASETS_FILE_PATH
    # data cols to use
    data_cols: [str] = field(default_factory=lambda: SyntheticDataVariates.columns())
    # backend to use for visualisations
    backend: str = Backends.none.value

    # drop p percent of observations, 0<p<1
    p: float = 0.3

    # resample rule - see pandas resampling rules
    rs_rule = "1min"

    def as_dict(self):
        return asdict(self)


def create_irregular_datasets(config: CreateIrregularDSConfig, ds_name: str, seed: int):
    """
    Wandb generate irregular data according to the config provided. It will be created for all raw, nn,
    and nc version, and a rs version of the irregular data will be saved too for the given run. The run will
    use the ds_name
    :param config: CreateIrregularDSConfig that configures the run
    :param ds_name: run name of the synthetic data
    :param seed: seed used to drop observations
    :return: dictionary of DescribeSyntheticData class for all data variations, wandb summary dict
    """
    raw_desc, nc_desc, nn_desc, rs_desc = None, None, None, None
    try:
        p_string = str(int(100 * config.p))
        project_name = config.wandb_project_name + "_p" + p_string
        wandb.init(project=project_name,
                   entity=config.wandb_entity,
                   name=ds_name,
                   mode=config.wandb_mode,
                   notes=config.wandb_notes,
                   tags=config.tags,
                   config=config.as_dict())

        exit_code = 0
        # we drop the same amount of data but not the same observations for different runs
        wandb.log({"seed": seed})

        # check that result dir matches p
        error_msg = "Root result dir: " + config.root_result_data_dir + ". does not match given p: " + p_string
        assert p_string in config.root_result_data_dir, error_msg

        print("1. LOAD RAW DATASET: " + ds_name)
        raw_data_type = SyntheticDataType.raw
        irds = CreateIrregularDataset(run_name=ds_name, data_type=raw_data_type, data_dir=config.data_dir,
                                      data_cols=config.data_cols, seed=seed)

        print("2. CREATE RAW IRREGULAR DATA")
        raw_irregular_data, raw_irregular_labels = irds.drop_observation_with_likelihood(config.p)

        print("3. SAVE RAW IRREGULAR DATA AND LABELS")
        save_data_labels_to_file(config.root_result_data_dir, raw_data_type, raw_irregular_data,
                                 raw_irregular_labels, ds_name)

        print("4. CREATE NC IRREGULAR DATA")
        nc_datatype = SyntheticDataType.normal_correlated
        nc_irr_data, nc_irr_labels = irds.irregular_version_for_data_type(nc_datatype,
                                                                          raw_irregular_data, raw_irregular_labels)

        print("5. SAVE NC IRREGULAR DATA AND LABELS")
        save_data_labels_to_file(config.root_result_data_dir, nc_datatype, nc_irr_data, nc_irr_labels, ds_name)

        print("6. CREATE NN IRREGULAR DATA")
        nn_datatype = SyntheticDataType.non_normal_correlated
        nn_irr_data, nn_irr_labels = irds.irregular_version_for_data_type(nn_datatype,
                                                                          raw_irregular_data, raw_irregular_labels)

        print("7. SAVE NN IRREGULAR DATA AND LABELS")
        save_data_labels_to_file(config.root_result_data_dir, nn_datatype, nn_irr_data, nn_irr_labels, ds_name)

        print("8. CREATE RS IRREGULAR DATA")
        rs_datatype = SyntheticDataType.resample(config.rs_rule)
        print("...resampling rule: " + config.rs_rule)
        rs_irr_data, rs_irr_labels = irds.irregular_version_for_data_type(rs_datatype,
                                                                          raw_irregular_data, raw_irregular_labels)

        print("9. SAVE RS IRREGULAR DATA AND LABELS")
        save_data_labels_to_file(config.root_result_data_dir, rs_datatype, rs_irr_data, rs_irr_labels, ds_name)

        print("10. LOG RAW DESCRIPTION")
        raw_desc = DescribeSyntheticDataset(ds_name, data_type=SyntheticDataType.raw,
                                            data_dir=config.root_result_data_dir)
        log_dataset_description(raw_desc, "RAW")

        print("11. LOG NORMAL CORRELATED DESCRIPTION")
        nc_desc = DescribeSyntheticDataset(ds_name, data_type=SyntheticDataType.normal_correlated,
                                           data_dir=config.root_result_data_dir)
        log_dataset_description(nc_desc, "NC")

        print("12. LOG NON-NORMAL CORRELATED DESCRIPTION")
        nn_desc = DescribeSyntheticDataset(ds_name, data_type=SyntheticDataType.non_normal_correlated,
                                           data_dir=config.root_result_data_dir)
        log_dataset_description(nn_desc, "NN")

        print("13. LOG RESAMPLED DESCRIPTION")
        rs_desc = DescribeSyntheticDataset(ds_name, data_type=SyntheticDataType().resample(config.rs_rule),
                                           data_dir=config.root_result_data_dir)
        log_dataset_description(rs_desc, "RS")

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)  # try to get error into wandb log
        exit_code = 1

    wandb_summary_dic = dict(wandb.run.summary)
    wandb.finish(exit_code=exit_code)
    if exit_code == 1:
        raise
    return {"raw": raw_desc, "nc": nc_desc, "nn": nn_desc, "rs": rs_desc}, wandb_summary_dic


if __name__ == "__main__":
    config = CreateIrregularDSConfig()
    config.p = 0.3
    config.root_result_data_dir = IRREGULAR_P30  # ensure this matches your p!
    config.rs_rule = "1min"
    csv_file = config.csv_of_runs
    generated_ds = pd.read_csv(csv_file)['Name'].tolist()

    # we create a run for each ds and we name the run consistently
    for idx, ds_name in enumerate(generated_ds):
        np.random.seed(1661 + idx)
        dataset_seed = np.random.randint(low=100, high=1000000)
        create_irregular_datasets(config, ds_name, dataset_seed)
