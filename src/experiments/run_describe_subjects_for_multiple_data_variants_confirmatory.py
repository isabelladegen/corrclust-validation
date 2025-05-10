from src.evaluation.describe_subjects_for_data_variant import DescribeSubjectsForDataVariant, \
    combine_all_ds_variations_multiple_description_summary_dfs
from src.utils.configurations import CONF_IRREGULAR_P90_DATA_DIR, CONF_IRREGULAR_P30_DATA_DIR, \
    CONFIRMATORY_SYNTHETIC_DATA_DIR, CONF_ROOT_RESULTS_DIR, CONFIRMATORY_DATASETS_FILE_PATH
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == '__main__':
    # create summary for a dataset variation
    run_file = CONFIRMATORY_DATASETS_FILE_PATH
    overall_ds_name = "n30"
    root_results_dir = CONF_ROOT_RESULTS_DIR

    dataset_types = [SyntheticDataType.raw, SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated, SyntheticDataType.rs_1min]

    # do regular sampled ones
    for ds_type in dataset_types:
        ds = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=ds_type,
                                            data_dir=CONFIRMATORY_SYNTHETIC_DATA_DIR)
        ds.save_summary(root_results_dir)

    # do irregular p30 sampled ones
    for ds_type in dataset_types:
        ds = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=ds_type,
                                            data_dir=CONF_IRREGULAR_P30_DATA_DIR, load_data=True)
        ds.save_summary(root_results_dir)

    # do irregular p90 sampled ones
    for ds_type in dataset_types:
        ds = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name, data_type=ds_type,
                                            data_dir=CONF_IRREGULAR_P90_DATA_DIR, load_data=True)
        ds.save_summary(root_results_dir)

    # write combined results (this also reads all files and then writes a result)

    combine_all_ds_variations_multiple_description_summary_dfs(result_root_dir=root_results_dir,
                                                               overall_ds_name=overall_ds_name,
                                                               dataset_types=dataset_types,
                                                               data_dirs=[CONFIRMATORY_SYNTHETIC_DATA_DIR,
                                                                          CONF_IRREGULAR_P30_DATA_DIR,
                                                                          CONF_IRREGULAR_P90_DATA_DIR],
                                                               save_combined_results=True
                                                               )
