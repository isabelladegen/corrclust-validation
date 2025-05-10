from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, GENERATED_DATASETS_FILE_PATH, \
    IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_multiple_data_variants import VisualiseMultipleDataVariants

if __name__ == "__main__":
    # correlation_patterns for data variants
    root_result_dir = ROOT_RESULTS_DIR
    backend = Backends.none.value
    data_types = [SyntheticDataType.raw, SyntheticDataType.normal_correlated,
                  SyntheticDataType.non_normal_correlated, SyntheticDataType.rs_1min]
    data_dir = [SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR]
    pattern_id = 19

    vds = VisualiseMultipleDataVariants(run_file=GENERATED_DATASETS_FILE_PATH, overall_ds_name="n30",
                                        dataset_types=data_types,
                                        data_dirs=data_dir,
                                        additional_cor=[],
                                        backend=backend)

    # faster plot for testing
    fig = vds.correlation_pattern_for_pattern(save_fig=True, root_result_dir=ROOT_RESULTS_DIR, pattern_id=pattern_id)
