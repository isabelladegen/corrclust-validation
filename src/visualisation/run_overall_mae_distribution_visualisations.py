from src.data_generation.generate_synthetic_segmented_dataset import CorrType
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, GENERATED_DATASETS_FILE_PATH, \
    IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_multiple_data_variants import VisualiseMultipleDataVariants

if __name__ == "__main__":
    # visualise min segment length for different correlations
    root_result_dir = ROOT_RESULTS_DIR
    backend = Backends.none.value
    data_types = [SyntheticDataType.raw, SyntheticDataType.normal_correlated,
                  SyntheticDataType.non_normal_correlated, SyntheticDataType.rs_1min]
    data_dir = [SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR]
    correlations = [CorrType.spearman, CorrType.pearson, CorrType.kendall]

    vds = VisualiseMultipleDataVariants(run_file=GENERATED_DATASETS_FILE_PATH, overall_ds_name="n30",
                                        dataset_types=data_types,
                                        data_dirs=data_dir,
                                        additional_cor=[CorrType.pearson, CorrType.kendall],
                                        backend=backend)

    fig = vds.violin_plots_of_overall_mae(save_fig=True, cor_types=correlations, root_result_dir=ROOT_RESULTS_DIR)
