from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_multiple_data_variants import get_row_name_from


class VisualiseClusteringQualityMeasuresForDataVariants:
    def __init__(self, run_file: str, overall_ds_name: str, dataset_types: [str], data_dirs: [str],
                 distance_measure: str, clustering_quality_measure: [str], backend: str = Backends.none.value):
        self.run_file = run_file
        self.overall_ds_name = overall_ds_name
        self.dataset_types = dataset_types
        self.data_dirs = data_dirs
        self.distance_measure = distance_measure
        self.clustering_quality_measure = clustering_quality_measure
        self.backend = backend
        self.row_names = []
        self.all_data = {}
        for folder in data_dirs:
            row_name = get_row_name_from(folder)
            self.row_names.append(row_name)
            column_results = {}
            for ds_type in dataset_types:
                column_name = SyntheticDataType.get_display_name_for_data_type(ds_type)
                ds = DescribeMultipleDatasets(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                              data_type=ds_type, data_dir=folder)
                column_results[column_name] = ds
            self.all_data[row_name] = column_results
        self.col_names = [SyntheticDataType.get_display_name_for_data_type(ds_type) for ds_type in dataset_types]
