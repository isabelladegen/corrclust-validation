import copy
import os

import yaml

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols


class HFStructures:
    """ This class translates between the hugging face names and the file directories
    where the keys are the hf names and the values are the corresponding folder names,
    file extensions, etc. strings
    """
    generation_stages = {
        "raw": "raw",
        "correlated": "normal",
        "nonnormal": "non_normal",
        "downsampled": "resampled_1min"
    }

    reduced_generation_stages = ["correlated", "nonnormal"]

    completeness_levels = {
        "complete": "",
        "partial": "irregular_p30",
        "sparse": "irregular_p90",
    }

    splits = {
        "exploratory": "exploratory",
        "confirmatory": "confirmatory",
        "reduced_11_clusters_exploratory": "exploratory/reduced-data/clusters_dropped_12",
        "reduced_6_clusters_exploratory": "exploratory/reduced-data/clusters_dropped_17",
        "reduced_50_segments_exploratory": "exploratory/reduced-data/segments_dropped_50",
        "reduced_25_segments_exploratory": "exploratory/reduced-data/segments_dropped_75",
        "reduced_11_clusters_confirmatory": "confirmatory/reduced-data/clusters_dropped_12",
        "reduced_6_clusters_confirmatory": "confirmatory/reduced-data/clusters_dropped_17",
        "reduced_50_segments_confirmatory": "confirmatory/reduced-data/segments_dropped_50",
        "reduced_25_segments_confirmatory": "confirmatory/reduced-data/segments_dropped_75",
    }

    generation_stages_for_splits = {
        "exploratory": list(generation_stages.keys()),
        "confirmatory": list(generation_stages.keys()),
        "reduced_11_clusters_exploratory": reduced_generation_stages,
        "reduced_6_clusters_exploratory": reduced_generation_stages,
        "reduced_50_segments_exploratory": reduced_generation_stages,
        "reduced_25_segments_exploratory": reduced_generation_stages,
        "reduced_11_clusters_confirmatory": reduced_generation_stages,
        "reduced_6_clusters_confirmatory": reduced_generation_stages,
        "reduced_50_segments_confirmatory": reduced_generation_stages,
        "reduced_25_segments_confirmatory": reduced_generation_stages,
    }

    file_types = {
        "data": "*-data.parquet",
        "labels": "*-labels.parquet",
        "badclustering_labels": "bad_partitions/*-labels.parquet"
    }

    data_features = [
        {"name": "subject_id", "dtype": "string"},
        {"name": "datetime", "dtype": "string", "type": "timestamp"},
        {"name": "iob", "dtype": "float64"},
        {"name": "cob", "dtype": "float64"},
        {"name": "ig", "dtype": "float64"}
    ]

    downsampled_data_features = copy.deepcopy(data_features)
    downsampled_data_features.insert(2, {"name": "original_index", "dtype": "float64"})

    sparsified_data_features = copy.deepcopy(data_features)
    sparsified_data_features.insert(1, {"name": SyntheticDataSegmentCols.old_regular_id, "dtype": "int64"})

    label_features = [
        {"name": SyntheticDataSegmentCols.subject_id, "dtype": "string"},
        {"name": SyntheticDataSegmentCols.segment_id, "dtype": "int32"},
        {"name": SyntheticDataSegmentCols.start_idx, "dtype": "int32"},
        {"name": SyntheticDataSegmentCols.end_idx, "dtype": "int32"},
        {"name": SyntheticDataSegmentCols.length, "dtype": "int32"},
        {"name": SyntheticDataSegmentCols.pattern_id, "dtype": "int32"},
        {"name": SyntheticDataSegmentCols.correlation_to_model, "dtype": "string"},
        {"name": SyntheticDataSegmentCols.actual_correlation, "dtype": "string"},
        {"name": SyntheticDataSegmentCols.actual_within_tolerance, "dtype": "string"},
        {"name": SyntheticDataSegmentCols.mae, "dtype": "float32"},
        {"name": SyntheticDataSegmentCols.relaxed_mae, "dtype": "float32"}
    ]

    bad_partitions_features = copy.deepcopy(label_features)
    bad_partitions_features.insert(1, {"name": SyntheticDataSegmentCols.cluster_desc, "dtype": "string"})

    @classmethod
    def get_features_for(cls, file_key, comp_key, gen_key):
        """Returns the features based on the file type and completeness levels"""
        if file_key == "data":
            if gen_key == "downsampled":
                return cls.downsampled_data_features
            elif comp_key == "complete":
                return cls.data_features
            else:
                return cls.sparsified_data_features
        elif file_key == "labels":
            return cls.label_features
        elif file_key == "badclustering_labels":
            return cls.bad_partitions_features
        else:
            assert False, "Unknown file_key: " + str(file_key)


def generate_hf_configs():
    """Build as dictionary"""
    config = {
        "license": "cc-by-4.0",
        "configs": build_list_of_configs()
    }
    return config


def build_list_of_configs():
    configs = []

    # build standard configs for all 12 data variants
    for gen_key, gen_value in HFStructures.generation_stages.items():
        for comp_key, comp_value in HFStructures.completeness_levels.items():
            for file_key, file_value in HFStructures.file_types.items():
                config_name = "_".join([gen_key, comp_key, file_key])
                features = HFStructures.get_features_for(file_key, comp_key, gen_key)
                # create all splits and datafiles for a config
                data_files = []
                for split_key, split_value in HFStructures.splits.items():
                    # check if split has this generation stage
                    if gen_key not in HFStructures.generation_stages_for_splits[split_key]:
                        continue
                    # e.g. exploratory/irregular_p30/raw/*-data.csv"
                    # exploratory/irregular_p30/raw/bad_partitions/*-labels.csv"
                    # confirmatory/reduced-data/clusters_dropped_12/normal/*-labels.csv
                    path_value = os.path.join(split_value, comp_value, gen_value, file_value)
                    data_files.append(create_a_data_file(split_key, path_value))
                # create config
                a_config = create_a_config(name=config_name,
                                           data_files=data_files,
                                           features=features)
                configs.append(a_config)

    return configs


def create_a_config(name, data_files: [], features):
    a_config = {
        "config_name": name,  # e.g. raw_complete_data
        "data_files": data_files,
        "features": features
    }
    return a_config


def create_a_data_file(split, path):
    return {
        "split": split,
        "path": path
    }


class PathQuotingDumper(yaml.SafeDumper):
    """Custom YAML dumper that puts double quotes around paths with wildcards."""

    def represent_scalar(self, tag, value, style=None):
        # Check if this is a string that contains wildcards or other special chars
        if tag == 'tag:yaml.org,2002:str' and isinstance(value, str):
            if '*' in value or '?' in value:
                style = '"'  # Force double quotes for wildcards
        return super().represent_scalar(tag, value, style)


if __name__ == "__main__":
    dataset_config = generate_hf_configs()

    # Save to file
    output_file = "huggingface_config.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(dataset_config,
                  f,
                  default_flow_style=False,
                  sort_keys=False,
                  Dumper=PathQuotingDumper)

    print(f"Configuration saved to {output_file}")
