import copy
import os

import yaml
from datasets import splits

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

    completeness_levels = {
        "complete": "",
        "partial": "irregular_p30",
        "sparse": "irregular_p90",
    }

    data_variants = {
        "standard": "",
        "bad_clustering": "bad_partitions",
        "reduced_12_clusters": "clusters_dropped_12",
        "reduced_6_clusters": "clusters_dropped_17",
        "reduced_50_segments": "segments_dropped_50",
        "reduced_25_segments": "segments_dropped_75",
    }
    # todo change to parquet for new db
    file_types = {
        "data": "*-data.csv",
        "labels": "*-labels.csv",
    }

    # Path patterns
    path_patterns = {
        ("raw", "complete"): "exploratory/raw/*-{file_type}.csv",
        ("correlated", "complete"): "exploratory/normal/*-{file_type}.csv",
        ("correlated", "partial"): "exploratory/irregular_p30/normal/*-{file_type}.csv",
        ("nonnormal", "partial"): "exploratory/irregular_p30/non_normal/*-{file_type}.csv",
        ("correlated", "sparse"): "exploratory/irregular_p90/normal/*-{file_type}.csv"
    }

    # Special path patterns
    special_path_patterns = {
        (
            "correlated", "sparse",
            "badclusterings_labels"): "exploratory/irregular_p90/normal/bad_partitions/*-labels.csv"
    }

    data_features = [
        # {"name": "subject_id", "dtype": "string"}, # todo put back in for new db
        {"name": "datetime", "dtype": "string", "type": "timestamp"},
        {"name": "iob", "dtype": "float32"},
        {"name": "cob", "dtype": "float32"},
        {"name": "ig", "dtype": "float32"}
    ]

    label_features = [
        # {"name": SyntheticDataSegmentCols.subject_id, "dtype": "string"}, # todo put back in for new db
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

    bad_partitions_features = [{"name": SyntheticDataSegmentCols.cluster_desc, "dtype": "string"}] + copy.deepcopy(
        label_features)
    # bad_partitions_features = label_features


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
    for split in ["exploratory", "confirmatory"]:
        for gen_key, gen_value in HFStructures.generation_stages.items():
            for comp_key, comp_value in HFStructures.completeness_levels.items():
                for file_key, file_value in HFStructures.file_types.items():
                    path_value = os.path.join(split, comp_value, gen_value, file_value)
                    # need to make sure we add quotes for * path values
                    # if "*" in path_value:
                    #     path_value = f'"{path_value}"'  # Make sure it's quoted
                    a_config = {
                        "config_name": "_".join([gen_key, comp_key, file_key]),  # e.g. raw_complete_data
                        "data_files": [
                            {
                                "split": split,
                                # e.g. exploratory/irregular_p30/raw/*-data.csv"
                                "path": path_value
                            }
                        ],
                        "features": HFStructures.data_features if file_key == "data" else HFStructures.label_features,
                    }
                    configs.append(a_config)

                # add bad-partitions
                a_config = {
                    "config_name": "_".join([gen_key, comp_key, "labels"]),  # e.g. raw_complete_data
                    "data_files": [
                        {
                            "split": split,
                            # e.g. exploratory/irregular_p30/raw/bad_partitions/*-labels.csv"
                            "path": os.path.join(split, comp_value, gen_value, "bad_partitions",
                                                 HFStructures.file_types["labels"])
                        }
                    ],
                    "features": HFStructures.bad_partitions_features,
                }
                configs.append(a_config)

    return configs


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
