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

    reduced_splits = {
        "reduced_11_clusters": "clusters_dropped_12",
        "reduced_6_clusters": "clusters_dropped_17",
        "reduced_50_segments": "segments_dropped_50",
        "reduced_25_segments": "segments_dropped_75",
    }

    file_types = {
        "data": "*-data.parquet",
        "labels": "*-labels.parquet",
    }

    data_features = [
        {"name": "subject_id", "dtype": "string"},
        {"name": "datetime", "dtype": "string", "type": "timestamp"},
        {"name": "iob", "dtype": "float64"},
        {"name": "cob", "dtype": "float64"},
        {"name": "ig", "dtype": "float64"}
    ]

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
                    # e.g. exploratory/irregular_p30/raw/*-data.csv"
                    config_name = "_".join([gen_key, comp_key, file_key])
                    path_value = os.path.join(split, comp_value, gen_value, file_value)
                    features = HFStructures.data_features if file_key == "data" else HFStructures.label_features
                    a_config = create_a_config(name=config_name,
                                               split=split,
                                               path=path_value,
                                               features=features)
                    configs.append(a_config)

                # add bad-partitions
                # path e.g. exploratory/irregular_p30/raw/bad_partitions/*-labels.csv"
                bad_part_path = os.path.join(split, comp_value, gen_value, "bad_partitions",
                                             HFStructures.file_types["labels"])
                bad_part_config = "_".join([gen_key, comp_key, "badclustering_labels"])
                a_config = create_a_config(name=bad_part_config,
                                           split=split,
                                           path=bad_part_path,
                                           features=HFStructures.bad_partitions_features)
                configs.append(a_config)

    # load reduced data as separate splits
    for split in ["exploratory", "confirmatory"]:
        for reduced_key, reduced_value in HFStructures.reduced_splits.items():
            # reduced data exists only for correlated and nonnormal data variants
            split_name = "_".join([reduced_key, split])
            for gen_key in ["correlated", "nonnormal"]:
                gen_value = HFStructures.generation_stages[gen_key]
                for comp_key, comp_value in HFStructures.completeness_levels.items():
                    for file_key, file_value in HFStructures.file_types.items():
                        # e.g. exploratory/reduced-data/clusters_dropped_12/irregular_p30/normal/*-data.csv"
                        config_name = "_".join([gen_key, comp_key, file_key])
                        path_value = os.path.join(split, "reduced-data", reduced_value, comp_value, gen_value,
                                                  file_value)
                        features = HFStructures.data_features if file_key == "data" else HFStructures.label_features
                        a_config = create_a_config(name=config_name,
                                                   split=split_name,
                                                   path=path_value,
                                                   features=features)
                        configs.append(a_config)

                    # add bad-partitions
                    # path e.g. exploratory/reduced-data/clusters_dropped_12/irregular_p30/raw/bad_partitions/*-labels.csv"
                    bad_part_path = os.path.join(split, "reduced-data", reduced_value, comp_value, gen_value,
                                                 "bad_partitions",
                                                 HFStructures.file_types["labels"])
                    bad_part_config = "_".join([gen_key, comp_key, "badclustering_labels"])
                    a_config = create_a_config(name=bad_part_config,
                                               split=split_name,
                                               path=bad_part_path,
                                               features=HFStructures.bad_partitions_features)
                    configs.append(a_config)

    return configs


def create_a_config(name, split, path, features):
    a_config = {
        "config_name": name,  # e.g. raw_complete_data
        "data_files": [
            {
                "split": split,
                "path": path
            }
        ],
        "features": features
    }
    return a_config


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
