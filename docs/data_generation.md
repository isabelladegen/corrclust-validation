from tests.evaluation.test_impact_of_reduction_on_internal_indices import root_reduced_data_dir

# Data Generation Guide

This section provides instructions for generating your own synthetic time series data with known correlation structures 
using our framework. If you want to modify or extend the CSTS benchmark, this guide will help you navigate our 
data generation pipeline.

## 1. Generating Complete Data Variants

To generate complete data variants (raw, correlated, non-normal, and downsampled):

```python
from src.data_generation.wandb_create_synthetic_data import SyntheticDataConfig, one_synthetic_creation_run

config = SyntheticDataConfig() # modify to adjust
dataset_seed = 12345 # add a seed to control random generation

# create a single subject
one_synthetic_creation_run(config, seed=dataset_seed)
```

**Customisation options:**
- Number of subjects created by running this as many times as you need varying the parameters
- Number of time series variates via `SyntheticDataConfig().number_of_variates` and names of variates via `SyntheticDataConfig().columns`
- Number of segments per subject via `SyntheticDataConfig().number_of_segments` 
- Segment durations (short and long) via `SyntheticDataConfig().segment_durations_short` and `SyntheticDataConfig().segment_durations_long` 
- Distribution types and parameters via `SyntheticDataConfig().distributions_for_variates`, `SyntheticDataConfig().distributions_args_<iob/cob/ig>`, 
- `SyntheticDataConfig().distributions_kwargs_<iob/cob/ig>` if you change the number and or names of variates you will have to modify the method slightly
- Correlation structures to model via providing an updated [correlation_patterns_to_model.csv](https://github.com/isabelladegen/corrclust-validation/blob/main/src/data_generation/config/correlation_patterns_to_model.csv)
- Downsampling rate via `SyntheticDataConfig().resample_rule` (using Pandas resample rules)

See [wandb_create_synthetic_data.py](https://github.com/isabelladegen/corrclust-validation/blob/main/src/data_generation/wandb_create_synthetic_data.py) for full implementation details. 
Note that this script uses Weights & Biases (WANDB) for experiment tracking and requires a specific configuration structure.
You might need to make modifications if you make loads of changes to make everything work. Use the test to 
check if things work in principle.

## 2. Generating Irregular Data Variants

To create sparsified data variants variants:

```python
import numpy as np
from src.data_generation.wandb_create_irregular_datasets import create_irregular_datasets, CreateIrregularDSConfig
from src.utils.configurations import WandbConfiguration, DataCompleteness, get_data_dir

# Configure which irregular variants to create, the following creates partial and sparse data,
# p defines how many observations to drop 0.3=30% 
irregular_pairs = [(0.3, DataCompleteness.irregular_p30), (0.9, DataCompleteness.irregular_p90)]
main_seed = 12345 # pick a seed for regeneration

for p, data_comp in irregular_pairs:
    config = CreateIrregularDSConfig()
    config.p = p  # Probability of dropping observations
    config.data_dir = '<data root dir>' # data root dir
    config.rs_rule = '<downsample rule, defaults to 1min>'
    config.data_cols = [] # list of strings of the data cols you used
    # this will add folder such as irregular_p30 in your main data folder which contain directories raw, normal, non-normal, resampled_<rule> 
    config.root_result_data_dir = get_data_dir(root_data_dir=config.data_dir, extension_type=data_comp)
    config.wandb_project_name = WandbConfiguration.wandb_irregular_project_name

    # Generate for each dataset
    subjects = [] # list of subject names from pipeline stage 1
    for idx, ds_name in enumerate(subjects):
        np.random.seed(main_seed + idx)
        dataset_seed = np.random.randint(low=100, high=1000000)
        create_irregular_datasets(config, ds_name, dataset_seed)
```

**Customisation options:**
- Sparsification versions via `irregular_pairs`
Note that the sparse versions are created from the complete version. The downsample rules and variate column names 
and data directories need to match. You might need to extend the `DataCompleteness` to allow for
other completeness levels.

See [wandb_create_irregular_datasets.py](https://github.com/isabelladegen/corrclust-validation/blob/main/src/data_generation/wandb_create_irregular_datasets.py) for implementation details. 
This script also uses Weight & Biases but this can easily be avoided with a few changes.

## 3. Generating Degraded Clustering Results

To create controlled bad clusterings for validation method evaluation:

```python
from src.data_generation.wandb_create_bad_partitions import create_bad_partitions, CreateBadPartitionsConfig
from src.utils.configurations import WandbConfiguration
from src.utils.load_synthetic_data import SyntheticDataType

# Configure data variants to create bad partitions for you might need to SyntheticDataType if
# you changed downsampling rules
dataset_types = [SyntheticDataType.raw, 
                 SyntheticDataType.normal_correlated,
                 SyntheticDataType.non_normal_correlated,
                 SyntheticDataType.rs_1min]
data_dirs = [] # list of folder for complete, partial and sparse (or your sparsified) data versions

config = CreateBadPartitionsConfig()
config.wandb_project_name = WandbConfiguration.wandb_partitions_project_name
config.seed = 12345 # set to a different number
config.data_cols = [] # list of strings of the data cols you used
config.leave_obs = 100  # how many observations to leave (not shift) in segment (this is based on your min segment length)
# how many partitions to generate for each of the three strategies, will result in 3x this number of degraded clusterings
# for each subject
config.n_partitions = 22
run_names = [] # list of subject names

# Default n_partitions is set in the configuration
for data_dir in data_dirs:
    for data_type in dataset_types:
        config.data_dir = data_dir
        config.data_type = data_type

        for idx, ds_name in enumerate(run_names):
            create_bad_partitions(config, ds_name=ds_name, idx=idx)
```

**Customisation options:**
- Number of partitions to generate for each of the three strategy via `CreateBadPartitionsConfig().n_partitions`
- How many observations not to shift in each segment (> min segment length) via `CreateBadPartitionsConfig().leave_obs`

See [wandb_create_bad_partitions.py](https://github.com/isabelladegen/corrclust-validation/blob/main/src/data_generation/wandb_create_bad_partitions.py) for implementation details. 
This script also uses Weight & Biases but this can easily be avoided with a few changes.

## 4. Generating Reduced Datasets

To create datasets with reduced cluster or segment counts:

```python
from src.data_generation.reduced_datasets.run_create_reduced_datasets import create_reduced_datasets
from src.utils.configurations import SYNTHETIC_DATA_DIR
from src.utils.configurations import DataCompleteness
from src.utils.load_synthetic_data import SyntheticDataType

run_names = [] # list of subject names
seed = 12345 # set to a different number

# choose which ones you want to generate reduced versions for
data_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated]
# you might need to adjust to your sparsification names
completeness = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]

root_reduced_data_dir = '' #full path on where you want to store these versions e.g ROOT_DIR/csts/exploratory/reduced-data

create_reduced_datasets(root_data_dir=SYNTHETIC_DATA_DIR, data_types=data_types, completeness=_completeness,
                        run_names=run_names, seed=seed,
                        root_reduced_data_dir=root_reduced_data_dir,
                        n_dropped_clusters=[12, 17],
                        n_dropped_segments=[50, 75])
```

**Customisation:**
- How many clusters to drop via `n_dropped_clusters`, e.g 12 keeps 11 clusters of 23, 17 keeps 6 clusters of 23
- How many segments to drop via `n_dropped_segments` (same as clusters, if empty no versions with dropped segments are created)

See [run_create_reduced_datasets.py](https://github.com/isabelladegen/corrclust-validation/blob/main/src/data_generation/reduced_datasets/run_create_reduced_datasets.py) for implementation details.
You need to make sure everything fits with your previous configurations which might require some modifications.

### Important Notes:

1. All generation scripts use Weights & Biases (WANDB) for experiment tracking - you'll need to set up WANDB or modify the scripts to run without it. To configure wandb create a file called `private.yaml` from [config template](https://github.com/isabelladegen/corrclust-validation/blob/main/private-yaml-template.yaml) which will make wandb automatically work for you as long as you have it installed and are logged in.
2. The scripts follow a sequential pipeline: complete datasets → sparsified versions → degraded/reduced versions. The configurations between these needs to match your customisation.
3. Each stage requires specific configuration objects that control the generation parameters. Some lookup and global variables we have defined for convenience mostly in [configurations.py](https://github.com/isabelladegen/corrclust-validation/blob/main/src/utils/configurations.py) might need modifying based on the configuration you choose.
4. The generated datasets are saved to specific directory structures we use based on completeness level (complete → data root folder, partial → `irregular_p30` in data root folder, sparse → `irregular_p90` in data root folder) each of these have a `raw`, `normal`, `non_normal`, and `resampled_<rs_rule.` subfolder which each have a `bad_partitions` subfolder.