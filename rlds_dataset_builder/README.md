# RoboCerebra RLDS Dataset Conversion

Convert RoboCerebra benchmark dataset to RLDS format for training and research.

**Note**: This project is adapted from [rlds_dataset_builder](https://github.com/moojink/rlds_dataset_builder) and modified specifically for RoboCerebra dataset conversion.

## 配置步骤

**重要**：使用前请先配置以下占位符路径：

1. **在 `regenerate_robocerebra_dataset.py` 中**：
   - `<LIBERO_ROOT_PATH>` → LIBERO安装目录的绝对路径

2. **在 `RoboCerebraDataset/RoboCerebraDataset_dataset_builder.py` 中**：
   - `<CONVERTED_HDF5_PATH>` → 转换后HDF5文件的存储路径

3. **在 `environment_macos.yml` 中**（仅macOS用户）：
   - `<CONDA_ENV_PATH>` → 您的conda环境安装路径

示例配置：
```python
# regenerate_robocerebra_dataset.py
LIBERO_ROOT = Path("/path/to/your/LIBERO")

# RoboCerebraDataset_dataset_builder.py  
'train': glob.glob('/path/to/converted_hdf5/robocerebra_ideal/all_hdf5/*.hdf5')
```

## Overview

The conversion process consists of two steps:
1. **HDF5 Conversion**: Use `regenerate_robocerebra_dataset.py` to convert raw RoboCerebra data to HDF5 format
2. **RLDS Conversion**: Use `RoboCerebraDataset` builder to convert HDF5 data to RLDS format

## Installation

Create a conda environment using the provided environment.yml file:
```bash
# For Ubuntu
conda env create -f environment_ubuntu.yml

# For macOS
conda env create -f environment_macos.yml
```

Activate the environment:
```bash
conda activate rlds_env
```

Alternatively, install packages manually:
```bash
pip install tensorflow tensorflow_datasets tensorflow_hub apache_beam matplotlib plotly wandb
```

## Step 1: Convert RoboCerebra to HDF5

### Prerequisites
- Raw RoboCerebra dataset with demo.hdf5 files
- LIBERO environment installed and configured
- Task description files (task_description*.txt)

### Run HDF5 Conversion

```bash
python regenerate_robocerebra_dataset.py \
  --dataset_name "robocerebra_dataset" \
  --robocerebra_raw_data_dir "/path/to/RoboCerebra_Bench/Ideal" \
  --robocerebra_target_dir "./converted_hdf5/robocerebra_ideal"
```

**Note**: Scene type is now automatically detected from BDDL files. The script will automatically determine whether to use `coffee_table` or `kitchen_table` based on the problem definition in each BDDL file.

### Parameters
- `--dataset_name`: Output dataset name
- `--robocerebra_raw_data_dir`: Path to raw RoboCerebra task directory (e.g., `<ROBOCEREBRA_BENCH_PATH>/Random_Disturbance`)
- `--robocerebra_target_dir`: Output directory for converted HDF5 files
- `--scene`: (Optional) Override auto-detected scene type.

### HDF5 Output Structure
```
converted_hdf5/
├── robocerebra_ideal/
│   ├── per_step/           # Step-wise organized data
│   └── all_hdf5/          # Flattened HDF5 files
├── robocerebra_random_disturbance/
│   ├── per_step/
│   └── all_hdf5/
└── ...
```

## Step 2: Convert HDF5 to RLDS

### Configure Dataset Builder

The `RoboCerebraDataset_dataset_builder.py` handles the RLDS conversion from the HDF5 files generated in Step 1.

### Run RLDS Conversion

```bash
cd RoboCerebraDataset
tfds build --overwrite
```

### Dataset Features

The converted RLDS dataset includes:
- **Images**: Agent view and wrist camera observations (256x256)
- **Actions**: 7-DOF robot actions (position, orientation, gripper)
- **Language**: Natural language task descriptions
- **Metadata**: Task type, success labels, episode information

### Output Location

Converted RLDS dataset will be saved to:
```
~/tensorflow_datasets/robo_cerebra_dataset/
├── 1.0.0/
│   ├── dataset_info.json
│   ├── features.json
│   └── train/
│       └── *.tfrecord
```

## Complete Conversion Pipeline

### Full Example
```bash
# 1. Convert raw RoboCerebra to HDF5 (scene auto-detected)
python regenerate_robocerebra_dataset.py \
  --dataset_name "robocerebra_complete" \
  --robocerebra_raw_data_dir "<ROBOCEREBRA_BENCH_PATH>/Random_Disturbance" \
  --robocerebra_target_dir "./converted_hdf5/robocerebra_random_disturbance"

# 2. Convert HDF5 to RLDS
cd RoboCerebraDataset
tfds build --overwrite
```

## Parallelizing RLDS Conversion

For large datasets, enable parallel processing:

1. **Install Package**:
```bash
pip install -e .
```

2. **Run with Parallel Processing**:
```bash
export CUDA_VISIBLE_DEVICES=
tfds build --overwrite --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=10"
```

## Dataset Specifications

### RoboCerebra Task Types
- **Ideal**: Baseline performance tests
- **Random_Disturbance**: Dynamic object disturbance
- **Mix**: Combined disturbances  
- **Observation_Mismatching**: Description mismatches
- **Memory_Execution**: Memory-based execution
- **Memory_Exploration**: Memory-based exploration

### Episode Structure
```python
episode = {
    'episode_metadata': {
        'task_type': tf.string,        # RoboCerebra task type
        'case_name': tf.string,        # Case identifier
        'episode_id': tf.int64,        # Episode number
    },
    'steps': [{
        'observation': {
            'image': tf.uint8[256, 256, 3],           # Agent view camera
            'wrist_image': tf.uint8[256, 256, 3],     # Wrist camera  
            'state': tf.float32[8],                   # Robot proprioception
        },
        'action': tf.float32[7],                      # Robot actions
        'reward': tf.float32,                         # Step reward
        'is_first': tf.bool,                          # First step flag
        'is_last': tf.bool,                           # Last step flag  
        'is_terminal': tf.bool,                       # Terminal step flag
        'language_instruction': tf.string,            # RoboCerebra task description
    }]
}
```

### Action Space (RoboCerebra Format)
- **Position**: [x, y, z] end-effector position
- **Orientation**: [rx, ry, rz] axis-angle rotation  
- **Gripper**: [grip] open/close command

### Image Observations
- **Resolution**: 256x256x3 RGB
- **Cameras**: Agent view + wrist-mounted
- **Format**: uint8 normalized images

## Verification

Verify conversion success by checking:
1. HDF5 files in Step 1 output directory
2. Dataset summary in RLDS build terminal output
3. TFRecord files in `~/tensorflow_datasets/robo_cerebra_dataset/`
4. Dataset info and feature specifications

## File Structure

```
rlds_dataset_builder/
├── LICENSE                           # Project license
├── README.md                         # This conversion guide
├── setup.py                          # Package setup
├── environment_ubuntu.yml            # Ubuntu conda environment
├── environment_macos.yml             # macOS conda environment
├── regenerate_robocerebra_dataset.py # Step 1: Raw to HDF5 converter
└── RoboCerebraDataset/               # Step 2: HDF5 to RLDS converter
    ├── CITATIONS.bib                 # Dataset citation
    ├── README.md                     # Dataset-specific readme
    ├── __init__.py                   # Package init
    ├── conversion_utils.py           # Conversion utilities
    └── RoboCerebraDataset_dataset_builder.py  # Main RLDS builder
```

## Usage Examples

### Convert Specific Task Type
```bash
# Step 1: Convert Random_Disturbance to HDF5 (scene auto-detected)
python regenerate_robocerebra_dataset.py \
  --dataset_name "robocerebra_random_disturbance" \
  --robocerebra_raw_data_dir "<ROBOCEREBRA_BENCH_PATH>/Random_Disturbance" \
  --robocerebra_target_dir "./converted_hdf5/random_disturbance"

# Step 2: Convert to RLDS
cd RoboCerebraDataset
tfds build --overwrite
```

### Batch Process Multiple Task Types
```bash
# Process all RoboCerebra task types
for task_type in Ideal Random_Disturbance Mix Observation_Mismatching Memory_Execution Memory_Exploration; do
  echo "Converting $task_type..."
  python regenerate_robocerebra_dataset.py \
    --dataset_name "robocerebra_${task_type,,}" \
    --robocerebra_raw_data_dir "<ROBOCEREBRA_BENCH_PATH>/$task_type" \
    --robocerebra_target_dir "./converted_hdf5/robocerebra_${task_type,,}" \
    --scene "coffee_table"
done

# Convert all HDF5 to RLDS
cd RoboCerebraDataset && tfds build --overwrite
```