# RoboCerebra

A comprehensive robot learning benchmark featuring dynamic environment disturbances and memory-based task evaluation.

## Overview

RoboCerebra provides two main components:

1. **Evaluation Suite** (`evaluation/`) - Model evaluation on RoboCerebra benchmark tasks
2. **Dataset Builder** (`rlds_dataset_builder/`) - Convert RoboCerebra data to RLDS format for training

## 配置步骤

**重要**：使用前请先配置以下占位符路径：

1. **编辑 `evaluation/config.py`**：
   - `<PRETRAINED_CHECKPOINT_PATH>` → 您的预训练模型检查点路径
   - `<ROBOCEREBRA_BENCH_PATH>` → RoboCerebra基准数据集路径
   - `<WANDB_ENTITY>` → 您的WandB实体名称（如使用WandB）
   - `<WANDB_PROJECT>` → 您的WandB项目名称（如使用WandB）

2. **编辑 `rlds_dataset_builder/environment_macos.yml`**（仅macOS用户）：
   - `<CONDA_ENV_PATH>` → 您的conda环境路径

3. **编辑 `rlds_dataset_builder/regenerate_robocerebra_dataset.py`**：
   - `<LIBERO_ROOT_PATH>` → LIBERO安装目录路径

4. **编辑 `rlds_dataset_builder/RoboCerebraDataset/RoboCerebraDataset_dataset_builder.py`**：
   - `<CONVERTED_HDF5_PATH>` → 转换后的HDF5文件路径

## Quick Start

### Model Evaluation

Evaluate OpenVLA-OFT on RoboCerebra benchmark:

```bash
cd evaluation/
python eval_openvla.py --task_types ["Ideal", "Random_Disturbance"]
```

### Dataset Conversion

Convert RoboCerebra data to RLDS format for training:

```bash
cd rlds_dataset_builder/

# Step 1: Convert to HDF5
python regenerate_robocerebra_dataset.py \
  --robocerebra_raw_data_dir "/path/to/RoboCerebra_Bench/Ideal" \
  --robocerebra_target_dir "./converted_hdf5/robocerebra_ideal"

# Step 2: Convert to RLDS
cd RoboCerebraDataset && tfds build --overwrite
```

## Directory Structure

```
RoboCerebra/
├── README.md                          # This overview guide
├── evaluation/                        # Model evaluation suite
│   ├── README.md                      # Evaluation documentation
│   ├── eval_openvla.py               # Main evaluation script
│   ├── config.py                     # Configuration management
│   ├── robocerebra_logging.py        # Logging and results
│   ├── task_runner.py                # Task-level execution
│   ├── episode.py                    # Episode-level execution
│   ├── resume.py                     # Resume mechanism
│   └── utils.py                      # Utility functions
└── rlds_dataset_builder/             # Dataset conversion tools
    ├── README.md                     # Conversion documentation
    ├── regenerate_robocerebra_dataset.py  # HDF5 conversion
    └── RoboCerebraDataset/           # RLDS builder
        └── RoboCerebraDataset_dataset_builder.py
```

## RoboCerebra Task Types

### Core Task Categories

| Task Type | Description | Dynamic Features |
|-----------|-------------|------------------|
| **Ideal** | Baseline performance under optimal conditions | None |
| **Random_Disturbance** | Random object movements during execution | Dynamic object movement |
| **Mix** | Combined disturbances and mismatches | All features enabled |
| **Observation_Mismatching** | Task description doesn't match current scene | Description shifts |
| **Memory_Execution** | Multi-step tasks requiring memory | Resume mechanisms |
| **Memory_Exploration** | Exploration-based memory tasks | Resume mechanisms |

### Dynamic Features

- **Dynamic Object Movement**: Objects randomly move during task execution
- **Observation Mismatching**: Task descriptions change mid-execution
- **Resume Mechanisms**: Models can recover from intermediate states
- **Progress Tracking**: Real-time completion monitoring

## Dependencies

### Core Requirements
```bash
pip install draccus numpy tqdm robosuite h5py imageio wandb
```

### LIBERO Environment
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e .
```

### For RLDS Conversion
```bash
pip install tensorflow tensorflow_datasets apache_beam
```

## Usage Examples

### Evaluate on All Task Types
```bash
cd evaluation/
python eval_openvla.py \
  --task_types ["Ideal", "Random_Disturbance", "Mix", "Observation_Mismatching", "Memory_Execution", "Memory_Exploration"] \
  --num_trials_per_task 5
```

### Convert Complete Dataset
```bash
cd rlds_dataset_builder/

# Convert all task types
for task_type in Ideal Random_Disturbance Mix Observation_Mismatching Memory_Execution Memory_Exploration; do
  python regenerate_robocerebra_dataset.py \
    --dataset_name "robocerebra_${task_type,,}" \
    --robocerebra_raw_data_dir "<ROBOCEREBRA_BENCH_PATH>/$task_type" \
    --robocerebra_target_dir "./converted_hdf5/robocerebra_${task_type,,}"
done

# Convert to RLDS
cd RoboCerebraDataset && tfds build --overwrite
```

## Data Paths

Default paths (can be configured):
- **RoboCerebra Benchmark**: `<ROBOCEREBRA_BENCH_PATH>/`
- **Initial States**: `<ROBOCEREBRA_BENCH_PATH>/init_files/`
- **Evaluation Logs**: `./evaluation/experiments/logs/`
- **Videos**: `./evaluation/rollouts/`
- **RLDS Output**: `~/tensorflow_datasets/robo_cerebra_dataset/`

## Key Features

### Evaluation Suite
- **Multi-task Benchmark**: 6 distinct task categories
- **Dynamic Testing**: Real-time environment changes
- **Memory Assessment**: Multi-step task execution
- **Video Recording**: Complete episode playbacks
- **Progress Tracking**: Real-time completion monitoring

### Dataset Builder
- **Automatic Scene Detection**: Detects scene type from BDDL files
- **Parallel Processing**: Multi-threaded conversion
- **RLDS Compatibility**: Standard format for robot learning
- **Metadata Preservation**: Task types, success labels, episode info

## Getting Help

For detailed usage instructions:
- **Evaluation**: See `evaluation/README.md`
- **Dataset Conversion**: See `rlds_dataset_builder/README.md`

## Citation

If you use RoboCerebra in your research, please cite:
```bibtex
@misc{robocerebra2024,
  title={RoboCerebra: A Comprehensive Benchmark for Robot Learning with Dynamic Environments},
  author={Your Author},
  year={2024}
}
```