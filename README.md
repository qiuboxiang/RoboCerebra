# RoboCerebra

A comprehensive robot learning benchmark featuring dynamic environment disturbances and memory-based task evaluation.

## Overview

RoboCerebra provides two main components:

1. **Evaluation Suite** (`evaluation/`) - Model evaluation on RoboCerebra benchmark tasks
2. **Dataset Builder** (`rlds_dataset_builder/`) - Convert RoboCerebra data to RLDS format for training

## Installation

### Initial Setup

First, clone the RoboCerebra repository:

```bash
git clone https://github.com/qiuboxiang/RoboCerebra/tree/main
cd RoboCerebra
```

### Dataset Download

Download the RoboCerebra benchmark dataset from Hugging Face:

```bash
# Install Hugging Face Hub if not already installed
pip install huggingface_hub

# Download the dataset
huggingface-cli download qiukingballball/RoboCerebraBench --local-dir ./RoboCerebra_Bench
```

### Option 1: Benchmark-Only Usage (LIBERO)

For running benchmarks using the LIBERO environment:

```bash
# Create and activate conda environment
conda create -n libero python=3.8.13
conda activate libero

# Clone and install LIBERO from RoboCerebra_Release
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install the libero package
pip install -e .
```

### Option 2: OpenVLA Evaluation

For evaluation using OpenVLA:

```bash
# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation

# Install LIBERO from RoboCerebra_Release
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # From openvla-oft base dir
```

## Configuration

**Important**: Configure the following placeholder paths before use:

1. **Edit `evaluation/config.py`**:
   - `<PRETRAINED_CHECKPOINT_PATH>` → Your pretrained model checkpoint path
   - `<ROBOCEREBRA_BENCH_PATH>` → RoboCerebra benchmark dataset path
   - `<WANDB_ENTITY>` → Your WandB entity name (if using WandB)
   - `<WANDB_PROJECT>` → Your WandB project name (if using WandB)

2. **Edit `rlds_dataset_builder/environment_macos.yml`** (macOS users only):
   - `<CONDA_ENV_PATH>` → Your conda environment path

3. **Edit `rlds_dataset_builder/regenerate_robocerebra_dataset.py`**:
   - `<LIBERO_ROOT_PATH>` → LIBERO installation directory path

4. **Edit `rlds_dataset_builder/RoboCerebraDataset/RoboCerebraDataset_dataset_builder.py`**:
   - `<CONVERTED_HDF5_PATH>` → Converted HDF5 files path

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

### Additional Dependencies


#### For RLDS Conversion
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