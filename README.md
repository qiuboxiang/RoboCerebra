# RoboCerebra

[![arXiv](https://img.shields.io/badge/arXiv-2506.06677-red)](https://www.arxiv.org/pdf/2506.06677) [![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/qiukingballball/RoboCerebraBench)

Recent advances in vision-language models (VLMs) have enabled instructionconditioned robotic systems with improved generalization. However, most existing work focuses on reactive System 1 policies, underutilizing VLMs’ strengths
in semantic reasoning and long-horizon planning. These System 2 capabilities—characterized by deliberative, goal-directed thinking—remain underexplored
due to the limited temporal scale and structural complexity of current benchmarks.
To address this gap, we introduce RoboCerebra, a benchmark for evaluating highlevel reasoning in long-horizon robotic manipulation

## Overview

<p align="center">
<img src="https://github.com/qiuboxiang/RoboCerebra/blob/main/assets/overview.png?raw=true" alt="RoboCerebra Overview" width="100%">
</p>

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

# Download the dataset (specify dataset type and enable resume)
huggingface-cli download qiukingballball/RoboCerebraBench --repo-type dataset --local-dir ./RoboCerebra_Bench --resume-download
```

### Option 1: Benchmark-Only Usage (LIBERO)

For running benchmarks using the LIBERO environment:

```bash
# Create and activate conda environment
conda create -n libero python=3.8.13
conda activate libero

# Clone and install LIBERO from RoboCerebra
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

# Install LIBERO from RoboCerebra repository
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
pip install "numpy>=1.23.5,<2.0.0"
pip install "peft>=0.17.0"
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

# Step 2: Convert to RLDS (disable CUDA to avoid initialization errors)
cd RoboCerebraDataset && CUDA_VISIBLE_DEVICES="" tfds build --overwrite
```

## Directory Structure

```
RoboCerebra/
├── README.md                          # This overview guide
├── LIBERO/
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

## Citation

If you use RoboCerebra in your research, please cite:
```bibtex
@article{han2025robocerebra,
  title={RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation},
  author={Han, Songhao and Qiu, Boxiang and Liao, Yue and Huang, Siyuan and Gao, Chen and Yan, Shuicheng and Liu, Si},
  journal={arXiv preprint arXiv:2506.06677},
  year={2025}
}
```