# RoboCerebra OpenVLA Evaluation Suite

Evaluation tool for OpenVLA-OFT model on RoboCerebra benchmark, designed with modular architecture.

## Setup

### Dataset Download

First, download the RoboCerebra benchmark dataset from Hugging Face:

```bash
# Install Hugging Face Hub if not already installed
pip install huggingface_hub

# Download the dataset
huggingface-cli download qiukingballball/RoboCerebraBench --local-dir ./RoboCerebra_Bench
```

## Configuration

**Important**: Before running evaluation, configure the following placeholders in `config.py`:

- `<PRETRAINED_CHECKPOINT_PATH>` → Your OpenVLA-OFT model checkpoint path
- `<ROBOCEREBRA_BENCH_PATH>` → RoboCerebra benchmark dataset root directory (e.g., `./RoboCerebra_Bench`)

Example:
```python
pretrained_checkpoint = "/path/to/your/openvla-checkpoint"
robocerebra_root = "/path/to/RoboCerebra_Bench"
```

## File Structure

```
evaluation/
├── eval_openvla.py           # OpenVLA evaluation main entry (396 lines)
├── config.py                 # Configuration management (87 lines)
├── utils.py                  # Data processing tools (422 lines)
├── robocerebra_logging.py    # Logging and results saving (141 lines)
├── task_runner.py            # Task-level management (200 lines)
├── episode.py                # Episode execution logic (280 lines)
├── resume.py                 # Resume mechanism (69 lines)
└── README.md                 # Usage instructions
```

## Quick Start

### Basic Usage

```bash
# Navigate to evaluation directory
cd <ROBOCEREBRA_PATH>/evaluation

# Run with default configuration (evaluate all task types)
python eval_openvla.py

# Specify GPU and task types
CUDA_VISIBLE_DEVICES=0 python eval_openvla.py --task_types ["Ideal", "Random_Disturbance"]
```

### Common Configurations

```bash
# Random_Disturbance task evaluation
python eval_openvla.py \
  --task_types ["Random_Disturbance"] \
  --num_trials_per_task 5

# Mix task evaluation 
python eval_openvla.py \
  --task_types ["Mix"] \
  --num_trials_per_task 3

# Specify OpenVLA-OFT model checkpoint
python eval_openvla.py \
  --pretrained_checkpoint "/path/to/openvla-oft/checkpoint" \
  --task_types ["Ideal"]

# Use specific task description suffix
python eval_openvla.py \
  --task_types ["Random_Disturbance"] \
```

## Configuration Parameters

### Task Configuration
- `--task_types`: List of task types to evaluate
- `--robocerebra_root`: RoboCerebra dataset root directory (default: `<ROBOCEREBRA_BENCH_PATH>`)
- `--init_files_root`: Initial state files directory (default: `<ROBOCEREBRA_BENCH_PATH>/init_files`)
- `--num_trials_per_task`: Number of trials per task (default: 5)

### OpenVLA-OFT Model Configuration
- `--pretrained_checkpoint`: OpenVLA-OFT model checkpoint path
- `--model_family`: Model type (fixed: "openvla")
- `--use_l1_regression`: Use L1 regression head (default: True)
- `--use_proprio`: Whether to use proprioceptive information (default: True)
- `--center_crop`: Image center cropping (default: True)
- `--num_open_loop_steps`: Open-loop execution steps (default: 8)

### Dynamic Features Configuration
- `--dynamic`: Enable dynamic disturbance object movement
- `--dynamic_shift_description`: Enable observation mismatching mode
- `--resume`: Enable task resume mechanism

### Logging Configuration
- `--local_log_dir`: Local log directory (default: "./experiments/logs")
- `--use_wandb`: Enable Weights & Biases logging (default: False)
- `--wandb_entity`: WandB entity name
- `--wandb_project`: WandB project name
- `--run_id_note`: Run ID notes

## Output Files

### Log Files
- `./experiments/logs/EVAL-robocerebra-openvla-{timestamp}.txt`: Detailed execution logs
- `./experiments/logs/EVAL-robocerebra-openvla-{timestamp}_results.json`: JSON format results

### Video Files
```
rollouts/{timestamp}/{task_type}/{case_name}/
├── {timestamp}--{task_type}--episode={idx}--success={0|1}--task={description}.mp4
└── {timestamp}--{task_type}--episode={idx}--success={0|1}--task={description}.json
```

## Module Architecture

### Core Modules
- **eval_openvla.py**: OpenVLA-OFT model evaluation main logic
- **config.py**: Evaluation parameter configuration and validation
- **utils.py**: Data loading and environment processing tools

### Functional Modules  
- **robocerebra_logging.py**: Logging setup and results saving
- **task_runner.py**: Task-level management and validation
- **episode.py**: Single episode execution and dynamic disturbance features
- **resume.py**: Intelligent resume mechanism handling