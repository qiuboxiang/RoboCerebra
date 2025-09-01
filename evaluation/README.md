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
- `<WANDB_ENTITY>` → Your WandB entity name (if using WandB logging)
- `<WANDB_PROJECT>` → Your WandB project name (if using WandB logging)

Example:
```python
pretrained_checkpoint = "/path/to/your/openvla-checkpoint"
robocerebra_root = "/path/to/RoboCerebra_Bench"
wandb_entity = "your-wandb-username"
wandb_project = "robocerebra-evaluation"
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
# Random_Disturbance task evaluation (automatically enables dynamic disturbance)
python eval_openvla.py \
  --task_types ["Random_Disturbance"] \
  --num_trials_per_task 5

# Mix task evaluation (enables all dynamic features)
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
  --task_description_suffix "_cosmos_matched"
```

## Configuration Parameters

### Task Configuration
- `--task_types`: List of task types to evaluate
  - `"Ideal"`: Ideal condition testing
  - `"Random_Disturbance"`: Random disturbance testing (automatically enables dynamic)
  - `"Mix"`: Mixed mode (automatically enables all dynamic features)
  - `"Observation_Mismatching"`: Observation mismatching (automatically enables dynamic_shift_description)
  - `"Memory_Execution"`: Memory execution testing
  - `"Memory_Exploration"`: Memory exploration testing

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
- `--task_description_suffix`: Task description suffix (e.g., "_cosmos_matched")

### Logging Configuration
- `--local_log_dir`: Local log directory (default: "./experiments/logs")
- `--use_wandb`: Enable Weights & Biases logging (default: False)
- `--wandb_entity`: WandB entity name
- `--wandb_project`: WandB project name
- `--run_id_note`: Run ID notes

## Automatic Configuration

Each task type automatically configures corresponding OpenVLA-OFT evaluation parameters:

| Task Type | dynamic | dynamic_shift_description | resume | Description |
|---------|---------|---------------------------|--------|------|
| Ideal | False | False | True | OpenVLA performance under ideal conditions |
| Random_Disturbance | **True** | False | True | Robustness testing under random disturbance |
| Mix | **True** | **True** | True | Comprehensive testing under mixed conditions |
| Observation_Mismatching | False | **True** | True | Adaptability testing for observation mismatching |
| Memory_Execution | False | False | True | Memory execution capability testing |
| Memory_Exploration | False | False | True | Memory exploration capability testing |

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

## Dependencies

### Environment Configuration
Ensure the following paths are accessible:
- OpenVLA-OFT code path: Must be in system path
- RoboCerebra dataset: `<ROBOCEREBRA_BENCH_PATH>`
- Initialization files: `<ROBOCEREBRA_BENCH_PATH>/init_files`

### Python Dependencies
```bash
pip install draccus numpy tqdm robosuite h5py imageio wandb
pip install "numpy>=1.23.5,<2.0.0"
pip install "peft>=0.17.0"
```

### LIBERO Environment
```bash
# Install LIBERO from RoboCerebra repository
pip install -e LIBERO
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

## Evaluation Process

1. **Model Initialization**: Load OpenVLA-OFT pretrained model
2. **Task Discovery**: Scan RoboCerebra dataset directory
3. **Environment Setup**: Create LIBERO simulation environment for each task
4. **Episode Execution**: Execute specified number of evaluation episodes
5. **Results Statistics**: Generate success rate and subtask completion rate statistics
6. **Video Saving**: Save execution videos for each episode

## Advanced Features

### Dynamic Disturbance Object Movement
Automatically enabled for Random_Disturbance and Mix tasks, moves objects in the scene during episode execution:
```
[Dynamic] Moved related object at step 1, Δy=+0.15
[Dynamic] Moved unrelated object at step 2, Δy=-0.15
```

### Intelligent Resume Mechanism
Step-based resume functionality that can continue execution from intermediate task states:
```
[Resume] Reverting to Step 2, auto-completed 1 prior subtasks: ['object_1_0']
```

### Observation Mismatching
Observation_Mismatching and Mix tasks test the model's adaptability when observation descriptions don't match.

## Performance Features

- **Efficient Modularization**: 7 specialized modules, 75% code reduction
- **Full Compatibility**: 100% functional consistency with original OpenVLA evaluation
- **Automatic Configuration**: Task types automatically configure OpenVLA evaluation parameters
- **Intelligent Resume**: Step-based intelligent resume mechanism
- **Dynamic Testing**: Supports dynamic object movement and observation mismatching
- **Detailed Logging**: Complete evaluation process recording and video saving

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure all module files are in the evaluation directory
2. **Path errors**: Check dataset and initialization file paths
3. **OpenVLA model loading failure**: Verify checkpoint path and format
4. **Dynamic features not working**: Check task_description.json files

### Check Configuration
```bash
# View complete configuration options
python eval_openvla.py --help

# Test single task
python eval_openvla.py --task_types ["Ideal"] --num_trials_per_task 1
```

### Debug Mode
Uncomment debug information in each module to get detailed log output.

## Contact Information

If you encounter issues, please check:
1. Whether OpenVLA-OFT model is correctly loaded
2. Whether RoboCerebra dataset path is correct
3. Whether LIBERO environment is correctly installed
4. Whether all Python dependencies are satisfied

This evaluation suite is specifically designed for OpenVLA-OFT model to ensure accurate evaluation on RoboCerebra benchmark.