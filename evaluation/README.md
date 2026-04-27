# RoboCerebra Evaluation

`evaluation/` contains the OpenVLA-based evaluation entrypoint for the
RoboCerebra benchmark.

## Prerequisites

Before running the evaluation code, make sure:

- OpenVLA-OFT and its dependencies are installed
- `LIBERO/` from this repository is installed with `pip install -e LIBERO`
- the RoboCerebra benchmark data has been downloaded locally

The repository-level setup is documented in
[../README.md](/Users/qiuboxiang/RoboCerebra/README.md).

## Configuration

The default configuration is defined in
[config.py](/Users/qiuboxiang/RoboCerebra/evaluation/config.py).

Recommended environment variables:

```bash
export ROBOCEREBRA_PRETRAINED_CHECKPOINT=/path/to/openvla/checkpoint
export ROBOCEREBRA_BENCH_ROOT=/path/to/RoboCerebra_Bench
```

Optional variables:

```bash
export ROBOCEREBRA_INIT_FILES_ROOT=/path/to/RoboCerebra_Bench/init_files
export WANDB_ENTITY=your_wandb_entity
export WANDB_PROJECT=your_wandb_project
```

You can override any of these at runtime with CLI flags such as
`--pretrained_checkpoint` and `--robocerebra_root`.

## Quick Start

Run from this directory:

```bash
cd /path/to/RoboCerebra/evaluation
python eval_openvla.py \
  --task_types '["Ideal", "Random_Disturbance"]' \
  --num_trials_per_task 1
```

Single-task examples:

```bash
python eval_openvla.py --task_types '["Random_Disturbance"]'
python eval_openvla.py --task_types '["Mix"]' --num_trials_per_task 3
```

Explicit path overrides:

```bash
python eval_openvla.py \
  --pretrained_checkpoint /path/to/openvla/checkpoint \
  --robocerebra_root /path/to/RoboCerebra_Bench
```

## Important Arguments

- `--task_types`: task categories to evaluate
- `--robocerebra_root`: root directory of the downloaded benchmark
- `--init_files_root`: directory containing the benchmark init files
- `--num_trials_per_task`: number of rollouts per task instance
- `--use_wandb`: enable Weights & Biases logging
- `--local_log_dir`: local directory for text logs and JSON results

## Outputs

If you run from `evaluation/`, outputs are written to:

- `experiments/logs/`: run logs and JSON summaries
- `rollouts/`: rollout videos and per-episode metadata
