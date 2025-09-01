#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robocerebra_logging.py

Logging and result saving utilities for RoboCerebra evaluation.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import imageio
import wandb

from config import GenerateConfig
from experiments.robot.robot_utils import DATE_TIME


logger = logging.getLogger(__name__)
BASE_DIR = Path.cwd() / "rollouts" / DATE_TIME
BASE_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    results_log_filepath = os.path.join(cfg.local_log_dir, run_id + "_results.json")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to {local_log_filepath}")
    logger.info(f"Results will be saved to {results_log_filepath}")
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
    return log_file, local_log_filepath, run_id, results_log_filepath


def log_message(msg: str, log_file=None):
    logger.info(msg)
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()


def save_results_log(results_log_filepath: str, cfg: GenerateConfig, results_by_task_type: Dict, 
                     total_eps: int, total_success: int, total_agent_subtasks: int, 
                     total_possible_subtasks: int, run_id: str, task_results: List[Dict] = None):
    """Save detailed evaluation results to a JSON file."""
    
    overall_success_rate = total_success / total_eps if total_eps > 0 else 0
    overall_subtask_rate = total_agent_subtasks / total_possible_subtasks if total_possible_subtasks > 0 else 0
    
    results_data = {
        "evaluation_info": {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model_family": cfg.model_family,
            "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
            "task_suite_name": cfg.task_suite_name,
            "seed": cfg.seed,
            "num_trials_per_task": cfg.num_trials_per_task,
            "task_types_evaluated": cfg.task_types,
        },
        "configuration": {
            "use_l1_regression": cfg.use_l1_regression,
            "use_diffusion": cfg.use_diffusion,
            "use_proprio": cfg.use_proprio,
            "center_crop": cfg.center_crop,
            "num_open_loop_steps": cfg.num_open_loop_steps,
            "switch_steps": cfg.switch_steps,
            "resume": cfg.resume,
            "dynamic": cfg.dynamic,
            "dynamic_shift_description": cfg.dynamic_shift_description,
            "complete_description": cfg.complete_description,
            "use_init_files": cfg.use_init_files,
            "intelligent_resume": True,  # Indicates that step-based intelligent resume is enabled
            "dynamic_shift_exclusion": True,  # Indicates that forced completions from dynamic shift are excluded from totals
        },
        "overall_results": {
            "total_episodes": total_eps,
            "total_successes": total_success,
            "overall_success_rate": overall_success_rate,
            "total_agent_subtasks": total_agent_subtasks,
            "total_possible_subtasks": total_possible_subtasks,
            "overall_subtask_rate": overall_subtask_rate,
        },
        "results_by_task_type": results_by_task_type,
        "detailed_task_results": task_results or []
    }
    
    try:
        with open(results_log_filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {results_log_filepath}")
    except Exception as e:
        logger.error(f"Failed to save results log: {e}")


def save_rollout_video(
    rollout_images,
    idx,
    success,
    task_description,
    log_file=None,
    task_suite: str = "",
    task_name: str = ""
):
    """Saves an MP4 replay of an episode with organized directory structure."""
    # Create hierarchical directory structure: rollouts/{DATE_TIME}/{task_suite}/{case_name}
    if task_suite and task_name:
        # Clean directory names by replacing spaces with underscores
        clean_task_suite = task_suite.replace(' ', '_')
        clean_task_name = task_name.replace(' ', '_')
        rollout_dir = BASE_DIR / clean_task_suite / clean_task_name
    elif task_name:
        clean_task_name = task_name.replace(' ', '_')
        rollout_dir = BASE_DIR / clean_task_name
    else:
        rollout_dir = BASE_DIR
    
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    
    # Include task suite in video filename
    suite_prefix = f"{task_suite.replace(' ', '_')}--" if task_suite else ""
    video_name = f"{DATE_TIME}--{suite_prefix}episode={idx}--success={int(success)}--task={processed_task_description}.mp4"
    mp4_path = rollout_dir / video_name
    
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return str(mp4_path)