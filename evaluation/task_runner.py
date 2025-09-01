#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robocerebra_task_runner.py

Task-level execution logic for RoboCerebra evaluation.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import tqdm
from robosuite import load_controller_config
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *  # noqa: F403

from config import GenerateConfig
from robocerebra_logging import log_message
from utils import load_actions_with_steps, parse_task_description, load_init_state
from resume import create_step_based_resume_handler


logger = logging.getLogger(__name__)


def setup_task_environment(task_dir: Path, log_file=None) -> Tuple[any, str, str]:
    """Setup environment for a single task."""
    # Find BDDL file
    bddl_files = list(task_dir.glob("*.bddl"))
    if not bddl_files:
        log_message(f"[WARN] .bddl file not found in {task_dir}, skipping.", log_file)
        return None, None, "No BDDL file found in task directory"
    
    bddl_file_path = str(bddl_files[0])

    try:
        # Environment initialization
        problem_info = BDDLUtils.get_problem_info(bddl_file_path)
        problem_name = problem_info["problem_name"]
        controller_config = load_controller_config(default_controller="OSC_POSE")
        env = TASK_MAPPING[problem_name](
            bddl_file_name=bddl_file_path,
            robots=["Panda"],
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            ignore_done=True,
            use_camera_obs=True,
            reward_shaping=True,
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
        )
        return env, bddl_file_path, None
    except Exception as e:
        error_msg = f"Failed to initialize environment: {e}"
        log_message(f"[ERROR] {error_msg}", log_file)
        return None, bddl_file_path, error_msg


def load_task_data(task_dir: Path, log_file=None) -> Tuple[any, any, any, str]:
    """Load demonstration data and goal information for a task."""
    dir_path = str(task_dir)
    
    # Read demo.hdf5
    h5_path = os.path.join(dir_path, "demo.hdf5")
    if not os.path.exists(h5_path):
        log_message(f"[WARN] demo.hdf5 not found in {task_dir}, skipping.", log_file)
        return None, None, None, "No demo.hdf5 file found in task directory"
    
    try:
        with h5py.File(h5_path, "r") as h5f:
            demo = h5f["data"]["demo_1"]
            orig_states = demo["states"][()]
    except Exception as e:
        error_msg = f"Failed to load demo.hdf5: {e}"
        log_message(f"[ERROR] {error_msg}", log_file)
        return None, None, None, error_msg

    # Load goal with step information
    goal_json = os.path.join(dir_path, "goal.json")
    if os.path.isfile(goal_json):
        try:
            goal, goal_steps = load_actions_with_steps(goal_json)
        except Exception as e:
            error_msg = f"Failed to load goal.json: {e}"
            log_message(f"[ERROR] {error_msg}", log_file)
            return orig_states, None, None, error_msg
    else:
        goal, goal_steps = None, None

    return orig_states, goal, goal_steps, None


def setup_task_descriptions(cfg: GenerateConfig, task_dir: Path, log_file=None) -> Tuple[List[str], List[str], List[int], str, str]:
    """Setup task descriptions for naming and model input."""
    dir_path = str(task_dir)
    
    # 1) First, read the description with suffix (used for video naming)
    if cfg.task_description_suffix:
        suff_txt = os.path.join(dir_path, f"task_description{cfg.task_description_suffix}.txt")
        if not os.path.isfile(suff_txt):
            error_msg = f"Task description file not found: {suff_txt}"
            log_message(f"[ERROR] {error_msg}", log_file)
            return [], [], [], "", error_msg
        naming_step_desc, suff_start_idx = parse_task_description(suff_txt)
    else:
        canon_txt = os.path.join(dir_path, "task_description.txt")
        if not os.path.isfile(canon_txt):
            error_msg = f"Task description file not found: {canon_txt}"
            log_message(f"[ERROR] {error_msg}", log_file)
            return [], [], [], "", error_msg
        naming_step_desc, suff_start_idx = parse_task_description(canon_txt)

    # 2) Then, read the canonical description (for model input)
    canon_txt = os.path.join(dir_path, "task_description.txt")
    if not os.path.isfile(canon_txt):
        error_msg = f"Task description file not found: {canon_txt}"
        log_message(f"[ERROR] {error_msg}", log_file)
        return [], [], [], "", error_msg
    model_step_desc, canon_start_idx = parse_task_description(canon_txt)

    # Extract task line
    task_line = ""
    try:
        with open(canon_txt, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln.startswith("Task:"):
                    task_line = ln.split(":", 1)[1].strip()
                    break
    except Exception as e:
        log_message(f"[WARN] Failed to read task line: {e}", log_file)

    # Choose which set of start_indices to use
    if cfg.resume and cfg.task_description_suffix:
        start_indices = suff_start_idx
    else:
        start_indices = canon_start_idx

    # Validate consistency
    if len(naming_step_desc) != len(start_indices):
        min_len = min(len(naming_step_desc), len(start_indices))
        log_message(
            f"[WARN] Number of video naming descriptions ({len(naming_step_desc)}) and indices ({len(start_indices)}) are inconsistent, truncating to {min_len}",
            log_file
        )
        naming_step_desc = naming_step_desc[:min_len]
        start_indices = start_indices[:min_len]

    return naming_step_desc, model_step_desc, start_indices, task_line, None


def validate_task_configuration(cfg: GenerateConfig, naming_step_desc: List[str], start_indices: List[int], 
                               model_step_desc: List[str], goal: any, goal_steps: any, 
                               bddl_file_path: str, task_type: str, case_name: str, 
                               log_file=None) -> Tuple[bool, Dict]:
    """Validate task configuration and return error result if invalid."""
    
    base_result = {
        "task_type": task_type,
        "case_name": case_name,
        "episodes": cfg.num_trials_per_task,
        "successes": 0,
        "success_rate": 0,
        "agent_subtasks": 0,
        "possible_subtasks": 0,
        "subtask_rate": 0,
        "bddl_file": bddl_file_path,
        "used_init_files": cfg.use_init_files,
        "has_step_annotations": bool(goal_steps),
    }
    
    # Check for empty suffixed description
    if cfg.task_description_suffix and len(naming_step_desc) == 0:
        total_goals = sum(len(v) for v in goal.values()) if goal else 0
        log_message(f"[WARN] The suffixed task description file is empty; all trials will be considered as failures", log_file)
        base_result.update({
            "possible_subtasks": cfg.num_trials_per_task * total_goals,
            "error": "Suffixed task description file is empty"
        })
        return False, base_result

    # Check dynamic shift description compatibility
    if cfg.dynamic_shift_description and len(start_indices) == 1:
        log_message(
            f"[WARN] dynamic_shift_description=True and only one start index available, treating all episodes as failures",
            log_file
        )
        base_result.update({
            "possible_subtasks": cfg.num_trials_per_task * len(model_step_desc),
            "error": "Only one start index available with dynamic_shift_description enabled"
        })
        return False, base_result

    # Check if start_indices is empty
    if not start_indices:
        log_message(f"[ERROR] No start indices found for {task_type}/{case_name}, treating as failure", log_file)
        total_goals = sum(len(v) for v in goal.values()) if goal else 0
        base_result.update({
            "possible_subtasks": cfg.num_trials_per_task * total_goals,
            "error": "No start indices found in task description file"
        })
        return False, base_result

    return True, base_result