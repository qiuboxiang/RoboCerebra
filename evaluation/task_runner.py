#!/usr/bin/env python3
"""Task-level execution logic for RoboCerebra evaluation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import libero.libero.envs.bddl_utils as BDDLUtils
from config import GenerateConfig
from libero.libero.envs import TASK_MAPPING
from robocerebra_logging import log_message
from robosuite import load_controller_config
from utils import load_actions_with_steps, parse_task_description


def setup_task_environment(
    task_dir: Path,
    log_file=None,
) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """Create the LIBERO environment for a single task directory."""
    bddl_files = list(task_dir.glob("*.bddl"))
    if not bddl_files:
        log_message(
            f"[WARN] .bddl file not found in {task_dir}, skipping.",
            log_file,
        )
        return None, None, "No BDDL file found in task directory"

    bddl_file_path = str(bddl_files[0])

    try:
        problem_info = BDDLUtils.get_problem_info(bddl_file_path)
        problem_name = problem_info["problem_name"]
        controller_config = load_controller_config(
            default_controller="OSC_POSE")
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
    except Exception as exc:
        error_msg = f"Failed to initialize environment: {exc}"
        log_message(f"[ERROR] {error_msg}", log_file)
        return None, bddl_file_path, error_msg


def load_task_data(
    task_dir: Path,
    log_file=None,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[str]]:
    """Load demonstration states and goal metadata for a task."""
    h5_path = task_dir / "demo.hdf5"
    if not h5_path.exists():
        log_message(
            f"[WARN] demo.hdf5 not found in {task_dir}, skipping.",
            log_file,
        )
        return None, None, None, "No demo.hdf5 file found in task directory"

    try:
        with h5py.File(h5_path, "r") as h5_file:
            demo = h5_file["data"]["demo_1"]
            orig_states = demo["states"][()]
    except Exception as exc:
        error_msg = f"Failed to load demo.hdf5: {exc}"
        log_message(f"[ERROR] {error_msg}", log_file)
        return None, None, None, error_msg

    goal_json = task_dir / "goal.json"
    if goal_json.is_file():
        try:
            goal, goal_steps = load_actions_with_steps(str(goal_json))
        except Exception as exc:
            error_msg = f"Failed to load goal.json: {exc}"
            log_message(f"[ERROR] {error_msg}", log_file)
            return orig_states, None, None, error_msg
    else:
        goal, goal_steps = None, None

    return orig_states, goal, goal_steps, None


def setup_task_descriptions(
    cfg: GenerateConfig,
    task_dir: Path,
    log_file=None,
) -> Tuple[List[str], List[str], List[int], str, Optional[str]]:
    """Load task descriptions for video naming and model input."""
    if cfg.task_description_suffix:
        naming_desc_path = (
            task_dir / f"task_description{cfg.task_description_suffix}.txt")
    else:
        naming_desc_path = task_dir / "task_description.txt"

    if not naming_desc_path.is_file():
        error_msg = f"Task description file not found: {naming_desc_path}"
        log_message(f"[ERROR] {error_msg}", log_file)
        return [], [], [], "", error_msg

    naming_step_desc, suff_start_idx = parse_task_description(
        str(naming_desc_path))

    canonical_desc_path = task_dir / "task_description.txt"
    if not canonical_desc_path.is_file():
        error_msg = f"Task description file not found: {canonical_desc_path}"
        log_message(f"[ERROR] {error_msg}", log_file)
        return [], [], [], "", error_msg

    model_step_desc, canon_start_idx = parse_task_description(
        str(canonical_desc_path))

    task_line = ""
    try:
        with canonical_desc_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                stripped = line.strip()
                if stripped.startswith("Task:"):
                    task_line = stripped.split(":", 1)[1].strip()
                    break
    except Exception as exc:
        log_message(f"[WARN] Failed to read task line: {exc}", log_file)

    if cfg.resume and cfg.task_description_suffix:
        start_indices = suff_start_idx
    else:
        start_indices = canon_start_idx

    if len(naming_step_desc) != len(start_indices):
        min_len = min(len(naming_step_desc), len(start_indices))
        log_message(
            "[WARN] Number of video naming descriptions "
            f"({len(naming_step_desc)}) and indices "
            f"({len(start_indices)}) are inconsistent, truncating to "
            f"{min_len}",
            log_file,
        )
        naming_step_desc = naming_step_desc[:min_len]
        start_indices = start_indices[:min_len]

    return naming_step_desc, model_step_desc, start_indices, task_line, None


def validate_task_configuration(
    cfg: GenerateConfig,
    naming_step_desc: List[str],
    start_indices: List[int],
    model_step_desc: List[str],
    goal: Optional[Dict[str, Any]],
    goal_steps: Optional[Any],
    bddl_file_path: str,
    task_type: str,
    case_name: str,
    log_file=None,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate task metadata before running rollout episodes."""
    base_result: Dict[str, Any] = {
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

    if cfg.task_description_suffix and not naming_step_desc:
        total_goals = sum(len(value) for value in goal.values()) if goal else 0
        log_message(
            "[WARN] The suffixed task description file is empty; "
            "all trials will be considered as failures.",
            log_file,
        )
        base_result.update({
            "possible_subtasks": cfg.num_trials_per_task * total_goals,
            "error": "Suffixed task description file is empty",
        })
        return False, base_result

    if cfg.dynamic_shift_description and len(start_indices) == 1:
        log_message(
            "[WARN] dynamic_shift_description=True and only one start "
            "index is available, treating all episodes as failures.",
            log_file,
        )
        base_result.update({
            "possible_subtasks":
            (cfg.num_trials_per_task * len(model_step_desc)),
            "error": ("Only one start index available with "
                      "dynamic_shift_description enabled"),
        })
        return False, base_result

    if not start_indices:
        log_message(
            f"[ERROR] No start indices found for {task_type}/{case_name}, "
            "treating as failure.",
            log_file,
        )
        total_goals = sum(len(value) for value in goal.values()) if goal else 0
        base_result.update({
            "possible_subtasks":
            cfg.num_trials_per_task * total_goals,
            "error": ("No start indices found in task description file"),
        })
        return False, base_result

    return True, base_result
