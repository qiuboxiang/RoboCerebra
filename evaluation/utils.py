#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robocerebra_utils.py

Utility functions for RoboCerebra evaluation including data loading,
environment handling, and task processing.
"""

import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
from robosuite import load_controller_config
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *  # noqa: F403

from config import GenerateConfig, SCENE_MAPPINGS, MOVABLE_OBJECT_LIST
from experiments.robot.libero.libero_utils import (
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import resize_image_for_policy
from experiments.robot.robot_utils import (
    invert_gripper_action,
    normalize_gripper_action,
)


logger = logging.getLogger(__name__)


def load_actions(json_path: str) -> Dict[str, List[List[str]]]:
    """Load actions from goal.json, supporting both old and new formats with task_step annotations."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result: Dict[str, List[List[str]]] = {}
    for obj_id, relations in data.items():
        processed = []
        for item in relations:
            # Check if this is the new format with task_step
            if isinstance(item, dict) and 'state_pair' in item and 'task_step' in item:
                # New format: {"state_pair": [action_type, obj, target], "task_step": step_number}
                triple = item['state_pair']
                if len(triple) == 2:
                    verb, subj = triple
                    processed.append([verb.lower(), subj])
                elif len(triple) == 3:
                    verb, subj, obj = triple
                    processed.append([verb.lower(), subj, obj])
                else:
                    continue
            elif isinstance(item, list):
                # Old format: direct list
                if len(item) == 2:
                    verb, subj = item
                    processed.append([verb.lower(), subj])
                elif len(item) == 3:
                    verb, subj, obj = item
                    processed.append([verb.lower(), subj, obj])
                else:
                    continue
            else:
                continue
        result[obj_id] = processed
    return result


def load_actions_with_steps(json_path: str) -> Tuple[Dict[str, List[List[str]]], Dict[str, List[int]]]:
    """Load actions and their corresponding task steps from goal.json."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    actions_result: Dict[str, List[List[str]]] = {}
    steps_result: Dict[str, List[int]] = {}
    
    for obj_id, relations in data.items():
        processed_actions = []
        processed_steps = []
        
        for item in relations:
            if isinstance(item, dict) and 'state_pair' in item and 'task_step' in item:
                # New format with task_step
                triple = item['state_pair']
                task_step = item['task_step']
                
                if len(triple) == 2:
                    verb, subj = triple
                    processed_actions.append([verb.lower(), subj])
                    processed_steps.append(task_step)
                elif len(triple) == 3:
                    verb, subj, obj = triple
                    processed_actions.append([verb.lower(), subj, obj])
                    processed_steps.append(task_step)
            elif isinstance(item, list):
                # Old format - assign step as index
                if len(item) == 2:
                    verb, subj = item
                    processed_actions.append([verb.lower(), subj])
                    processed_steps.append(len(processed_actions) - 1)  # Use index as step
                elif len(item) == 3:
                    verb, subj, obj = item
                    processed_actions.append([verb.lower(), subj, obj])
                    processed_steps.append(len(processed_actions) - 1)  # Use index as step
        
        actions_result[obj_id] = processed_actions
        steps_result[obj_id] = processed_steps
    
    return actions_result, steps_result


def determine_scene_type(bddl_file: Path) -> str:
    """Determine scene type from BDDL filename."""
    filename = bddl_file.name
    for scene_prefix in SCENE_MAPPINGS.keys():
        if filename.startswith(scene_prefix):
            return scene_prefix
    
    # Default to coffee table if no match
    logger.warning(f"Could not determine scene type for {filename}, defaulting to COFFEE_TABLESCENE")
    return "COFFEE_TABLESCENE"


def load_init_state(cfg: GenerateConfig, task_type: str, case_name: str, log_file=None) -> Optional[np.ndarray]:
    """Load initial state for a specific case (based on run_libero_eval.py logic).
    
    Modified to use Ideal init files for specific test scenarios:
    - Ideal, Observation_Mismatching, and Random_Disturbance all use init files from the Ideal directory
    """
    if not cfg.use_init_files:
        return None
        
    init_files_dir = Path(cfg.init_files_root)
    
    # Task types that should use Ideal directory init files
    use_ideal_files = {"Ideal", "Observation_Mismatching", "Random_Disturbance"}
    
    if task_type in use_ideal_files:
        # Use init files from the Ideal directory
        init_dir_name = "Ideal"
        # Avoid circular import by importing locally
        from robocerebra_logging import log_message
        log_message(f"Using Ideal init files for task type: {task_type}", log_file)
    else:
        # Use init files from the original task type directory
        init_dir_name = task_type.replace(" ", "_")
    
    # Try the .init file (simplified structure)
    init_file = init_files_dir / init_dir_name / f"{case_name}.init"
    
    if init_file.exists():
        try:
            with open(init_file, 'rb') as f:
                init_state = pickle.load(f)
            # Avoid circular import by importing locally
            from robocerebra_logging import log_message
            log_message(f"Loaded init state from {init_file}", log_file)
            return init_state
        except Exception as e:
            logger.error(f"Failed to load init state from {init_file}: {e}")
    
    logger.warning(f"No init state found for {init_dir_name}/{case_name}")
    return None


def get_task_directories(cfg: GenerateConfig) -> List[Tuple[str, Path]]:
    """Get all task directories for specified task types.
    
    Modified to use Ideal files for specific test scenarios:
    - Ideal, Observation_Mismatching, and Random_Disturbance all use files from the Ideal directory
    - Other task types use their original directories
    """
    robocerebra_root = Path(cfg.robocerebra_root)
    task_dirs = []
    
    # Task types that should use Ideal directory files
    use_ideal_files = {"Ideal", "Observation_Mismatching", "Random_Disturbance"}
    
    for task_type in cfg.task_types:
        if task_type in use_ideal_files:
            # Use files from the Ideal directory
            source_dir = robocerebra_root / "Ideal"
            logger.info(f"Using Ideal directory files for task type: {task_type}")
        else:
            # Use files from the original task type directory
            source_dir = robocerebra_root / task_type
            logger.info(f"Using original directory files for task type: {task_type}")
        
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            continue
            
        for case_dir in source_dir.iterdir():
            if case_dir.is_dir():
                # Check if it contains a BDDL file
                bddl_files = list(case_dir.glob("*.bddl"))
                if bddl_files:
                    task_dirs.append((task_type, case_dir))
                    
    logger.info(f"Found {len(task_dirs)} total task directories")
    return task_dirs


def load_environment(task_dir: Path):
    """Load the environment for a given task directory."""
    # Find BDDL file
    bddl_files = list(task_dir.glob("*.bddl"))
    if not bddl_files:
        logger.error(f"No BDDL file found in {task_dir}")
        return None, None, None
        
    bddl_file = bddl_files[0]
    
    try:
        # Get problem info from BDDL file
        problem_info = BDDLUtils.get_problem_info(str(bddl_file))
        problem_name = problem_info["problem_name"]
        
        # Determine scene type from filename
        scene_type = determine_scene_type(bddl_file)
        
        # Map to the expected class name
        expected_class = SCENE_MAPPINGS.get(scene_type, problem_name)
        
        # Load controller config
        controller_config = load_controller_config(default_controller="OSC_POSE")
        
        # Create environment
        env = TASK_MAPPING[expected_class](
            bddl_file_name=str(bddl_file),
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
        
        return env, scene_type, str(bddl_file)
        
    except Exception as e:
        logger.error(f"Failed to load environment for {task_dir}: {e}")
        return None, None, None


def prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


# ★★ Find the address (addr) of an object's y-axis in qpos ★★
def _find_obj_y_addr(sim, obj_name: str) -> Optional[int]:
    """
    Infers the index of the y-coordinate of the object's position in qpos based on the joint naming convention.
    Returns None if not found (the object will be ignored).
    """
    # Three common naming conventions: <name>_1_joint0 / <name>_joint0 / <name>_joint
    patterns = [f"{obj_name}_1_joint0", f"{obj_name}_joint0", f"{obj_name}_joint"]
    # logger.debug(f"[DEBUG] Searching for object '{obj_name}' joints: {patterns}")
    
    for jn in patterns:
        if jn in sim.model.joint_names:
            qpos_addr = sim.model.get_joint_qpos_addr(jn)[0]  # x
            # logger.debug(f"[DEBUG] Found joint '{jn}' for object '{obj_name}' at qpos address {qpos_addr}, y-addr = {qpos_addr + 1}")
            return qpos_addr + 1  # x,y,z -> take y
    
    # logger.debug(f"[DEBUG] No joint found for object '{obj_name}', available joints: {list(sim.model.joint_names)[:10]}...")
    return None


# ★★ Parse task_description(.suffix).json ★★
def _load_step_objects(json_path: str, step_desc: Sequence[str]) -> List[str]:
    """
    Args:
        json_path: Path to task_description{suffix}.json
        step_desc: List obtained from parse_task_description() (with the "Step:" prefix removed)
    Returns:
        A list of object names corresponding to step_desc (if not found, an empty string is used as a placeholder)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)      # list[dict{step, object}]
    # Create a mapping from 'Step: xxx' to object
    mapping = {item["step"] : item["object"] for item in data if "step" in item}
    objs = []
    for desc in step_desc:
        key = f"Step: {desc}"
        objs.append(mapping.get(key, ""))   # Use an empty string as placeholder if not found
    return objs


def parse_task_description(txt_path: str) -> Tuple[List[str], List[int]]:
    """Parse task_description*.txt and return (step_descriptions, start_indices)."""
    step_desc: list[str] = []
    start_indices: list[int] = []
    BRACKET_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")
    
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
        logger.error(f"Task description file not found: {txt_path}")
        return step_desc, start_indices
    except Exception as e:
        logger.error(f"Error reading task description file {txt_path}: {e}")
        return step_desc, start_indices
    
    if not lines:
        logger.warning(f"Task description file is empty: {txt_path}")
        return step_desc, start_indices
    
    logger.debug(f"Parsing task description file: {txt_path} with {len(lines)} lines")
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Step"):
            # Step line
            desc = line.split(":", 1)[1].strip()
            step_desc.append(desc)
            logger.debug(f"Found step: {desc}")
            # The next line should be [start, end]
            if i + 1 < len(lines):
                m = BRACKET_RE.match(lines[i + 1])
                if m:
                    start_idx = int(m.group(1))
                    start_indices.append(start_idx)
                    logger.debug(f"Found start index: {start_idx}")
                    i += 1  # Skip the bracket line
                else:
                    logger.warning(f"Expected bracket line after step, but found: {lines[i + 1]}")
            else:
                logger.warning(f"Missing bracket line after step: {desc}")
            i += 1
        else:
            i += 1
    
    logger.info(f"Parsed {len(step_desc)} steps and {len(start_indices)} indices from {txt_path}")
    return step_desc, start_indices


def setup_dynamic_distractor_info(cfg: GenerateConfig, task_dir: Path, env, naming_step_desc: List[str], log_file=None) -> Optional[Dict[str, Any]]:
    """Setup dynamic distractor information for dynamic object movement."""
    from robocerebra_logging import log_message
    
    # log_message(f"[DEBUG] Dynamic setup: cfg.dynamic={cfg.dynamic}, cfg.resume={cfg.resume}", log_file)
    
    if not (cfg.dynamic and cfg.resume):
        # log_message(f"[DEBUG] Dynamic disabled: dynamic={cfg.dynamic}, resume={cfg.resume}", log_file)
        return None
    
    dir_path = str(task_dir)
    # log_message(f"[DEBUG] Checking dynamic setup for task: {task_dir.name}", log_file)
    
    # Read JSON
    if cfg.task_description_suffix:
        json_name = f"task_description{cfg.task_description_suffix}.json"
    else:
        json_name = "task_description.json"
    json_path = os.path.join(dir_path, json_name)
    
    # log_message(f"[DEBUG] Looking for JSON file: {json_path}", log_file)
    
    if not os.path.isfile(json_path):
        log_message(f"[WARN] {json_path} not found, dynamic functionality disabled.", log_file)
        return None
    
    # log_message(f"[DEBUG] Found JSON file, parsing step objects", log_file)
    # Parse step -> object mapping
    step_objects = _load_step_objects(json_path, naming_step_desc)
    # log_message(f"[DEBUG] Step objects: {step_objects}", log_file)

    # For each step, collect the movable related object's address/base
    step_addr_y: List[Optional[int]] = []
    step_base_y: List[Optional[float]] = []
    # log_message(f"[DEBUG] Processing {len(step_objects)} step objects for related addresses", log_file)
    
    for i, obj_name in enumerate(step_objects):
        # log_message(f"[DEBUG] Step {i}: object '{obj_name}' in MOVABLE_LIST: {obj_name in MOVABLE_OBJECT_LIST if obj_name else False}", log_file)
        
        if not obj_name or obj_name not in MOVABLE_OBJECT_LIST:
            step_addr_y.append(None)
            step_base_y.append(None)
            continue
            
        addr = _find_obj_y_addr(env.sim, obj_name)
        if addr is None:
            log_message(f"[WARN] Could not find joint for {obj_name}, ignoring the related object.", log_file)
            step_addr_y.append(None)
            step_base_y.append(None)
        else:
            # log_message(f"[DEBUG] Found address {addr} for object {obj_name}", log_file)
            step_addr_y.append(addr)
            step_base_y.append(env.sim.data.qpos[addr].copy())

    # log_message(f"[DEBUG] Related objects found: {sum(1 for x in step_addr_y if x is not None)}/{len(step_addr_y)}", log_file)

    # Collect the pool of unrelated movable objects
    unrelated_addr = []
    # log_message(f"[DEBUG] Searching for unrelated objects in {len(MOVABLE_OBJECT_LIST)} candidates", log_file)
    
    for name in MOVABLE_OBJECT_LIST:
        if name in step_objects:
            continue
        addr = _find_obj_y_addr(env.sim, name)
        if addr is not None:
            # log_message(f"[DEBUG] Found unrelated object {name} at address {addr}", log_file)
            unrelated_addr.append((addr, env.sim.data.qpos[addr].copy()))

    # log_message(f"[DEBUG] Unrelated objects found: {len(unrelated_addr)}", log_file)
    
    has_related = any(a is not None for a in step_addr_y)
    has_unrelated = len(unrelated_addr) > 0
    
    # log_message(f"[DEBUG] Dynamic validation: has_related={has_related}, has_unrelated={has_unrelated}", log_file)
    
    if has_related and has_unrelated:
        # log_message("[DEBUG] Dynamic distractor setup successful!", log_file)
        return {
            "step_addr": step_addr_y,
            "step_base": step_base_y,
            "unrel": unrelated_addr,
        }
    else:
        log_message(f"[WARN] Insufficient dynamic information (related: {has_related}, unrelated: {has_unrelated}), feature disabled.", log_file)
        return None