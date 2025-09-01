#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robocerebra_episode.py

Episode-level execution logic for RoboCerebra evaluation.
"""

import json
import logging
import os
import random
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import GenerateConfig, MOVABLE_OBJECT_LIST
from robocerebra_logging import log_message, save_rollout_video
from resume import simulate_resume_completion
from utils import prepare_observation, process_action, _find_obj_y_addr, _load_step_objects
from experiments.robot.libero.libero_utils import get_libero_dummy_action
from experiments.robot.robot_utils import get_action


logger = logging.getLogger(__name__)


def setup_dynamic_distractor_info(cfg: GenerateConfig, env, naming_step_desc: Sequence[str], 
                                  dir_path: str, log_file=None) -> Optional[Dict[str, Any]]:
    """Setup dynamic distractor information for the episode."""
    if not (cfg.dynamic and cfg.resume):
        return None
        
    # Read JSON
    if cfg.task_description_suffix:
        json_name = f"task_description{cfg.task_description_suffix}.json"
    else:
        json_name = "task_description.json"
    json_path = os.path.join(dir_path, json_name)
    
    if not os.path.isfile(json_path):
        log_message(f"[WARN] {json_path} not found, dynamic functionality disabled.", log_file)
        return None
    
    # Parse step -> object mapping
    step_objects = _load_step_objects(json_path, naming_step_desc)

    # For each step, collect the movable related object's address/base
    step_addr_y: List[Optional[int]]   = []
    step_base_y: List[Optional[float]] = []
    for obj_name in step_objects:
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
            step_addr_y.append(addr)
            step_base_y.append(env.sim.data.qpos[addr].copy())

    # Collect the pool of unrelated movable objects
    unrelated_addr = []
    for name in MOVABLE_OBJECT_LIST:
        if name in step_objects:
            continue
        addr = _find_obj_y_addr(env.sim, name)
        if addr is not None:
            unrelated_addr.append((addr, env.sim.data.qpos[addr].copy()))

    if any(a is not None for a in step_addr_y) and unrelated_addr:
        return {
            "step_addr": step_addr_y,
            "step_base": step_base_y,
            "unrel": unrelated_addr,
        }
    else:
        log_message("[WARN] Insufficient dynamic information (no related or unrelated objects), feature disabled.", log_file)
        return None


def initialize_episode_state(cfg: GenerateConfig, env, goal: Any, step_states: Sequence[np.ndarray] | None,
                             initial_state: Optional[np.ndarray], task_type: str, case_name: str,
                             resume_handler: Optional[Dict[str, Any]], log_file=None) -> Tuple[Any, Dict]:
    """Initialize episode state including dynamic shift and resume logic."""
    # Reset environment
    obs = env.reset()

    # Apply init state if available
    if initial_state is not None:
        try:
            env.sim.set_state_from_flattened(initial_state)
            env.sim.forward()
            env._post_process()
            env._update_observables(force=True)
            obs = env._get_observations()
            log_message(f"Applied init state from init files for {task_type}/{case_name}", log_file)
        except Exception as e:
            logger.error(f"Failed to apply init state: {e}")
            log_message(f"Continuing with default reset state", log_file)

    # Initialize statistics
    episode_stats = {
        'total_agent_subtasks': 0,
        'total_goals': sum(len(v) for v in goal.values()) if goal else 0,
        'total_resume_skipped': 0,
        'total_resume_completed': 0,
        'total_dynamic_shift_excluded': 0,
        'init_state_idx': 0,
        'skip_increment': False,
        'pending_catch_up': False,
    }

    if cfg.dynamic_shift_description:
        episode_stats['init_state_idx'] = 1
        episode_stats['skip_increment'] = True
        if cfg.resume:
            episode_stats['pending_catch_up'] = True

    if (cfg.resume or cfg.dynamic_shift_description) and step_states:
        env.sim.set_state_from_flattened(step_states[episode_stats['init_state_idx']])
        env.skip_pick_quat_once = True
        env.sim.forward(); env._post_process(); env._update_observables(force=True)
        obs = env._get_observations()
        
        # Apply simulate_resume_completion for dynamic_shift_description
        if cfg.dynamic_shift_description and resume_handler and episode_stats['init_state_idx'] > 0:
            # Step 1: Reset completion tracking to avoid counting conflicts
            if goal:
                comp_before_shift, total_completed_before_shift, _ = env._check_success(goal)
                log_message(f"[Dynamic Shift] Initial completion state: {total_completed_before_shift} total, details: {comp_before_shift}", log_file)
                
                # Reset _state_progress to initial state to avoid double counting
                if hasattr(env, '_state_progress'):
                    for obj in env._state_progress.keys():
                        env._state_progress[obj] = 0
                    log_message(f"[Dynamic Shift] Reset state progress to initial values", log_file)
            
            # Step 2: Apply simulate_resume_completion to mark prior subtasks as completed
            resume_completed_count, completed_subtasks = simulate_resume_completion(
                env, goal, resume_handler, episode_stats['init_state_idx']
            )
            
            # Step 3: Get the new completion state after resume simulation
            if goal:
                comp_after_shift, total_completed_after_shift, _ = env._check_success(goal)
                log_message(f"[Dynamic Shift] Post-resume completion state: {total_completed_after_shift} total, details: {comp_after_shift}", log_file)
            
            episode_stats['total_resume_completed'] += resume_completed_count
            episode_stats['total_dynamic_shift_excluded'] += resume_completed_count
            log_message(f"[Dynamic Shift] Jumped to Step {episode_stats['init_state_idx']}, logically completed {resume_completed_count} prior subtasks: {completed_subtasks}", log_file)
            log_message(f"[Dynamic Shift] Excluding {resume_completed_count} subtasks from total goals due to forced completion", log_file)

    return obs, episode_stats


def handle_dynamic_movement(cfg: GenerateConfig, env, distractor_info: Dict[str, Any], 
                           step_idx: int, resume_trigger_step: Optional[int], t: int, 
                           rng, toggle_dir: int, seg_mid_moved: bool, log_file=None) -> Tuple[Any, int, Optional[int], bool]:
    """Handle dynamic object movement during episode execution."""
    from robocerebra_logging import log_message
    
    if not (cfg.dynamic and distractor_info and not seg_mid_moved and 
            resume_trigger_step is not None and t == resume_trigger_step + 10):
        return env._get_observations(), toggle_dir, resume_trigger_step, seg_mid_moved
    
    step_addr_y = distractor_info["step_addr"]
    step_base_y = distractor_info["step_base"]
    unrelated_set = distractor_info["unrel"]
    
    # Decide whether to use a related or an unrelated object
    use_related = rng.random() < 0.5
    if use_related and step_idx < len(step_addr_y) and step_addr_y[step_idx] is not None:
        addr_y = step_addr_y[step_idx]
        base_y = step_base_y[step_idx]
        target_name = "related"
    else:
        # Use Python's random.choice to avoid converting to NumPy float64
        addr_y, base_y = random.choice(unrelated_set)
        target_name = "unrelated"

    # Execute movement
    offset = 0.15 * toggle_dir
    env.sim.data.qpos[addr_y] = base_y + offset
    env.sim.forward(); env._post_process(); env._update_observables(force=True)
    obs = env._get_observations()
    log_message(f"[Dynamic] Moved {target_name} object at step {step_idx}, Δy={offset:+.2f}", log_file)
    
    return obs, toggle_dir * -1, None, True


def handle_segment_transition(cfg: GenerateConfig, env, goal: Any, step_idx: int, prev_step_idx: int,
                             seg_increment_accum: int, replay_images_seg: List[np.ndarray],
                             episode_idx: int, naming_step_desc: Sequence[str], task_type: str, 
                             case_name: str, comp_start_dict: Dict, step_states: Sequence[np.ndarray] | None,
                             episode_stats: Dict, resume_handler: Optional[Dict[str, Any]], 
                             log_file=None) -> Tuple[Dict, List[np.ndarray], int, bool, bool, Optional[int]]:
    """Handle transition between segments including video saving and resume logic."""
    # Save the previous segment's video
    seg_success = seg_increment_accum
    log_message(f"[Segment {prev_step_idx}] Subtasks completed = {seg_success}", log_file)
    suffix = "_complete_description" if cfg.complete_description else ""
    name = f"{episode_idx}_step{prev_step_idx}{suffix}"
    
    # Save the video and get its path
    mp4_path = save_rollout_video(
        replay_images_seg,
        name,
        success=seg_success > 0,
        task_description=naming_step_desc[prev_step_idx],
        log_file=log_file,
        task_suite=f"{cfg.task_suite_name}_{task_type}",
        task_name=case_name,
    )
    
    comp_end_dict, _, _ = env._check_success(goal)
    completed_objects = [
        obj for obj, rate in comp_end_dict.items()
        if rate > comp_start_dict.get(obj, 0)
    ]

    json_path = mp4_path.rsplit('.', 1)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({
            "video": os.path.basename(mp4_path),
            "completed_subtasks": completed_objects,
            "success": len(completed_objects) > 0
        }, jf, ensure_ascii=False, indent=2)

    # Prepare the starting completion baseline for the next segment
    comp_start_dict = comp_end_dict
    replay_images_seg.clear()
    seg_increment_accum = 0
    just_resumed = False

    # Dynamic shift / resume logic
    do_resume_now = False
    if episode_stats['pending_catch_up']:
        if step_idx == 1:
            episode_stats['skip_increment'] = False
        else:
            do_resume_now = True
            episode_stats['pending_catch_up'] = False
    else:
        do_resume_now = cfg.resume and step_states is not None

    resume_trigger_step = None
    if do_resume_now and step_states is not None:
        env.sim.set_state_from_flattened(step_states[step_idx])
        env.sim.forward(); env._post_process(); env._update_observables(force=True)
        env.skip_pick_quat_once = True 
        episode_stats['skip_increment'] = True
        just_resumed = True
        resume_trigger_step = True  # Signal that resume occurred
        
        # Intelligent step-based resume completion
        if resume_handler and step_idx < len(naming_step_desc):
            resume_completed_count, completed_subtasks = simulate_resume_completion(
                env, goal, resume_handler, step_idx
            )
            episode_stats['total_resume_completed'] += resume_completed_count
            log_message(f"[Resume] Reverting to Step {step_idx}, auto-completed {resume_completed_count} prior subtasks: {completed_subtasks}", log_file)
        else:
            log_message(f"[Resume] Reverting to the initial state of Step {step_idx}", log_file)

    return comp_start_dict, replay_images_seg, seg_increment_accum, just_resumed, episode_stats['skip_increment'], resume_trigger_step


def execute_policy_step(cfg: GenerateConfig, model, observation: Dict, desc: str, 
                       action_queue: deque, processor=None, action_head=None, 
                       proprio_projector=None, noisy_action_projector=None) -> np.ndarray:
    """Execute a single policy inference step and return the action."""
    if not action_queue:
        actions = get_action(
            cfg,
            model,
            observation,
            desc,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            use_film=cfg.use_film,
        )
        action_queue.extend(actions)
    return action_queue.popleft()


def update_completion_tracking(env, goal: Any, total_completed_prev: int, episode_stats: Dict,
                              step_idx: int, log_file=None) -> Tuple[int, int]:
    """Update completion tracking and return difference and new total."""
    _, total_completed_now, _ = env._check_success(goal)
    diff = total_completed_now - total_completed_prev

    if diff > 0:
        step_no = step_idx + 1
        if episode_stats['skip_increment']:
            episode_stats['total_resume_skipped'] += diff
            log_message(f"[Step {step_no}] (Skip) Directly completed {diff} subtasks by resuming at the first frame", log_file)
        else:
            episode_stats['total_agent_subtasks'] += diff
            log_message(f"[Step {step_no}] Completed {diff} new subtasks", log_file)
            return diff, total_completed_now

    return 0, total_completed_now


def finalize_episode(cfg: GenerateConfig, env, goal: Any, replay_images_all: List[np.ndarray],
                    replay_images_seg: List[np.ndarray], episode_idx: int, prev_step_idx: int,
                    seg_increment_accum: int, naming_step_desc: Sequence[str], task_type: str,
                    case_name: str, episode_stats: Dict, log_file=None) -> Tuple[bool, int, int]:
    """Finalize episode and return success status and statistics."""
    # Handle final segment
    seg_success = seg_increment_accum
    seg_no = prev_step_idx + 1
    log_message(f"[Segment {seg_no}] Subtasks counted = {seg_success} (final segment)", log_file)
    suffix = "_complete_description" if cfg.complete_description else ""
    name = f"{episode_idx}_step{prev_step_idx}{suffix}"
    save_rollout_video(
        replay_images_seg,
        name,
        success=seg_success > 0,
        task_description=naming_step_desc[prev_step_idx],
        log_file=log_file,
        task_suite=f"{cfg.task_suite_name}_{task_type}",
        task_name=case_name,
    )

    # Determine overall task success
    _, _, all_done = env._check_success(goal)

    suffix = "_complete_description" if cfg.complete_description else ""
    name = f"{episode_idx}{suffix}"
    mp4_path = save_rollout_video(
        replay_images_all,
        name,
        success=all_done,
        task_description=naming_step_desc[prev_step_idx],
        log_file=log_file,
        task_suite=f"{cfg.task_suite_name}_{task_type}",
        task_name=case_name,
    )
    
    # Generate JSON for the complete video
    json_path = mp4_path.rsplit('.', 1)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({
            "video": os.path.basename(mp4_path),
            "completed_subtasks": [],  # This would need to be tracked separately
            "episode_success": all_done
        }, jf, ensure_ascii=False, indent=2)
    
    # Adjust effective total goals to exclude dynamic shift forced completions
    effective_total_goals = episode_stats['total_goals'] - episode_stats['total_dynamic_shift_excluded']
    log_message(f"[Statistics] Original total goals: {episode_stats['total_goals']}, Dynamic shift excluded: {episode_stats['total_dynamic_shift_excluded']}, Effective total goals: {effective_total_goals}", log_file)
    
    return all_done, episode_stats['total_agent_subtasks'], effective_total_goals