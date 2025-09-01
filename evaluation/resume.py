#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robocerebra_resume.py

Resume mechanism utilities for RoboCerebra evaluation.
"""

import logging
from typing import Any, Dict, List, Tuple



logger = logging.getLogger(__name__)


def create_step_based_resume_handler(goal: Dict[str, List[List[str]]], goal_steps: Dict[str, List[int]]) -> Dict[str, Any]:
    """Create a handler for step-based resume functionality."""
    if not goal or not goal_steps:
        return {}
    
    # Create mapping from step to all subtasks that should be completed at that step
    step_to_subtasks = {}
    
    for obj_id, actions in goal.items():
        steps = goal_steps.get(obj_id, [])
        for i, (action, step) in enumerate(zip(actions, steps)):
            if step not in step_to_subtasks:
                step_to_subtasks[step] = []
            step_to_subtasks[step].append({
                'object': obj_id,
                'action_index': i,
                'action': action
            })
    
    # Create mapping from step to all subtasks that should be completed BEFORE that step
    step_to_prior_subtasks = {}
    for current_step in step_to_subtasks.keys():
        prior_subtasks = []
        for step in step_to_subtasks.keys():
            if step < current_step:
                prior_subtasks.extend(step_to_subtasks[step])
        step_to_prior_subtasks[current_step] = prior_subtasks
    
    return {
        'step_to_subtasks': step_to_subtasks,
        'step_to_prior_subtasks': step_to_prior_subtasks,
        'max_step': max(step_to_subtasks.keys()) if step_to_subtasks else 0
    }


def simulate_resume_completion(env, goal: Dict[str, List[List[str]]], resume_handler: Dict[str, Any], current_step: int) -> Tuple[int, List[str]]:
    """Simulate completion of all subtasks that should be done before current_step when resuming."""
    if not resume_handler or current_step not in resume_handler['step_to_prior_subtasks']:
        return 0, []
    
    prior_subtasks = resume_handler['step_to_prior_subtasks'][current_step]
    completed_by_resume = []
    
    # Get current environment state to manipulate completion tracking
    if hasattr(env, '_state_progress'):
        for subtask in prior_subtasks:
            obj_id = subtask['object']
            action_index = subtask['action_index']
            
            # Mark this subtask as completed in the environment's progress tracking
            if obj_id in env._state_progress:
                if env._state_progress[obj_id] <= action_index:
                    env._state_progress[obj_id] = action_index + 1
                    completed_by_resume.append(f"{obj_id}_{action_index}")
    
    return len(completed_by_resume), completed_by_resume