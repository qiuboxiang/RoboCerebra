#!/usr/bin/env python3
"""Configuration classes and constants for RoboCerebra evaluation."""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

PRETRAINED_CHECKPOINT_ENV = "ROBOCEREBRA_PRETRAINED_CHECKPOINT"
BENCH_ROOT_ENV = "ROBOCEREBRA_BENCH_ROOT"
INIT_FILES_ROOT_ENV = "ROBOCEREBRA_INIT_FILES_ROOT"
WANDB_ENTITY_ENV = "WANDB_ENTITY"
WANDB_PROJECT_ENV = "WANDB_PROJECT"

DEFAULT_TASK_TYPES = [
    "Ideal",
    "Memory_Execution",
    "Memory_Exploration",
    "Mix",
    "Observation_Mismatching",
    "Random_Disturbance",
]


def _read_env(name: str) -> Optional[str]:
    """Return a stripped environment variable or ``None``."""
    value = os.getenv(name)
    if value is None:
        return None

    value = value.strip()
    return value or None


def _default_init_files_root() -> Optional[str]:
    """Resolve the init files root from explicit or derived configuration."""
    explicit_root = _read_env(INIT_FILES_ROOT_ENV)
    if explicit_root:
        return explicit_root

    bench_root = _read_env(BENCH_ROOT_ENV)
    if not bench_root:
        return None

    return str(Path(bench_root) / "init_files")


class TaskSuite(str, Enum):
    ROBOCEREBRA = "robocerebra"


TASK_MAX_STEPS: Dict[TaskSuite, int] = {
    TaskSuite.ROBOCEREBRA: 400,
}

SCENE_MAPPINGS = {
    "COFFEE_TABLESCENE": "libero_coffee_table_manipulation",
    "KITCHEN_TABLESCENE": "libero_kitchen_tabletop_manipulation",
    "STUDY_TABLESCENE": "libero_study_tabletop_manipulation",
}

MOVABLE_OBJECT_LIST = [
    "alphabet_soup",
    "bbq_sauce",
    "butter",
    "chocolate_pudding",
    "cookies",
    "cream_cheese",
    "ketchup",
    "macaroni_and_cheese",
    "milk",
    "orange_juice",
    "popcorn",
    "salad_dressing",
    "new_salad_dressing",
    "tomato_sauce",
    "white_bowl",
    "akita_black_bowl",
    "plate",
    "glazed_rim_porcelain_ramekin",
    "red_coffee_mug",
    "porcelain_mug",
    "white_yellow_mug",
    "chefmate_8_frypan",
    "bowl_drainer",
    "moka_pot",
    "window",
    "faucet",
    "black_book",
    "yellow_book",
    "desk_caddy",
    "wine_bottle",
]


@dataclass
class GenerateConfig:
    # Model-specific parameters
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path, None] = field(
        default_factory=lambda: _read_env(PRETRAINED_CHECKPOINT_ENV))
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    unnorm_key: Union[str, Path] = "robocerebra"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # RoboCerebra environment-specific parameters
    robocerebra_root: Optional[str] = field(
        default_factory=lambda: _read_env(BENCH_ROOT_ENV))
    init_files_root: Optional[str] = field(
        default_factory=_default_init_files_root)
    task_suite_name: str = TaskSuite.ROBOCEREBRA.value
    task_types: Optional[List[str]] = None
    num_steps_wait: int = 15
    num_trials_per_task: int = 5
    env_img_res: int = 256
    switch_steps: int = 150
    resume: bool = False
    dynamic_shift_description: bool = False
    complete_description: bool = False
    task_description_suffix: str = ""
    dynamic: bool = False
    use_init_files: bool = True
    initial_states_path: str = "DEFAULT"

    # Logging and utilities
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_entity: Optional[str] = field(
        default_factory=lambda: _read_env(WANDB_ENTITY_ENV))
    wandb_project: Optional[str] = field(
        default_factory=lambda: _read_env(WANDB_PROJECT_ENV))
    seed: int = 7

    def __post_init__(self) -> None:
        if self.task_types is None:
            self.task_types = list(DEFAULT_TASK_TYPES)

        if self.use_init_files and not self.init_files_root:
            if self.robocerebra_root:
                self.init_files_root = str(
                    Path(self.robocerebra_root) / "init_files")

        self._configure_dynamic_parameters()

    def _configure_dynamic_parameters(self) -> None:
        """Set baseline dynamic settings before per-task overrides."""
        self.dynamic = False
        self.dynamic_shift_description = False


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, ("Set --pretrained_checkpoint or export "
                                       f"{PRETRAINED_CHECKPOINT_ENV}.")
    assert cfg.robocerebra_root, ("Set --robocerebra_root or export "
                                  f"{BENCH_ROOT_ENV}.")
    if cfg.use_init_files:
        assert cfg.init_files_root, ("Set --init_files_root or export "
                                     f"{INIT_FILES_ROOT_ENV}.")
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, (
            "Expected center_crop=True because the checkpoint "
            "was trained with image augmentations.")
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), (
        "Cannot use both 8-bit and 4-bit quantization.")
    if cfg.dynamic:
        assert cfg.resume, "dynamic=True requires resume=True."
    if cfg.use_wandb:
        assert cfg.wandb_entity, (
            "Set --wandb_entity or export WANDB_ENTITY when "
            "use_wandb=True.")
        assert cfg.wandb_project, (
            "Set --wandb_project or export WANDB_PROJECT when "
            "use_wandb=True.")
