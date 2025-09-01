#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regenerate RoboCerebra dataset with full compatibility to original LIBERO format.
Converts RoboCerebra benchmark data to HDF5 format for training and evaluation.

Key features:
1. Extract complete observation data from RoboCerebra demonstrations
2. Process multi-step task descriptions and frame mappings
3. Generate texture variants for data augmentation
4. Maintain LIBERO-compatible HDF5 structure for training
5. Support dynamic distractor object processing
"""

import argparse, json, os, shutil, xml.etree.ElementTree as ET
from pathlib import Path
import re, bisect

import torch
import h5py, imageio, numpy as np, tqdm
import robosuite.utils.transform_utils as T
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING
from robosuite import load_controller_config
import numpy as np

# RoboCerebra BDDL problem -> scene mapping
BDDL_SCENE_MAPPING = {
    "LIBERO_Coffee_Table_Manipulation": "coffee_table",
    "LIBERO_Kitchen_Tabletop_Manipulation": "kitchen_table",
}

def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

seed = 7
set_seed_everywhere(seed)

# --------------------------------------------------------------------------- #
#                    RoboCerebra Scene -> MJCF -> Texture Utilities           #
# --------------------------------------------------------------------------- #
LIBERO_ROOT = Path("<LIBERO_ROOT_PATH>")  # TODO: Set path to LIBERO installation directory

SCENES = {
    "coffee_table": {
        "mjcf_path": Path("libero/libero/assets/scenes/libero_coffee_table_base_style.xml"),
        "texture_name": "tex-short_coffee_table",
        "texture_options": [
            "../textures/martin_novak_wood_table.png",
            "short_coffee_table/Marble062_COL_4K.png",
            "../textures/table_light_wood.png",
        ],
    },
    "kitchen_table": {
        "mjcf_path": Path("libero/libero/assets/scenes/libero_kitchen_tabletop_base_style.xml"),
        "texture_name": "tex-table",
        "texture_options": [
            "../textures/martin_novak_wood_table.png",
            "short_coffee_table/Marble062_COL_4K.png",
            "../textures/table_light_wood_512.png",
        ],
    },
    "living_room_table": {
        "mjcf_path": Path("libero/libero/assets/scenes/libero_living_room_tabletop_base_style.xml"),
        "texture_name": "tex-living_room_table",
        "texture_options": [
            "../textures/martin_novak_wood_table.png",
            "short_coffee_table/Marble062_COL_4K.png",
            "../textures/table_light_wood.png",
        ],
    },
    "study_table": {
        "mjcf_path": Path("libero/libero/assets/scenes/libero_study_base_style.xml"),
        "texture_name": "tex-table",
        "texture_options": [
            "../textures/martin_novak_wood_table.png",
            "short_coffee_table/Marble062_COL_4K.png",
            "../textures/table_light_wood.png",
        ],
    },
}

# --------------------------------------------------------------------------- #
#                                Utility Functions                           #
# --------------------------------------------------------------------------- #

def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.
    """
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action

def list_texture_options(scene_key: str):
    return SCENES[scene_key]["texture_options"]


def set_scene_texture(scene_key: str, texture_path: str):
    """Directly modify the texture path in scene MJCF file."""
    cfg = SCENES[scene_key]
    mjcf_file = LIBERO_ROOT / cfg["mjcf_path"]

    tree = ET.parse(mjcf_file)
    root = tree.getroot()
    for tex in root.findall("./asset/texture"):
        if tex.get("name") == cfg["texture_name"]:
            tex.set("file", texture_path)
            break
    else:
        raise RuntimeError(f"Texture node not found: {cfg['texture_name']}")

    tree.write(mjcf_file, encoding="utf-8", xml_declaration=True)
    print(f"[Scene={scene_key}] Switched texture -> {texture_path}")


def detect_scene_from_bddl(bddl_path: str) -> str:
    """
    Automatically detect scene type from BDDL file problem definition.
    Returns scene key (e.g., 'coffee_table', 'kitchen_table').
    """
    try:
        with open(bddl_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # Extract problem name from "(define (problem PROBLEM_NAME)"
        match = re.search(r'\(define\s+\(problem\s+([^)]+)\)', first_line)
        if not match:
            raise ValueError(f"Cannot parse problem definition in {bddl_path}")
        
        problem_name = match.group(1).strip()
        
        if problem_name not in BDDL_SCENE_MAPPING:
            available_scenes = list(BDDL_SCENE_MAPPING.keys())
            raise ValueError(f"Unknown scene type '{problem_name}' in {bddl_path}. Available: {available_scenes}")
        
        scene_key = BDDL_SCENE_MAPPING[problem_name]
        print(f"[Auto-detected] {bddl_path} → scene: {scene_key}")
        return scene_key
    
    except Exception as e:
        raise RuntimeError(f"Failed to detect scene from {bddl_path}: {e}")


def parse_step_file(txt_path: str):
    """
    Parse task_description*.txt file, returns List[(step_desc, end_frame)].
    Supports both old/new formats, ensures end_frame is monotonically increasing.
    """
    steps, lines = [], Path(txt_path).read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith("Step:"):
            i += 1
            continue
        desc = lines[i].split(":", 1)[1].strip()
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        # New format
        if j < len(lines) and lines[j].lstrip().startswith("["):
            m = re.match(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", lines[j].strip())
            if not m:
                raise ValueError(f"{txt_path}: Frame interval format error: {lines[j]}")
            end_frame = int(m.group(2))
            i = j + 1
            if i < len(lines) and lines[i].startswith("Related Objects"):
                i += 1
        # Old format
        else:
            try:
                desc, end_frame_str = desc.rsplit(" ", 1)
                end_frame = int(end_frame_str)
                desc = desc.strip()
            except ValueError:
                raise ValueError(f"{txt_path}: Old format parsing failed: {lines[i]}")
            i += 1
        steps.append((desc, end_frame))

    steps.sort(key=lambda x: x[1])
    for p, n in zip(steps[:-1], steps[1:]):
        if n[1] <= p[1]:
            raise RuntimeError(f"{txt_path}: End frames not strictly increasing")
    return steps

# --------------------------------------------------------------------------- #
#                          RoboCerebra Conversion Pipeline                   #
# --------------------------------------------------------------------------- #
def main(args):
    # Create output directories for RoboCerebra dataset
    if os.path.exists(args.robocerebra_target_dir):
        if input("Target dir exists. Overwrite? (y/N): ").lower() != "y":
            return
        shutil.rmtree(args.robocerebra_target_dir)
    per_step_root = Path(args.robocerebra_target_dir, "per_step")
    all_hdf5_root = Path(args.robocerebra_target_dir, "all_hdf5")
    per_step_root.mkdir(parents=True)
    all_hdf5_root.mkdir()

    metainfo = {}
    # Ensure metainfo directory exists
    metainfo_dir = Path("./experiments/robot/libero")
    metainfo_dir.mkdir(parents=True, exist_ok=True)
    metainfo_path = metainfo_dir / f"{args.dataset_name}_metainfo.json"

    raw_root = Path(args.robocerebra_raw_data_dir)
    subdirs = sorted([d for d in raw_root.iterdir() if d.is_dir()])

    # Outer loop: process subdirectories
    for subdir in tqdm.tqdm(subdirs, desc="[SUBDIR] Raw subfolders"):
        case_name = subdir.name

        # Check required files
        files_ok = {
            "bddl": next(subdir.glob("*.bddl"), None),
            "h5f":  next(subdir.glob("*.hdf5"), None),
            "step_txt": next(subdir.glob("task_description*.txt"), None),
        }
        if not all(files_ok.values()):
            print(f"Skip {case_name}: missing files")
            continue

        # Auto-detect scene type from BDDL file
        detected_scene = detect_scene_from_bddl(files_ok["bddl"])
        if args.scene and args.scene != detected_scene:
            print(f"[Warning] Manual scene '{args.scene}' overrides detected scene '{detected_scene}' for {case_name}")
            scene_to_use = args.scene
        else:
            scene_to_use = detected_scene

        steps = parse_step_file(files_ok["step_txt"])

        # Process distractor joint
        distract_txt = next(subdir.glob("distractor*.txt"), None)
        distractor = None
        if distract_txt:
            distract_val = next(
                (ln.strip() for ln in distract_txt.read_text(encoding="utf-8").splitlines() if ln.strip()),
                None,
            )
            if distract_val:
                distractor = f"{distract_val}_1_joint0"

        # Inner loop: process textures
        for tex_idx, tex_rel in enumerate(list_texture_options(scene_to_use)):
            tex_suffix = f"tex{tex_idx}"
            set_scene_texture(scene_to_use, tex_rel)

            # Create environment (after texture change)
            pb_info = BDDLUtils.get_problem_info(files_ok["bddl"])
            env = TASK_MAPPING[pb_info["problem_name"]](
                bddl_file_name=str(files_ok["bddl"]),
                robots=["Panda"],
                controller_configs=load_controller_config(default_controller="OSC_POSE"),
                has_renderer=False,
                has_offscreen_renderer=True,
                camera_names=["agentview", "robot0_eye_in_hand"],
                ignore_done=True,
                use_camera_obs=True,
                reward_shaping=True,
                camera_heights=256,  # Consistent with scale_up_nonoop version
                camera_widths=256,
                control_freq=20,
            )

            # Get distractor qpos addresses
            mj_model = env.sim.model
            if distractor and distractor in mj_model.joint_names:
                dq0 = mj_model.get_joint_qpos_addr(distractor)[0]
                dq1 = dq0 + 1
            else:
                dq0 = dq1 = None

            # Hide green auxiliary sites
            for sn in ("gripper0_grip_site_cylinder", "gripper0_grip_site"):
                if sn in mj_model.site_names:
                    mj_model.site_rgba[mj_model.site_name2id(sn)][3] = 0.0

            variants = [("orig", [])]
            if dq0 is not None:
                variants += [
                    ("dxp005", [(dq0, +0.05)]),
                    ("dyp005", [(dq1, +0.05)]),
                    ("dym005", [(dq1, -0.05)]),
                ]

            # Load original demo (reuse for all variants)
            with h5py.File(files_ok["h5f"], "r") as f:
                demo = f["data"]["demo_1"]
                orig_states = demo["states"][()]
                orig_actions = demo["actions"][()]

            # Process variants and steps
            for var, delta_spec in variants:
                var_full = f"{var}_{tex_suffix}"
                env.reset()

                # Replay and filter no-op actions
                keep_idx = []
                states, actions = [], []
                # LIBERO format observation data
                gripper_states, joint_states = [], []
                ee_pos, ee_ori, ee_states = [], [], []
                robot_states = []
                agent_imgs, eye_imgs, vid_frames = [], [], []
                prev_action = None

                for i, (st, act) in enumerate(zip(orig_states, orig_actions)):
                    if is_noop(act, prev_action):
                        prev_action = act
                        continue
                    keep_idx.append(i)
                    prev_action = act

                    env.sim.set_state_from_flattened(st)
                    for addr, delta in delta_spec:
                        env.sim.data.qpos[addr] += delta
                    env.sim.forward(); env._post_process(); env._update_observables(force=True)
                    obs = env._get_observations()

                    # Store original states and actions
                    states.append(st)
                    actions.append(act)

                    # Extract observation data for RoboCerebra (LIBERO-compatible format)
                    # 1. gripper_states (2D)
                    if "robot0_gripper_qpos" in obs:
                        gripper_states.append(obs["robot0_gripper_qpos"])
                    else:
                        gripper_states.append(np.zeros(2))  # Fallback
                    
                    # 2. joint_states (7D joint positions)
                    joint_states.append(obs["robot0_joint_pos"])
                    
                    # 3. ee_pos (3D end-effector position)
                    ee_pos.append(obs["robot0_eef_pos"])
                    
                    # 4. ee_ori (3D end-effector orientation in axis-angle)
                    ee_ori.append(T.quat2axisangle(obs["robot0_eef_quat"]))
                    
                    # 5. ee_states (6D: position + orientation)
                    ee_states.append(np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]))))
                    
                    # 6. robot_states (9D) - same computation as original script
                    robot_states.append(
                        np.concatenate([
                            obs.get("robot0_gripper_qpos", np.zeros(2)),  # 2D gripper positions
                            obs["robot0_eef_pos"],                        # 3D end-effector position  
                            obs["robot0_eef_quat"]                        # 4D quaternion
                        ])
                    )
                    
                    # 7. Image data
                    agent_imgs.append(obs["agentview_image"])
                    eye_imgs.append(obs["robot0_eye_in_hand_image"])
                    vid_frames.append(obs["agentview_image"][::-1])  # flip vertical

                # Convert to numpy arrays
                states, actions = np.stack(states), np.stack(actions)
                gripper_states, joint_states = map(np.stack, (gripper_states, joint_states))
                ee_pos, ee_ori, ee_states = map(np.stack, (ee_pos, ee_ori, ee_states))
                robot_states = np.stack(robot_states)
                agent_imgs, eye_imgs = map(np.stack, (agent_imgs, eye_imgs))
                dones   = np.zeros(len(actions), dtype=np.uint8); dones[-1]   = 1
                rewards = np.zeros(len(actions), dtype=np.uint8); rewards[-1] = 1

                # Map old end_frame to new end_frame
                new_step_ends = [bisect.bisect_right(keep_idx, old_end)
                                 for _, old_end in steps]

                prev_end = 0
                for (desc, _), new_end in zip(steps, new_step_ends):
                    step_name = desc.replace(" ", "_")
                    idx = slice(prev_end, new_end)

                    step_dir = per_step_root / case_name / step_name
                    step_dir.mkdir(parents=True, exist_ok=True)
                    h5_path  = step_dir / f"{step_name}_{var_full}_{case_name}.hdf5"
                    mp4_path = step_dir / f"{step_name}_{var_full}.mp4"

                    # Write HDF5 (RoboCerebra format, compatible with LIBERO)
                    with h5py.File(h5_path, "w") as hf:
                        grp = hf.create_group("data").create_group("demo_0")
                        og = grp.create_group("obs")
                        
                        # Observation group data - RoboCerebra format (LIBERO-compatible)
                        og.create_dataset("gripper_states", data=gripper_states[idx])
                        og.create_dataset("joint_states",   data=joint_states[idx])
                        og.create_dataset("ee_pos",         data=ee_pos[idx])
                        og.create_dataset("ee_ori",         data=ee_ori[idx])
                        og.create_dataset("ee_states",      data=ee_states[idx])
                        og.create_dataset("agentview_rgb",  data=agent_imgs[idx])
                        og.create_dataset("eye_in_hand_rgb",data=eye_imgs[idx])

                        # Demo group data
                        grp.create_dataset("actions",      data=actions[idx])
                        grp.create_dataset("states",       data=states[idx])
                        grp.create_dataset("robot_states", data=robot_states[idx])  # Robot states for RoboCerebra
                        grp.create_dataset("rewards",      data=rewards[idx])
                        grp.create_dataset("dones",        data=dones[idx])

                    # Write video
                    imageio.mimsave(mp4_path, vid_frames[prev_end:new_end], fps=30)

                    # Copy to flattened all_hdf5 directory
                    flat_name = f"{step_name}_{var_full}_{case_name}.hdf5"
                    shutil.copy(h5_path, all_hdf5_root / flat_name)

                    # Update metainfo
                    meta_key = f"{case_name}/{step_name}_{var_full}"
                    metainfo.setdefault(meta_key, {})["demo_0"] = {
                        "success": False,
                        "initial_state": states[prev_end].tolist(),
                    }

                    prev_end = new_end

    # Save metainfo
    metainfo_path.write_text(json.dumps(metainfo, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n✅  Finished")
    print("Output root :", args.robocerebra_target_dir)
    print("Metainfo    :", metainfo_path)


# --------------------------------------------------------------------------- #
#                                  CLI                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RoboCerebra dataset to HDF5 format with automatic scene detection")
    parser.add_argument("--dataset_name", required=True, 
                       help="Output dataset name")
    parser.add_argument("--robocerebra_raw_data_dir", required=True,
                       help="Path to raw RoboCerebra task directory (e.g., <ROBOCEREBRA_BENCH_PATH>/Random_Disturbance)")
    parser.add_argument("--robocerebra_target_dir", required=True,
                       help="Output directory for converted HDF5 files")
    parser.add_argument("--scene", required=False, choices=list(SCENES),
                       help="Optional: Override auto-detected scene type. Available: %(choices)s")
    main(parser.parse_args())