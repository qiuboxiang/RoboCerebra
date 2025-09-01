import os
# 跳过 GCE 元数据服务器检查，避免 TFDS 在没有凭证时长时间等待
os.environ['NO_GCE_CHECK'] = 'true'

# 禁用 TFDS 从 GCS 初始化 DatasetInfo 的全局行为
try:
    from tensorflow_datasets.core.utils import gcs_utils
    gcs_utils._is_gcs_disabled = True
except ImportError:
    pass

from typing import Iterator, Tuple, Any
import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from RoboCerebraDataset.conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # 每个 worker 各自创建模型，避免死锁
    def _parse_example(episode_path, demo_id):
        with h5py.File(episode_path, "r") as F:
            if f"demo_{demo_id}" not in F['data'].keys():
                return None
            actions = F['data'][f"demo_{demo_id}"]["actions"][()]
            states = F['data'][f"demo_{demo_id}"]["obs"]["ee_states"][()]
            gripper_states = F['data'][f"demo_{demo_id}"]["obs"]["gripper_states"][()]
            joint_states = F['data'][f"demo_{demo_id}"]["obs"]["joint_states"][()]
            images = F['data'][f"demo_{demo_id}"]["obs"]["agentview_rgb"][()]
            wrist_images = F['data'][f"demo_{demo_id}"]["obs"]["eye_in_hand_rgb"][()]

        raw_file_string = os.path.basename(episode_path)
        # 去掉后缀，比如 ".h5"
        name = raw_file_string[:-5]
        # 2. 按 "_" 分割成列表
        parts = name.split("_")
        # 3. 最后两段永远是 [variant, "tex0"]，截掉它们
        instr_words = parts[:-3]

        # 4. 如果最后一个 instr word 以 "." 结尾，就去掉它
        if instr_words and instr_words[-1].endswith('.'):
            instr_words[-1] = instr_words[-1][:-1]

        # 4. 合成指令
        instruction = " ".join(instr_words)
        print('instruction:', instruction)

        command = instruction
        # for w in instruction:
        #     if "SCENE" in w:
        #         command = ''
        #         continue
        #     command += w + ' '
        # command = command.strip()
        # print('commmand:', command)

        episode = []
        for i in range(actions.shape[0]):
            episode.append({
                'observation': {
                    'image': images[i][::-1, ::-1],
                    'wrist_image': wrist_images[i][::-1, ::-1],
                    'state': np.asarray(
                        np.concatenate((states[i], gripper_states[i]), axis=-1), np.float32),
                    'joint_state': np.asarray(joint_states[i], dtype=np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),
                'discount': 1.0,
                'reward': float(i == (actions.shape[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': command,
            })

        sample = {
            'steps': episode,
            'episode_metadata': {'file_path': episode_path}
        }
        return episode_path + f"_{demo_id}", sample

    for sample in paths:
        with h5py.File(sample, "r") as F:
            n_demos = len(F['data'])
        idx = 0
        cnt = 0
        while cnt < n_demos:
            ret = _parse_example(sample, idx)
            if ret is not None:
                cnt += 1
            idx += 1
            yield ret


class RoboCerebraDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for LIBERO10 dataset."""
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release.'}

    N_WORKERS = 40
    MAX_PATHS_IN_MEMORY = 80
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format='jpeg'),
                        'wrist_image': tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format='jpeg'),
                        'state': tfds.features.Tensor(shape=(8,), dtype=np.float32),
                        'joint_state': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    }),
                    'action': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'discount': tfds.features.Scalar(dtype=np.float32),
                    'reward': tfds.features.Scalar(dtype=np.float32),
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_last': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                    'language_instruction': tfds.features.Text(),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(),
                }),
            })
        )

    def _split_paths(self):
        return {
            'train': glob.glob('<CONVERTED_HDF5_PATH>/robocerebra_ideal/all_hdf5/*.hdf5'),  # TODO: Set converted HDF5 path
        }
