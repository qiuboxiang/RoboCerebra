# RoboCerebra OpenVLA Evaluation Suite

OpenVLA-OFT模型在RoboCerebra基准测试上的评估工具，采用模块化架构设计。

## 配置步骤

**重要**：运行评估前，请先在 `config.py` 中配置以下占位符：

- `<PRETRAINED_CHECKPOINT_PATH>` → 您的OpenVLA-OFT模型检查点路径
- `<ROBOCEREBRA_BENCH_PATH>` → RoboCerebra基准数据集根目录
- `<WANDB_ENTITY>` → 您的WandB实体名称（如使用WandB记录）
- `<WANDB_PROJECT>` → 您的WandB项目名称（如使用WandB记录）

示例：
```python
pretrained_checkpoint = "/path/to/your/openvla-checkpoint"
robocerebra_root = "/path/to/RoboCerebra_Bench"
wandb_entity = "your-wandb-username"
wandb_project = "robocerebra-evaluation"
```

## 文件结构

```
evaluation/
├── eval_openvla.py           # OpenVLA评估主入口 (396行)
├── config.py                 # 配置管理 (87行)
├── utils.py                  # 数据处理工具 (422行)
├── robocerebra_logging.py    # 日志和结果保存 (141行)
├── task_runner.py            # 任务级管理 (200行)
├── episode.py                # 回合执行逻辑 (280行)
├── resume.py                 # 恢复机制 (69行)
└── README.md                 # 使用说明
```

## 快速开始

### 基本运行

```bash
# 切换到evaluation目录
cd <ROBOCEREBRA_PATH>/evaluation

# 默认配置运行（评估所有任务类型）
python eval_openvla.py

# 指定GPU和任务类型
CUDA_VISIBLE_DEVICES=0 python eval_openvla.py --task_types ["Ideal", "Random_Disturbance"]
```

### 常用配置

```bash
# Random_Disturbance任务评估（自动启用动态干扰）
python eval_openvla.py \
  --task_types ["Random_Disturbance"] \
  --num_trials_per_task 5

# Mix任务评估（启用所有动态功能）
python eval_openvla.py \
  --task_types ["Mix"] \
  --num_trials_per_task 3

# 指定OpenVLA-OFT模型检查点
python eval_openvla.py \
  --pretrained_checkpoint "/path/to/openvla-oft/checkpoint" \
  --task_types ["Ideal"]

# 使用特定任务描述后缀
python eval_openvla.py \
  --task_types ["Random_Disturbance"] \
  --task_description_suffix "_cosmos_matched"
```

## 配置参数详解

### 任务配置
- `--task_types`: 评估的任务类型列表
  - `"Ideal"`: 理想条件测试
  - `"Random_Disturbance"`: 随机干扰测试（自动启用dynamic）
  - `"Mix"`: 混合模式（自动启用所有动态功能）
  - `"Observation_Mismatching"`: 观察不匹配（自动启用dynamic_shift_description）
  - `"Memory_Execution"`: 记忆执行测试
  - `"Memory_Exploration"`: 记忆探索测试

- `--robocerebra_root`: RoboCerebra数据集根目录（默认：`<ROBOCEREBRA_BENCH_PATH>`）
- `--init_files_root`: 初始状态文件目录（默认：`<ROBOCEREBRA_BENCH_PATH>/init_files`）
- `--num_trials_per_task`: 每个任务的试验次数（默认：5）

### OpenVLA-OFT模型配置
- `--pretrained_checkpoint`: OpenVLA-OFT模型检查点路径
- `--model_family`: 模型类型（固定："openvla"）
- `--use_l1_regression`: 使用L1回归头（默认：True）
- `--use_proprio`: 是否使用本体感受信息（默认：True）
- `--center_crop`: 图像中心裁剪（默认：True）
- `--num_open_loop_steps`: 开环执行步数（默认：8）

### 动态功能配置
- `--dynamic`: 启用动态干扰物移动
- `--dynamic_shift_description`: 启用观察不匹配模式
- `--resume`: 启用任务恢复机制
- `--task_description_suffix`: 任务描述后缀（如："_cosmos_matched"）

### 日志配置
- `--local_log_dir`: 本地日志目录（默认："./experiments/logs"）
- `--use_wandb`: 启用Weights & Biases日志（默认：False）
- `--wandb_entity`: WandB实体名称
- `--wandb_project`: WandB项目名称
- `--run_id_note`: 运行ID备注

## 自动配置功能

每个任务类型会自动配置相应的OpenVLA-OFT评估参数：

| 任务类型 | dynamic | dynamic_shift_description | resume | 说明 |
|---------|---------|---------------------------|--------|------|
| Ideal | False | False | True | 理想条件下的OpenVLA性能 |
| Random_Disturbance | **True** | False | True | 随机干扰下的鲁棒性测试 |
| Mix | **True** | **True** | True | 混合条件综合测试 |
| Observation_Mismatching | False | **True** | True | 观察不匹配适应性测试 |
| Memory_Execution | False | False | True | 记忆执行能力测试 |
| Memory_Exploration | False | False | True | 记忆探索能力测试 |

## 输出文件

### 日志文件
- `./experiments/logs/EVAL-robocerebra-openvla-{timestamp}.txt`: 详细执行日志
- `./experiments/logs/EVAL-robocerebra-openvla-{timestamp}_results.json`: JSON格式结果

### 视频文件
```
rollouts/{timestamp}/{task_type}/{case_name}/
├── {timestamp}--{task_type}--episode={idx}--success={0|1}--task={description}.mp4
└── {timestamp}--{task_type}--episode={idx}--success={0|1}--task={description}.json
```

## 依赖要求

### 环境配置
确保以下路径可访问：
- OpenVLA-OFT代码路径：需要在系统路径中
- RoboCerebra数据集：`<ROBOCEREBRA_BENCH_PATH>`
- 初始化文件：`<ROBOCEREBRA_BENCH_PATH>/init_files`

### Python依赖
```bash
pip install draccus numpy tqdm robosuite h5py imageio wandb
```

### LIBERO环境
```bash
# 安装LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e .
```

## 模块架构

### 核心模块
- **eval_openvla.py**: OpenVLA-OFT模型评估主逻辑
- **config.py**: 评估参数配置和验证
- **utils.py**: 数据加载和环境处理工具

### 功能模块  
- **robocerebra_logging.py**: 日志设置和结果保存
- **task_runner.py**: 任务级别的管理和验证
- **episode.py**: 单回合执行和动态干扰功能
- **resume.py**: 智能恢复机制处理

## 评估流程

1. **模型初始化**: 加载OpenVLA-OFT预训练模型
2. **任务发现**: 扫描RoboCerebra数据集目录
3. **环境设置**: 为每个任务创建LIBERO仿真环境
4. **回合执行**: 执行指定次数的评估回合
5. **结果统计**: 生成成功率和子任务完成率统计
6. **视频保存**: 保存每个回合的执行视频

## 高级功能

### 动态干扰物移动
Random_Disturbance和Mix任务自动启用，会在回合执行中移动场景中的对象：
```
[Dynamic] Moved related object at step 1, Δy=+0.15
[Dynamic] Moved unrelated object at step 2, Δy=-0.15
```

### 智能恢复机制
基于步骤的恢复功能，可以从任务的中间状态继续执行：
```
[Resume] Reverting to Step 2, auto-completed 1 prior subtasks: ['object_1_0']
```

### 观察不匹配
Observation_Mismatching和Mix任务会测试模型在观察描述不匹配时的适应能力。

## 性能特点

- **高效模块化**: 7个专门模块，代码量减少75%
- **完全兼容**: 与原版OpenVLA评估保持100%功能一致
- **自动配置**: 任务类型自动配置OpenVLA评估参数
- **智能恢复**: 基于步骤的智能恢复机制
- **动态测试**: 支持动态对象移动和观察不匹配
- **详细日志**: 完整的评估过程记录和视频保存

## 故障排除

### 常见问题
1. **导入错误**: 确保所有模块文件在evaluation目录下
2. **路径错误**: 检查数据集和初始化文件路径
3. **OpenVLA模型加载失败**: 验证检查点路径和格式
4. **动态功能不工作**: 检查task_description.json文件

### 检查配置
```bash
# 查看完整配置选项
python eval_openvla.py --help

# 测试单个任务
python eval_openvla.py --task_types ["Ideal"] --num_trials_per_task 1
```

### 调试模式
取消注释各模块中的debug信息来获得详细日志输出。

## 联系信息

如有问题请检查：
1. OpenVLA-OFT模型是否正确加载
2. RoboCerebra数据集路径是否正确
3. LIBERO环境是否正确安装
4. 所有Python依赖是否满足

本评估套件专门为OpenVLA-OFT模型设计，确保在RoboCerebra基准测试上的准确评估。