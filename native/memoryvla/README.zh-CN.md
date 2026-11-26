# MemoryVLA × SimplerEnv-Bridge

MemoryVLA 原版（[shihao1895/MemoryVLA](https://github.com/shihao1895/MemoryVLA) 的 `openvla-codebase` 分支，ICLR 2026）接入平台，用官方发布的 `shihao1895/memvla-bridge` 检查点在 SimplerEnv-Bridge 四个 WidowX 抓放任务上闭环评测。模型直接看相机 RGB 图像，每一步都请求一次模型（自适应动作集成，窗口 7），每个回合开始时清空记忆。

| 文件 | 作用 |
|---|---|
| `recipes.json` | 配方 `memoryvla-simpler-bridge-eval`：参数与官方 `script/eval/bridge/eval_bridge.sh` 相同 |
| `prepare_server.sh` | 在服务器上下载固定版本的源码、建环境、下载权重、写 `runtime.json`、做导入检查 |
| `requirements-inference.txt` | 推理所需依赖，版本取自 MemoryVLA 的 `pyproject.toml` |
| `runtime.example.json` | 机器配置模板；服务器上的 `runtime.json` 由 `prepare_server.sh runtime` 生成，不入库 |
| `../../src/wms/engines/memoryvla/` | 任务入口 `native.py`、评测 `simpler.py`、独立评价 `metrics.py` |

## 固定版本

| 组件 | 版本 |
|---|---|
| MemoryVLA | `d732ea9072bc063399ccc817aed74ab172eb50be`（openvla-codebase） |
| SimplerEnv | `eddb5690749a1e34267807c0f7b5eede323918d7` |
| ManiSkill2_real2sim | `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682`（SimplerEnv 在该提交指向的子模块） |
| dlimp_openvla | `040105d256bd28866cc6620621a3d5f7b6b91b46` |
| 检查点 `shihao1895/memvla-bridge` | `82b2b6196e331270c80e2b2e853ecf18635b6b53`（`checkpoints/memvla-bridge.pt`，33.5 GB） |
| Llama-2 配置和分词器 | `NousResearch/Llama-2-7b-hf`（`meta-llama/Llama-2-7b-hf` 的公开副本；权重来自 MemoryVLA 检查点） |
| Python / torch | 3.10 / 2.2.0（CUDA 12.1 版） |

## 在服务器上准备

```bash
cd ~/world-model/deployments/wm-safetylab
export WMS_ROOT=$HOME/world-model && source scripts/linux/use-linux.sh
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple UV_CONCURRENT_DOWNLOADS=3 UV_HTTP_TIMEOUT=90 \
  bash native/memoryvla/prepare_server.sh all       # 也可分步：sources env checkpoint weights runtime check
```

服务器不能直连 GitHub 和 Hugging Face：源码用 `codeload.github.com` 的压缩包，权重和视觉骨干用 `HF_ENDPOINT=https://hf-mirror.com`。评测时 `HF_HUB_OFFLINE=1`，不再联网。

## 运行

服务启动时才读取新配方。服务未重启时，可以提交完整配置：

```bash
.venv/bin/python -m wms run --config <完整配置.json> --wait
```

先跑冒烟测试（每个任务 1 回合）：`params.episodes = [0, 1]`。正式复现用默认的 `[0, 24]`，四个任务共 96 回合。

## 与官方脚本的差异

这些差异会写进每次运行的 `evaluation.json`：

- 默认 bf16 权重（官方 Bridge 脚本用 fp32，需要约 30 GB 显存；RTX 4090 只有 24 GB）。官方代码本身支持 `--use_bf16`。
- 模型对四个任务只加载一次（官方每个任务加载一次）。
- 每个回合固定随机种子（官方不设种子，diffusion 采样每次不同）。
- Llama-2 的配置和分词器从本地目录读取。
- `VK_ICD_FILENAMES` 由运行配置给出（官方脚本写死为 `/usr/share/vulkan/icd.d/nvidia_icd.json`，本服务器在 `/etc/vulkan/icd.d/`）；TensorFlow 不占 GPU。

## 复现结果（2026-10-15，stair-4090-server，RTX 4090）

正式运行 `run_96c428cdd0e04ae6`（自动评价 `run_54b322f345334232`），每任务 24 回合，bf16，种子 0：

| 任务 | 本平台 | 官方 | 差 |
|---|---|---|---|
| 勺子放到毛巾上 | 75.0（18/24） | 75.0 | 0 |
| 胡萝卜放到盘子上 | 83.3（20/24） | 75.0 | +8.3（2 回合） |
| 绿方块叠到黄方块上 | 33.3（8/24） | 37.5 | −4.2（1 回合） |
| 茄子放进篮子 | 100.0（24/24） | 100.0 | 0 |
| **平均** | **72.9** | **71.9** | **+1.0** |

模型加载 101 秒，显存峰值 15.7 GB，96 回合共 51 分钟（每回合约 30 秒）。冒烟运行 `run_9752af1b33f94fdb`（每任务 1 回合）。安装日志与实际安装的包版本在 `results/memoryvla-integration-20261015/`（不入库）。

安装中遇到并已写进脚本的问题：服务器解析 `files.pythonhosted.org` 不稳定，改用 `UV_INDEX_URL`（清华源；中科大源并发下载会返回 429）；Python 3.10 下 `tensorflow-metadata` 新版需要 protobuf 5，需固定 `protobuf<4.21` 与 `tensorflow-metadata==1.14.0`；uv 建的环境没有 setuptools（`pkg_resources`）。

## 结果怎么看

官方成绩（README，每任务 24 回合）：勺子 75.0、胡萝卜 75.0、方块 37.5、茄子 100.0，平均 71.9。每任务只有 24 回合，一个回合就是 4.2 个百分点；官方建议每 2.5k 步评测一个检查点并取最好成绩，而这里只有发布的一个检查点。因此验收看四任务平均，单个任务差 ±10 个百分点以内都属正常波动。

`evaluation.json` 有逐任务成功率、与官方的差值、版本和差异清单；`episodes.jsonl` 每回合一行；`videos/` 是每回合的录像和动作曲线。自动评价 `memoryvla_simpler` 从 `episodes.jsonl` 重新统计成功率，与 `evaluation.json` 不一致时报错。

## 在主平台页面运行配对实验

服务启动时，将 `--native-configs` 对应 JSON 的 `memoryvla` 键指向上述机器 runtime 文件。页面的 `MemoryVLA` 与 `MemoryVLA（关记忆）` 使用同一权重；后者每步清空记忆，是消融选项。任务选择 SimplerEnv-Bridge 四任务之一，不能直接沿用 TD-MPC2 的 PushCube 状态策略环境。

胡萝卜放盘可选种子 0–9、每回合 120 步，添加 `image-perturbation` 的固定遮挡：

```json
{"schedule":[{"type":"occlude","start":10,"end":40,"camera":"3rd_view_camera","box":[0.25,0.25,0.75,0.75],"value":0}]}
```

仅勾选单独扰动组且关闭“监控挂到所有组”，得到完全无插件 A0 和仅图像遮挡 A1。遮挡覆盖 t=10..39 的中心 25% 面积，不读取物体真值。该设置用于输入扰动对照，不代表最优攻击或记忆有效性的消融证明。

实验详情按相同环境和种子匹配基线，播放实际保存的 `3rd_view_camera` 图像。普通帧是 S1 处理后送入策略的图像；T+1 末帧只是动作后的记录。平台图像实验同时保存 `raw_3rd_view_camera.npz`，用于核查扰动前图像。`policy_info` 记录执行器、相机、记忆开关、是否重置和两类记忆库长度。

Simpler 的 `success@120` 表示最后评价步成功，不是首次到达目标。默认环境在第 120 步评价；仅缩短内核 `max_steps` 会提前截断，不应把这样的冒烟回合作为正式成功率。查看回放时以最后一步的任务完成标志为准，目标距离不是任务成功判据。
