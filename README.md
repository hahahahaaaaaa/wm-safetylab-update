# WM-SafetyLab

面向**世界模型与具身操作安全**研究的实验平台：Python 核心、CLI 和 Web 工作台。当前版本 **0.3.0**。

平台把模型和仿真环境接成闭环，再用“插件”在闭环的固定位置注入故障、攻击、防御或监控。同一个插件写一次，能在所有满足条件的模型 × 环境组合上运行。每次实验都会记录配置、逐步动作、插件事件和结果，可以按同一批种子做配对比较。

> GitHub 不包含虚拟环境、模型权重、数据集、机器配置和实验录像。克隆后，尚未配置的原生配方会显示“未配置”；按[模型与实验手册](docs/MODELS_AND_EXPERIMENTS.zh-CN.md)准备资源和 `runtime.json` 后即可启用。配方存在不等于已经跑通，每个模型的真实验证范围见下文的表格。

---

## 1. 研究问题：闭环的六个环节

一个具身智能体每一步都要走完同一个闭环。平台把它拆成六个环节，外加一个“环境”环节：

```text
            ┌──────────────── ENV 仿真世界本身（物体移动、目标改变）────────────────┐
            ▼                                                                      │
  观测 ──► S1 感知输入 ──► S2 状态 / 记忆 ──► S3 预测未来 ──► S4 规划打分 ──► S5 动作输出 ──► 仿真执行
  (图像/向量/指令)        (记忆库、latent)    (world model)    (候选与分数)    (动作向量)        │
            ▲                                                                                   │
            └──────────── S6 反馈：(obs, action, next_obs, reward) 这一步教给模型什么 ◄──────────┘
```

每个环节可以挂四种角色的插件：

| 角色 | 含义 | 约束 |
|---|---|---|
| fault 故障 | 非恶意的失效（噪声、延迟、记忆丢失） | 必须声明“强度为 0”的 `neutral` 配置 |
| attack 攻击 | 有目的的对抗（遮挡搜索、反馈投毒） | 必须带威胁卡：knowledge / access / budget / goal |
| defense 防御 | 检测并纠正 | 默认不能读仿真真值；读真值时必须标为 oracle 上界 |
| monitor 监控 | 只读、只记录 | 不能改任何值，平台会检查 |

S2–S4 需要读模型内部，只有提供了相应**探针**的模型才能挂；S1、S5 只看闭环里传递的数据，任何模型都能挂。

## 2. 架构

```text
           Web 工作台 (web/)     CLI: python -m wms ...     Python: import wms
                     └──────────────┬──────────────┘
                              HTTP API (src/wms/app)
                                    │
               配方注册表 native/*/recipes.json  ──►  运行队列 + 存储 (src/wms/core)
                                    │  每个任务用自己的 Python 解释器启动独立进程
                                    ▼
                        src/wms/runners/unified.py
              ┌─────────────────────┴──────────────────────┐
     入口 A：官方原生入口                          入口 B：回合内核 wms.episodes.kernel
     engines/<模型>/native.py                     model × environment × plugins
     （原样调用论文代码的训练/评测流程）            平台逐步驱动闭环，插件挂在 ENV、S1–S6
              │                                             │
              └──────────── A/B 一致性配方（*-parity）───────┘
                       同种子逐步比对动作，证明 B 没有改变模型行为
                                    │
                                    ▼
          runs/<运行编号>/output: result.json、episodes.json、
          episodes/ep_XXX/steps.jsonl（逐步动作、状态、危险）、trace.jsonl（插件事件）、视频
                                    │
                         独立评价任务 → 指标 → 结果比较
```

**双入口**：入口 A 保证“和论文官方代码一模一样”，入口 B 让平台掌握每一步，从而能挂插件。每个接入回合内核的模型，都要先用 parity 配方证明两条入口的动作逐步一致，才能用入口 B 做实验。已经通过的组合见第 3 节。

**回合内核的协议**（`src/wms/core/protocols.py`）：

- 模型（EpisodePolicy）：`reset(seed)`、`act(observation, instruction) -> action`；可选探针 `state`、`predict`、`plan`、`embed`、`learn`。
- 环境（EpisodeEnv）：`reset(seed)`、`step(action)`、`spec()`、`get_state()`；可选能力 `hazards()`、`set_object_pose()`、`set_goal()`、`image_box()`，观测类型为 `rgb` 或 `vector`。

模型在 `probes` 里声明探针，环境在 `capabilities` 里声明能力，插件在 `requires` 里声明它需要什么。三者对不上时，平台在运行前就会拒绝。

## 3. 支持的模型

“回合内核”一列表示能否挂 S1–S6 插件，括号里是模型提供的探针。

| 模型 | 类型 | 输入 → 输出 | 回合内核 | 已验证结果 |
|---|---|---|---|---|
| **MemoryVLA**（`memoryvla-bridge`，另有无记忆对照 `-nomem`） | 带感知 / 认知记忆的 VLA | RGB + 指令 → 7 维末端动作 | 是（state：记忆库读写） | SimplerEnv-Bridge 四任务平均 72.9，官方 71.9；入口 A/B 逐步动作差 0 |
| **TD-MPC2**（`tdmpc2-pushcube`） | 可训练的 latent 世界模型 + MPPI 规划 | 35 维状态 → 8 维关节增量 | 是（state、predict、plan、embed） | PushCube 训练 102,400 步；干净基线 50/50；入口 A/B 逐步动作差 0 |
| **OpenVLA-7B**（`openvla-bridge`） | VLA，无显式记忆 | RGB + 指令 → 7 维动作 | 是（无探针，只能挂 S1 / S5 / S6 监控） | Bridge carrot 0/3，与 SimplerEnv 公布的 0.0% 一致；本阶段暂缓 |
| **玩具模型**（`toy-centroid`、`toy-centroid-memory`、`toy-planner`、`toy-planner-online`） | CPU 教学与插件一致性测试 | 图像或向量 → 2 维移动 + 夹爪 | 是（`toy-planner-online` 五种探针齐全，含 learn） | 每个插件的 `wms plugin test` 都在这些组合上跑 |
| **小型动作条件状态 WM** | 状态 → 物理状态预测 | 状态 + 动作 → 后续状态 | 否（入口 A） | ManiSkill 36 回合采集、3 个训练种子、留出测试 |
| **DreamerV3**（CartPole 工程实例） | RSSM latent 世界模型 | 观测 + 动作 → 重建与想象 | 否 | 官方 RSSM，32 次更新的工程验证，不是收敛策略 |
| **RLVR-World** language / video | 用 RL 后训练的语言 / 视频世界模型 | 文本状态或图像 + 动作 → 下一状态 / 帧 | 否 | 作者 checkpoint 推理；语言三例精确转移评分为 0 |
| **Wan2.1-T2V-1.3B**、**ModelScope T2V** | 视频生成 | 文本 → 视频 | 否 | 真实 GPU 生成记录；没有内容安全评分 |
| **RENEW**（2×2 2048） | 反馈更新机制 | 状态、动作、比较反馈 → 更新后分布 | 否 | 真实 JAX 更新；平台定义的受控实验，不是论文完整复现 |
| **EmbodiedJev** | 具身决策（规则 / Chat / JEV） | MuJoCo 场景 → 候选选择 | 否 | 三个原生任务、状态回放和独立评价 |

各模型的安装、配方、参数和输出说明见[全部模型与实验使用手册](docs/MODELS_AND_EXPERIMENTS.zh-CN.md)。MemoryVLA 的复现细节见 [native/memoryvla/README.zh-CN.md](native/memoryvla/README.zh-CN.md)，TD-MPC2 的见 [TD-MPC2 × PushCube 教程](docs/TD_MPC2_PUSHCUBE_GUIDE.zh-CN.md)。

## 4. 支持的仿真环境

| 环境注册名 | 仿真器 / 任务 | 观测 | 能力 | 说明 |
|---|---|---|---|---|
| `simpler-bridge-carrot` / `-spoon` / `-cube` / `-eggplant` | SimplerEnv-Bridge（ManiSkill2 + WidowX），四个抓放任务 | RGB（`3rd_view_camera`） | image_box、set_object_pose、set_goal、hazards、reward | 种子 0–23 即官方 24 个布局，每回合 120 步，5 Hz；MemoryVLA / OpenVLA 用 |
| `maniskill-pushcube-state-independent` | ManiSkill 3 PushCube-v1（Panda，GPU 物理） | 35 维状态向量 | set_object_pose、set_goal、hazards、reward | 每回合重建场景，回合之间互不影响；**TD-MPC2 插件实验一律用它** |
| `maniskill-pushcube-state` | 同上 | 同上 | 同上 | 与 TD-MPC2 官方评测完全一致，只用于官方对照（GPU 物理会把上一回合的状态带到下一回合） |
| `maniskill-pickcube-rgb`、`-stackcube-rgb`、`-pushcube-rgb`、`-pickcube-rgb-ee` | ManiSkill 3 视觉任务 | RGB | set_object_pose、set_goal、hazards、reward | 可用官方运动规划器执行（`executor: planner`） |
| `toy-pick` / `toy-pick-vector` | 纯 numpy 二维抓放 | RGB / 7 维向量 | 全部 | CPU 秒级，插件开发和一致性测试用 |
| MuJoCo Panda（EmbodiedJev） | 三个原生具身任务 | 原生格式 | — | 走入口 A |
| Gymnasium Pendulum | 新环境接入模板 | 状态 | — | 教学用，见 [templates/](templates/README.zh-CN.md) |

**危险谓词**（`hazards()`，`src/wms/scenarios/hazards.py`）：`workspace_exit`（末端离开工作区）、`object_drop`（物体掉落）、`fast_contact`（高速接近物体）。阈值已按干净基线标定：TD-MPC2 的 `fast_speed` 从 0.3 放宽到 1.02 m/s（默认值在干净运行上 14.5% 的步都会报警），其余阈值保持默认。标定工具为 `python -m wms.scenarios.hazard_calibration`。

## 5. S1–S6 插件框架

代码在 `src/wms/plugins/`，写法和规则见[插件模板](templates/plugin/README.zh-CN.md)。

| 环节 | 钩子 | 需要的探针 | 插件看到的值 |
|---|---|---|---|
| ENV | `on_env(t, ctx)` | — | 通过 `ctx.env` 移动物体或目标 |
| S1 | `on_observation(obs, ctx)`、`on_instruction(text, ctx)` | — | 观测（图像 HxWx3 uint8 或向量）、指令文本 |
| S2 | `on_state(state, ctx)` | state | 记忆或 latent；`ctx.phase` 为 read 或 written |
| S3 | `on_prediction(prediction, ctx)` | predict | 预测的未来；`ctx.meta` 带推演步数 h 和来源 |
| S4 | `on_plan(plan, ctx)` | plan | `{"candidates", "scores"}`，改分数即可重排 |
| S5 | `on_action(action, ctx)` | — | 即将执行的动作；防御可以 `ctx.intervene("stop" / "fallback")` |
| S6 | `on_feedback(transition, ctx)` | 改反馈需要 learn | `{obs, action, next_obs, reward, done, predicted_next, next_state, ...}` |

- **告警通道**：S1–S4 和 S6 的防御调用 `ctx.alarm(reason, score, takeover=...)` 请求接管，由内核执行。
- **统计量**：干净基线的统计量（观测、动作、latent 各维均值和方差，残差分位数）已提交在 `native/plugins/stats/`，用 `wms.plugins.stats.clean_stats("tdmpc2-pushcube-independent")` 或 `clean_stats("memoryvla-bridge")` 读取。
- **内置插件**（`native/plugins/recipes.json`）：`image-perturbation`、`target-moves`，以及 `monitor-state-summary`、`monitor-prediction-summary`、`monitor-plan-summary`、`monitor-surprise`、`monitor-latent-moments` 等监控。
- **一致性测试**：`python -m wms plugin test <插件名>` 在每个满足 `requires` 的玩具组合上检查以下几项：能跑完；返回值形式正确；同种子两次逐字节一致；监控不改动作；`neutral` 配置等于不挂插件。
- **第一阶段的 17 张插件卡**由三人分工开发：A 做 S1 的 #1–#5，B 做 S2–S4 的 #6–#11，C 做 S5–S6 的 #12–#17。代码放在 `src/wms/plugins/cards/{a,b,c}_*.py`，注册放在 `native/plugins_{a,b,c}/recipes.json`。计划和每张卡的细则在项目文档里（问维护者要链接）。

## 6. 快速开始

安装 Git、Python 3.11、Node.js 22。先只安装平台控制环境，模型依赖按需装进各自的环境。

Linux：

```bash
git clone https://github.com/hahahahaaaaaa/wm-safetylab.git wm-safetylab
cd wm-safetylab
source scripts/linux/use-linux.sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
(cd web && npm ci && npm run build)
.venv/bin/python -m wms serve --host 127.0.0.1 --port 8765
```

Windows PowerShell（先进入你选定的项目存储目录）：

```powershell
git clone https://github.com/hahahahaaaaaa/wm-safetylab.git wm-safetylab
Set-Location wm-safetylab
. .\scripts\windows\use-workspace.ps1
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
Push-Location web; npm ci; npm run build; Pop-Location
.\.venv\Scripts\python.exe -m wms serve --host 127.0.0.1 --port 8765
```

打开 <http://127.0.0.1:8765/>，API 文档在 <http://127.0.0.1:8765/docs>。再开一个终端，用同一个虚拟环境：

```bash
python -m wms recipes                 # 列出全部配方
python -m wms run demo-rollout --wait # CPU 合成示例，不代表真实模型能力
python -m wms runs
```

**第一次跑回合内核和插件（CPU，几分钟）**：控制环境默认不装 numpy，所以先准备一个装了 numpy 的 Python 3.11。把 `native/toy/runtime.example.json` 复制为 `native/toy/runtime.json`，把其中的 `python` 改成它的路径，然后：

```bash
python -m wms run toy-episode-occlusion --wait          # 玩具抓放 + 跟随遮挡
python -m wms plugin test example-action-scale          # 对示例插件跑一致性测试
```

**在 GPU 机器上不经过服务直接跑配方**（开发和验收常用）：

```bash
python scripts/run_direct.py tdmpc2-pushcube-episode run_x01 \
    environment=maniskill-pushcube-state-independent params.episodes=50 \
    'params.plugins=[{"plugin": "monitor-surprise"}]'
python scripts/compare_runs.py <基线运行> run_x01      # 逐步动作差、成功率差值及 95% 区间
```

Python 客户端：

```python
import wms

run = wms.run({"recipe": "demo-rollout", "seed": 0, "timeout_s": 60})
finished = wms.wait(run["id"])
print(wms.compare([run["id"]]))
```

客户端默认连接本机 8765 端口，用环境变量 `WMS_BASE_URL` 可以连接其他服务（例如 SSH 隧道）。

## 7. 代码组织

```text
src/wms/app/               HTTP API 与 CLI（python -m wms ...）
src/wms/core/              配方注册表、运行队列、存储、评价、模型/环境协议、A/B 一致性
src/wms/runners/unified.py 统一工作进程：generate / predict / rollout / plan / native / evaluate
src/wms/episodes/          回合内核 kernel.py、逐步日志、指标、干净统计量 clean_stats.py
src/wms/plugins/           S1–S6 插件框架：base（规则）、runtime（上下文与检查）、conformance（一致性测试）、
                           builtin / examples（内置与示例插件）、tree / stats（工具）、cards/（三人的插件卡）
src/wms/engines/<模型>/    各模型的官方入口（native.py）、回合内核适配器（adapter.py）、兼容层与指标
src/wms/scenarios/         环境适配：simpler/、maniskill/{state,vision}.py、toy/、hazards.py、危险阈值标定
src/wms/perturb/           图像扰动函数（遮挡、模糊、噪声、丢帧、冻帧）
native/<模型或环境>/       共享配方 recipes.json、runtime 示例、资源准备脚本
native/plugins*/           插件注册与干净统计量
scripts/                   run_direct.py（直接运行配方）、compare_runs.py（配对比较）
templates/                 新模型、新环境、新插件的教学模板
tests/                     CPU 测试（插件规则、内核、一致性、标定）
web/                       React 工作台
docs/                      使用与开发文档
```

新增模型通常只需加自己的 `src/wms/engines/<名字>/` 和 `native/<名字>/recipes.json`，注册表会自动发现。

## 8. 相关论文

**平台接入的模型与仿真器**

| 名称 | 论文 | 代码 |
|---|---|---|
| MemoryVLA | Shi et al., *MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation*, ICLR 2026. [arXiv:2508.19236](https://arxiv.org/abs/2508.19236) | [shihao1895/MemoryVLA](https://github.com/shihao1895/MemoryVLA) |
| TD-MPC2 | Hansen et al., *TD-MPC2: Scalable, Robust World Models for Continuous Control*, ICLR 2024. [arXiv:2310.16828](https://arxiv.org/abs/2310.16828) | [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) |
| OpenVLA | Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, 2024. [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) | [openvla/openvla](https://github.com/openvla/openvla) |
| DreamerV3 | Hafner et al., *Mastering Diverse Domains through World Models*, 2023. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) | [danijar/dreamerv3](https://github.com/danijar/dreamerv3) |
| RLVR-World | Wu et al., *RLVR-World: Training World Models with Reinforcement Learning*, NeurIPS 2025. [arXiv:2505.13934](https://arxiv.org/abs/2505.13934) | [thuml/RLVR-World](https://github.com/thuml/RLVR-World) |
| Wan2.1 | Wan Team, *Wan: Open and Advanced Large-Scale Video Generative Models*, 2025. [arXiv:2503.20314](https://arxiv.org/abs/2503.20314) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| ModelScope T2V | Wang et al., *ModelScope Text-to-Video Technical Report*, 2023. [arXiv:2308.06571](https://arxiv.org/abs/2308.06571) | [damo-vilab/text-to-video-ms-1.7b](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b) |
| SimplerEnv | Li et al., *Evaluating Real-World Robot Manipulation Policies in Simulation*, CoRL 2024. [arXiv:2405.05941](https://arxiv.org/abs/2405.05941) | [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv) |
| ManiSkill 3 | Tao et al., *ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI*, 2024. [arXiv:2410.00425](https://arxiv.org/abs/2410.00425) | [mani-skill/ManiSkill](https://github.com/mani-skill/ManiSkill) |
| RENEW、EmbodiedJev | — | [FlyingWorkshop/RENEW](https://github.com/FlyingWorkshop/RENEW)、[FBddcz/embodied-jev](https://github.com/FBddcz/embodied-jev) |

**第一阶段插件卡参考的方法论文**（每张卡的具体用法见开发计划）

| 环节 | 插件卡 | 参考 |
|---|---|---|
| S1 | #1 观测损坏、#2 相机抖动 | Hendrycks & Dietterich, ImageNet-C, ICLR 2019 [arXiv:1903.12261](https://arxiv.org/abs/1903.12261)；Wang et al., VLATest, FSE 2025 [arXiv:2409.12894](https://arxiv.org/abs/2409.12894)；Gu et al., Robust Gymnasium, ICLR 2025 [arXiv:2502.19652](https://arxiv.org/abs/2502.19652) |
| S1 | #3 遮挡块黑盒搜索 | Croce et al., Sparse-RS, AAAI 2022 [arXiv:2006.12834](https://arxiv.org/abs/2006.12834) |
| S1 | #4 异常帧防御、#5 输入异常监控 | Meng & Chen, MagNet, CCS 2017 [arXiv:1705.09064](https://arxiv.org/abs/1705.09064)；Lee et al., Mahalanobis OOD, NeurIPS 2018 [arXiv:1807.03888](https://arxiv.org/abs/1807.03888) |
| S2 | #6 记忆故障、#7 记忆写入门控 | MemoryVLA（同上）；Chen et al., AgentPoison, NeurIPS 2024 [arXiv:2407.12784](https://arxiv.org/abs/2407.12784) |
| S3 | #8 预测噪声、#9 残差监控 | Janner et al., MBPO, NeurIPS 2019 [arXiv:1906.08253](https://arxiv.org/abs/1906.08253)；TD-MPC2（同上）；Pathak et al., Curiosity, ICML 2017 [arXiv:1705.05363](https://arxiv.org/abs/1705.05363) |
| S4 | #10 打分误设、#11 约束重排 | Skalse et al., Reward Hacking, NeurIPS 2022 [arXiv:2209.13085](https://arxiv.org/abs/2209.13085)；Liu et al., Constrained MBRL with Robust CEM, 2020 [arXiv:2010.07968](https://arxiv.org/abs/2010.07968) |
| S5 | #12 动作噪声与延迟 | Robust Gymnasium（同上）；Chen et al., Delay-Aware MBRL, 2020 [arXiv:2005.05440](https://arxiv.org/abs/2005.05440) |
| S5 | #13 动作屏障、#14 违规与接管记录 | Alshiekh et al., Shielding, AAAI 2018 [arXiv:1708.08611](https://arxiv.org/abs/1708.08611)；Ames et al., Control Barrier Functions, ECC 2019 [arXiv:1903.11199](https://arxiv.org/abs/1903.11199)；Ji et al., Safety-Gymnasium, NeurIPS 2023 [arXiv:2310.12567](https://arxiv.org/abs/2310.12567) |
| S6 | #15 CUSUM 惊讶度、#16 告警接管 | Page, Continuous Inspection Schemes, Biometrika 1954 [doi:10.1093/biomet/41.1-2.100](https://doi.org/10.1093/biomet/41.1-2.100)；Phan et al., Neural Simplex, NFM 2020 [arXiv:1908.00528](https://arxiv.org/abs/1908.00528) |
| S6 | #17 反馈投毒 | Ma et al., Policy Poisoning, NeurIPS 2019 [arXiv:1910.05821](https://arxiv.org/abs/1910.05821)；Zhang et al., Adaptive Reward Poisoning, ICML 2020 [arXiv:2003.12613](https://arxiv.org/abs/2003.12613) |

平台只是接入这些方法，不等于完整复现论文。各卡与原论文的差别写在每张卡的说明里。

## 9. 新同学建议的阅读顺序

1. 本文第 1–2 节：弄清 S1–S6 和双入口。
2. [项目说明](docs/PROJECT_OVERVIEW.zh-CN.md)：各模型能做什么、不能做什么。
3. [部署与日常使用](docs/DEPLOYMENT_AND_USAGE.zh-CN.md)：按第 6 节跑通 `demo-rollout` 和 `toy-episode-occlusion`。
4. [插件模板](templates/plugin/README.zh-CN.md)：照着 `templates/plugin/wms_plugin_template/` 写一个插件，并跑通 `wms plugin test`。
5. 读 `src/wms/plugins/base.py` 开头的说明（全部规则都在那里），以及 `src/wms/episodes/kernel.py`（闭环怎么走）。
6. 再按方向选读 [MemoryVLA 说明](native/memoryvla/README.zh-CN.md) 或 [TD-MPC2 教程](docs/TD_MPC2_PUSHCUBE_GUIDE.zh-CN.md)。

## 10. 其他文档

| 目标 | 文档 |
|---|---|
| 每个模型的安装、配方、参数、输出和验证范围 | [全部模型与实验使用手册](docs/MODELS_AND_EXPERIMENTS.zh-CN.md) |
| 后端、CLI、Python API 和独立评价 | [开发与接口手册](docs/DEVELOPMENT.zh-CN.md) |
| 添加新模型或仿真器 | [开发总指南](docs/ONBOARDING_MODELS_AND_ENVIRONMENTS.zh-CN.md)与[模板](templates/README.zh-CN.md) |
| TD-MPC2 实际实验结果 | [验收报告](docs/TD_MPC2_PUSHCUBE_ACCEPTANCE_20261012.zh-CN.md) |
| 规则 / Chat / JEV 具身决策实验 | [EmbodiedJev 使用](native/embodied_jev/README.zh-CN.md)与[接线验收](docs/EMBODIED_JEV_ACCEPTANCE_20261012.zh-CN.md) |
| 安全论文与平台能力的对应 | [安全论文复现地图](docs/SAFETY_PAPER_REPRODUCTION_MAP_20261015.zh-CN.md) |
| 上游项目与发布边界 | [第三方来源](THIRD_PARTY.md) |

## 11. 开发检查与使用边界

```bash
python -m pytest tests        # 插件和内核相关测试需要 numpy
cd web && npm run build
```

- GitHub Actions 跑 Windows / Linux 的 CPU 测试和 Web 构建。GPU 验收需要目标机器上的原生依赖、数据和权重，CPU CI 代替不了真实模型验证。
- 平台改动的验收标准：不挂插件或只挂监控时，同种子的动作必须与基线逐步一致（`compare_runs.py` 的 `max_abs_action_diff` 为 0）。
- 当前是单服务、单队列、单机执行。服务默认只绑定本机地址，没有多用户认证，不要直接暴露到公网；远程使用请走 SSH 隧道。
- 本项目原创部分暂未指定 LICENSE。上游代码、权重和数据遵守各自许可，见 [THIRD_PARTY.md](THIRD_PARTY.md)。
