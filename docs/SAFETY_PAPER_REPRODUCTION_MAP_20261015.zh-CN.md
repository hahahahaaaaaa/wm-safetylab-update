# 主平台世界模型安全论文复现清单

核对日期：2026-10-15。对象是 `wm-safetylab` 主平台，不是 `wm-visual-sublab` 子平台。本轮只读核对代码、验收报告、服务器任务状态以及论文和官方仓库；没有重新训练、安装候选项目或运行新的安全实验。

当前可以直接开展预测失效、控制扰动、反馈更新机制等研究；尚未找到本清单安全论文在主平台完成完整复现的证据。下面的接入优先级是工程判断，不是实测安装成功率。

## 1. 当前真正可用的实验基础

| 基础 | 已验证范围 | 与安全研究的差距 |
|---|---|---|
| TD-MPC2＋ManiSkill PushCube | 真实训练、checkpoint 回载、闭环控制、动作噪声、固定动作一步／多步潜空间预测；另有独立圆心精度任务 | 尚无通用碰撞、接触力、风险 cost 或安全约束。状态输入，不是视觉 WM |
| 小型状态 WM＋ManiSkill | 动作条件模型／无动作模型／状态保持，真实物理位置预测对照和录像 | 工程自建模型；没有用学习到的 WM 规划，也不是安全方法复现 |
| RENEW 机制实验 | 官方 LatentCNN 在 2×2 2048 中的六组更新对照；完整有限支持上的概率与固定决策诊断 | 不是论文完整主动查询和模型利用修复实验 |
| DreamerV3 | CartPole 离线序列，工程 checkpoint、RSSM 状态与预测 | 不是收敛策略或完整 DMC 闭环，不是 SafeDreamer |
| RLVR-World 语言／视频 | 原生 checkpoint 的少量预测验证 | 未实现完整 RLVR 后训练；RT-1 视频预测不能直接替代 Panda 控制 |
| Wan／ModelScope | 真实文本到视频、工件和配置保存 | 尚未接入本文的视频安全与物理可靠性评价器 |
| OpenVLA | 主平台原路径支持单图与指令到动作 | 已验证 LIBERO 闭环和视觉扰动属于子平台，主平台还需迁回／注册 |
| MemoryVLA＋SimplerEnv | 本轮服务器核对时，四任务各一回合试跑及独立评价已完成；96 回合任务仍在运行 | 小规模试跑不是安全实验；尚无本轮确认的遮挡／模糊防护评价。任务状态可能随后变化 |
| EmbodiedJev＋MuJoCo | 本地有候选预演、拒绝、门槛及指定接触信号 | 不应称为学得的 WM 或校准危险概率；与正式安全论文协议不同 |

真实记录以 [TD-MPC2 验收](TD_MPC2_PUSHCUBE_ACCEPTANCE_20261012.zh-CN.md)、[状态 WM 实验](STATE_WM_MANISKILL_EXPERIMENT_20261010.zh-CN.md)、[模型手册](MODELS_AND_EXPERIMENTS.zh-CN.md)、[MemoryVLA 说明](../native/memoryvla/README.zh-CN.md)为依据。

MemoryVLA 核对时的服务器记录：试跑 `run_9752af1b33f94fdb`、评价 `run_d4f44930405e4225` 已完成，四任务共 2/4 成功；正式任务 `run_96c428cdd0e04ae6` 在运行。这里保留的是核对时快照，不代表正式实验最终结果。

## 2. 复现口径

- **已有实验**：代码、环境、实际工件已经验证，只能报告实际覆盖的设置。
- **评价接入**：可以复用已接入的生成模型，仍需加入论文数据、评价器、分组和协议。
- **原生复现**：新建论文独立环境，使用作者模型、任务、训练与评价协议，通过平台调度和保存结果。
- **方法迁移**：把论文方法改到现有 TD-MPC2／ManiSkill；这是新实验，不能直接宣称复现了原论文数值。

“官方有代码”不等于“本机已经能跑”。同属 Dreamer、TD-MPC2 或 ManiSkill 家族，也不保证版本、权重、观测和动作兼容。

## 3. 直接研究世界模型安全与鲁棒性的论文

### 3.1 SafeDreamer — 第一篇正式安全方法建议选它

**SafeDreamer: Safe Reinforcement Learning with World Models，ICLR 2024。** [论文](https://openreview.net/forum?id=tsE5HLYtYg) · [官方代码](https://github.com/PKU-Alignment/SafeDreamer)

- 研究对象：想象中的安全代价、奖励与风险权衡、安全规划／安全策略；主要 S3、S4、S5、S6。
- 复现路线：独立运行官方 SafeDreamer＋匹配 Safety-Gymnasium，先评估官方 checkpoint，再做训练复现。作者提供多组权重及 unsafe Dreamer 对照。
- 平台复用：原生任务、独立 Python、日志／视频／checkpoint、独立评价。
- 缺口：原版本 JAX 环境、cost 信号、原论文配方与评价数据。当前 Dreamer 的小型 checkpoint 不能代替。
- 首轮输出建议：同一任务与种子的 return、累计 cost、违规回合率、规划延迟。若只复现一个任务，应明确称为单任务子集。

### 3.2 PIGDreamer — 部分可观测与安全状态估计

**PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable Reinforcement Learning，ICML 2025。** [论文](https://proceedings.mlr.press/v267/huang25ai.html) · [官方代码](https://github.com/hggforget/PIGDreamer)

- 研究对象：训练时利用特权状态帮助视觉世界模型与安全控制；主要 S1、S2、S3、S5。
- 原生复现：先用作者安全环境和训练流程；机械臂遮挡、模糊和目标变化是后续扩展。
- 缺口：图像输入／训练专用特权状态双通道、cost 监督、对应学习器。不能把 GT 状态直接送入测试策略后声称实现视觉鲁棒性。
- 部署风险：仓库有真实训练代码，但 README 仍有旧项目名和版本不一致，未确认公开预训练权重；需要先做原生环境验证。

### 3.3 RePo — 最贴近视觉机械臂鲁棒性需求

**RePo: Resilient Model-Based Reinforcement Learning by Regularizing Posterior Predictability，NeurIPS 2023。** [论文](https://arxiv.org/abs/2309.00082) · [官方代码](https://github.com/zchuning/repo)

- 研究对象：视觉干扰下，如何学习保留任务相关信息的潜在表示；主要 S1、S2、S3，非对抗失效与鲁棒性改进。
- 作者提供 distracted DMC 和 ManiSkill 的 PushCubeMatterport、LiftCubeMatterport、TurnFaucetMatterport 实验入口。
- 原生复现需要作者版本仿真器、背景资产和视觉训练。迁入当前 ManiSkill 3／PushCube-v1 必须核对环境版本与控制语义，属于方法迁移。
- 当前 TD-MPC2 使用状态输入，给录像加模糊并不会改变它收到的观测。视觉实验必须把扰动施加到算法实际输入图像。
- 收益是表示与控制鲁棒性，不能把任务成功率直接称为物理安全率。

### 3.4 WAKER — 数据覆盖与环境分布变化

**Reward-Free Curricula for Training Robust World Models，ICLR 2024。** [论文](https://arxiv.org/abs/2306.09205) · [官方代码](https://github.com/marc-rigter/waker)

- 研究对象：根据世界模型误差选择训练环境，改善弱环境表现；主要 S1、S3、S6。
- 原生复现：作者的 TerrainWalker／TerrainHopper／CleanUp／CarCleanUp，以及 WAKER 与随机域采样等对照。
- 当前平台能够管理采样、训练和评价任务，但尚无 WAKER 调度与训练实现。迁到 ManiSkill 要额外定义场景参数分布和误差估计。
- 建议比较平均表现与困难环境尾部表现；这是覆盖和可靠性研究，不自动等于危险约束满足。

### 3.5 FOSP — 离线到在线安全更新

**FOSP: Fine-tuning Offline Safe Policy through World Models，ICLR 2025。** [论文](https://openreview.net/forum?id=dbuFJg7eaw) · [官方代码](https://github.com/Sunlighted/FOSP)

- 研究对象：离线安全学习后的在线适应、可达性指导、安全策略扩展；主要 S3、S4、S6。
- 作者实现基于 SafeDreamer，并提供离线数据及离线／在线入口。建议在 SafeDreamer 原生复现之后接入。
- 缺口：离线数据转换、两个训练阶段、迁移设置和持续 cost 记录。只比较更新前后任务奖励不足以支持安全微调结论。
- 仓库安装示例存在地址／版本不一致，实际接入需锁定可运行依赖。

### 3.6 UNISafe — 机械臂中的潜空间安全过滤

**Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures，CoRL 2025。** [论文](https://arxiv.org/abs/2505.00779) · [官方代码](https://github.com/CMU-IntentLab/UNISafe)

- 研究对象：世界模型不确定性、分布外失效与安全动作过滤；主要 S2、S3、S4、S5。
- 作者提供 Dreamer 与过滤器权重。机械臂任务在 IsaacLab 分支，README 指定旧 IsaacSim 4.2.0。
- 原生复现需新增该仿真环境；当前 ManiSkill 不能直接替换。迁移需要重新定义失败标签、视觉观测、模型及过滤器训练。
- 可以借鉴其过滤接口，但不能直接加载作者 Dreamer 权重到当前 TD-MPC2。

### 3.7 ActSafe — 安全探索

**ActSafe: Active Exploration with Safety Constraints for Reinforcement Learning，ICLR 2025。** [论文与项目](https://yardenas.github.io/actsafe/) · [官方代码](https://github.com/yardenas/actsafe)

- 研究对象：概率模型、不确定性驱动探索与保守安全约束；主要 S3、S4、S6。
- 需要作者 JAX 环境、RWRL／安全导航等原任务与训练流程。可复用平台长任务管理，但不是现有确定性模型的即插即用模块。
- 建议记录整个学习期的违规数，而不只统计最终策略；理论保证必须保留论文假设。

### 3.8 SPOWL — 与 TD-MPC2 路线相关的安全规划

**Safe Planning and Policy Optimization via World Model Learning，ECAI 2025。** [论文](https://journals.sagepub.com/doi/10.3233/FAIA251029) · [官方代码](https://github.com/AIRI-Institute/SPOWL)

- 研究对象：隐式 WM、安全策略与规划的结合及动态阈值；主要 S3、S4、S5、S6。
- 论文与 TD-MPC2 路线相关，但作者实现是 JAX、原环境为 Safety-Gymnasium。
- 需要独立运行作者代码。迁到当前 PyTorch TD-MPC2 时，要补安全代价学习、规划约束和回退策略并重新训练；不能直接使用已有 checkpoint 冒充安全版。

### 3.9 RENEW — 当前已有基础，但还缺原论文实验

**RENEW: Towards Learning World Models and Repairing Model Exploitation from Preferences，2026；官方仓库标注 RLC Finding the Frame Workshop。** [论文](https://arxiv.org/abs/2607.14180) · [官方代码](https://github.com/FlyingWorkshop/RENEW)

- 研究对象：偏好反馈、世界模型更新、模型利用现象；主要 S3、S4、S6。
- 主平台已有 `native-renew`，但仅为 2×2 2048 六组机制实验。
- 复现论文修复实验应加入作者的 Jumanji Maze 预训练、查询、更新和评价链路，并保持对应基线与预算。
- 当前可以先研究概率变化与固定决策变化；完整主动查询与论文主结果尚未完成。该工作不应列为顶会主会论文。

## 4. 可以复用现有视频生成模型的论文

| 论文／来源 | 可开展的实验 | 当前还缺什么 |
|---|---|---|
| **T2VSafetyBench: Evaluating the Safety of Text-to-Video Generative Models**，NeurIPS 2024 D&B。[论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/74eed5f568354c2e77dd9b018f38a9d4-Paper-Datasets_and_Benchmarks_Track.pdf) · [代码](https://github.com/yibo-miao/T2VSafetyBench) | 在 Wan／ModelScope 上进行内容风险和恶意提示评测；S1、S3及结果评价 | 提示数据、视频路径映射、裁判、正常提示对照。锁定数据版本；论文与当前仓库类别细分存在变化。更换原裁判必须说明。用 Wan 是新模型上的基准评测 |
| **WorldModelBench: Judging Video Generation Models As World Models**，NeurIPS 2025 D&B。[论文](https://papers.neurips.cc/paper_files/paper/2025/hash/4ec03ed08a3fcb59e1c815b5598beff1-Abstract-Datasets_and_Benchmarks_Track.html) · [代码](https://github.com/WorldModelBench-Team/WorldModelBench) | 指令遵循、常识、物理一致性；官方有 T2V 路径，可复用 Wan／ModelScope；主要 S3、评测 | 350 条测试配置、命名规范、VILA 裁判及权重。物理一致性分数不等同于闭环控制安全 |
| **VideoEraser: Concept Erasure in Text-to-Video Diffusion Models**，EMNLP 2025 主会。[论文](https://aclanthology.org/2025.emnlp-main.304/) · [代码](https://github.com/bluedream02/VideoEraser) | 原生成器与概念擦除防护对照；主要 S3、防御＋评测 | 官方方法、概念配置、匹配依赖与消融。优先核对现有 ModelScope 是否匹配官方 ModelScope 1.7B 路线；也可新增 CogVideoX-5B。官方未提供当前 Wan1.3B 直接路径 |
| **Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation（PhyGenBench）**，ICML 2025。[论文](https://proceedings.mlr.press/v267/meng25c.html) · [代码](https://github.com/OpenGVLab/PhyGenBench) | 物理规律失效及前后状态合理性评价；主要 S3、非对抗失效＋评测 | 测试集、PhyGenEval 及其模型／API 依赖；可先复用生成接口。仍属于物理可靠性，并非机器人物理风险成本 |

## 5. 新近的视觉 WM 失效研究候选

**Hallucination in World Models is Predictable and Preventable，2026 预印本。** [论文](https://arxiv.org/abs/2606.27326) · [官方代码](https://github.com/nicklashansen/mmbench2)

作者发布视觉动作条件世界模型、预训练／微调权重、幻觉检测信号和数据收集代码；数据任务包含 ManiSkill3。研究涉及 S1 数据覆盖、S3 想象失真及 S6 定向收集与更新，适合本平台长期目标。

但这需要新增 MMBench2 模型、对应预处理与任务控制协议，不是直接使用现有 Wan 或 TD-MPC2。可以先评估作者 checkpoint，再考虑局部微调；不能把加载权重等同于复现全量预训练。本轮未确认正式会议录用，按预印本列示。

## 6. 行动安全的相邻清单

这些工作可以补足平台的攻击／防御实验，但 OpenVLA 或 MemoryVLA 有动作输出或记忆，并不自动意味着它有显式世界模型。若声称 WM 安全，还需记录内部状态、预测未来和它们对动作的作用。

| 论文 | 主平台接入判断 |
|---|---|
| **Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics**，ICCV 2025。[论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Exploring_the_Adversarial_Vulnerabilities_of_Vision-Language-Action_Models_in_Robotics_ICCV_2025_paper.pdf) · [代码](https://github.com/William-wAng618/roboticAttack) | OpenVLA／LIBERO 对抗视觉实验候选，属于 S1 攻击与行为评价。先把子平台的 LIBERO 闭环接回主平台，再安装作者攻击与原协议；随机模糊不能替代对抗优化 |
| **VLA-Arena: An Open-Source Framework for Benchmarking Vision-Language-Action Models**，项目页标注 ICML 2026。[论文](https://arxiv.org/abs/2512.22539) · [代码](https://github.com/PKU-Alignment/VLA-Arena) · [项目页](https://vla-arena.github.io/) | 新增安全任务和对应微调模型，不能拿普通 LIBERO-Spatial 权重直接声称复现官方基线。属于系统行为安全评价 |
| **SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning**，NeurIPS 2025 Spotlight。[论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e185c7be603426028c32ae1003a59d78-Abstract-Conference.html) · [代码](https://github.com/PKU-Alignment/SafeVLA) | 原环境是 AI2THOR／Safety-CHORES，需匹配策略和安全对齐权重。ManiSkill／LIBERO 不能直接替代，建议后置 |

## 7. 哪些已有论文不应直接写成安全方法复现

TD-MPC2、DreamerV3、OpenVLA 和 MemoryVLA 首先是模型／控制基础。平台对它们的接入与工程验证，不证明安全方法有效。

**RLVR-World: Training World Models with Reinforcement Learning，NeurIPS 2025** 是通用世界模型后训练工作。[论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b63a24a1832bd14fa945c71f535c0095-Abstract-Conference.html) · [官方代码](https://github.com/thuml/RLVR-World)。可以作为预测错误与更新目标的研究基础，但目前只有推理接口；增加安全奖励、机械臂迁移或安全更新目标都是新研究，需要独立验证。

## 8. 推荐落地顺序

1. **先完成一篇明确的 WM 安全子集复现：SafeDreamer。** 使用作者权重和原安全任务，把 predicted cost、realized cost、reward 与轨迹接到独立评价；随后扩展训练种子和任务。
2. **并行补现有视频管线的评价：T2VSafetyBench＋WorldModelBench。** 复用现有生成器，分别报告内容安全与物理可靠性。先运行小样本连通性检查，再跑锁定的正式测试集。
3. **视觉机械臂方向优先 RePo，之后 PIGDreamer。** 先原生复现，再迁移当前 ManiSkill 3。必须让策略真正消费受扰图像，并将 GT 留给监督／评价。
4. **反馈更新方向用 FOSP 或补全 RENEW。** 记录数据、查询预算、更新目标、checkpoint、冻结的评价集与更新前后决策。
5. **安全过滤进阶考虑 UNISafe；TD-MPC2 安全规划参考 SPOWL。** 新任务的危险定义、cost 和控制接口确定后，再进行方法迁移。

本计划不要求重写四层核心。每篇论文可先采用独立原生任务：自己的 `native/<paper>/recipes.json`、运行环境配置、`src/wms/engines/<paper>/` 入口与指标。模型/环境需交互时，再增加实际必要的适配器。

## 9. 复现结果至少交付什么

- 固定论文版本、代码提交、权重、环境、观测与动作语义、数据划分及预算。
- 先验证原生论文脚本，再通过平台运行同一配置；环境或算法更改必须单列。
- 按论文要求运行训练与评价种子；缩小预算时明确只复现子集，保留方差和失败运行。
- 分别保存自然失效、攻击、防护和评价条件，不能把随机噪声命名为自适应攻击。
- 记录原始预测、实际动作、真实后果、任务指标与安全指标。无法观测的危险应标记“未测量”。
- 评价协议与阈值固定；对照应匹配初始条件。任务成功、视频质量、物理一致性和安全约束满足分别报告。

截至本轮核对，最准确的项目表述是：**主平台已经有支撑 WM 安全研究的实验底座与部分机制实验，下一步应补论文原生安全方法、危险真值与正式对照协议，形成可复核的论文复现。**
