# EmbodiedJev 接入验收（2026-10-12）

已在本机 WM-SafetyLab 完成 EmbodiedJev 原生实验接入。Chat 使用本次指定的智增增端点 `https://api.zhizengzeng.com/v1/`，请求与响应模型均为 `gpt-5.5`。JEV 使用独立 TypeSafe 连接，实际响应模型为 `jev-1.13.0`。密钥不包含在本文、配方或运行配置中。

工作台：<http://127.0.0.1:8765/>。运行数据保存在 `K:\world-model\deployments\wm-safetylab\data`；本轮验收工件在 `results/embodied-jev-integration-20261012/`。

## 已完成

- 注册 `embodied-jev-baseline`、`embodied-jev-chat`、`embodied-jev-jev` 三个配方。
- 复用 K: 现有 Python 3.12/MuJoCo 环境，加载干净上游源码 `K:\world-model\repos\embodied-jev`，不修改上游 checkout。
- 通过平台 FIFO 队列运行独立 worker；支持任务、种子、控制模式、观测模式、预演、概率门槛、循环和 API 预算。
- 模型请求串行，默认每回合最多 20 次 HTTP 尝试（包括重试），不加固定两秒间隔；Chat 明确使用非流式 JSON 候选响应。
- 凭据仅由 worker 从 Git 忽略的文件读取；HTTP 错误、超时、预算耗尽和协议错误独立记录，不回退为规则动作。
- 保存原始 episode、标准轨迹、候选决策、事件、元数据、摘要与 HTML 时间线。
- 按同一回合记录的 qpos/time 重放生成 MP4/WebM，明确标注为记录状态回放；不重新调用模型、不重新推进物理。录像前先保存核心证据。
- 自动创建独立评价任务；相同来源可以重新评价，原始实验文件不被改写。
- 补充取消处理：初始化后、开始请求前检查取消；显式取消的原生结果保留经过路径验证的附件。

使用方法及参数示例见 [EmbodiedJev 使用文档](../native/embodied_jev/README.zh-CN.md)。

## 真实接线验收

条件：三个原生 MuJoCo Panda 任务，seed=0，privileged 观测，skills 控制，preview=true，threshold=0.55。以下是小规模接线验证，不能用于正式模型排名或安全性能结论。

| Provider | 任务 | 回合结果 | 任务成功 | 实际 HTTP 请求 | 回合耗时（秒） | 原始运行 |
| --- | --- | --- | --- | --- | --- | --- |
| 规则 | transfer | completed | 是 | 0 | 1.61 | run_cbbbeb2653ac4ba7 |
| 规则 | stack | completed | 是 | 0 | 1.52 | run_ec1c054adb96472b |
| 规则 | barrier | completed | 是 | 0 | 1.78 | run_b46522c0cc7d440e |
| 智增增 GPT-5.5 | transfer | completed | 是 | 13 | 83.64 | run_fc350021c79345b7 |
| 智增增 GPT-5.5 | stack | completed | 是 | 13 | 61.75 | run_572f7db2b76d418b |
| 智增增 GPT-5.5 | barrier | completed | 是 | 13 | 77.48 | run_bf8c0ddd654d4b1f |
| JEV | transfer | uncertain | 否 | 10 | 4.81 | run_de4e0c42a85540e5 |
| JEV | stack | uncertain | 否 | 10 | 4.89 | run_ff501b346f5544cd |
| JEV | barrier | uncertain | 否 | 10 | 4.75 | run_2a8d483453ab48f0 |

合计九个来源实验、九个自动独立评价均正常完成归档，指标作业没有 metric_errors。这里的平台 completed 表示实验执行与记录完成，不等于机器人任务成功。规则与 Chat 各 3/3 成功；JEV 三次均触发不确定性停止。

JEV 停止时所选动作的原生概率分别为 0.50、0.52、0.51，低于配方的 0.55 门槛，因此按设计结束回合并保存未完成状态。没有降低默认门槛或换成规则动作来改写结果。Chat 没有原生候选分布，概率显示未知且不执行同一种概率门控，因此此表不能直接证明哪个模型能力更强。

实际请求合计 69 次（Chat 39，JEV 30），并发上限为 1。回合耗时来自上游 wall_seconds，不包含后续视频渲染和平台评价；这些接线结果不代表长期可用率或账户剩余额度。

完整原始记录及对应 evaluation_id 见 `results/embodied-jev-integration-20261012/live-acceptance.json`。

## 验证

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 平台完整回归 | 128 passed，7 skipped；1 条已有依赖弃用警告 | `pytest-full.xml` |
| 新接入模块的原生环境测试 | 64 passed，覆盖真实上游决策类与 MockTransport，无真实 API 调用 | `pytest-native.xml` |
| 显式取消附件与现有生命周期 | 11 passed | `cancelled-native-lifecycle.xml` |
| 智增增 transfer 重复独立评价 | 成功，指标相同，所有来源文件 SHA-256 不变 | `repeat-evaluation.json` |
| 凭据归档检查 | 初次检查 375 个本轮代码、配置、日志和导出文件，没有发现真实凭据 | `credential-scan.json` |
| 录像完整解码与来源 | 9 回合、18 个 MP4/WebM 共 2,736 帧解码通过，场景/时序一致，来源哈希不变 | `final-video-check.json` 与每局 `episode-replay.json` |

上述测试集合在不同环境中有重叠，数量不相加。平台环境跳过项来自可选 numpy/Gymnasium 以及未安装上游策略依赖；其中五个上游策略测试已在原生环境运行通过。没有为本次接入修改前端源码，沿用既有视频卡片和产物链接；本轮未完成浏览器自动化视觉检查。

重复评价的新运行是 `run_8e707b5bca5a4b76`，来源为 `run_fc350021c79345b7`，本次重复评价没有调用模型。

## 文件与运行

主要实现：

- `src/wms/engines/embodied_jev/native.py`：配置校验、上游加载、Session 生命周期及平台结果。
- `connections.py`：密钥引用、Chat/JEV 接口、预算、重试与逐请求用量记录。
- `artifacts.py`：原始与标准化产物、HTML 时间线。
- `metrics.py`：纯离线评价及指标语义。
- `video.py`：同回合状态重放录像。
- `native/embodied_jev/recipes.json`、`runtime.example.json`、`README.zh-CN.md`：注册与使用入口。
- `src/wms/runners/unified.py`：保留原生取消结果的合法附件。

本机服务已启动。若以后需要重启，在 PowerShell 中执行：

```powershell
. 'K:\world-model\use-k-drive.ps1'
Set-Location 'K:\world-model\deployments\wm-safetylab'
& '.\.venv\Scripts\python.exe' -m wms serve --host 127.0.0.1 --port 8765
```

服务已经运行时不要重复启动。另一个终端可提交：

```powershell
. 'K:\world-model\use-k-drive.ps1'
Set-Location 'K:\world-model\deployments\wm-safetylab'
& '.\.venv\Scripts\python.exe' -m wms run embodied-jev-chat --wait
```

机器 runtime、凭据、虚拟环境和运行数据保持本地并按 Git 规则忽略。实施过程中发现另有 TD-MPC2 圆心任务相关文件正在变更，本轮没有修改或覆盖这些文件，也没有提交或推送它们。

## 当前边界

- 本次真实验收覆盖 MuJoCo 原生三任务的 privileged + skills；RGB-D、原始视觉、incremental、hierarchical 的参数接口已接线，但不把它们计入本次真实验收结果。
- 尚未实现 ManiSkill 动作桥接，也没有新增世界模型训练、危险概率头或在线学习。JEV 候选概率不代表危险概率。
- 碰撞记录只反映上游已配置的接触检测，不能称全面机器人安全。
- 平台现有独立评价接口要求来源运行 completed。请求/协议异常或超时会标为 failed，保留原始证据；对此类记录可在新目录使用离线 measure 审计。统计正式成功率时必须包含全部计划回合，不能只挑有自动评价的运行。
- 大规模种子实验、preview 消融和 ManiSkill 迁移属于后续研究阶段；本次没有把接线验收当作完整论文实验。