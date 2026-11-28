# E 工作区 S1–S6 集成说明

日期：2026-11-24（America/Chicago）。目标工作区：`/home/suzhenjie2026/world-model/workspaces/wm-safetylab-e`，分支 `dev/e`。

本次将 A 的 S5/S6 实现接入 E 的现有 S1–S4，包含后端插件、统一执行/接管接口、Web 参数表单、指标重算和后续验收脚本。系统已覆盖 S1–S6；完整组合实验和安全效果验收按用户要求留待后续，不将 A 的历史结果当作 E 的验收。

## 六阶段功能

| 环节 | E 中的实现 |
|---|---|
| S1 输入/感知 | 观测损坏、向量噪声、图像平面抖动、遮挡攻击、异常帧防御、输入监控（a-*） |
| S2 状态/记忆 | b-memory-fault、b-write-gate |
| S3 预测 | b-prediction-noise、b-residual-monitor |
| S4 规划 | b-score-misspec、b-constrained-rerank |
| S5 动作 | 原有 example-action-scale；新增 s5-action-noise、s5-action-delay、s5-action-barrier |
| S6 反馈 | 新增 s6-residual-monitor、s6-cusum、s6-feedback-poison |

S5/S6 的源码保持 A 的实现：`src/wms/plugins/cards/c_action.py`、`c_feedback.py`，S6 残差定义在 `src/wms/plugins/residuals.py`。注册在 `native/plugins_c/recipes.json`。

共享运行时合入 A 的两个必要修正：严格只读检查正确处理未改变的 NaN；停止请求优先于备用动作，S5 临时覆盖不会消耗尚未执行的 S6 备用步数。E 已有的 S2 written 阶段检查、S3/S4 接口、TD-MPC2 检查点选择与独立场景、观察相机记录、回放和后端校验均保留。

## 使用 E

```bash
ssh suzhenjie2026@10.161.97.111
cd /home/suzhenjie2026/world-model/workspaces/wm-safetylab-e
./dev.sh info
```

E 继续监听服务器 `127.0.0.1:8785`。在自己的电脑开启隧道：

```bash
ssh -N -L 8785:127.0.0.1:8785 suzhenjie2026@10.161.97.111
```

打开 `http://127.0.0.1:8785`，刷新页面后，在原有八步实验向导的插件配置中选择 S5/S6。原页面结构、路由、样式以及 S1–S4 控件保留。

S5 新表单支持噪声强度、延迟、幅度/变化量限制、处理维度和生效窗口；S6 支持残差尺度、CUSUM 参数、只告警/下一步停止/备用动作，以及反馈投毒参数。更复杂参数仍可通过高级 JSON 设置。

## 模型能力约束

六阶段覆盖指平台能力，不代表每个模型支持每一张卡。Toy planner / TD-MPC2 可配置 S1–S6 的适用插件；TD-MPC2 无 written 和 learn，不能使用写入门控或反馈投毒。MemoryVLA 支持图像、记忆和动作相关插件，但不能凭此声明其具备 TD-MPC2 的预测/规划接口。反馈投毒当前仅用于在线学习 Toy，未接入真实模型训练更新。

控制器维度仍需正确设置：Toy 默认处理前 2 维；TD-MPC2 动作实验通常处理前 7 个关节维度并保留夹爪维度。S5 屏障约束控制指令，不等同于碰撞安全保证。CUSUM 阈值必须独立标定；A 历史 GPU 检测结果并未证明可靠检出。

## 已准备的组合配置

`examples/s1-s6/toy-neutral.json` 与 `tdmpc2-neutral.json` 分别包含每个阶段的一个兼容插件。故障强度为零，S6 只记录残差，供后续中性一致性检查使用；本次仅校验配置，没有提交实验。

只检查配置，不执行任务：

```bash
curl -s http://127.0.0.1:8785/api/v1/experiments/validate \
  -H 'Content-Type: application/json' \
  --data-binary @examples/s1-s6/toy-neutral.json
```

结果位置继续隔离：Web 实验在 E 的 `.workspace/store`，CLI 实验在 `.workspace/runs`。环境、模型、检查点继续共享，没有复制或重新下载。

## 验收工具与当前状态

S5/S6 验收工具已迁入 `scripts/s5_acceptance*.py`、`s6_acceptance*.py`、`s5_trace_audit.py`，指标重算为 `wms.episodes.s5_metrics`、`s6_metrics`。工具读取 `WMS_BASE_URL` 或当前工作区 `.workspace/settings.json` 的端口，不再写死 A 的 8781；存储根目录也跟随当前工作区。后续从 E 的 `./dev.sh toy-python` 调用。

本次完成：14 个 Python 文件语法检查、6 个新增插件的声明/构造检查、S1–S6 目录检查、两份六阶段组合配置校验、TypeScript/Vite 构建。未运行训练、GPU/CPU 实验矩阵、完整回归或浏览器全流程验收。

S1/S2 的既有原始验收数据保留。补测链已停止，后续继续时需先修复独立验收副本与 E 之间的 runtime_config 路径等价校验，再恢复补测；不应直接把旧的 failed 状态改成 completed。

A 的历史 S5/S6 验收文档随源码保留，并已明确标注来源。A 的工作树、Git 暂存区、服务和实验结果未修改。

## 来源与合并范围

A 来源基准为 `67ec869` 及当时尚未提交的 S5/S6 工作内容；E 基准为 `e428ef3`。来源文件的 SHA-256 记录在同目录 `S56-A-source-snapshot.json`。

没有照搬 A 中与 S5/S6 无关的回退差异，包括删除观察相机记录、撤销较新的模型适配器/检查点逻辑、减少校验字段/测试等。S5/S6 的历史验收链接仍指向 A 的历史存储，不伪造为 E 的新结果。
