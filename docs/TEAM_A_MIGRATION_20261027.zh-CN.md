# A 工作区：真实任务迁移验证

验证日期：2026-10-27。结论：本次 TD-MPC2 × ManiSkill PushCube 真实闭环任务通过迁移与运行链路验证。

| 项目 | 实测结果 |
| --- | --- |
| 工作区 | `/home/suzhenjie2026/world-model/workspaces/wm-safetylab-a` |
| 代码提交 | `fcbc34e49e651a7f3903ca15ee6f1c3f5457eb0e` |
| 运行编号 | `a_migration_tdmpc2_20261027_032622` |
| 配方 | `tdmpc2-pushcube-episode` |
| 环境 | `maniskill-pushcube-state-independent`，实际为 `PushCube-v1` |
| 种子 / 回合 / 步数 | 1000 / 1 / 50 |
| 插件、扰动、目标移动 | 均为空 |
| 完成状态 | 退出码 0，`completed` |
| 耗时 | 12.53 秒 |
| 任务结果 | 成功；第 11 步首次达到成功条件（逐步索引 t=10） |
| 危险事件记录 | 本回合 `hazard_steps={}` |
| 逐步证据 | `steps.jsonl` 与 `trace.jsonl` 各 50 行，t=0–49 连续 |
| 录像 | 51 帧，512×512 RGB，MP4 解码通过 |

执行时使用 A 的源码与 runtime 配置，复用已有 TD-MPC2 Python、CUDA/PhysX、模型源码和 checkpoint。未安装依赖、下载模型或训练新权重。

共享 checkpoint：

```text
/home/suzhenjie2026/world-model/deployments/wm-safetylab/data-linux/runs/run_d5fc89b69fb84542/output/model.pt
```

运行前后 SHA256 一致：

```text
f67ec9bb86d000c32017f588505757cbf3a5596e5cd43bd67047d5018d50bb32
```

运行产物目录：

```text
/home/suzhenjie2026/world-model/workspaces/wm-safetylab-a/.workspace/runs/a_migration_tdmpc2_20261027_032622
```

额外核对结果：实际执行器为 TD-MPC2 模型；动作维度 8、观测维度 35，动作/观测/奖励均为有限数，动作在归一化边界内；物体真实状态发生变化。每步 policy_info 指向同一个共享 checkpoint；无插件接管。录像相机仅用于记录，不作为此状态策略的图像输入。

现有 8765/8766 服务的进程、启动标识、健康状态和运行列表不变；受保护的部署源码与配置哈希一致。A 的 Git 工作区保持干净。

这里只运行了一个真实模型任务。该结果证明这条任务的环境、权重、仿真、日志和录像迁移可用，不构成安全插件验收、稳定成功率估计、跨回合隔离实证或全部模型兼容性结论。

本地机器可读证据位于本交付目录的 `a-migration-test/validation.json`、`a-migration-test/execution.json` 和 `a-migration-test/run/`；完整压缩包为 `a-migration-test-evidence.tar.gz`。
