# S5 动作与控制：第一版

本版复用动作缩放，新增随机动作噪声、控制步延迟和指令空间屏障。插件代码按现有 S5/S6 归属放在 `src/wms/plugins/cards/c_action.py`，注册于 `native/plugins_c/recipes.json`；服务器使用 A 工作区不会改变插件归属。

## 页面使用

沿用原版八步实验配置向导，在第 4 步“插件”点击 S5，选择动作缩放、动作随机噪声、动作延迟或动作屏障。新增参数控件直接放在原版插件卡片内部，使用原有样式；原版导航、布局、流程和其他阶段的参数表单保持原样。

勾选插件后配置写入原向导的实验配置，取消勾选则移除插件。本地前端已从服务器 `dev/a` 原版源码恢复；不使用另一套“配方＋JSON”应用替换此向导。首次错误部署的前端已备份并恢复，后续前端修改以原版为基准检查差异。

| 插件 ID | 参数 | 行为 |
| --- | --- | --- |
| `example-action-scale` | `scale=0.5, dims=2` | 复用已有动作缩放，前两维减半 |
| `s5-action-noise` | `std=0.1, dims=2` | 前两维加入高斯噪声；std=0 是无效扰动 |
| `s5-action-delay` | `steps=1, dims=2` | 前两维延迟一控制步，预热重复首个动作；steps=0 是无效扰动 |
| `s5-action-barrier` | `limit=1, max_delta=0.25, dims=2` | 与控制器动作边界取交集限幅，再限制相邻执行动作变化；首步只限幅 |

三个新插件支持 `start`（含）与 `end`（不含，null 表示持续）；`dims` 是前几个维度，JSON 中 null 表示全部维度。维度不得超过控制器动作维度。默认两维只是一项配置，真实机器人必须根据控制器语义调整，不能直接推广玩具动作含义。

噪声用控制命令单位，不代表物理位置误差；使用插件独立随机源。同一插件排列与种子下可复现，调整插件顺序可能改变随机流。噪声本身不隐藏裁剪，由后续屏障或环境处理。

真实 TD-MPC2 环境会拒绝超范围动作。噪声卡片新增 `clip_to_bounds`（默认关闭，保留原实验语义），在 TD-MPC2 噪声实验中必须开启。开启后按照控制器边界限幅，同时记录 `raw_noisy_action`、`output_action`、`clipped_dimensions`；这不是碰撞防护。与基线做零强度检查时仍保持原动作不变。

延迟是控制步队列，不是真正暂停程序。窗口外恢复当前命令，回合间清空队列；未选维度始终使用当前动作。

屏障不读取仿真真值，只约束控制命令，不保证无碰撞或物理安全。`max_delta=null` 关闭变化量限制。幅度与变化量约束冲突时优先幅度约束。每次修改调用现有 fallback 接管接口，备用命令就是约束后的动作；后续变化量基于实际执行命令。多个防护或外部告警同时使用时仍须验证组合行为。

## 配置示例

在支持 episode kernel 的配方中设置：

```json
{
  "params": {
    "plugins": [
      {"plugin": "s5-action-noise", "config": {"std": 0.1, "dims": 2}},
      {"plugin": "s5-action-barrier", "config": {"limit": 0.8, "max_delta": 0.25, "dims": 2}}
    ]
  }
}
```

此片段须合并至原配方，不能替代完整运行配置。故障先于防护运行；同角色按数组顺序执行。基线用空插件列表，防护单独组用于检查正常性能损失，故障组与故障加防护组使用配对种子。新增插件的 trace 事件包含每个插件的输入/输出动作；最终动作读取 trace 的顶层 `action`，原模型动作读取 `action_proposed`。

## 测试与指标

使用项目 Python（3.11+，含 numpy）运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -p test_s5_actions.py -v
python -m wms.plugins.conformance s5-action-noise --output test-output/s5-noise-check
python -m wms.plugins.conformance s5-action-delay --output test-output/s5-delay-check
python -m wms.plugins.conformance s5-action-barrier --output test-output/s5-barrier-check
```

每次一致性检查使用新的输出目录，以保留旧证据。服务器也可用 `./dev.sh plugin-test <ID>`。

从某次运行的 output 目录重算：

```bash
python -m wms.episodes.s5_metrics /path/to/run/output
```

结果打印为 JSON，不覆盖原文件。危险按每种每步去重；另算任一危险步数，避免同时发生多种危险时重复计算总步数。没有危险测量字段时标为 null。连续、同插件且同类型的接管算一个事件；停止单独计为事件，不算执行步。成功率依据 trace 的真实环境 success 事件计算，适用于当前 episode kernel 的事件约定。

重算结果现包含逐步 `takeover_records`、连续接管分组、干预比例、执行步数、首次成功步以及 `summary_checks`。屏障日志补充幅度/控制器边界限制、变化量限制的原因代码。`sent_action` 是发送给环境的命令，不等于测量到的关节运动；CPU 验收会额外核对玩具环境实际位置变化。

## 完整验收入口

使用服务器 A 的既有环境：

```bash
./dev.sh toy-python scripts/s5_acceptance.py cpu --output .workspace/results/新的验收目录
./dev.sh toy-python scripts/s5_acceptance.py gpu --output .workspace/results/同一验收目录
./dev.sh toy-python scripts/s5_acceptance_report.py .workspace/results/同一验收目录
```

CPU 阶段先在独立种子上校准屏障，再评估缩放、两种噪声强度、延迟与防护组合，同时核对零强度轨迹。GPU 阶段检查显存和利用率，从机器 runtime 读取已有 checkpoint（不训练），每回合独立重建场景，处理前7个关节动作维度并保留夹爪。GPU 阶段可跳过同一目录内已完成的组，保留失败运行用于追溯；不要同时重复启动同一验收目录。

报告工具输出 `acceptance.md`、`acceptance-summary.json`、`takeovers.csv`、逐运行重算/动作审计文件，并把已完成 GPU 运行登记为原版结果页的一份对照记录，不新增模型计算。动作审计也可单独执行：`./dev.sh toy-python scripts/s5_trace_audit.py .workspace/store/runs/运行编号`。

本轮完整结果见 `docs/S5_ACCEPTANCE_20261030.zh-CN.md`。指标核对通过与防护有效是两件事；报告保留屏障导致成功率下降的负面结果。

本版 CPU 检查验证程序行为与插件契约，不能替代服务器真实模型验收。后续需按服务器同步说明备份上传，再执行配对 GPU 实验，校准屏障阈值并核对统计结果。

### 本次本地验证（2026-10-30）

- Python 3.11：6 项针对性测试通过，包括两回合多插件组合测试（故意将屏障放在配置数组最前，确认运行时仍先故障后防护）。
- 三个插件在四种 CPU 玩具组合上共 74 项契约检查无失败，覆盖严格模式、可复现、随机源隔离和故障零强度等检查；包含 INFO 项，非 74 个独立测试用例。
- 后端 catalog 能发现三个新增插件，指标重算工具已读取实际生成的四种屏障测试轨迹。
- 前端 `npm run build` 通过，包含 TypeScript 类型检查；未完成浏览器交互验收。构建保留原有大型过程对比模块的包体积提示。
- 以上为第一版完成时的记录；随后已同步服务器并完成本轮 CPU/GPU 验收，详见上方完整验收记录。
