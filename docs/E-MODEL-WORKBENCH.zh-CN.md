# E 模型工作台

E 将已有模型的原生任务与 S1–S6 闭环实验放在同一服务中。任务由 E 的独立进程和队列执行，输出写入 E 自己的 `.workspace/store/runs`；共享已安装的环境、模型权重与仿真资产。

## 访问

在个人电脑保留这个 SSH 窗口：

```powershell
ssh -F NUL -N -L 8785:127.0.0.1:8785 suzhenjie2026@10.161.97.111
```

打开 `http://127.0.0.1:8785/#models/library`。服务器工作区为 `/home/suzhenjie2026/world-model/workspaces/wm-safetylab-e`。

「模型工作台」中选择模型和任务配方，编辑输入与参数，点击提交。下方可选择运行记录，查看状态、日志、结果 JSON、视频和下载文件。队列中或运行中的任务可取消。「新建实验」仍是原有八步安全实验配置流程，支持可兼容的插件对照、报告和逐帧回放。

## 模型与能力

| 模型 / 配置 | E 中的入口 | 能力范围 |
|---|---|---|
| MemoryVLA | 单步闭环、官方评测、接口一致性 | SimplerEnv 图像策略和记忆；S1/S2/S5/S6，S6 指反馈监控 |
| TD-MPC2 | 训练、评测、潜空间后果预测、逐步闭环 | ManiSkill 状态模型；具备 S1–S6 挂点，图像攻击不适用 |
| TD-MPC2 中心场景 | 独立检查点训练和评测 | 复用现有中心场景检查点 |
| OpenVLA Bridge | 原生动作推理、SimplerEnv 回合 | S1/S5 和适用的 S6 反馈监控，无状态/预测/规划探针 |
| OpenVLA LIBERO-Spatial | 微调模型视觉抓放 | 复用视觉子平台适配器、环境与权重；支持其图像干预，尚非通用插件内核 |
| Dreamer RSSM | 原生推理和预测适配器 | 工程检查点离线后果预测 |
| RLVR-World 语言 / 视频 | 原生预测 | 既有本地模型及测试输入 |
| Wan2.1 / ModelScope | 文本生成视频 | 轻量参数用于接入冒烟，不是生成质量基准 |
| Cosmos3-Edge | 图像生成视频 | 本地 Diffusers 模型；默认官方图像、提示词 |
| Cosmos3-Edge-Policy-DROID | 离线动作块推理 | 输出动作 JSON，不操控实体机械臂；Isaac Sim 闭环未通过 |
| RENEW | 2048 小规模反馈更新 | 机制实验，不等于机械臂安全训练更新 |
| Small State WM | 数据采集、训练、独立后果评价 | 小型状态模型工程验证 |
| EmbodiedJev | MuJoCo baseline / Chat / JEV | baseline 不调用远程模型；Chat/JEV 需私有凭据 |
| Toy、ManiSkill、SimplerEnv | 接口、仿真、规划器与回放 | 是辅助任务，不应重复统计成预训练模型 |

页面显示的挂点来自该配方实际模型的探针声明。资源就绪、Python 任务完成、机器人任务成功、安全性验收，是四个不同层次。少量种子上的运行不能代替 G2/G3 正式安全验收。

## 配置与扩展

- 共享配方：`native/*/recipes.json`。
- 共享模型说明和轻量运行参数：`native/model-library.json`。
- 本机解释器、权重、缓存：`native/*/runtime*.json`，不提交 Git。
- 本机输入模板：`model-library.local.json`，不提交 Git；用于指向现有示例图片、数据集与检查点。
- 本机连接检查说明：同一文件的 `availability_note` 字段；最近运行成功后页面自动隐藏。当前 Chat 的额度不足、JEV 的地区限制需要外部处理后重试。
- EmbodiedJev API 密钥通过环境变量或 `.local.json` 凭据文件引用；不要写入共享配方或页面配置。
- Cosmos 使用已有 `deployments/cosmos` 和 `deployments/cosmos-policy` 环境，模型适配层位于 `src/wms/engines/cosmos`。
- LIBERO 使用现有 `wm-visual-sublab` 适配器；新增入口仅改变任务所有权与输出目录，不修改子平台源码。

当新增机器配置时，同时检查间接资源：MemoryVLA 的 DINO/SigLIP 缓存、Cosmos Policy 的 Wan2.2 VAE、外部适配器源码、仿真资产。配方能够解析并不保证这些间接资源都可访问。

## 重跑接入检查

```bash
cd /home/suzhenjie2026/world-model/workspaces/wm-safetylab-e
./dev.sh info
./dev.sh python scripts/smoke_model_library.py \
  --port 8785 --report .workspace/results/model-smoke-new.json \
  --recipes native-cosmos tdmpc2-pushcube-episode memoryvla-bridge-episode
```

脚本逐个提交并等待任务和独立评价结束，可凭同一报告断点继续。要重测已结束或失败的项目，使用新的报告文件名。运行前查看 GPU 资源；不同开发工作区之间没有 GPU 预约或抢占协调系统。

默认模型参数是轻量验证参数。训练、轨迹回放及部分后果预测还需要明确输入检查点、源运行或轨迹文件。不要把小预算训练结果覆盖共享模型。

## 本轮验证

运行编号、原始失败、修复后重测、正式 E 服务验证见同目录的 `E-MODEL-ACCEPTANCE-20261125.md`。保留失败记录用于追溯，不把失败运行重标成成功。

2026-11-25 已实跑 22 个代表性配方，其中 20 个本地入口完成；2 个远程入口分别受额度、地区限制。40 个配方全部纳入目录，但没有重新跑全部训练和全量官方评测。S1–S6 中性配置已在 Toy 和真实 TD-MPC2 上检查，完整攻击/防御安全效果仍需按正式方案验收。
