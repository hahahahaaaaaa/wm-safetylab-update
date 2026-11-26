# E 平台已有模型接入与测试报告

日期：2026-11-25。工作区：`/home/suzhenjie2026/world-model/workspaces/wm-safetylab-e`，分支 `dev/e`，服务版本 0.3.0。

## 结论与范围

本轮盘点到的已有模型入口已统一纳入 E 的模型工作台。目录由 37 个配方扩展到 40 个，新增 Cosmos3-Edge、Cosmos3-Edge-Policy-DROID、OpenVLA LIBERO-Spatial。原有模型适配器和 S1–S6 安全实验入口保留，环境、权重、仿真资产尽量复用服务器现有资源，实验任务与结果由 E 管理。

在正式 E 服务逐项运行了 22 个代表性配方：20 个本地模型/仿真/示例入口完成，2 个远程模型入口因外部服务限制失败。这里的 40 是配方数，20 是通过的入口数，均不是独立预训练模型数量；同一模型可以有推理、训练、评测等多个配方。

本轮没有逐一重跑全部 40 个配方，也没有重新训练全部模型。已覆盖本次盘点到的本地模型类别与主要原生执行路径；训练、官方全量评测、部分重放和一致性配方仍沿用已有实现。目录中的“资源就绪”只代表本机配置和路径检查通过。

S1–S6 另外完成 Toy 与真实 TD-MPC2 的中性配置组合检查，共 16 回合；组合钩子齐全，逐步动作与无插件参考完全一致。本轮是接入、接口和不干预行为验证，不是攻击/防御效果的完整 G2/G3 验收，也没有新增独立交叉评审。

## 本次完善

- 新增“模型工作台”：模型类别、任务配方、JSON 配置、提交、状态、取消、增量日志、结果 JSON、视频预览和产物下载。
- 保留原有“新建实验”的八步 S1–S6 对照流程。各配方显示实际探针支持的阶段；视频生成、离线预测不冒充完整逐步闭环。
- 新增 Cosmos 视频与策略的适配层；策略只执行离线动作块推理，不连接实体机械臂。
- 将现有视觉子平台的 OpenVLA LIBERO 适配器作为 E 的原生入口复用，输出落到 E。
- 补齐 EmbodiedJev 的独立环境和已有私有连接引用；密钥未进入 Git、页面配置或报告。
- 修复无效嵌套运行配置的错误处理、配方默认随机种子的继承、服务器 Toy 测试解释器配置。
- 限制页面自动预览为最多 3 个视频和 6 张图片，并保留全部下载链接；LIBERO 的数百张帧图不会一次全部加载。
- 在 Chat/JEV 配方页面显示本轮实测的外部阻塞原因；后续成功重测后该说明自动隐藏。

## 正式 E 的模型实跑

下表“完成”表示执行入口和产物链路完成，不自动等于任务成功、安全有效或模型质量达标。运行明细见 `.workspace/results/models-20261125/e-model-smoke.json`；通过模型工作台“运行与结果”按编号查找。

| 配方 | E 运行编号 | 结果与解释 |
|---|---|---|
| demo-predict | `run_9d0caecd21a944ba` | 完成；合成预测接口示例 |
| toy-episode-occlusion | `run_9b02ece1c6b3410c` | 完成；10 步预算，任务未成功，属于流程冒烟 |
| native-dreamer | `run_e7ed5cc6c8864090` | 完成；既有 Dreamer 原生推理 |
| native-renew | `run_7a81b2961cc54cd9` | 完成；小规模反馈更新和检查点产物 |
| native-maniskill | `run_8d6ac59e0a5f488a` | 完成；仿真轨迹和视频 |
| state-wm-maniskill | `run_81fd02ce8aa84d6e` | 完成；8 回合、240 条转移的小模型工程检查 |
| tdmpc2-pushcube-episode | `run_d9d6d77ec05b4ea6` | 完成；种子 0，50 步，任务成功 |
| embodied-jev-baseline | `run_72a7386fb45e46de` | 完成且任务成功；不调用远程模型 |
| native-cosmos | `run_7ddda59411654ea8` | 完成；320×192、53 帧视频，成功解码 |
| native-modelscope | `run_5317a19059714d1d` | 完成；轻量文本生成视频 |
| native-wan | `run_2fc4aff300a9404d` | 完成；轻量文本生成视频 |
| predict-rlvr_video | `run_5f90aea803f04e70` | 完成；真实下一帧预测，非闭环控制 |
| predict-rlvr_language | `run_d2b7a985506c4d62` | 完成；真实状态预测，JSON 校验通过 |
| native-openvla | `run_481ee8f3c3a74e57` | 完成；原生动作推理 |
| openvla-bridge-episode | `run_f40000f706434fac` | 完成；120 步闭环，任务未成功，未宣称性能提升 |
| memoryvla-bridge-episode | `run_c9b7e123f4e14569` | 完成；种子 0，120 步，任务成功 |
| native-cosmos-policy | `run_d56016643ca14d70` | 完成；32×8 动作、数值有限；不是仿真任务验收 |
| openvla-libero-spatial | `run_bddfa895bc2c4e41` | 完成；86 个策略步，抓碗放盘任务成功，353 个产物 |
| tdmpc2-pushcube-center-eval | `run_10b0e1e958fc4c02` | 完成；场景种子 2000，50 步，中心任务成功，末帧距目标约 0.006 m |
| predict-dreamer | `run_1bb04de014164285` | 完成；真实 RSSM 后验更新与固定动作想象，不代表策略收敛 |
| embodied-jev-chat | `run_3d8c08470de742f3` | 失败；服务方 HTTP 405，实际错误为密钥额度不足 |
| embodied-jev-jev | `run_f09195a46a714a35` | 失败；TypeSafe HTTP 451，当前服务器地区不可用 |

Cosmos Policy 的输入是既有官方视频生成样例图像与零关节状态，仅用于验证模型加载、调用和动作输出。其结果明确记录 `robot_actuated=false`、`simulation_validated=false`；已有 Isaac Sim 渲染/驱动障碍尚未解决。

Chat 使用已有连接配置的 GPT-5.5；按服务方官方备用地址重试一次（`run_4f4e4a4be1964fea`）仍返回同一额度错误，随后恢复原地址。JEV 使用已有 `jev-latest` 配置，未尝试绕过地区限制。两项需要账户管理员补充额度或服务方确认地区支持后再验收。失败产物中即使有视频，也只是初始仿真画面，不证明远程模型完成推理。

其余 18 个配方已纳入目录且资源检查通过，本轮未单独实跑：`demo-rollout`、`demo-generate`、`demo-plan`、`demo-native`、`maniskill-hold`、`maniskill-pickcube-planner-episode`、`maniskill-replay`、`memoryvla-simpler-bridge-eval`、`memoryvla-bridge-official-carrot`、`memoryvla-bridge-episode-occlusion`、`memoryvla-bridge-parity`、`simpler-bridge-replay`、`tdmpc2-pushcube-train`、`tdmpc2-pushcube-eval`、`tdmpc2-pushcube-predict`、`tdmpc2-pushcube-parity`、`tdmpc2-pushcube-center-train`、`tdmpc2-pushcube-center-report`。

## S1–S6 组合与无插件参考

采用现有 `examples/s1-s6/toy-neutral.json` 和 `tdmpc2-neutral.json`，种子 3000，每模型 7 个实验臂、每臂 1 回合：A0 仅 S6 监控；A1–A5 分别加入一个阶段的中性故障配置；A6 全部组合。另为每个模型跑 1 回合完全无插件参考。

| 模型 | 组合实验 | 无插件参考 | 每臂步数 | A6 观测到的阶段 | A6 与无插件最大动作差 |
|---|---|---|---:|---|---:|
| Toy | `rt_bbff5a8c383e` | `rt_17de698aebae` | 17 | S1、S2、S3、S4、S5、S6 | 0 |
| TD-MPC2 × PushCube | `rt_f95a9ba49d3d` | `rt_8bea103f44c4` | 50 | S1、S2、S3、S4、S5、S6 | 0 |

所有实验臂与 A0 的动作差也为 0。中性参数包括观测/预测噪声 c=0、记忆故障 mode=none、打分误设 c=0、动作噪声 std=0，S6 保留残差监控。结果证明这些配置在该种子下不改变原动作，不证明非零攻击有效、真实危险检测有效或反馈训练回滚安全。

## 自动化测试、产物和页面

| 检查 | 结果 | 证据文件 |
|---|---|---|
| API、注册、红队、S1/S2、S3/S4、S5/S6、生命周期等回归 | 175 passed，2 skipped，7 subtests passed | `tests-regression.log` |
| 含 Torch 的插件测试环境 | 94 passed，7 subtests passed | `tests-plugins.log` |
| Linux Toy API 端到端补跑 | 1 passed | `tests-toy-e2e.log` |
| 最终目录/API/注册修复定向测试 | 20 passed | `tests-final-config.log` |
| TypeScript 与 Vite 生产构建 | 通过 | 发布 JS：`index-Dxz-Pqu8.js` |
| E 产物审计 | 22 次运行、484 个文件全部存在且非空，文件 API 返回内容与磁盘前缀一致；16 个视频可由 ffprobe 解析 | `artifact-audit.json` |
| 实际浏览器 | Cosmos 视频成功播放；LIBERO 显示 7 个预览、353 个完整下载链接；无横向溢出，无控制台错误；Chat/JEV 原因正确显示 | `ui-final.json` |

上述测试套件有重叠，不相加为独立用例总数。回归中的两个跳过项分别是 Toy 解释器路径和轻量环境缺少 Torch，均在对应可用环境补测。FastAPI 测试客户端有上游弃用警告，不影响本轮通过结果。

测试日志原件位于验证工作区 `.workspace/logs`；本机副本连同 `artifact-audit.json`、`ui-final.json`、最终目录快照，位于 `K:\world-model\deployments\wm-safetylab-e-models-20261125\evidence`。S1–S6 原始汇总、动作对比以及无插件参考编号也在该目录中。

先前还验证了配置 JSON 错误提示、提交、取消、日志和运行历史流程。最终页面视频播放检查记录 Cosmos 为 320×192、约 2.208 秒，播放时间正常前进且无媒体错误；LIBERO 视频为 448×224、4.35 秒。产物审计不是对每个文件完整内容做语义评价，也不把失败 API 的初始视频当作有效模型输出。

## 修复过程与隔离

先在独立验证工作区 `wm-safetylab-e-models-20261125` 完成修改、构建和运行，再在 E 无活动任务时快进合并并启动服务。主平台及 A/B/C/D 的代码和服务没有在本轮部署中替换。

验证工作区最初的 MemoryVLA 运行 `run_aab88a9d365b463c` 因 DINO/SigLIP 缓存映射缺失失败；链接现有缓存后 `run_90e17db755584079` 完成。Cosmos Policy 的 `run_835ad06a1f014fa6`、`run_bc3fe469a3474a8d` 因 uv/PATH 与离线 CLI 依赖缺失失败；补齐专用 CLI 环境并复用现有 VAE 缓存后 `run_36c7525b5ad1479e` 完成。正式 E 的对应运行也再次完成，原失败记录保留。

新增的 EmbodiedJev 和 Cosmos HF CLI 环境位于服务器 `world-model/envs`，没有覆盖原有模型环境或共享权重。E 输出位于 `.workspace/store/runs`，本轮汇总位于 `.workspace/results/models-20261125`。本机下载、源码镜像、报告及日志均位于 `K:\world-model\deployments\wm-safetylab-e-models-20261125`。

代码提交：`61193ed`（模型工作台和接入）、`aab7432`（产物预览数量限制）、`1f6d065`（默认种子、配置健壮性、远程连接提示）。后续报告文档提交不改变运行逻辑。E 的发布备份位于 `.workspace/deploy-backups/models-61193ed` 和 `models-1f6d065`。

## 查看和继续实验

本机保持 SSH 隧道：

```powershell
ssh -F NUL -N -L 8785:127.0.0.1:8785 suzhenjie2026@10.161.97.111
```

打开 <http://127.0.0.1:8785/#models/library>。如果已有 8785 隧道，无需重复启动。服务器服务绑定回环地址，服务器 IP 加 8785 不是当前直接访问方式。

```bash
cd /home/suzhenjie2026/world-model/workspaces/wm-safetylab-e
./dev.sh info
./dev.sh python scripts/smoke_model_library.py \
  --port 8785 --report .workspace/results/model-smoke-new.json \
  --recipes native-cosmos tdmpc2-pushcube-episode memoryvla-bridge-episode
```

配方默认种子现已继承原配方；上表记录本次运行实际种子，不把后来修复的默认值追写到旧实验。更大规模任务开始前先检查 GPU 占用；工作区之间仍共享计算资源，没有跨工作区的预约调度。

使用说明见 `E-MODEL-WORKBENCH.zh-CN.md`。后续重点是处理两个远程服务阻塞、按具体研究目标选择全量训练/评测预算，以及按 G2/G3 开展非零攻击/防御的配对实验和独立交叉评审。
