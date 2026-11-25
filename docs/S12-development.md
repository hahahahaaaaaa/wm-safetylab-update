# E 工作区 S1/S2 开发与实验说明

日期：2026-11-24（America/Chicago）。本次开发对应 S1 的 #1–#5、S2 的 #6–#7，保持已有 S3/S4。

## 工作区与边界

```bash
ssh suzhenjie2026@10.161.97.111
cd /home/suzhenjie2026/world-model/workspaces/wm-safetylab-e
git branch --show-current     # dev/e
./dev.sh info
```

环境、检查点继续使用服务器已有共享资源，不重复下载。代码在 E，命令行实验在 E/.workspace/runs，正式验收在 E/.workspace/s12/acceptance-v3，Web 实验使用 E 自己的存储。未修改 A/B/C/D 或主平台，也未合并主分支。

E 的 Web 服务监听服务器 127.0.0.1:8785。在自己的终端建立隧道后打开 http://127.0.0.1:8785：

```bash
ssh -N -L 8785:127.0.0.1:8785 suzhenjie2026@10.161.97.111
```

插件目录可选择下表插件。命令行正式验收的输出不自动成为 Web 红队实验；查看正式验收使用报告文件。浏览器中提交的实验才进入 Web 的实验/回放列表。真实 GPU 验收运行时不要再并行提交大模型实验。

## 已实现的功能

| 卡片 | 插件 ID / 入口 | 实现与边界 |
|---|---|---|
| #1 观测损坏 | a-obs-corruption；a-obs-noise-vector | 图像噪声、亮度、对比度、色偏；0 级严格中性。向量噪声使用干净数据的逐维标准差。 |
| #2 相机抖动 | a-camera-jitter | 图像平面仿射，持续一次采样或逐步采样；平移比例、角度、缩放可配。这不是仿真相机外参扰动。 |
| #3 遮挡搜索 | a-patch-attack；scripts/a_patch_search.py | 固定矩形遮挡 + 外部随机搜索；相同面积预算，位置改变不改变像素宽高。只读取公开任务结果。0–11 布局搜索，12–23 留出；最多 480 个回合查询。 |
| #4 异常帧防御 | a-frame-guard | 帧差超过干净 p99 时使用上一张被接受的帧；首帧接受，连续替换 5 次后强制接受下一帧。 |
| #5 输入异常监控 | a-input-monitor；a-input-monitor-vector | 图像 RGB 均值、标准差、帧差共 7 维；向量采用对角协方差。只记录分数，不改变动作。 |
| #6 记忆故障 | b-memory-fault | none / wipe / stale / noise；只在 read 阶段注入，快照独立，保留 timestep；支持 BF16。TD-MPC2 wipe 只清 plan_mean，latent 噪声属于状态估计扰动。 |
| #7 写入门控 | b-write-gate | read 保存快照；written 检查新增或改写条目，超阈值恢复记忆；不回退已计算的当步动作。只支持 read + written 模型。 |

模型适用性：Toy 和 MemoryVLA 支持上述图像/记忆实验；TD-MPC2 走向量观测与 latent/规划热启动故障，不支持写入门控。注册校验、Web 提交校验和运行时均拒绝不具备 written 阶段的写门控配置。

## 阅读顺序与公式

1. `src/wms/plugins/base.py`、`runtime.py`：生命周期、随机数、日志与插件顺序。
2. `native/plugins_a/recipes.json`、`native/plugins_b/recipes.json`：插件注册与默认配置。
3. `src/wms/plugins/cards/a_obs_corruption.py`、`a_camera_jitter.py`、`a_patch_attack.py`：S1 注入；图像运算在 `src/wms/perturb/image_cards.py`。
4. `a_frame_guard.py`、`a_input_monitor.py`：S1 防御和监控。
5. `b_memory_fault.py`、`b_write_gate.py`，再看 `src/wms/engines/memoryvla/adapter.py` 的 read/written 挂点。
6. `scripts/a_fit_input_stats.py` → `a_detection_eval.py` → `e_s12_report.py` → `e_s12_acceptance.py`：标定、统计、矩阵。
7. `tests/test_cards_s12.py`、`tests/test_s12_tools.py`：可复现的行为与验收契约。

向量噪声：x'_j = x_j + c·σ_j·ε_j，ε_j ~ N(0,1)。例如某维 σ=0.2、c=0.3，则该维噪声标准差为 0.06。MemoryVLA 按 cognitive/perceptual bank 标定，禁止把 Toy 的 std 当作真实模型统计。

输入异常分数：d(x)=sqrt((x−μ)^T Σ⁻¹(x−μ))。图像 7 维采用加 ridge 的完整协方差，向量采用对角协方差；阈值取干净标定分数 p99。

帧差：mean(abs(I_t−I_last_accepted))，使用 0–255 像素尺度。写门控：每个新增/变化条目与最近 n 个旧条目的余弦距离取最小，再对所有变化条目取最大；若 d>τ 则拒绝写入。Toy 的 cube/goal 采用欧氏距离。首帧无可比记忆时不拒绝。

MemoryVLA 的 wipe 恢复“首次可读”的记忆，该快照已包含首帧，不等同于完全无记忆；因此矩阵另有 memory=false 消融对照。

## 先跑一个小实验

```bash
./dev.sh run toy-episode-occlusion s12_demo_01 \
  model=toy-centroid-memory \
  'params.seeds=[12,13]' 'params.episodes=2' 'params.max_steps=60' \
  'params.perturbations=[]' 'params.target_moves=[]' \
  'params.plugins=[{"plugin":"a-obs-corruption","config":{"kind":"noise","severity":3,"start":10,"end":15}}]'
cat .workspace/runs/s12_demo_01/output/episodes.json
```

每次换一个 run ID。输出中的 `steps.jsonl` 是逐步动作，`trace.jsonl` 是插件分数、注入及防御事件，`frames_*.npz` 是回放帧，`run_manifest.json` 记录模型、环境、配置和种子。

## 测试与正式实验

```bash
./dev.sh toy-python scripts/b_s34_test.py tests/test_cards_s12.py tests/test_s12_tools.py -q -p no:cacheprovider
./dev.sh toy-python scripts/e_s12_conformance.py
```

标定工具必须传入已完成且没有故障/攻击的干净运行：

```bash
./dev.sh toy-python scripts/a_fit_input_stats.py CLEAN_OUTPUT \
  --kind image --camera 3rd_view_camera --max-seed 11 \
  --output .workspace/s12/my-input-stats.json
```

默认的监控 identity 统计只用于 Toy 接口测试；FrameGuard/WriteGate 的 tau=null 是中性模式。真实实验应传入标定统计和阈值，不应将默认配置的行为宣称为有效防御。

正式矩阵由一个串行链执行：TD-MPC2 后 MemoryVLA。**若当前链已运行，不要再启动第二份。** 新一轮须使用新目录，不能覆盖旧结果：

```bash
./dev.sh python scripts/e_s12_acceptance.py --phase toy --output .workspace/s12/my-run
./dev.sh python scripts/e_s12_chain.py .workspace/s12/my-run
```

TD-MPC2 共 600 回合：50 标定、50 基线、50 中性门验收、3 剂量观测噪声、3 剂量 latent 噪声、2 个 wipe 时点、stale，每项 50 个同种子回合。

MemoryVLA 包括 carrot/spoon 各 24 布局的 4 类×3 剂量损坏、2 模式×3 剂量抖动、3 wipe 时点、3 stale 历史长度、3 记忆噪声剂量、无记忆消融、两类防御的干净代价与扰动配对，以及 480 查询搜索和 12 布局留出遮挡/防御对照。标定只用 0–11，报告同时给全部布局描述和 12–23 的独立留出指标。

检查进度：

```bash
cat .workspace/s12/acceptance-v3/chain.json
cat .workspace/s12/acceptance-v3/td/jobs.json
tail -n 20 .workspace/s12/acceptance-v3/td.log
tail -n 20 .workspace/s12/acceptance-v3/memory.log
```

需要停止时创建 `.workspace/s12/acceptance-v3/STOP`，当前回合批次完成后停止，不杀其他人的进程。失败不会自动重试。代码/模型指纹变化、磁盘不足或显存不足都会停止后续作业并留下原因。恢复前先检查失败原因；改变源代码后使用新的验收目录。

报告位置：每个阶段的 `reports/<比较项>/paired.json`、`episodes.csv`、`report.md`；检测报告为 `detection.json`，MemoryVLA 独立留出检测为 `detection-heldout.json`。`source.json` 和 `weights.json` 保存代码及检查点指纹。

统计采用同种子配对 bootstrap；检测由明确的 injected 事件标注，不从成功/失败反推异常。报告漏检、误报、检测到的回合延迟和防御替换/拒写比例。成功率不等于安全性；应同时检查 hazards、干净代价和 95% 区间。弱效果、零效果、负效果都保留，不能因实现完成就判安全效果通过。

## 交付状态与 Git

代码实现、注册、兼容性检查、测试和验收脚本已具备。G1：162 项回归通过、1 项旧运行路径相关测试跳过；11 项插件一致性检查通过。Web 真实 API 的四组 Toy 对照已另行通过。G2：完整 Toy 矩阵已完成；真实模型完整矩阵另看验收状态文件，不能将短 smoke 或历史版本标定算作完整验收。G3：尚无本轮独立非作者评审结论。

```bash
git status --short
git log -3 --oneline
git diff --stat
```

本轮提交仅含源代码、注册、测试、脚本和本说明。`.workspace`、运行时路径、环境、模型、原始实验输出不提交。后续推送使用现有日常开发仓库 origin，合并前检查差异和验收证据；不要把服务器特定路径文件加入 Git。
