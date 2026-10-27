# 学习路径：从读代码到交付一张 S1–S6 插件卡

> 面向"要接手这个平台、并参与 S1–S6 后续研究"的新成员。
> 结论先行：**这个项目的难点不在模型代码，在契约。** 读代码应围绕三件事转——内核在什么时机调你、你怎么证明自己没改变基线、一次实验怎么配对比较。12 个 engine 适配器只需要精读两个。

---

## 0. 开始之前：先纠正三个预期

### 0.1 README 里的"17 张卡"目前是 0 张

`README.md` 第 5 节写"第一阶段的 17 张插件卡由三人分工开发"，但当前代码状态是：

| 证据 | 实际 |
|---|---|
| `src/wms/plugins/cards/` | 只有 `__init__.py`（0.2 KB），无任何 `*_*.py` |
| `native/plugins_a/`、`_b/`、`_c/recipes.json` | 三个文件均为 **0 字节** |
| 2026-10-20 审查包 | 明确记录"17 张卡完成数为 **0/17**" |

所以：**S1–S6 的"后续研究计划"= 待实现的工作，不是待学习的既有功能。** 学习路径的终点必须落在"亲手做出一张卡"，否则读完仍然没有产出。

### 0.2 七个契约缺口（R1–R7）挡在第一批卡前面

审查包 `04-review-findings.zh-CN.md` 列了 7 项，不是 bug，而是**开写前的规格问题**：

| 编号 | 一句话 | 影响哪张卡 |
|---|---|---|
| R1 | `requires=state` 分不出 read / written 阶段 | #6、#7 |
| R2 | `map_floats` 不是深复制，整数张量共享 | #6、#7 |
| R3 | `after_step` 的 strict 只读检查可被绕过；缺统一 ledger 视图 | #13、#14、#15 |
| R4 | 最终接管动作没走 shape/finite 校验；接管冲突口径未定 | #13、#16 |
| R5 | `compare_runs` 按位置配对，`parity.passed` 只看首步与结果 | **所有卡** |
| R6 | clean stats 缺单位、模型版本、phase 字段 | #1、#4、#5、#8、#9、#11、#13、#15 |
| R7 | #11 公式是软惩罚，不保证硬可行性；#13 neutral 语义歧义 | #11、#13 |

> 读代码时把这张表放手边：**你读到的每一处"能跑"，都要问它属于"已实现"还是"只是可承载"。**

### 0.3 命令分两类：要服务的 / 不要服务的（实测）

| 命令 | 需要 `wms serve` | 说明 |
|---|---|---|
| `python -m wms recipes` / `runs` / `logs` | **要** | 它们都是 HTTP 客户端，没服务会直接抛 httpx 连接错误 |
| `python -m wms run <recipe> --wait` | **要** | `wms.run()` = `POST /runs` |
| `python -m wms plugin test <name>` | **不要** | 子进程跑 `wms.plugins.conformance`，用 `native/toy/runtime.json` 的 python |
| `python scripts/run_direct.py <recipe> <run_id>` | **不要** | 绕过队列，直接按 runtime python 跑 |
| `python -m pytest tests` | **不要** | 但需要 numpy |

`.venv` **没有** numpy；`native/toy/runtime.json` 已指向 `K:/world-model/envs/wms-onboarding/Scripts/python.exe`（3.11 + numpy 2.4.6）。所以 `pytest` 和 `plugin test` 要用那个解释器，或者直接让 `plugin test` 自己转过去。

**已验证可用**（2026-10-21 实跑）：`python -m wms plugin test example-action-scale` 全绿——4 个 toy 组合，每个 5 项 PASS（runs / own randomness / called / reproducible / neutral）+ 1 项 INFO(effect)。组合数取决于插件的 `requires`。

另一个环境问题：`C:\Users\hhh\.gitconfig` 第一行是损坏的 section header，导致**所有 git 命令失败**（`fatal: bad config line 1`）。想用 `git log` / `git diff` 学历史，先修它。

---

## 1. 学习目标：不是"读完代码"，是能回答五个问题

学完必须能不看目录树回答：

1. 一份 `recipes.json` 里的条目，怎么变成一次 run？配置覆盖顺序是什么？（`core/registry.py`）
2. 一次 run 的进程是谁起的、产物落在哪、`state` 和"实验成功"是不是一回事？（`core/runner.py`、`runners/unified.py`、`core/storage.py`）
3. 内核在每一步的哪几个点调用插件？每个点能读什么、能改什么、能不能读仿真真值？（`episodes/kernel.py`、`plugins/base.py`、`plugins/runtime.py`）
4. 我挂上插件后，凭什么证明"除了我想改的，其它都没变"？（conformance 的 neutral、`core/parity.py`、`compare_runs.py`）
5. 一张卡从规格到验收要交什么？（计划文档 + 配对协议 + 剂量曲线）

问题 3、4 是分水岭。答不出 4，你写的所有实验结论都不成立。

---

## 2. 两周路径（全职约 5 天，业余约 10 个半天）

### 阶段 A｜让它跑起来（半天）

| 步骤 | 命令 | 通过标准 |
|---|---|---|
| A1 | 修 gitconfig，或 `git -c` 临时绕过 | `git -C . log --oneline -3` 有输出 |
| A2 | `python -m wms serve --port 8765`（另开终端） | <http://127.0.0.1:8765/docs> 打开 |
| A3 | `python -m wms recipes` | 能列出配方，不再抛 httpx |
| A4 | `python -m wms run demo-rollout --wait` | 状态 `succeeded`，`runs/<id>/output` 有 `result.json` |
| A5 | `python -m wms plugin test example-action-scale` | 全 PASS（已验证） |
| A6 | `python -m pytest tests`（用 numpy 解释器） | 全绿；记下跳过了哪些 |

**A 阶段的产出**：能说清"哪些命令离不开服务"，以及 `runs/<id>/output` 里 `episodes.json`、`steps.jsonl`、`trace.jsonl` 各是什么。

### 阶段 B｜沿一次实验读代码（1.5 天，约 240 分钟）

直接复用审查包里的 **10 站清单**：`K:\world-model\deployments\wm-safetylab-review\2026-10-20-reading-pack\00-code-reading.zh-CN.md`。它已经按"一次 TD-MPC2 PushCube 实验"设计好了顺序和每站的提问。不要另起一套。

但**不要只是读**。每站结束后逼自己产出一个东西：

| 站 | 读什么 | 必须产出（写在自己的笔记里） |
|---|---|---|
| 1 | `registry.py` + `native/tdmpc2/recipes.json` | 画一张配置合并顺序图：默认值 → 配方 → 请求覆盖 → runtime |
| 2 | `runner.py`、`unified.py`、`storage.py` | 一句话回答"进程退出码 = 任务状态 = 实验成功？" |
| **3** | `protocols.py`、`kernel.py`、`steplog.py` | **手画一步的时序图**：obs → S1 → act → S2/S3/S4 → S5 → env.step → S6 → log。标出 `proposed` vs `executed`、`stop` 时 transition 为 None |
| **4** | `base.py`、`runtime.py`、`conformance.py`、`tree.py` | 列出"插件绝对不能做的事"，并指出哪些只是**约定**、哪些有**代码强制** |
| 5 | `toy/env.py`、`toy/planner.py`、`examples.py` | 说明为什么 toy 通过 ≠ 真实模型有效 |
| 6 | `tdmpc2/adapter.py`、`native.py`、`maniskill/state.py` | 标出 4 个探针（state/predict/plan/embed）各自在哪一行被填充 |
| 7 | `memoryvla/adapter.py`、`simpler/env.py`、`parity.py` | 回答"首帧有没有 read？written 拒写能改变同一步动作吗？" |
| 8 | `hazards.py`、`compare_runs.py`、`redteam/stats.py` | 写清每个危险谓词的输入、阈值来源、误报率 |
| 9 | `redteam/validation.py`、`experiments.py`、`Setup.tsx`、`Results.tsx` | 一次"攻击×防御"组合实际生成几组实验 |
| 10 | `test_episode_kernel.py`、`test_plugins_feedback.py`、`test_redteam_api.py` | 哪些行为**没有**测试 |

**B 阶段只精读 3、4、6 三站**，其余按需。站 7（MemoryVLA）等你真的要做 S2 的卡再读。

> 关键判断：`engines/` 下 12 个模型目录是**同构的**（`native.py` 原样调论文代码 + `adapter.py` 接内核 + `metrics.py`）。读懂 TD-MPC2 一个，剩下的只需要看它的 `probes` 声明。**不要逐个读。**

### 阶段 C｜写一个插件（1 天）

以 `templates/plugin/` 为起点，写一个**故意无用**的插件（比如"把动作第 0 维乘以 0.99"），走完整个回路：

1. 照 `templates/plugin/README.zh-CN.md` 建包，注册到自己的 `recipes.json`
2. `python -m wms plugin test <你的插件> --registry <你的 recipes.json>`
3. 故意违反一次契约（在 monitor 里改 `next_obs`），看它**是否**被拦住——这一步直接对应 R3
4. 故意把 neutral 配成"不等于不挂"，看 conformance 是否报错

**通过标准**：五项 PASS（runs / own randomness / called / reproducible / neutral），且你能解释 neutral 检查到底比对了什么。做完这一步，你就理解了 R5 为什么重要。

### 阶段 D｜做一张真卡（3–5 天）

**第一张卡选 #1（S1 观测损坏）**，理由：

- S1 不需要任何探针，任何模型都能挂，不会被 R1/R2 卡住；
- 剂量直观（按损坏类型 × 5 级严重度）；
- 它同时是 #4、#5、#7、#9、#15 的**共同故障源**，做完立刻被复用；
- 验收标准现成：强度 0 必须与基线逐字节一致。

**第二张选 #14（S5 违规与接管记录）**，因为它是全组的危险指标来源；但它依赖 R3（ledger）和 R6（clean stats），所以放第二。

执行顺序（每张卡都一样）：

```
读该卡的"任务计划"文档  →  读它引用的 1–2 篇论文  →  查它依赖的 R 项是否已修
→  先在 50 个 toy 种子上做完  →  再上一个真实模型（TD-MPC2 或 MemoryVLA）
→  剂量曲线 3 档 + 强度 0 对齐  →  写一页说明  →  合入
```

**不要跳过 toy 直接上真实模型**：真实模型一轮 50 种子要几小时，而 toy 几分钟就能暴露接口错误。

---

## 3. S1–S6 计划地图

计划文档在 `K:\world-model\deployments\claude-export-analysis-2026-10-19\S1-S6任务内容\`，每阶段一份；17 张卡的总表在 `S1-S6分工计划\00-第一阶段总计划.md` 第 128–150 行。

| 阶段 | 卡 | 计划文档 | 代码落点（待建） | 验收模型 | 前置 |
|---|---|---|---|---|---|
| **S1** | #1 观测损坏、#2 相机抖动 | `S1-输入与感知任务计划.md` | `cards/a_*.py` | MemoryVLA / OpenVLA / TD-MPC2 | R6 |
| S1 | #3 遮挡块攻击、#4 帧守卫、#5 OOD 监控 | 同上 | `cards/a_*.py` | MemoryVLA | R6（#5） |
| **S2** | #6 记忆故障、#7 写入门控 | `S2-状态与记忆任务计划.md` | `cards/b_*.py` | MemoryVLA + toy | **R1、R2** |
| **S3** | #8 latent 噪声、#9 残差监控 | `S3-预测与残差监控任务计划.md` | `cards/b_*.py` | TD-MPC2 | R6（#8、#9） |
| **S4** | #10 cost 误设、#11 候选重排 | `S4-规划打分与约束任务计划.md` | `cards/b_*.py` | TD-MPC2 | **R7（#11）** |
| **S5** | #12 动作噪声/延迟、#13 动作屏障、#14 违规接管记录 | `S5-动作执行与安全记录任务计划.md` | `cards/c_*.py` | MemoryVLA + TD-MPC2 | **R3/R4（#13、#14）、R7（#13）** |
| **S6** | #15 CUSUM、#16 告警接管、#17 反馈投毒 | `S6-回合反馈监控与干预任务计划.md` | `cards/c_*.py` | TD-MPC2；#17 仅 toy | **R6（#15）**、R3（告警记账）、#15 先于 #16 |

**依赖链**（决定你做卡的顺序）：

```
前置闸门：R5（验收口径）→ R1/R2（接入与状态复制）→ R3/R4（记账与接管）

做卡顺序：
  #1 ─┬─→ #4 / #5 ─→ #9 ─→ #15 ─→ #16
      ├─→ #7
      └─→ #12 ─→ #13 ─→ #14
```

> #14 是"全组危险指标"的来源，但它自己依赖 R3/R6。因此**最省事的第一步是 #1**，同时把 #14 的字段名先定下来。

---

## 4. 三条学习原则

### 4.1 以"契约"为轴，不以"功能"为轴

| 层 | 要不要精读 | 理由 |
|---|---|---|
| `core/`（registry / runner / storage / evaluation / parity） | **要** | 决定"实验是什么" |
| `episodes/kernel.py` | **要，逐行** | 决定"每一步发生什么" |
| `plugins/base.py` + `runtime.py` + `conformance.py` | **要，逐行** | 决定"你能做什么、怎么被验证" |
| `engines/tdmpc2/` + `engines/memoryvla/` | **各一个就够** | 两种极端：全探针 vs 记忆读写 |
| 其余 10 个 engines | 只看 `probes` / `capabilities` 声明 | 同构 |
| `web/` | 最后再看 | 它只读已有证据，不产生证据 |
| `redteam/` | 做攻击类卡前再看 | 是实验编排，不是内核 |

### 4.2 一切以"配对比较"为单位

平台的验收语言只有一种：**同种子、同布局，挂插件 vs 不挂插件，报告差值与 95% 区间**。单独报一个成功率不算结果。学到任何结论，先问"基线和它的种子对得上吗"。

### 4.3 先无 GPU，后有 GPU

CPU 能覆盖：toy 全链路、插件一致性、`compare_runs` 配对、`pytest`。GPU 只用于真实模型验收。把接口错误留在 CPU 阶段暴露。

---

## 5. 常见误区

| 误区 | 为什么错 |
|---|---|
| 从 `web/` 或 `redteam/` 开始读 | 它们是证据的展示与编排层，读完仍不知道一步怎么走 |
| 通读 12 个 engines | 同构；读两个足够 |
| 把 README 的"17 张卡"当成已有功能 | 实际 0/17 |
| 把 `max_abs_action_diff = 0` 当成通过 | R5：按位置配对，缺回合、多维动作都会被忽略；`compare_runs` 也不返回失败退出码 |
| 把 `parity.passed` 当成"全轨迹一致" | 它只看首动作与结果 |
| 用随机噪声冒充攻击 | 攻击必须有威胁卡（knowledge / access / budget / goal），且要报查询预算 |
| 把任务成功率当安全率 | TD-MPC2 的 latent MSE 没有物理单位；成功率不度量接触力或越界 |
| 用整个动作范围的百分比当扰动强度 | 应该用干净运行的 `action.std` / 分位数 |

---

## 6. 一页 checklist

**第 1 天**
- [ ] 修 gitconfig，`git log` 可用
- [ ] 起服务，`wms recipes` / `run demo-rollout` 跑通
- [ ] `wms plugin test example-action-scale` 全 PASS
- [ ] `pytest tests` 全绿，记下跳过项

**第 2–3 天**
- [ ] 读完 10 站清单的 1–5 站
- [ ] 手画一步的时序图（含 proposed/executed、stop 无 transition）
- [ ] 能说出"插件绝对不能做的事"里哪些有代码强制
- [ ] 精读 `kernel.py`、`base.py`、`runtime.py`

**第 4 天**
- [ ] 精读 `tdmpc2/adapter.py` + `maniskill/state.py`
- [ ] 写一个玩具插件并通过 conformance 五项
- [ ] 故意违反一次契约，确认拦截行为并对应到 R3

**第 5 天**
- [ ] 读 `S1-输入与感知任务计划.md` 的 #1 段 + ImageNet-C / LIBERO-Plus
- [ ] 读 R6，确认 #1 需要的统计字段是否齐
- [ ] 在 toy 上实现 #1，强度 0 与基线逐字节一致
- [ ] 3 档剂量曲线跑完

**第 2 周**
- [ ] #1 在 MemoryVLA 或 TD-MPC2 上验收，写一页说明
- [ ] 开始 #14 的字段定义（先定字段，再实现）
- [ ] 挑一张同阶段的卡做第二张

---

## 7. 读完应该能反驳的一句话

如果有人在组会上说"我们的平台已经支持 S1–S6 的安全插件了"，你应该能准确回答：

> 框架支持在 S1–S6 注入，TD-MPC2 和 MemoryVLA 已通过 A/B 逐步一致；但 17 张卡是 0/17，危险阈值刚标定过一轮，配对比较的门禁（R5）和回合记账（R3/R4）还没收紧。所以现在能说的是"可承载"，不是"已完成"。

能说出这句话，这个项目你就学明白了。
