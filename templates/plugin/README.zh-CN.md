# S1–S6 插件模板

适用 WM-SafetyLab 回合内核（`wms.episodes.kernel`）与插件框架（`wms.plugins`），2026-10-18（第一阶段第 0 周：加入 S6 on_feedback、告警、embed / learn、rgb / vector / reward、neutral 检查）。

一个插件就是一个 Python 类，再加一条注册和一个 CPU 测试，不用改模型代码。它挂在闭环的某一个环节上：要么改这里流过的东西（故障、攻击、防御），要么只看不改（监控）。改完的东西和插件写下的事件都会记进每步的 `trace.jsonl`。

## 五步上手

1. 把 `templates/plugin/` 复制到自己的开发目录，把包名 `wms_plugin_template` 和注册名 `template-*` 都改成自己的前缀，避免与别人冲突。
2. 在 `gripper_flip.py` 的位置写自己的类：设好 `stage`、`role`，只实现本环节的钩子。
3. 在 `registry.json` 的 `plugins` 里登记：`entrypoint`、`stage`、`role`、默认 `config`；S2–S4 插件还要写 `side` 和 `requires`，攻击插件要写 `threat`。服务提交时就按这条登记做检查。故障和攻击还要在类上写 `neutral`：让插件不起作用的那组配置（强度 0），例如 `neutral = {"p": 0.0}`。
4. 跑一致性测试（几秒）：

   ```powershell
   . K:\world-model\use-k-drive.ps1
   Set-Location K:\world-model\deployments\wm-safetylab
   .\.venv\Scripts\python.exe -m wms plugin test <你的注册名> --registry <你的目录>\registry.json
   ```

   `--config` 可以给一段 JSON，或一个 JSON 文件的路径。测试用 `native/toy/runtime.json` 里那个带 numpy 的 Python 跑玩具环境，结果写在 `test-output/plugin-test/<注册名>/report.json`。
5. 在配方里挂上插件，到任务卡指定的真实模型上跑一次：

   ```json
   "params": {"plugins": [{"plugin": "<你的注册名>", "config": {"p": 0.2}}]}
   ```

## 环节与钩子

| 环节 | 钩子 | 收到什么 | 返回什么 |
|---|---|---|---|
| ENV | `on_env(t, ctx)` | 第 t 步开始；`ctx.env` 是环境本身 | 不返回；改了世界就 `ctx.log(...)`（内核据此重新渲染） |
| S1 | `on_observation(obs, ctx)` | 将送进模型的观测（`obs["rgb"][相机]` 为 HxWx3 uint8） | 新观测 |
| S1 | `on_instruction(text, ctx)` | 语言指令 | 新指令 |
| S2 | `on_state(state, ctx)` | 模型的记忆或潜状态；`ctx.phase` 为 `"read"`（读之前）或 `"written"`（写之后） | 新状态，或原样返回 |
| S3 | `on_prediction(pred, ctx)` | 一次预测；`ctx.meta["kind"]` 为 `"rollout"`（规划里的每步推演）或 `"chosen_next"`（选中动作的下一步预测，只作报告） | 新预测 |
| S4 | `on_plan(plan, ctx)` | `{"candidates", "scores", ...}` | 改 `scores` 即重排候选 |
| S5 | `on_action(action, ctx)` | 将送进仿真的动作 | 新动作；防御可调 `ctx.intervene("stop" 或 "fallback", action=...)` 接管 |
| S6 | `on_feedback(transition, ctx)` | 仿真走完一步后的 transition（见下） | 新 transition；会在线学习的模型从它学习，故障 / 攻击改它就是投毒 |
| S6 | `on_update` | 第二轮（P2）才接入（训练过程），目前内核会拒绝 | |

所有插件都可以另外实现：`reset(episode, ctx)`（每回合开始）、`after_step(step, ctx)`（仿真步之后，只读，`step` 含建议动作、执行动作、奖励、事件和本步的 `transition`）、`end_episode(ctx)`（返回的字典写进回合汇总 `plugin_reports`）、`close()`。`after_step` 里 `ctx.log` 的事件也写进本步 trace。

每步顺序：ENV → S1 → 模型内部 S2 / S3 / S4 → S5（之后执行告警要求的接管）→ 仿真一步 → 危险谓词 → S6 `on_feedback` → 模型 `learn` → `after_step` → 写日志。

### transition

```python
{"t": 3, "obs": ..., "action": array([...]), "next_obs": ..., "reward": 0.1, "done": False,
 "taken_over": False,          # 这一步是否被防御接管
 "predicted_next": ...,         # 模型对它所选动作的下一步预测；requires 含 "predict" 才有，否则为 None
 "next_state": ...}             # model.embed(next_obs)，与 predicted_next 同一空间；requires 含 "embed" 才有
```

`obs`、`next_obs` 是环境原样返回的观测（S1 改动之前）。`action` 是实际执行的动作（含接管）。危险谓词不在 transition 里，读它仍要 `ctx.hazards`。

同一钩子上的执行顺序：先 fault / attack，再 defense，最后 monitor。所以防御看到的是被攻击后的值，监控看到的是最终值。

## ctx 能用什么

| 名称 | 说明 |
|---|---|
| `ctx.t`、`ctx.episode`、`ctx.seed` | 步号、回合序号、回合种子 |
| `ctx.rng` | 本插件自己的随机数（numpy Generator，按回合种子和插件位置设定）。不要用 `numpy.random` 或 `random` |
| `ctx.config` | 本插件的配置 |
| `ctx.spec` | 环境规格：动作维度与范围、相机、步数上限、`safety`（危险判据的阈值） |
| `ctx.log(dict 或 key=value)` | 写一条事件到本步 trace。S1 / ENV 的 fault、attack 写的事件也会进 `steps.jsonl` 的 `perturbations` / `target_moves` 列 |
| `ctx.truth()` | 仿真真值（`env.get_state()`），需要 `privileged = True` |
| `ctx.hazards` | 上一步之后成立的危险谓词，需要 `privileged = True` |
| `ctx.locate(name)` | 物体在图像中的框，需要 `privileged = True` 且 `requires` 含 `"image_box"` |
| `ctx.env` | 环境本身，只给 ENV 插件 |
| `ctx.intervene(kind, action=None, reason="")` | 只给 S5 防御：`stop` 本步动作不执行并结束回合；`fallback` 改执行给定动作 |
| `ctx.alarm(reason, score=None, takeover=None, steps=1, action=None)` | 只给 S1–S4、S6 的防御：报告异常。`takeover` 为 `"stop"` 或 `"fallback"` 时，内核从下一个动作起接管 `steps` 步（若 S5 防御已接管则只记录），记在告警插件名下 |
| `ctx.alarms` | 自上一个动作以来的告警（本步 S1–S4 的，上一步 S6 / `after_step` 的）。S5 插件据此决定怎么处理 |

监控插件读 `ctx.truth()` 和 `ctx.hazards` 不需要声明 `privileged`。

## 必须遵守的规则（`wms plugin test` 会检查）

- 不要原地修改收到的值，返回一个新值。测试在严格模式下运行，原地修改会直接报错。
- 随机数只用 `ctx.rng`。测试会检查全局随机状态有没有被动过。
- 只实现自己环节的钩子。
- 防御默认不能读仿真真值。确实要读，写 `privileged = True` 并在配置里设 `"oracle": true`，结果会标成 oracle 上界。
- 攻击必须有威胁模型卡 `threat`：`knowledge`（`black_box` / `gray_box` / `white_box`）、`access`（从哪里进入）、`budget`（预算）、`goal`（目标），可选 `adaptive`（是否知道已挂的防御）。只有预算和知识相同的攻击才能比较成功率。
- 监控不能改变任何动作。测试会和不挂插件的运行逐步对比。
- 返回值必须和收到的值同一形式：同样的键、长度、数组形状和 dtype（S2 只查顶层键，因为清空记忆会改变长度；S4 只查 `scores`；S5 查动作长度和是否有限）。图像是 uint8 就返回 uint8。
- 故障和攻击必须声明 `neutral`，并且用 `neutral` 配置跑出来的结果和不挂插件逐字节一致（除了插件自己写进 `perturbations` / `target_moves` 列的记录）。强度参数用相对单位，例如动作范围的比例（`wms.plugins.tree.action_span(ctx.spec)`）、图像边长的比例，不写死米或像素。
- S6 的故障和攻击必须 `requires` 含 `"learn"`：不在线学习的模型不受 transition 改动影响。
- 处理记忆或潜状态时，把它当作整体快照：清空记忆 = 换成回合开始后第一次 `read` 时的快照；陈旧记忆 = 换成 k 步前的快照；加噪声只动浮点叶子（`wms.plugins.tree.map_floats`）。这样同一个插件对 MemoryVLA 的记忆库和 TD-MPC2 的潜状态都适用。
- 模块顶层只 import 轻量的东西（`wms.plugins`、标准库）。numpy、torch 放到钩子里 import，或在注册条目里写全 `stage` / `role` / `requires`，因为服务的 Python 里没有这些库。

## 各模型提供哪些探针

`requires` 里写需要的探针和环境能力。模型或环境不提供，提交时就会被拒绝（rgb / vector / reward 在第一次观测和第一步时再查一次）。

| 名称 | 类别 | 含义 |
|---|---|---|
| `state`、`predict`、`plan` | 模型探针 | 记忆或潜状态、预测、规划经过 S2 / S3 / S4 钩子；`predict` 还提供 `transition["predicted_next"]` |
| `embed` | 模型探针 | `model.embed(obs)`：把观测编码到预测所在空间，提供 `transition["next_state"]` |
| `learn` | 模型探针 | 模型每步用 transition 在线学习 |
| `image_box`、`set_object_pose`、`set_goal`、`hazards` | 环境方法 | 物体图像框、移动物体、移动目标、危险谓词 |
| `rgb`、`vector`、`reward` | 环境能力 | 观测带相机图像 / 观测是一维状态向量 / 环境给奖励 |

| 模型（注册名） | 探针 | 值的格式 |
|---|---|---|
| 玩具 `toy-centroid` / `toy-centroid-memory` | state | `{"cube": [x, y] 或 None, "goal": ..., "age": {...}}` |
| 玩具 `toy-planner` | state、predict、plan、embed | state：`{"latent": (7,), "memory", "plan_mean"}`；predict：`(N, 7)` 或 `(7,)`；plan：`candidates (N, H, 2)`、`predicted (N, H, 7)`、`scores (N,)`；embed：`(7,)`，看不见的物体为 NaN |
| 玩具 `toy-planner-online` | 以上加 learn | 同上；动作增益从 transition 在线拟合（初值为真值一半），state 另有 `gain` |
| TD-MPC2 `tdmpc2-pushcube` | state、predict、plan、embed | state：`{"latent": (512,), "plan_mean": (H, A) 或 None}`，把 `plan_mean` 换成 `None` 即清空热启动、本步按回合第一步规划；predict：`(N, 512)`，`ctx.meta` 含 `h`（推演第几步）和 `source`（`"prior"` 策略先验推演 / `"value"` 每轮 MPPI 的价值估计）；plan：每轮 MPPI 一次，`candidates (N, H, A)`、`scores (N,)`；embed：`model.encode(obs)` `(512,)`。均为 float32 numpy |
| MemoryVLA `memoryvla-bridge` | state | `{"timestep", "cognitive": [(t, tensor)], "perceptual": [...]}`，GPU 上的 bf16 torch 张量，返回新列表，不要原地改张量 |
| OpenVLA `openvla-bridge` | 无 | 只能挂 ENV、S1、S5 |

玩具 `toy-planner` 的探针与 TD-MPC2 含义相同。S2–S4、S6 插件可以先在它上面写测试，再上 GPU。

各环境的能力：玩具 `toy-pick`、SimplerEnv Bridge、ManiSkill 视觉环境为 rgb + reward；玩具 `toy-pick-vector`、TD-MPC2 的 ManiSkill 状态环境为 vector + reward。

## 一致性测试查什么

`wms plugin test` 在每一个满足 `requires` 的玩具组合上各跑一遍：`toy-centroid-memory + toy-pick`、`toy-planner + toy-pick`、`toy-planner + toy-pick-vector`、`toy-planner-online + toy-pick`。在每个组合上检查：能跑完（严格模式，返回值形式正确）、钩子被调用、同种子两次逐字节一致、没动全局随机状态；监控不改动作；故障 / 攻击 / 防御的 `neutral` 配置等于不挂插件，并报告改变了多少步动作。卡上写"任意模型"的插件必须在所有组合上通过。

## 强度和阈值从哪里来

不要把阈值写死在插件里。用干净运行（不挂故障和攻击）算统计量，再通过 `config` 传给插件：

```bash
python -m wms.episodes.clean_stats <运行输出目录> [...] --output stats.json
```

得到的统计量包括：
- 向量观测和动作各维的均值、标准差。
- 动作幅度和相邻两步动作差的分位数。
- 图像各通道的均值、标准差，以及相邻帧差的分位数。
- 挂了 `monitor-latent-moments` 时，latent 各维的均值、标准差。
- 监控插件记下的数值（例如 `monitor-surprise` 的 `surprise`）的分位数。

检测器"误报 ≤ 1%"的阈值就取对应量的 99% 分位数。服务器上已算好的两份见计划文档。危险谓词的阈值用 `python -m wms.scenarios.hazard_calibration` 标定。

## 在服务器上验收

GPU 服务器上不经过服务、直接跑配方：

```bash
python scripts/run_direct.py <配方> <运行编号> [键=JSON ...] [--registry 你的目录/registry.json]
python scripts/run_direct.py tdmpc2-pushcube-episode run_a01 params.episodes=50     'params.plugins=[{"plugin": "a-obs-corruption", "config": {"severity": 3}}]' --registry ~/a/registry.json
python scripts/compare_runs.py <基线运行> <插件运行>
```

`compare_runs.py` 按回合配对，输出：
- 动作最大差。
- 成功数，以及成功率差值和 95% 置信区间。
- 两边的危险步数和接管次数。

不挂插件或只挂监控的运行，与基线比动作差必须为 0。开跑前先用 `nvidia-smi` 看 GPU 是否空闲：这个脚本不经过服务的排队。

## 输出在哪里

- `episodes/ep_XXX/trace.jsonl`：每步一行，包括各环节跑了哪些插件、插件事件、建议动作与执行动作、接管（告警引起的接管带 `"via": "alarm"`）、本步告警、危险谓词。
- `episodes/ep_XXX/steps.jsonl`：原有格式不变。执行的动作在 `action` 列；被 stop 的那一步 `action` 为 `null`，事件为 `plugin_stop`。
- `episodes.json`：每回合汇总，另有 `takeovers`、`hazard_steps`、`plugin_reports`。
- `run_manifest.json`：本次挂的插件及其配置。

任务卡模板见 [TASK_CARD.zh-CN.md](TASK_CARD.zh-CN.md)。
