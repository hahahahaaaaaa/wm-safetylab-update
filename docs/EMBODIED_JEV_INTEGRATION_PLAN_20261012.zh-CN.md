# embodied-jev 接入 WM-SafetyLab：现状核查与开发计划

日期：2026-10-12。范围：读取现有代码、文档和保存的验收工件，制定接入方案。本轮没有安装、启动服务、重跑实验或调用模型 API；下列开发阶段尚未实施。本文不包含密钥。

## 1. 结论

建议接入，且不需要另建一套实验管理平台。WM-SafetyLab 已具备配方注册、进程隔离、排队运行、产物归档、独立评价和网页展示；embodied-jev 已有本地独立部署。主要工作是把现有决策实验接入这些能力，并补齐配置、生命周期、指标和证据链。

第一阶段将 embodied-jev 定位为“MuJoCo 具身决策实验后端”，保留原生 transfer / stack / barrier 三任务；第二阶段才将模型决策接口迁移到 ManiSkill。它当前负责从候选动作中做选择，物理结果由 MuJoCo 计算；不能直接注册成已训练的“历史＋动作 → 下一状态”世界模型。候选概率也不能直接解释成任务成功概率或危险发生概率。

## 2. 当前平台开发情况

### 2.1 工程基础

核查目录：`K:\world-model\deployments\wm-safetylab`。当前版本 0.3.0，初次核查时 HEAD 为 `cc57e97`，工作树干净。报告写入后的检查发现 `src/wms/engines/tdmpc2/native.py` 出现其他未提交修改；本轮未修改该文件，后续实施应保留这些变化。日常开发仓库为 `hahahahaaaaaa/wm-safetylab-update`；较早的 wm-safetylab 仓库为归档仓库。

现有调用链：

```text
Web / CLI / Python client
  → FastAPI /api/v1
  → Registry 解析配方及机器配置
  → Runner 单工作进程 FIFO 队列
  → 配方指定的 Python 解释器 + UnifiedWorker
  → 原始结果、轨迹、视频、检查点
  → 独立 evaluate 作业
  → 指标记录、比较页面
```

| 部分 | 已有能力 | 接入时如何复用 |
| --- | --- | --- |
| 注册与配置 | `native/*/recipes.json` 自动发现，机器 runtime 配置与配方分离 | 新增 embodied_jev 配方和独立解释器配置 |
| 执行 | 独立子进程、超时、取消、stdout 事件、运行状态 | 运行原生 Session，适配停止和资源释放 |
| 产物 | result、轨迹、视频等输出归档 | 同时保存上游 episode 原件与统一摘要 |
| 评价 | 原始实验与独立指标作业分开保存 | 新增离线 metrics，不再次调用模型 API |
| 前端 | 配方、运行详情、视频、产物和指标比较 | 首期直接使用；后续增加逐步决策展示 |
| 环境隔离 | 每个作业选择自己的 Python | 避免把 MuJoCo 依赖装进平台服务或 ManiSkill 环境 |

当前是本地/单服务研究平台，未见可直接复用的分布式调度、多租户权限或跨解释器通用模型—环境 RPC。此次接入不需要以这些功能为前置条件。

主要代码入口：

- `src/wms/core/registry.py`：配方与运行时解析。
- `src/wms/core/runner.py`：队列和子进程生命周期。
- `src/wms/core/storage.py`：配置、运行记录及结果保存。
- `src/wms/runners/unified.py`：native / predict / rollout 等统一执行入口。
- `src/wms/core/protocols.py`：环境、模型和规划器协议。
- `web/src/pages/Cases.tsx`：现有运行及产物展示。

### 2.2 已接入的研究能力与实际证据

代码已有 Wan、ModelScope、Dreamer、RLVR language/video、RENEW、OpenVLA、ManiSkill、小型状态世界模型和 TD-MPC2 等引擎。这些入口覆盖不同能力，不能因为都能注册，就当成相同任务的可互换模型。

| 项目 | 本轮读取到的证据 | 可以得出的结论 |
| --- | --- | --- |
| 平台回归 | 10/12 发布阶段 JUnit：51 passed、2 skipped；保存的 clean-install 与 Web build 记录 | 已有工程测试基础；本轮未重新运行 |
| TD-MPC2 训练 | 原始 training.json：102,400 transitions、12,800 updates，参数确有变化 | 已有真实训练和检查点链路 |
| TD-MPC2 闭环 | 初始 0/10、训练后 10/10、所测动作噪声条件 10/10；逐回合轨迹和六段视频记录 | PushCube 上的受限工程验收，不能推广成鲁棒性或安全结论 |
| 独立评价 | 保存了独立指标作业、重复评价及来源文件哈希 | 可以复用产物与评价分离模式 |
| embodied-jev | 平台 src/native/recipes 中未找到相关接入 | 尚未进入平台统一管理 |

现有 TD-MPC2 是状态输入路线：35 维特权状态、8 维 `pd_joint_delta_pos` 动作，内含 MPPI。潜状态预测标为 `tdmpc2_latent`，没有物理状态解码；安全信号明确为 `not_measured`。另一条普通 PandaPushCube 环境适配器使用 `pd_joint_pos`。两条路径控制模式不同，后续接入不能直接互换动作。

历史 10/10 文档“没有 Git/托管 CI”的说明已过时；10/12 保存了旧仓库 Actions 成功记录，但它不能证明新日常仓库的最新 CI 已成功。

证据文件：

- `docs/MODELS_AND_EXPERIMENTS.zh-CN.md`
- `docs/ONBOARDING_MODELS_AND_ENVIRONMENTS.zh-CN.md`
- `docs/TD_MPC2_PUSHCUBE_ACCEPTANCE_20261012.zh-CN.md`
- `results/github-publication-20261012/publication-summary.json` 与 `pytest.xml`
- `results/tdmpc2-integration-20261012/acceptance/acceptance.json`
- `results/tdmpc2-integration-20261012/server-evidence/` 下原始训练、评测和重复评价工件

### 2.3 embodied-jev 已经存在的本地工作

| 路径 | 状态 | 处理原则 |
| --- | --- | --- |
| `K:\world-model\repos\embodied-jev` | 干净上游 Git checkout，HEAD `f08de2e`，包版本 0.1.0 | 用作锁版本源码；不直接修改 |
| `K:\world-model\embodied-jev` | 历史独立部署，包含 `.venv312`、前端、启动脚本、运行记录和本地文档 | 保留，用于提取兼容经验与历史证据 |
| `K:\world-model\deployments\wm-safetylab` | 当前统一平台 | 新增研究适配、配方、评价和报告 |

历史 `README-VALIDATION.zh-CN.md` 报告规则基线三任务×三种子 9/9、JEV skills 三任务各 seed 0 为 3/3，并保留对应 JSON 文件。`LOCAL-DEPLOYMENT.md` 还记录过 Chat 接口兼容修正和不同服务商试跑。它们说明已有可复用成果，不代表当前凭据、额度、网络、模型名或解释器仍然可用。

历史 Windows 部署采用 Python 3.12，原因是当时的 Conda 3.11 出现 MuJoCo DLL 加载问题。落地时先探测既有解释器；需要新环境再建立 `K:\world-model\envs\wms-embodied-jev`。不要覆盖旧环境，也不要直接复制一个可能含旧绝对路径的虚拟环境。

## 3. 接入后的架构与能力范围

```text
平台配方：规则 / GPT-5.5 / JEV + 任务、种子、观测、控制、安全预演配置
  → 平台独立 worker
  → embodied_jev 适配层
      ├─ 读取连接配置与密钥引用
      ├─ 创建原生 Session
      ├─ DecisionPolicy：观察 + 候选菜单 → 合法选择
      ├─ RobotWorld / MuJoCo：技能或短步动作 → 物理状态
      └─ 逐步记录、停止、资源释放
  → episode 原件 + 统一轨迹 + 决策事件 + 视频
  → 平台离线评价
  → 任务结果 / 接口可用性 / 已定义安全事件 / 调用量与延迟
```

在 WM-SafetyLab 六阶段中的位置：

| 阶段 | 本次能够提供的内容 |
| --- | --- |
| S1 数据与观测 | 结构化状态、RGB-D 检测结果、相机帧、动作和物理结果 |
| S2 状态与记忆 | 上游场景状态与有限历史上下文，记录其实际来源 |
| S3 动力学与想象 | MuJoCo 副本候选预演；明确标注为仿真真值辅助，不称学习世界模型 |
| S4 评估与规划 | 规则 / Chat / JEV 候选选择及拒绝、调用预算 |
| S5 动作与控制 | 原生技能、incremental 或 hierarchical 控制与执行 |
| S6 反馈与更新 | 轨迹反馈、离线分析、回归实验；当前没有 JEV 权重训练/在线更新 |

原生任务为搬运入盘 transfer、方块堆叠 stack、越障搬运 barrier。控制模式 skills / incremental / hierarchical 含不同程度的程序先验，必须分组比较。JEV 接口主要接受结构化状态；原始 RGB 直接输入不能未经验证地套到 JEV 上。

## 4. 两类模型连接与凭据处理

### 4.1 使用本次指定的连接

| Profile（拟新增） | Provider | 端点 | 模型与协议 |
| --- | --- | --- | --- |
| `chat_gpt55` | `chat` | `https://112.125.121.34/v1` | `gpt-5.5`；适配后请求 `/chat/completions` |
| `jev_native` | `jev` | 上游默认 `https://api.typesafe.ai/v1/systemone` | 上游默认 `jev-latest`；记录实际响应模型 |
| `rules` | `baseline` | 无 | 零模型调用 |

Chat 与 JEV 是两个不同服务，分别使用其密钥；不能将 JEV 请求发送到 Chat 端点。JEV 模型名如需固定版本，应以实际可用接口验证为准。旧的智增增/gpt-5.6-sol 配置保留，不作为本次新配方默认值。

上线前做最小预检：解释器和依赖 → 本地无 API 规则任务 → TLS/鉴权/模型名/JSON 候选响应 → 单局真实调用。数字 IP 的 HTTPS 证书可能需要排查；不能静默关闭证书验证。历史出现过 `stream: false` 和候选字段格式问题，应以请求契约测试确定兼容修正，保留合法候选校验。

### 4.2 不能把密钥直接写进平台运行配置

核查发现 `core/storage.py` 会保存解析后的 config，Registry 也会将 runtime 配置中的 env 合并进去。因此，即使 runtime.json 被 Git 忽略，把密钥放到 `runtime.env` 仍可能进入 `config.json` 和数据库。

实现方式：配方只保存 profile、端点、模型和 `credentials_file` / `api_key_env` 等引用；工作进程在创建 DecisionPolicy 时读取密钥，并仅放在不参与序列化的连接对象中。密钥文件如需落地，放在 K: 下受控、Git 忽略的位置；所有日志、异常、结果和 HTML 导出检查不含密钥或 Authorization。不要读取并重新输出旧 `.env`。

默认模型请求并发为 1，不人为加入固定两秒间隔。429 按 Retry-After/指数退避做有上限的请求重试；鉴权、余额和不支持的模型错误直接分类结束。请求重试不得重复执行机械臂动作。配置最大请求次数、单次超时、总时长和可用的用量预算；缺少价格或 usage 时显示未知，不能显示为零费用。

上游 Windows 默认连接持久化会使用 C: 用户目录；本次通过 worker 显式连接对象运行，避免启用该默认存储。若保留独立交互 UI，采用 memory-only 或明确的 K: 存储适配。

## 5. 分阶段开发任务与验收

### P0：冻结输入和复用已有部署

任务：记录平台和上游 commit、Python/MuJoCo 版本；核对历史部署中真正需要的兼容差异；验证旧 `.venv312` 是否可复用；确定新连接 profile 和密钥引用；写资源预检。

交付：运行时配置样例、兼容差异说明、预检报告。平台 `configured=true` 只代表配置可读，预检必须另外证明依赖和输入协议可用。

验收：源码 checkout 保持干净；所有新文件、缓存、日志位于 K:；baseline 不依赖 API；端点和凭据错误能定位且不泄露密钥。

### P1：原生 worker 与生命周期

任务：注册三个基础配方 `embodied-jev-baseline`、`embodied-jev-chat`、`embodied-jev-jev`。使用 `task=native` 与 `entrypoint(config, output_dir, context)`，直接封装 Session，不依赖 8090 Web 服务的共享当前 session。

适配 seed、task、scene、observation_mode、control_mode、preview、threshold、max_cycles、timeout、相机、候选打乱和扰动参数。每局创建新 Session，避免历史与随机数状态串局。

执行中轮询平台取消信号。取消、超时或错误时通知 Session 停止、阻止迟到响应执行动作，并在有限等待后结束进程；显式释放 HTTP/仿真/相机资源。原生路径不能假设 UnifiedWorker 会替它关闭 Session 内部资源。`uncertain` 不得静默改用规则动作。

验收：规则三任务 smoke 能运行；等待 API、执行动作和正常完成三种情况下取消均有明确结局；输入非法候选、服务错误、超时被准确分类。任务未成功可以是已完成的实验，但不能把请求异常伪装成物理失败或成功。

### P2：产物统一与独立评价

拟输出：

| 产物 | 内容 |
| --- | --- |
| `episode.raw.json` | 原生 `embodied-jev-episode-v1`，保留来源语义 |
| `trajectory.jsonl` | 观测、选择、实际执行、前后状态、时间与终止原因 |
| `decisions.jsonl` | 候选集合、选择、可用概率、输入/公开说明、延迟和 usage |
| `events.jsonl` | 拒绝、预演拦截、实际接触、抓取丢失、任务成功及接口失败 |
| `video.mp4` / 相机文件 | 实际仿真回放或采集帧，标注帧源与采样方式 |
| `manifest.json` | 源码/配置/场景哈希、种子、请求与实际模型、schema 版本、文件索引 |
| 平台 `result.json` | 指向输出目录内产物的标准 outputs 清单及摘要 |

视频不是上游 episode JSON 自动附带的 MP4，需单独复用渲染工具或添加录像适配。视频与指标必须来自同一回合，不用另一次随机重跑冒充原回合。

独立 `metrics:measure` 只读取已保存产物，输出任务成功率、步骤/仿真时长、调用量/延迟、接口失败率、拒绝率和已明确定义的安全事件指标。重复评价不调用模型、不覆盖原始产物，保存版本与来源哈希。

JEV 候选概率保留原值及上游 warning；Chat 没有候选分布时填 null/不可用，不补零或自行造 softmax。概率指标需记录有效覆盖率。

验收：原始 episode 可回溯每个汇总数；源文件哈希在重复评价前后不变；零调用、缺失 usage、缺失概率和失败回合均正确处理；视频可解码且时序与轨迹一致。

### P3：三方对照与研究变量

先做小规模接线验收，再扩展实验。建议开发矩阵：三任务 × seeds 0/1/2 × 规则/Chat/JEV × preview 开/关 = 54 回合；这是拟议开发规模，当前没有这些新结果，也不足以直接支持论文结论。

运行前统一场景、可见信息、候选集合、控制模式、步数和请求预算。先使用 privileged + skills，随后分别研究 RGB-D、incremental 与 hierarchical，避免一次混入多个变量。种子控制仿真初始化，但不能保证远程模型跨时间完全确定；需记录实际模型、请求设置并为正式评测安排重复与置信区间。

建议研究问题：

| 问题 | 实验设计 | 指标与限制 |
| --- | --- | --- |
| 模型能否完成任务 | 相同配置的三方配对比较 | 成功率、回合数、调用量、延迟 |
| 预演贡献多大 | preview 开/关消融 | 实际接触、成功、拦截；说明其使用仿真真值 |
| 观测误差影响 | 检测偏移/缺失、延迟，逐项固定强度 | 性能退化、拒绝、恢复步骤 |
| 决策对呈现是否敏感 | 候选顺序打乱、语义等价描述 | 选择一致性、最终结果变化 |
| 扰动后能否恢复 | 可重复的物体位置或执行扰动 | 恢复成功率、额外步骤、明确安全事件 |

同时报告全部已启动试验的端到端结果和有效模型请求条件下的任务表现，附样本数与排除原因。API 429/鉴权/额度问题单独计入服务可用性，不能默默剔除，也不能归为机器人碰撞。

预演拦截数不是实际碰撞数；上游碰撞检查只覆盖配置的末端/手指与桌面/障碍，不能写成全机器人安全。要扩展完整安全研究，另行定义接触对象、力阈值、持续时间和允许接触，不用任务成功率替代安全指标。

### P4：展示与工程验收

先复用平台运行列表、视频、文件链接和指标比较。增量增加模型选择、任务/控制/观测/预演标签，以及逐步“观测—候选—选择—执行—物理结果”时间线；原生概率仅在可用时显示。

首期不整体搬入 embodied-jev 的 Three.js 页面。它可作为独立交互工作台保留；统一平台负责批量实验和证据管理。

完成接口契约、取消/超时、指标、脱敏测试，以及现有平台回归；修改前端时执行 Web build。验收报告列明真实运行数、跳过项和失败原因。提交只包含适配代码、样例、测试及文档，运行环境和密钥不入仓库。

### P5（后续选做）：迁移 ManiSkill 并连接世界模型实验

原生 MuJoCo 三任务验收后，再新增“候选生成/选择 → ManiSkill 控制器”桥接。起步可用 PushCube，后续选择抓取或堆叠任务；不能把 transfer 与 PushCube 的成功率直接作模型排名。

ManiSkill 输入提供双方相同的任务状态或检测结果；GPT/JEV 输出合法候选 ID；平台把候选转换为已选控制模式的动作并执行。需要明确坐标系、归一化、动作维数、重复次数和回合长度。MuJoCo 的技能或 XYZ 短步不能直接作为 ManiSkill 关节命令。

与现有世界模型联合的进一步方案：使用 TD-MPC2 对同一动作序列预测潜状态/奖励，由统一规划器评分；GPT/JEV 做高层子目标或候选选择。先定义两者分工，避免在 TD-MPC2 的 MPPI 上无意再叠一层同义规划。若研究物理状态误差或危险概率，需另加可验证的物理读出/风险预测头与校准数据；不能直接使用 latent MSE 或 JEV 选择概率替代。

本阶段是新的环境和控制适配工作，不属于本次 MuJoCo MVP 的完成条件。

## 6. 建议新增文件

以下均为拟新增路径，不代表已实现；正常接入优先不修改平台核心调度协议。

```text
K:\world-model\deployments\wm-safetylab\
  src\wms\engines\embodied_jev\
    __init__.py
    native.py       # 平台入口、单局/套件调度、预算
    backend.py      # Session 创建、运行、取消和释放
    compat.py       # 经验证的必要兼容差异，避免修改上游 checkout
    connections.py  # profile 及密钥引用，只在 worker 内解析
    artifacts.py    # 原始导出、标准轨迹、录像、manifest
    metrics.py      # 离线独立评价
  native\embodied_jev\
    recipes.json
    runtime.example.json
    README.zh-CN.md
  tests\test_embodied_jev_integration.py
  tests\test_embodied_jev_metrics.py
  docs\EMBODIED_JEV_INTEGRATION_PLAN_20261012.zh-CN.md
  results\embodied-jev-integration-<date>\  # 实施后产生
```

常规运行产物继续进入平台配置的 K: 数据目录；验收汇总放入 results。若确需上游修补，使用可审查的补丁/独立派生源码目录并记录基准 commit，不修改 `repos\embodied-jev`。

## 7. 完成定义、优先级与工作量

MVP 必须完成 P0–P2，并包含三个 provider 的最小真实接线验证：平台能选择任务/模型、串行运行、停止、保存原始轨迹与视频、独立评价；成功/失败/未知字段含义明确，密钥不进入运行配置或导出。

完整研究接入完成 P0–P4：再加可复现的对照与消融、决策展示、回归测试和验收报告。P5 单独安排，避免把基础接入扩成多个仿真器的大改造。

单人开发量粗估：可用端点与现有环境兼容的前提下，MVP 约 3–5 个工作日；完成 P0–P4 约 5–8 个工作日。ManiSkill 迁移另估 4–7 个工作日，具体取决于控制模式与安全指标；以上不包括大规模正式实验、训练和等待外部服务恢复。

主要未决事项可在 P0/P1 中验证，无需提前改造平台：当前 API 额度与模型可用性、IP HTTPS 证书、历史环境迁移后的有效性、上游兼容差异、真实视频导出的成本。当前没有这些项目的重新验收结果。

## 8. 参考

本地代码与历史证据是平台现状结论的依据；公开资料用于核对上游接口。上游网页可能随 main 变化，实施应以固定 commit 为准。

- [embodied-jev 官方仓库](https://github.com/FBddcz/embodied-jev)
- [运行与 episode 导出](https://github.com/FBddcz/embodied-jev/blob/main/src/embodied_jev/runtime.py)
- [模型连接与候选选择](https://github.com/FBddcz/embodied-jev/blob/main/src/embodied_jev/policies.py)
- [技术说明](https://github.com/FBddcz/embodied-jev/blob/main/docs/TECHNICAL_GUIDE.md)
- [规划与控制模式](https://github.com/FBddcz/embodied-jev/blob/main/docs/PLANNING.md)
- [TypeSafe API](https://docs.typesafe.ai/api)
- 本地历史部署：`K:\world-model\embodied-jev\LOCAL-DEPLOYMENT.md`、`README-VALIDATION.zh-CN.md`、`PROJECT-GUIDE.zh-CN.md`