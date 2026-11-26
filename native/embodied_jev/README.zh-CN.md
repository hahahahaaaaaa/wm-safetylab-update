# embodied-jev 原生接入使用说明

本接入复用上游 `embodied_jev.runtime.Session`，在独立 worker 中运行一回合 MuJoCo Panda 任务，归档决策、实际轨迹和事件，再由独立评价任务读取工件计算指标。它是候选动作决策与仿真执行接入；模拟器预演不是学习到的世界模型。

**当前状态：已完成三配方在 privileged + skills 下的原生三任务接线验收。** 规则与智增增 GPT-5.5 各完成 3/3；JEV 接口已连通，三回合按默认概率门槛停止为 uncertain。其他模式的真实验证范围见 [验收报告](../../docs/EMBODIED_JEV_ACCEPTANCE_20261012.zh-CN.md)。

## 1. 三个配方

| 配方 ID | 决策来源 | 本次共享配置默认值 |
|---|---|---|
| `embodied-jev-baseline` | 上游规则基线，不发模型请求 | `provider=baseline`，`max_api_calls=0` |
| `embodied-jev-jev` | 上游 TypeSafe JEV 原生决策协议 | `jev-latest`，原生 `https://api.typesafe.ai/v1/systemone` |
| `embodied-jev-chat` | Chat Completions 返回 JSON 选择 | 智增增 `https://api.zhizengzeng.com/v1/`，`gpt-5.5` |

三个配方默认都是 `params.task=transfer`、`control_mode=skills`、`observation_mode=privileged`、`preview=true`、`seed=0`，并登记 `embodied-jev/1` 独立指标。默认模型名是本次配置选择，不是对远程服务可用性或模型身份的额外保证；返回的模型标识会随请求记录保存。

源码入口为 [native.py](../../src/wms/engines/embodied_jev/native.py)，共享配方为 [recipes.json](recipes.json)。

## 2. 运行环境与机器配置

平台服务环境与模型/仿真 worker 环境可以分开。API 决策不要求本机 CUDA GPU；MuJoCo 仿真、相机采集和视频回放仍需要相应依赖和可用的本机渲染后端。关闭视频不等于关闭 `rgbd/vision` 所需的相机渲染。

本机 Windows 可以复用已有解释器：

```text
K:/world-model/embodied-jev/.venv312/Scripts/python.exe
```

这是已有机器路径，不是其他使用者必须创建的目录。上游解释器需能导入 MuJoCo、NumPy、HTTPX 等上游依赖；视频另需 Pillow 与 imageio-ffmpeg。平台通过 worker 加载适配器，并从 `source_dir/src` 加载上游。不要修改上游源码来填写连接配置。

以 [runtime.example.json](runtime.example.json) 为模板，在服务机器创建自己的 runtime 文件。以下跨机器示例中的路径必须替换；JSON 不展开 `~`、`$WMS_ROOT` 或其他 shell 变量：

```json
{
  "python": "/path/to/embodied-jev-environment/bin/python",
  "source_dir": "/path/to/clean/embodied-jev",
  "timeout_s": 420,
  "env": {
    "EMBODIED_JEV_PERSISTENCE": "memory",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1"
  },
  "profiles": {
    "jev": {
      "url": "https://api.typesafe.ai/v1/systemone",
      "model": "jev-latest",
      "api_key_env": "TYPESAFE_API_KEY"
    },
    "chat": {
      "base_url": "https://api.zhizengzeng.com/v1/",
      "model": "gpt-5.5",
      "api_key_env": "EMBODIED_API_KEY",
      "json_mode": true
    }
  }
}
```

本机对应的两个路径可填写为：

```json
{
  "python": "K:/world-model/embodied-jev/.venv312/Scripts/python.exe",
  "source_dir": "K:/world-model/embodied-jev"
}
```

将它们替换进完整 runtime，而不是把这个两字段片段当成 API 配置。不要覆盖已经存在的机器文件、凭据或环境。

密钥只通过引用解析，有两种方式：

- `api_key_env`：服务/worker 进程继承的环境变量名。仅在提交请求的客户端终端设置变量，不会把密钥自动传到远程服务。
- `credentials_file` 与 `api_key_field`：指向服务机器的私密 JSON 文件及其字段。文件名必须以 `.local.json` 结尾，内容是“字段名 → 密钥”的平面映射；相对路径以 runtime 文件所在目录为基准。

私密文件引用的 profile 示例：

```json
{
  "base_url": "https://api.zhizengzeng.com/v1/",
  "model": "gpt-5.5",
  "credentials_file": "credentials.local.json",
  "api_key_field": "chat",
  "json_mode": true
}
```

每个 profile 选择一种引用方式；若同时填写，当前实现优先读取 `api_key_env`。实际密钥不应写入共享 runtime、配方、run 请求、`runtime.env`、日志或 Git。profile 中的 `key/api_key/token/secret/password/authorization/headers` 等内联凭据字段会被拒绝。配置文件中也不使用带凭据、查询参数或片段的 URL。

默认配方读取本目录的 `runtime.json`，也可以在既有服务映射文件中增加：

```json
{
  "embodied_jev": "native/embodied_jev/runtime.json"
}
```

映射值相对映射文件解析。服务使用 `wms serve --native-configs <映射文件>` 加载；保留已有映射项。也可以在单次请求的 `runtime.config_file` 指定配置，路径由服务机器解释。

## 3. 提交一回合与切换任务

本机 Windows 运行平台命令前：

```powershell
. K:/world-model/use-k-drive.ps1
Set-Location K:/world-model/deployments/wm-safetylab
$env:WMS_BASE_URL = 'http://127.0.0.1:8765'
& ./.venv/Scripts/python.exe -m wms recipes
& ./.venv/Scripts/python.exe -m wms run embodied-jev-baseline --wait
```

`WMS_BASE_URL` 改成实际服务地址；上面的回环地址仅适用于该端口确有服务的机器。其他平台使用已安装 wms 客户端的解释器执行同样的 `python -m wms ...` 命令。

需要调整参数时，将以下 JSON 保存到项目存储目录，例如本机的 `K:/world-model/deployments/wm-safetylab/embodied-jev-request.local.json`：

```json
{
  "recipe": "embodied-jev-chat",
  "name": "Chat barrier seed 1",
  "seed": 1,
  "timeout_s": 420,
  "runtime": {
    "config_file": "native/embodied_jev/runtime.json"
  },
  "params": {
    "task": "barrier",
    "control_mode": "skills",
    "observation_mode": "privileged",
    "preview": true,
    "max_cycles": 30,
    "max_api_calls": 20,
    "timeout_seconds": 40,
    "episode_timeout_s": 300,
    "record_video": true
  }
}
```

```powershell
& ./.venv/Scripts/python.exe -m wms run embodied-jev-request.local.json --wait
```

要使用 JEV，改为 `recipe=embodied-jev-jev`；规则对照改为 `embodied-jev-baseline`，并将显式 `max_api_calls` 改为 0。**物理任务写在 `params.task`，不要把顶层平台 `task=native` 改成 `transfer/stack/barrier`。**

| 参数位置 | 可选值或用途 |
|---|---|
| 顶层 `seed` | 非负整数，传入上游场景与控制过程；远程 API 不因此承诺逐次可复现 |
| `params.task` | `transfer` 搬运入盘、`stack` 堆叠、`barrier` 越障搬运 |
| `params.control_mode` | `skills` 候选技能；`incremental` 逐次短动作；`hierarchical` 子目标与独立运动通道 |
| `params.observation_mode` | `privileged` 仿真状态；`rgbd` 上游已知颜色 RGB-D 感知；`vision` 原始图像 |
| `params.preview` | 是否启用上游模拟器预演；不改变底层实际执行接触检查的含义 |
| `params.threshold` | 默认 0.55；仅对提供了候选概率的决策适用 |
| `params.shuffle_candidates` | 默认 false，是否打乱候选顺序 |
| `params.connection_profile` | 选择 runtime 的 `profiles` 键，默认取 provider 名称 |

参数名是 `connection_profile`。不要使用 `profile`；该名称由平台注册表保留为配方别名。

当前 `vision` 组合只接受 `provider=chat` 与 `control_mode=incremental`，同时要求所选 Chat 服务实际支持图像输入。可在 `params.camera_views` 指定上游支持的 `["external", "wrist"]`。JEV、规则基线或其他控制模式与 `vision` 的组合会被拒绝。可接受的配置组合并不等于已经完成对应模式验收。

## 4. 接入自己的 Chat JSON 服务

在机器 runtime 的 `profiles` 中新增一个命名配置：

```json
{
  "my_chat": {
    "provider": "chat",
    "base_url": "https://your-chat-service.example/v1",
    "model": "your-model-name",
    "api_key_env": "MY_CHAT_API_KEY",
    "json_mode": true
  }
}
```

这是 `profiles` 内的片段。运行请求仍用 `embodied-jev-chat`，增加 `params.connection_profile="my_chat"`。连接器会为未带后缀的 Chat URL 追加 `/chat/completions`；完整 endpoint 也可直接填入 `url`。JEV 连接仍固定沿用原生 TypeSafe endpoint，不能用 Chat URL 替换。

当前 Chat 路径发送非流式 `messages` 请求，读取 `choices[0].message.content` 中的 JSON 字符串。技能选择需要类似：

```json
{"choice": "本次请求实际提供的候选ID"}
```

候选 ID 必须来自该次请求，不能返回任意动作文本。增量规划还允许上游规定的短 `intent/visual_evidence` 字段；分层模式会使用运动通道选择协议，不能用固定的单 `choice` 响应代替所有请求。没有提供概率的 Chat 输出会保留 `selected_probability=null`，不会生成置信度或把它补成 1。

`json_mode=true` 会请求 `response_format={"type":"json_object"}`。服务不支持该参数时可显式设置 false，但返回内容仍须是符合上述上游协议的 JSON。不能解析、返回未知候选或异常结束的响应会保存为协议错误，不会自动换成规则策略。

## 5. 请求预算、超时与状态

每回合的 HTTP 请求串行执行，并发数为 1；**正常请求之间没有固定 2 秒间隔**。只有遇到 429 时，按配置及 `Retry-After` 做有界退避。

| 字段 | 默认值 | 口径 |
|---|---:|---|
| `params.max_api_calls` | JEV/Chat 为 20，baseline 为 0 | 实际 HTTP 尝试上限，429 重试也计入；支持 0..500 |
| `params.max_cycles` | 30 | 上游决策周期上限，支持 1..200 |
| `params.timeout_seconds` | 40 | 单次 HTTP 请求超时，当前入口上限 120 秒 |
| `params.episode_timeout_s` | 300 | 回合驱动预算，当前入口上限 3600 秒 |
| 顶层 `timeout_s` | 420 | 平台 worker 总预算，还要容纳初始化、归档及视频回放 |
| `params.max_429_retries` | 1 | 仅 429 重试，支持 0..3 次 |
| `params.retry_backoff_seconds` | 1 | 重试退避起点，单次等待最多 10 秒 |

不要将 `model_calls` 与实际 HTTP 次数混用：一个逻辑决策可以遇到重试，也可以因为预算用尽而没有真正发出请求。实际请求、状态码、延迟、usage 和剩余预算在 `run_metadata.json` 的 `transport` 中。认证失败、其他 HTTP 错误、请求超时不会无限重试。

回合 `uncertain` 会停止并保留原状态，不自动等待人工操作或退回规则策略。平台任务完成不等于物理任务成功：例如决策停滞或预算内未完成任务，应继续查看 `data.episode_status` 和 `data.task_success`。当前回合 `error/timeout` 映射为平台 `failed`，取消映射为 `cancelled`；其他已结束的回合可作为 `completed` 工程记录保留。平台硬终止进程时，末尾归档可能来不及完成。

## 6. 输出与录像含义

已走到归档阶段的回合会登记：

| 文件 | 内容 |
|---|---|
| `episode.raw.json` | 上游原始 episode 导出，保持 JSON 值，不保证原始空白排版 |
| `trajectory.json` | 逐决策周期的前后观测、动作、执行标记与保存的帧 |
| `decisions.json` | 阶段/动作选择、候选、拒绝和最后一次可能未执行的决策 |
| `events.json` | 上游事件时间线 |
| `run_metadata.json` | 源码版本/哈希、依赖版本、公开连接字段、请求计量与录像状态 |
| `summary.json` | 派生摘要与指标，未知值保留 null |
| `visualization.html` | 已转义的决策与事件时间线 |
| `cameras.zip` | 仅当存在原始相机帧时生成 |

`trajectory.steps` 的单位是 `recorded_decision_cycles`；增量/分层模式的记录可能包含拒绝而未执行的动作。它不是 MuJoCo 积分步数。奖励、风险成本及未测量的终止信息保留 null，不为套用通用指标补零。

默认 `record_video=true` 会尝试生成 `episode-replay.mp4`、浏览器兼容的 `episode-replay.webm` 及 `episode-replay.json`。这是对保存的 `qpos` 重新渲染：验证场景哈希、按仿真时间采样，不再次调用策略、不积分新物理步、不插值轨迹，也不包含 API 等待时间。回放不是模型生成视频，也不是模型当时看到的原始相机图像。文件和实际帧选择以 manifest 为准。

录像依赖缺失或渲染失败会留下 `video_status=unavailable` 等状态，不把缺失录像描述成成功。显式关闭录像、取消或 worker 尚未退出时不会进行该回放。网页结果页通过登记的 outputs 查看 JSON、时间线和可用视频。

## 7. 独立离线评价

三个配方都登记 `wms.engines.embodied_jev.metrics:measure`，版本 `embodied-jev/1`。平台仅对 `completed` 来源自动排入独立评价；`wms run --wait` 只等待来源任务。客户端应另外等待评价：

```python
import wms

source = wms.wait(wms.run({"recipe": "embodied-jev-baseline", "seed": 0})["id"])
print(source["state"], source.get("data", source.get("result", {}).get("data")))
evaluation_id = source.get("automatic_evaluation_id")
if evaluation_id:
    evaluation = wms.wait(evaluation_id)
    if evaluation["state"] != "completed" or evaluation.get("evaluation_status") == "partial_failed":
        raise RuntimeError("独立评价没有完整完成")
```

对已完成来源重评可以运行：

```text
python -m wms evaluate RUN_ID --wait
```

评价读取 `episode.raw.json` 与 `run_metadata.json`，重新计算后写入新的评价目录，不重跑模型、不启动 MuJoCo、不改写来源，也不直接采用 `summary.json` 的已有分数。输出为 `embodied_jev_metrics.json`，平台另登记通用指标工件。

指标区分任务成功、回合状态、决策周期、执行/拒绝动作段、模型调用、延迟、token 覆盖、候选拒绝和配置接触信号。候选拒绝可能来自可达性或预演检查，不能计作实际碰撞；`configured_contact_observed` 只反映已记录的上游禁止接触信号，不是全面安全率。缺失安全信号或 usage 时，摘要保留 null，数值指标不伪造 0。token 覆盖按已观察 HTTP 请求与逻辑尝试的较大数作分母，同时报告已有 usage 的覆盖范围；历史累计 token 计数不能独自证明完整用量。

平台 API 暂不接受对 `failed/cancelled` 来源创建评价。若失败回合已经留下完整 raw/metadata，可在单独目录直接调用纯离线函数审计，不能把该结果冒称平台自动评价：

```python
from wms.engines.embodied_jev.metrics import measure

metrics = measure({
    "source_output_dir": "/absolute/path/to/existing/source/output",
    "output_dir": "/absolute/path/to/new/offline-evaluation"
}, {})
```

本机两个路径都应在 `K:/world-model` 下，评价目录必须位于来源目录之外。后续科学对比还需固定场景、seed、控制与观测模式、预演开关和请求预算，并报告所有失败回合；单回合任务成功或候选概率不构成安全性结论。



## 本机接线验收

2026-10-12 的三任务测试中，规则与智增增 GPT-5.5 各完成 3/3；JEV 三次触发默认概率门槛，保留 uncertain。完整条件、调用量及限制见 [验收报告](../../docs/EMBODIED_JEV_ACCEPTANCE_20261012.zh-CN.md)。
