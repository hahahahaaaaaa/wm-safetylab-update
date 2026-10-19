# 模型与仿真环境接入模板

适用 WM-SafetyLab 0.3.0 当前简化接口，2026-10-11。

| 你要做什么 | 从哪里开始 |
|---|---|
| 加新世界模型、生成模型或动作策略 | [模型接入代码与教程](model/README.zh-CN.md) |
| 加新仿真器、任务或机器人场景 | [环境接入代码与教程](environment/README.zh-CN.md) |
| 写 S1–S6 插件（故障、攻击、防御、监控） | [插件模板与教程](plugin/README.zh-CN.md) |
| 理解两者如何与现有平台协作 | [接入总指南](../docs/ONBOARDING_MODELS_AND_ENVIRONMENTS.zh-CN.md) |
| 一次验证两套模板的 HTTP/Python/CLI 路径 | `verify_onboarding.py` |

模型模板默认使用明确标注的标量教学模型，实际支持预测、最小二乘训练、checkpoint 回载、独立评价；真实论文模型另有需填写的骨架。环境模板实际运行 Gymnasium Pendulum-v1，支持轨迹、独立评价和可选 MP4。不要把教学模型称作神经 WM 或安全 baseline。

两套模板各自可复制到独立开发目录，通过 `registry.json` 接入，不需要为每个新组件改平台核心。复制后务必改 Python 包名、组件 ID 和配方 ID，避免多个组员互相覆盖。

本轮另打包了 `results/onboarding-20261011/wms-model-template.zip` 和 `wms-environment-template.zip`。压缩包保留 `templates/model/` 或 `templates/environment/` 路径，解压到平台项目根目录即可；已有同名模板时先使用新的开发目录。

当前有两个选择：开发服务启动时用 `--registry` 加载配方名称；或在代码已部署的前提下直接提交完整内联配置，不要求服务重启。服务端必须能访问实际模块、解释器和模型资源，HTTP 不会自动上传这些文件。

## 已完成的端到端验证

在隔离本地服务运行以下五条模型/仿真任务，其中四条自动创建独立评价；Python 客户端、CLI 查询和视频 HTTP Range 读取也已通过：

1. 教学模型预测。
2. 教学模型训练并生成 checkpoint。
3. 新 checkpoint 回载后预测。
4. 真实 Pendulum rollout。
5. 真实 Pendulum 录像。

验证时模型任务使用平台 `.venv`，仿真任务使用 `K:/world-model/envs/wms-onboarding`，证明不同任务可选择不同 Python。没有验证两个互不兼容环境在同一个配对任务中自动协作；当前配对仍使用单一解释器。

完整记录：`results/onboarding-20261011/http-acceptance/summary.json`。隔离服务验证完成后已关闭，原生产服务保持不变。

重新运行，请使用一个不存在的输出目录：

```powershell
. K:\world-model\use-k-drive.ps1
Set-Location K:\world-model\deployments\wm-safetylab
.\.venv\Scripts\python.exe templates/verify_onboarding.py --sim-python K:/world-model/envs/wms-onboarding/Scripts/python.exe --output K:/world-model/deployments/wm-safetylab/results/my-onboarding-check
```

脚本自动启动并关闭独立服务，默认端口 19021；若端口被占用，使用 `--port` 选择其他空闲端口。只读教程不需要运行该脚本。
