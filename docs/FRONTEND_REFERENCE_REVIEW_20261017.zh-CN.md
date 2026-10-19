# 主平台前端重设计：历史开源项目调研整理

整理日期：2026-10-17。本文件回溯已有调研和界面检查记录，不是对上游最新版本的重新评测，也不表示本轮已经开始修改主平台前端。

## 1. 最重要的历史结论

此前主平台的设计取向是：**前端使用体验主要参考 promptfoo；案例证据与执行过程参考 Inspect AI；Python 公共入口参考 DeepEval；最终保留自己的 React Web。** 没有将 DeepEval 的托管 Dashboard 作为可直接私有部署的开源前端。

2026-09-30 的[八项目工程对照报告](K:/world-model/deployments/COMPARATIVE_FRAMEWORK_RESEARCH.zh-CN.md:116)包含完整界面核查表。八个项目中，七个有浏览器／终端界面或报告检查记录；Isaac Lab 仅验证了无界面仿真，未验收原生 GUI。

## 2. 真正检查过的七个界面／报告

| 项目与当时版本 | 当时看了什么 | 可供主平台重设计借鉴的交互 | 实际验证边界与资料 |
|---|---|---|---|
| [promptfoo](https://github.com/promptfoo/promptfoo)，0.123.1 | React 评测矩阵、失败详情、配置与红队工作流 | 案例为行、模型／实验条件为列；筛选失败；从单元格展开输入、输出、判定与配置 | 真实查看了合成轨迹结果和失败详情；没有验证全部红队功能。[分析](K:/world-model/deployments/promptfoo/ARCHITECTURE_AND_EMBODIED_SAFETY.zh-CN.md) · [矩阵截图](K:/world-model/deployments/promptfoo/runs/ui-results.png) · [失败详情](K:/world-model/deployments/promptfoo/runs/ui-failure-detail.png) |
| [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)，0.3.273 | Inspect View，样例详情与 transcript | 运行→案例→逐步执行记录→评分理由；失败证据定位 | 浏览器进入碰撞样例，检查执行和评分记录；原始数据是 fixture，不是机器人安全验证。[部署记录](K:/world-model/deployments/inspect-ai/README.zh-CN.md) · [截图](K:/world-model/deployments/inspect-ai/runs/ui-collision-transcript.png) |
| [PyRIT / CoPyRIT](https://github.com/microsoft/PyRIT)，1.1.0 | React＋FastAPI 图形界面，目标选择、交互测试、历史与评分 | 攻击／防护实验的目标选择、变换配置、连续过程与历史追踪 | 用本地 fixture 完成界面发送及响应查看；当时 converter 面板为占位，不应照搬成“已实现功能”。[分析](K:/world-model/deployments/pyrit/ARCHITECTURE.zh-CN.md) · [历史截图](K:/world-model/deployments/pyrit/runs/ui-history.png) · [交互截图](K:/world-model/deployments/pyrit/runs/ui-local-target-response.png) |
| [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)，源码声明0.7.3 | 官方 React/Vite Web，模型与任务配置、命令预览、日志、逐样例结果 | 实验配置表单、模型／任务选择、可复制命令、日志与结果分离 | 实测结果页与逐例分数；外部任务图片预览返回404，START/STOP未实测。当时 `tui/` 目录实际上包含 Web 界面。[部署记录](K:/world-model/deployments/lmms-eval/README.zh-CN.md) · [截图](K:/world-model/deployments/lmms-eval/runs/ui-native-results.png) |
| [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)，固定源码提交 | Gradio 数据浏览、MMBench 专项评分页 | 预览图片、问题、参考答案，切换样本索引；适合数据与案例页 | 数据加载和索引切换通过；评分页只确认控件，没有调用裁判。主要是数据浏览工具，不能当作完整实验管理后台。[证据汇总](K:/world-model/deployments/COMPARATIVE_FRAMEWORK_RESEARCH.zh-CN.md:36) |
| [garak](https://github.com/NVIDIA/garak)，0.17.0 | React/ECharts 生成的静态 HTML 风险报告 | 按风险／探针分组的摘要、分项分数、展开详细结果 | 浏览器展开探针并检查分数；属于报告浏览，不是任务控制后台。[分析](K:/world-model/deployments/garak/ARCHITECTURE.zh-CN.md) · [截图](K:/world-model/deployments/garak/runs/ui-report.png) |
| [DeepEval](https://github.com/confident-ai/deepeval)，4.2.7 | 静态 HTML 报告、Textual 本地终端查看器；研究了托管产品边界 | 指标卡、案例判定、解释与 trace；主要借鉴其 TestCase→Metric→Result 的组织 | TUI 加载、切换、搜索和静态 HTML 检查通过；Confident AI Dashboard 是独立托管产品。[源码与边界分析](K:/world-model/deployments/deepeval/FRAMEWORK_AND_SIMPLIFICATION.zh-CN.md:96) · [HTML截图](K:/world-model/deployments/deepeval/runs/ui-html.png) |

截图保留的是当时界面，不是现在主平台的新设计。历史本地服务端口本轮没有检查，不应假定仍在线。

## 3. 也讨论过，但应与上述前端检查区分

| 项目／方案 | 当时用途 | 本轮应如何定位 |
|---|---|---|
| [MLflow](https://github.com/mlflow/mlflow) | 早期方案用于参数、指标、工件索引与实验比较；syhj 中有同步／读取验证 | 可参考运行表格、指标比较和工件目录；这不等于已完成独立前端交互验收。当前主平台使用 SQLite＋文件＋自有 Web，MLflow 不是强制依赖 |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | 机器人仿真任务及 Sim/Kit 原生调试界面 | 已有无界面物理任务验证，但此前没有原生 GUI 验收；不作为 Web 管理前端的主要参考 |
| 现有 embodied-jev 控制台 | 早期讨论复用其界面资源，后来有机械臂场景和逐步决策的过程对比 | 属于现有项目的场景交互资源，适合参考三维回放；不要让只读结果查看依赖仿真器启动 |
| ClawKeeper | 拆解论文、工具贡献和 ACL Demonstration 展示方法 | 本轮未找到把其前端作为主平台 UI 基座或完成界面实测的记录；不列入主要前端调研清单 |

MLflow 的历史角色见[旧版验证计划](K:/world-model/syhj/docs/OPEN_SOURCE_VALIDATION.zh-CN.md:25)；当前取舍见[主平台项目说明](K:/world-model/deployments/wm-safetylab/docs/PROJECT_OVERVIEW.zh-CN.md:106)。

## 4. 当时的 A／B／C 方案如何演变

| 阶段 | 原设计 | 界面取向 |
|---|---|---|
| 方案 A | Python 研究包，原生产物，按需接 MLflow／报告 | 轻量、自主，最初推荐作为工程起点 |
| 方案 B | Inspect Task＋原生 worker | 复用 Inspect Viewer；资产、训练队列和三维回放另补 |
| 方案 C | promptfoo 风格配置与实验矩阵 | 借鉴配置和对比体验，WM 的媒体／轨迹仍需自建 |
| C 正式规格 | 自有前端＋Python 平台＋promptfoo 桥接 | 不要求用户进入 promptfoo；不长期复制／维护上游 React 页面 |
| DeepEval 风格 C v2 | 一个 Python 包＋独立 Web＋CLI | Python 学 DeepEval，UI 保留 promptfoo 式体验；移除首版独立 Node 后端与重复结果投影 |

回溯材料：

- [A／B／C 总对比](K:/world-model/designs/three-track-frameworks-v2/README.zh-CN.md:5)
- [Inspect 方案的页面边界](K:/world-model/designs/three-track-frameworks-v2/B-inspect-native-workers.zh-CN.md:107)
- [promptfoo 方案的复用与自建](K:/world-model/designs/three-track-frameworks-v2/C-promptfoo-campaigns.zh-CN.md:125)
- [C 正式版前端规格](K:/world-model/designs/scheme-c-final-v1/docs/06-frontend-and-cli.md:7)
- [DeepEval 风格简化方案](K:/world-model/designs/scheme-c-deepeval-style-v2/README.md:11)

这些是历史设计，不应把旧文档中的全部接口、权限和页面承诺当作当前实现。C v2 曾收敛为实验、运行、案例、对比四页；当前主平台源码还包含过程对比入口。

## 5. 用于这次重设计的建议组合

以下是根据历史调研和当前主平台功能给出的设计建议，不是已实现的新页面。

| 主平台页面／区域 | 优先参考 | 需要保留的 WM 专有内容 |
|---|---|---|
| 实验配置 | lmms-eval＋promptfoo | 模型、环境、任务、输入类型、实验条件、可复制 CLI／配置 |
| 运行列表与详情 | MLflow 的记录组织＋lmms-eval 的日志呈现 | 模型运行与独立评价的状态分开、checkpoint、失败日志 |
| 案例与证据 | Inspect AI | 输入→状态／记忆→预测→规划→动作→反馈；按模型实际能力展示，不虚构缺失阶段 |
| 方法与条件对比 | promptfoo | 相同案例和种子的 clean／扰动／防护条件、任务效用与安全指标分开 |
| 安全报告 | garak；交互研究按需借鉴 PyRIT | 风险类别、干预权限、危险事件、判定依据及未测量项 |
| 数据与样本预览 | VLMEvalKit | 原始图像与算法实际收到的图像、动作、GT 状态及数据来源 |
| 仿真过程对比 | 现有 embodied-jev／主平台过程视图 | 视频、三维场景、真实／预测轨迹同步，误差及干预时刻定位 |

建议先确定 **promptfoo 的比较方式＋Inspect 的证据详情＋lmms-eval 的配置流程** 三个主要参考，再统一视觉风格。DeepEval 继续用于理解后台接口组织，不能据其名字推断需要更换前端技术栈。
