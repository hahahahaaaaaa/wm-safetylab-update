# E 合并回主平台（2026-11-25）

## 代码来源

- 目标：日常仓库 `wm-safetylab-update` 的现有开发分支 `dual-entry`；没有向旧 `archive` 仓库发布。
- E 来源：`dev/e`，`a9ec8fd3baf88761a8c14986be8fefdf31cb2563`，原工作区干净，无遗漏待提交代码。
- 主平台原 HEAD：`af3e758b855466a436bb3e762986ebcc58a7571f`。
- 主平台原有未提交的前端、蜡烛场景及文档先保存为 `2173cad`。两个与平台无关的 pelican 文件保留在原目录，未纳入提交。
- 完整合并在独立 `integrate/main-e-20261125` 工作区完成；以双方实际共享的 2026-10-27 源码快照辅助处理早期未提交文件造成的 add/add 冲突，保留双方功能。

## 合并结果

保留 E 的 S1–S6 插件实现、校准阈值显示、验收脚本、模型工作台、Cosmos 视频/策略与 OpenVLA LIBERO 入口。保留主平台的蜡烛场景、场景专属危险指标、任务与蜡烛附加选项的互斥/恢复逻辑、历史记录兼容和回放。

EnvironmentEntry 同时包含场景说明、hazard_ids 和当前环境的 hazard_thresholds/hazard_note；结果页同时显示场景危险指标和 E 的报告产物。重新从后端生成 OpenAPI 与 TypeScript 类型，前端生产构建通过。

合并后的目录包含 41 个配方、28 个插件；这不是模型数量或正式验收卡数量。两个远程模型入口仍受此前确认的额度/地区限制，资源就绪不代表外部服务可用。

## 验证

- 隔离服务器首次全套回归：450 passed、5 failed、11 skipped、7 subtests passed；5 个失败均为新隔离目录没有 Toy 的机器配置。
- 补齐隔离机器配置后重跑这 5 项：5 passed，其中包含真实 CPU Toy API 端到端任务。
- 在已有 Torch 环境补跑 bfloat16 记忆不别名检查：1 passed。
- 其余跳过项属于外部 EmbodiedJev/MuJoCo/Gymnasium 依赖或部署基线证据，不伪称通过。
- 新增合并回归断言：蜡烛场景 metadata、TD-MPC2 校准阈值、S4/S5/S6 插件和模型工作台配方同时存在。
- 前端 TypeScript/Vite 构建通过；蜡烛任务选择的 8 项回调/渲染检查通过。
- 本机最初控制环境缺少 NumPy，无法收集插件测试；未修改共享控制环境，改用服务器已有独立测试环境完成上述验证。

未在本轮重跑所有模型训练或正式 G2/G3 安全效果实验。E 的真实模型接入证据见 `E-MODEL-ACCEPTANCE-20261125.md`。

## 部署与证据

主平台固定目录为 `/home/suzhenjie2026/world-model/deployments/wm-safetylab`，端口 8765；E 仍使用 `/home/suzhenjie2026/world-model/workspaces/wm-safetylab-e` 和 8785。服务器主平台是部署副本，不把它重新初始化为 Git 仓库。

源码、数据库、前端与机器配置在部署前单独备份；环境、模型权重、历史任务结果保留。机器配置与凭据不进入 Git。运行时复用已安装的模型环境，主平台新任务继续使用自己的 `data-linux` 存储。

本机审计目录：`K:\world-model\deployments\wm-safetylab-main-merge-20261125`。服务器审计目录：`/home/suzhenjie2026/world-model/deployments/wm-safetylab-main-merge-20261125`。最终运行检查和发布提交编号以该目录下的部署收据、测试日志及 API 检查结果为准。
