# EmbodiedJev 前端过程对比验收 · 2026-10-12

已接入平台导航“过程对比”，地址：<http://127.0.0.1:8765/#embodied>。

## 交付功能

- 两路或三路真实 MuJoCo Panda 三维场景；网格与位姿来自原始实验，支持旋转、缩放、俯视及复位。
- 原始实验选择与新建共同参数对比；规则基线、平台 Chat 连接、TypeSafe JEV 连接按单队列逐路执行。
- 实时快照、停止所选未完成实验、完成后的同步时间轴、逐动作跳转及倍速回放。
- 逐段动作/阶段决策、候选概率、分层通道概率、预演候选、仿真状态、全回合输入事件、评价与视频链接。
- 原始条件不匹配提醒、未知概率保留、较短回合末帧保持、未执行终局决策标注，以及对比清单导出。
- 历史 qpos 只经 mj_forward 还原，缓存写入 data/visual-cache，不改原始产物。

## 实测结果

| 验证 | 结果 |
| --- | --- |
| 历史真实实验 | 9 条原始回合全部可交互回放；99 个原输出文件 SHA256 保持一致 |
| 新建实时对比 | 两路规则基线先后成功，均 8 个动作、201 帧、0 模型 API 调用；浏览器捕获连续变化的快照 |
| 停止对比 | 正在执行与排队中的两个回合均取消成功 |
| 浏览器 | 三场景、时间定位、动作跳转、播放暂停、未执行决策、视角、条件差异、导出、页面切换通过；无 pageerror |
| 移动端 | 390 px 无页面横向溢出；滚动到三场景分别截图，视口重新绘制正常 |
| 平台 Python 全回归 | 152 passed，10 skipped；跳过项为可选 worker 依赖，与浏览器成功无混淆 |
| MuJoCo worker 原生/可视化测试 | 26 passed；禁止 mj_step/RobotWorld 初始化后，历史姿态重建仍通过 |
| 前端纯映射测试 | 18 passed，包含九份真实回合的证据审查 |
| 生产构建 | TypeScript + Vite 通过；3D 页面按需加载，包含 Three.js 的页面包约 533 kB，产生非阻断包大小提示 |

本次没有新增付费模型调用。Chat/JEV 使用此前真实归档验证显示链路；新增执行和停止链路用规则基线验证。未将这些工程检查当作模型性能基准。

首次实时验收在轨迹已就绪但录像仍写入时过早断言运行 completed，修正为等待平台终态后，完整工作流已通过。浏览器 range 的步长亦改为任意值，使非整步长的最终时间可到达。

## 验收文件

本目录 `../results/embodied-jev-frontend-20261012/`：

- `acceptance.json`：详细汇总，历史哈希与准备状态。
- `browser-history-check.json`、`browser-live-check.json`、`browser-mobile-check.json`：实测结果及运行 ID。
- `comparison-desktop.png`、`comparison-top-terminal.png`：三模型真实场景和结束决策。
- `comparison-live.png`、`comparison-new-baselines.png`：新建基线实时执行及回放。
- `mobile-scene-0.png` 至 `mobile-scene-2.png`：窄屏滚入后实际渲染。
- `check-history.cjs`、`check-live.cjs`、`check-mobile.cjs`：浏览器验收脚本。live 脚本会创建基线实验；重跑前确认有此需要。

全页截图中离屏 WebGL 画布可能不进入截图缓存，移动端以各通道滚入视口后的单场景截图为准；运行时无需持续保存 GPU 绘图缓冲。

使用方式与实现位置参见 [过程对比说明](EMBODIED_JEV_PROCESS_COMPARISON.zh-CN.md)。密钥继续保存在本地忽略的凭据文件，页面与导出清单不承载密钥。上游 K:/world-model/repos/embodied-jev 保持干净。