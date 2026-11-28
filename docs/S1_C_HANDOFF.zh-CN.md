# S1 输入与感知：C 工作区代码审查与交付报告

日期：2026-11-28
任务：S1（#1–#5）
工作区：wm-safetylab-c
分支：dev/c

## 1. 命名约定

S1 是任务阶段，C 是个人工作区和分支。
保留 c_*.py 文件命名及 native/plugins_c/recipes.json。
插件注册 ID 保留 a- 前缀。

## 2. 七个插件

| 卡 | 插件 ID | 实现类 |
|---|---|---|
| #1 | a-obs-corruption | ObsCorruption |
| #1 | a-obs-noise-vector | VectorObsNoise |
| #2 | a-camera-jitter | CameraJitter |
| #3 | a-patch-attack | FixedPatch |
| #4 | a-frame-guard | FrameGuard |
| #5 | a-input-monitor | InputMahalanobis |
| #5 | a-input-monitor-vector | InputMahalanobisVector |

## 3. 本轮源码修订

- 修订 CameraJitter 的中性配置随机数行为及参数边界。
- 修订 FrameGuard 和 InputMahalanobis 的相机选择检查。
- 加强 affine 图像变换参数检查。
- 增强干净运行统计拟合的数据门禁和来源记录。
- 增强检测评价的漏报和检测延迟统计。
- 修订 patch search 的 seed 配对和运行状态检查。
- 新增 CPU Toy 组合验证脚本及分析测试。

## 4. 服务器验证证据

### 4.1 源码部署

Reviewed overlay：18 个文件。
部署结果：OVERLAY_APPLIED=PASS。

备份：
.workspace/backups/s1-reviewed-v2-20261128T070032Z-7f09898c

### 4.2 CPU Toy 主动组合验证

结果：两个组合均 PASS。

随机种子：17。

图像组合：
- 五个插件均被调用。
- action_difference_steps_vs_baseline = 0。

向量组合：
- 两个插件均被调用。
- action_difference_steps_vs_baseline = 2。

报告：
.workspace/results/s1-active-toy-20261128T070032Z-8d414165/report.json

注意：这是 CPU Toy 功能组合验证，不是真实模型效果实验。图像组合动作变化为零，不能据此声称攻击或防御有效。

### 4.3 服务器定向测试

Python：3.11
NumPy：1.26.4
pytest：9.1.1

测试：
- test_cards_c.py
- test_s1_analysis.py
- test_s1_composition.py

结果：
26 passed in 0.32s
PYTEST_EXIT=0

测试使用已有 Toy Python 和临时 PYTHONPATH。
没有安装或修改服务器依赖。

### 4.4 代码检查

git diff --check：PASS。

## 5. 尚未完成的科研实验

- MemoryVLA 真实模型实验尚未完成。
- 2026-11-28 clean baseline 因 CUDA OOM 失败，未重复启动。
- 尚无完整的非中性剂量曲线和真实模型配对实验。
- 尚无完整的 95% 置信区间及真实模型检测效果报告。
- a-input-image.json 尚未由可验证的干净 RGB 原始数据拟合。
- a-input-vector.json 尚未由确认干净的 TD-MPC2 原始数据拟合。
- patch search 仍须确认真实实验输出中的 seed 字段。

现有 fallback statistics 不等同于重新拟合的 S1 检测统计。

## 6. 维护者交接

代码交付分支：dev/c

插件 ID：
a-obs-corruption
a-obs-noise-vector
a-camera-jitter
a-patch-attack
a-frame-guard
a-input-monitor
a-input-monitor-vector

服务器测试：26 passed。
CPU Toy 组合报告：见第 4.2 节。
随机种子：17。

Git commit ID 应以最终 git log 为准。

本报告所证明的是源码修订、CPU Toy 组合运行及定向测试通过。
不将其表述为 S1 真实模型科研实验全部完成。
