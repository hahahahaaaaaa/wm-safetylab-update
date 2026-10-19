# 插件任务卡（模板）

复制本文件，每个插件填一张。

| 项 | 填写 |
|---|---|
| 卡号 / 插件名（个人前缀） | |
| 负责人 | |
| 环节 / 角色 | 例：S1 fault |
| 挂在哪个钩子 | 例：`on_observation` |
| requires | 例：—；`state`；`plan`、`hazards` |
| privileged / oracle | 例：否 |
| 威胁模型卡（攻击必填） | knowledge / access / budget / goal / adaptive |
| 对应文献 | 论文或文档链接，借鉴了什么 |
| 开发用玩具 | `toy-centroid-memory` 或 `toy-planner` |
| 验收模型 / 环境 | 例：MemoryVLA + SimplerEnv carrot，24 个布局 |
| 参数 | 默认值与取值范围 |

## 交付物

- [ ] 插件代码与注册条目（个人前缀）
- [ ] `wms plugin test <注册名>` 全部 PASS，附 `report.json`
- [ ] 至少一个单元测试（参照 `tests/test_gripper_flip.py`）
- [ ] 一次真实模型运行的运行编号及其 trace，说明插件效果（成功率、危险步数等与不挂插件时对比）
- [ ] 一页说明：对应文献、参数、威胁模型卡、已知局限

## 审查看三件事

1. 只写自己环节的对象，不原地修改输入。
2. 结果可复现：同种子两次运行逐字节一致。
3. `privileged` 使用合理：防御不读真值，除非明确标为 oracle 上界。
