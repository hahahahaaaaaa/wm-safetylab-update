# 上游项目与发布范围

本项目提供实验编排、模型/环境兼容层、证据展示与评价接口。以下项目的源码、权重、数据和许可证保持独立；安装或下载时应遵守各自仓库及模型/数据卡的条款。

| 接入 | 官方来源 |
|---|---|
| ManiSkill 与其中 TD-MPC2 baseline | https://github.com/mani-skill/ManiSkill |
| TD-MPC2 | https://github.com/nicklashansen/tdmpc2 |
| DreamerV3 | https://github.com/danijar/dreamerv3 |
| Wan2.1 | https://github.com/Wan-Video/Wan2.1 |
| RLVR-World | https://github.com/thuml/RLVR-World |
| OpenVLA | https://github.com/openvla/openvla |
| ModelScope text-to-video | https://huggingface.co/damo-vilab/text-to-video-ms-1.7b |
| Gymnasium | https://github.com/Farama-Foundation/Gymnasium |
| EmbodiedJev | https://github.com/FBddcz/embodied-jev |

RENEW 使用的确切源码提交由其 `configure_linux.py` 生成的本地环境锁记录；安装说明见模型手册。上游论文方法的接入不表示完整论文复现。

Git 仓库只发布平台源代码、共享配方、示例、当前文档和 CI。不会发布本机 `source-audit/` 上游快照、下载压缩包、环境目录、权重、机器配置、运行数据库和原始实验目录。具体记录可由团队从自己的实验存储导出。

目前没有为本项目原创部分指定独立 LICENSE 文件；公开仓库不自动授予 MIT/Apache 等许可。上游许可证也不会自动覆盖本平台原创代码。对外再分发前，由项目维护者确定平台许可。

EmbodiedJev 接入保留独立上游 checkout；记录状态录像参考上游 render_demo.py 的重放方法。对应 MIT 许可见 [上游许可副本](native/embodied_jev/UPSTREAM_LICENSE.txt)，具体源码提交随实验元数据保存。
