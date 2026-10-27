# A / B / C / D 主平台开发工作区

服务器：`suzhenjie2026@10.161.97.111`。所有开发文件和结果在 `~/world-model` 下。

| 人员 | 工作目录 | 分支 | 开发服务端口 |
| --- | --- | --- | --- |
| A | `~/world-model/workspaces/wm-safetylab-a` | `dev/a` | 8781 |
| B | `~/world-model/workspaces/wm-safetylab-b` | `dev/b` | 8782 |
| C | `~/world-model/workspaces/wm-safetylab-c` | `dev/c` | 8783 |
| D | `~/world-model/workspaces/wm-safetylab-d` | `dev/d` | 8784 |

各目录都是同一主仓库的独立 Git worktree，源码独立、共同保留原仓库历史。公共基线分支为 `team/base-20261027`。D 的具体插件范围尚未指定。

## 开发入口

以 A 为例：

```bash
cd ~/world-model/workspaces/wm-safetylab-a
./dev.sh info
./dev.sh run toy-episode-occlusion a_toy_001 params.episodes=1 params.max_steps=4 params.save_raw=false
./dev.sh plugin-test monitor-state-summary
```

真实模型实验前先确认 GPU 排班和 `nvidia-smi`。多个工作区的队列没有跨服务 GPU 调度，共享环境和权重也不共享显存中的模型实例。

```bash
# GPU 已安排给当前实验时运行；这一条会使用真实模型和仿真。
./dev.sh run tdmpc2-pushcube-episode a_tdmpc2_001 environment=maniskill-pushcube-state-independent params.episodes=1 params.max_steps=5
```

可选启动自己工作区的 Web/API（前台运行，Ctrl+C 结束自己的服务）：

```bash
./dev.sh serve
# 另一个终端，仍在同一工作区：
./dev.sh wms recipes
```

Windows 访问 A 的示例隧道（其余人替换端口）：

```powershell
ssh -F NUL -N -L 18781:127.0.0.1:8781 suzhenjie2026@10.161.97.111
```

浏览器访问 `http://127.0.0.1:18781`。工作区初始复制当前 8766 已部署的静态 Web 构建；前端源码若修改，需自行构建后更新本工作区的 `web/dist`。

## Git 协作

```bash
git status
git add <本次修改的插件、测试和文档>
git commit -m "Implement plugin ..."
# 需要发布自己的开发分支时执行：
git push -u origin dev/a
```

origin 为日常仓库 `https://github.com/hahahahaaaaaa/wm-safetylab-update.git`。初始化没有向 GitHub 推送任何分支；远程写入需要个人已有授权。提交到公共开发基线前评审，涉及内核/适配器的修改单独提交并验证。不要在别人的工作区切换分支或修改文件。

同一服务器账号下的四个 worktree 提供开发目录隔离，不提供用户权限隔离；本次未创建 Linux 用户。需要不同身份时由每个人使用自己的 Git 作者信息（例如提交时 `git -c user.name=... -c user.email=... commit`），避免修改共享仓库级作者配置影响他人。

## 共享与独立数据

- 共享并按固定版本使用：已有 Python 环境、模型权重、上游源码、仿真资产。不要在共享环境中 `pip install -e`、升级或卸载包。
- 独立保存：`.workspace/store/`（服务数据库与运行结果）、`.workspace/runs/`（直接运行结果）、`.workspace/results/`、`.workspace/logs/`、`.workspace/tmp/`、`.workspace/cache/`。
- 每人的 `native/*/runtime*.json` 和 `native-configs.local.json` 是独立机器配置；它们保留共享输入路径并将缓存/临时写入重定向到本工作区。
- Hugging Face 使用独立缓存目录和离线模式；已有大模型缓存文件通过链接复用，缓存元数据独立。不要覆盖共享链接指向的权重文件；训练新权重写入自己的 output。
- TD-MPC2 使用共享 checkpoint 的绝对路径；不依赖旧平台的 `source_run` 在新结果库中存在。
- `./dev.sh` 固定本工作区的 PYTHONPATH、客户端地址和存储目录；直接使用共享环境中的裸 `wms` 命令可能指向旧平台，应使用上述入口。
- 新依赖需要独立环境时，在自己的 `.workspace/` 下建立并修改自己的 runtime；无需复制其他模型环境。

现有 8765、8766 主平台服务及其源码、配置、结果目录不属于开发工作区的写入范围。
