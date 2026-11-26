# WM-SafetyLab：A / B / C / D 开发使用说明

更新时间：2026-10-27。本文可以直接转发给开发成员。

大家使用同一主仓库的四个独立工作区，复用服务器已经安装好的 Python、模型权重和仿真资产。进入自己的目录后，通过 `./dev.sh` 运行；不用重新下载模型或安装整个平台。

2026-10-27 已在 A 完成一次真实 TD-MPC2 × PushCube 迁移测试：种子 1000、1 回合、50 步，任务完成且录像可解码。该测试证明这条运行链路可用，不代表所有模型、插件或安全验收均已通过。

## 1. 先找到自己的工作区

服务器地址：`10.161.97.111`；SSH 账号：`suzhenjie2026`。

| 人员 | 服务器工作区 | Git 分支 | 服务器 Web/API 端口 | 建议本地转发端口 |
| --- | --- | --- | --- | --- |
| A | `/home/suzhenjie2026/world-model/workspaces/wm-safetylab-a` | `dev/a` | 8781 | 18781 |
| B | `/home/suzhenjie2026/world-model/workspaces/wm-safetylab-b` | `dev/b` | 8782 | 18782 |
| C | `/home/suzhenjie2026/world-model/workspaces/wm-safetylab-c` | `dev/c` | 8783 | 18783 |
| D | `/home/suzhenjie2026/world-model/workspaces/wm-safetylab-d` | `dev/d` | 8784 | 18784 |

A 负责 S1，B 负责 S2–S4，C 负责 S5–S6；D 的具体任务另行分配。四人目前共用已有 Linux 账号，没有分别建立 Linux 用户，因此要各自在分配的目录里工作。

## 2. 从自己的电脑连接服务器

连接需要能访问服务器所在内网，并且自己的 SSH 公钥已获得该账号授权。如果还没有访问权限，请先让维护者登记自己的公钥；这份说明不包含登录密钥。

Windows PowerShell：

```powershell
ssh -F NUL suzhenjie2026@10.161.97.111
```

如果已授权密钥不是默认文件，可加 `-i "你的已有私钥路径"`。首次连接应向维护者核对主机指纹；已有指纹发生变化时先核实原因。

Linux / macOS 终端：

```bash
ssh suzhenjie2026@10.161.97.111
```

登录后执行下面几行。每个人只改第一行的小写字母：

```bash
DEV_ID=a
cd "/home/suzhenjie2026/world-model/workspaces/wm-safetylab-${DEV_ID}"
pwd
git branch --show-current
./dev.sh info
```

A 应看到 `wm-safetylab-a`、`dev/a`、端口 `8781`。B/C/D 对应替换。新开一个 SSH 终端后重新设置 `DEV_ID` 并进入目录。

## 3. 跑第一个 CPU 实验

下面命令不启动 Web 服务，也不占 GPU。运行编号会加时间戳，避免覆盖已有结果。

```bash
RUN_ID="${DEV_ID}_toy_$(date +%Y%m%d_%H%M%S)"
./dev.sh run toy-episode-occlusion "$RUN_ID" \
  params.episodes=1 params.max_steps=4 params.save_raw=false
```

命令最后应出现 `"exit": 0` 和 `"status": "completed"`。直接运行通常把详细日志写入文件，终端短时间不刷新并不等于卡住。

## 4. 跑真实 TD-MPC2 实验

先查看 GPU 使用情况，并按团队排班运行：

```bash
nvidia-smi
```

四个工作区共用一张 GPU，自己的服务队列不会自动协调别人的任务。不要结束不属于自己的 GPU 进程。真实模型与依赖已经配置好，直接运行：

```bash
RUN_ID="${DEV_ID}_tdmpc2_$(date +%Y%m%d_%H%M%S)"
./dev.sh run tdmpc2-pushcube-episode "$RUN_ID" \
  environment=maniskill-pushcube-state-independent \
  seed=1000 params.seeds='[1000]' \
  params.episodes=1 params.max_steps=50 \
  params.plugins='[]' params.perturbations='[]' params.target_moves='[]' \
  params.save_raw=false params.video=true params.fps=10
```

这会加载共享的已训练 checkpoint，不重新训练模型。不要填旧平台的 `source_run`；新工作区已经通过绝对 checkpoint 路径完成配置。

A 实测运行编号为 `a_migration_tdmpc2_20261027_032622`，耗时约 12.5 秒。其他人首次运行、GPU 负载和缓存状态不同，耗时可能不同。

## 5. 查看日志、结果和录像

在刚刚运行实验的终端继续执行：

```bash
RUN_DIR="$PWD/.workspace/runs/$RUN_ID"
cat "$RUN_DIR/output/result.json"
cat "$RUN_DIR/output/episodes/ep_000/summary.json"
```

想实时看日志，可以另开 SSH 终端，进入同一工作区并使用准确的运行编号：

```bash
RUN_ID=a_migration_tdmpc2_20261027_032622
tail -f "$PWD/.workspace/runs/$RUN_ID/run.log"
```

上面的编号属于 A 的已完成迁移测试；查看自己任务时替换为自己的编号。`Ctrl+C` 退出日志查看，不会结束另一个终端中运行的实验。

| 文件 | 用途 |
| --- | --- |
| `config.json` | 本次解析后的完整实验配置 |
| `run.log` | 模型加载、运行日志和错误信息 |
| `output/result.json` | 是否完成、任务摘要和产物清单 |
| `output/run_manifest.json` | Python、包版本、模型、环境、种子与插件配置 |
| `output/episodes.json` | 各回合结果和环境定义 |
| `output/episodes/ep_000/steps.jsonl` | 逐步动作、观测、奖励和状态 |
| `output/episodes/ep_000/trace.jsonl` | 插件事件、危险事件和接管记录 |
| `output/episodes/ep_000/render_camera.mp4` | 本次 TD-MPC2 任务的录像 |

文件表中的路径均相对于本次 `RUN_DIR`。不同任务、不同录像开关产生的文件可能不同。

将 A 的实测录像下载到 Windows 本机的示例（在本机 PowerShell 执行）：

```powershell
New-Item -ItemType Directory -Force -Path 'K:\world-model\results\team-a' | Out-Null
scp -F NUL suzhenjie2026@10.161.97.111:/home/suzhenjie2026/world-model/workspaces/wm-safetylab-a/.workspace/runs/a_migration_tdmpc2_20261027_032622/output/episodes/ep_000/render_camera.mp4 'K:\world-model\results\team-a\migration.mp4'
```

下载自己的结果时替换人员目录和运行编号。本机项目结果继续保存在 `K:\world-model` 下。

## 6. 使用 Web/API 工作台（可选）

先在服务器自己的工作区中运行：

```bash
./dev.sh serve
```

服务在前台运行，保持这个终端打开。要关闭自己这个开发服务，按 `Ctrl+C`；先确认自己的服务里没有尚未结束的任务。

再在本机新开 PowerShell，为自己建立隧道。A 示例：

```powershell
ssh -F NUL -o ExitOnForwardFailure=yes -N -L 18781:127.0.0.1:8781 suzhenjie2026@10.161.97.111
```

保持隧道窗口打开，浏览器访问 `http://127.0.0.1:18781`。B/C/D 使用表格中的对应端口。

也可以在另一个服务器终端、自己的工作区里通过 CLI 访问同一个服务：

```bash
./dev.sh wms recipes
./dev.sh wms run toy-episode-occlusion --wait
./dev.sh wms runs
```

根据返回的 `run_...` 编号查看服务任务：

```bash
./dev.sh wms get-run run_替换为实际编号
./dev.sh wms logs run_替换为实际编号
```

**两种运行入口的结果位置不同：**

- `./dev.sh run ...` 直接执行，结果在本工作区 `.workspace/runs/`，不会自动进入 Web 的任务列表。
- Web 或 `./dev.sh wms run ...` 经本人的服务提交，结果在 `.workspace/store/runs/`，可以在本人的 Web 工作台查看。

不要使用共享环境里的裸 `wms` 命令；`./dev.sh` 会确保 CLI 连到自己的端口，而不是默认的旧主平台。

## 7. 开发和验证插件

按已分配的卡修改插件代码、注册文件和测试。A/B/C 的现有位置为：

| 人员 | 插件代码 | 注册文件 |
| --- | --- | --- |
| A | 本工作区 `src/wms/plugins/cards/a_*.py` | 本工作区 `native/plugins_a/recipes.json` |
| B | 本工作区 `src/wms/plugins/cards/b_*.py` | 本工作区 `native/plugins_b/recipes.json` |
| C | 本工作区 `src/wms/plugins/cards/c_*.py` | 本工作区 `native/plugins_c/recipes.json` |

D 工作区已建好，但本次没有分配具体卡，也没有新增 `plugins_d` 注册文件；按分配任务确认代码位置。

先用已有监控插件熟悉一致性测试入口：

```bash
./dev.sh plugin-test monitor-state-summary
```

自己的插件完成注册后，把名称换为自己的注册 ID。测试入口会打印结果目录，每次使用新的 `.workspace/results/plugin-test-.../` 子目录，避免覆盖旧证据。

给真实任务挂载已注册插件的参数示例：

```bash
params.plugins='[{"plugin":"monitor-state-summary"}]'
```

把它作为 `./dev.sh run ...` 的参数使用；实际实验应换成自己卡的插件 ID 和配置。任务成功率不能替代插件一致性、同种子配对和组合验收。

## 8. 在自己的分支提交代码

每次开始和提交前，确认自己位于正确目录和分支：

```bash
git branch --show-current
git status --short
git diff
```

不要在别人的工作区切换分支。每张卡尽量单独提交，公共内核或模型适配器修改另作提交。

提交示例（文件名替换成实际文件）：

```bash
git add src/wms/plugins/cards/a_example.py native/plugins_a/recipes.json tests/test_a_example.py
git diff --cached --stat
git diff --cached
git -c user.name="你的姓名" -c user.email="你的邮箱" \
  commit -m "Implement S1 example plugin and its tests"
git log -1 --oneline
```

B/C/D 按自己的实际文件调整。不要照抄尚不存在的 `a_example.py`；这里是提交格式示例。

四个 worktree 共用仓库配置，因此身份信息用上面的单次 `git -c` 方式传入，不要修改共享仓库的 `user.name` / `user.email`。模型、环境、结果和机器 runtime 配置已被 Git 忽略，不要用 `git add -f` 把它们强行提交。

提交后，把“人员分支、commit ID、插件 ID、测试结果目录和实际使用的种子”交给维护者。**同一服务器仓库中的个人提交立即可被维护者读取，不需要先 push 到 GitHub 才能协作。**

## 9. 合并回团队主仓库：当前可直接使用的本地流程

公共集成分支已建立：`team/integration-20261027`。固定起点仍为 `team/base-20261027`，用于追溯初始化版本。

以下由维护者执行，开发成员无需进入管理目录：

```bash
cd /home/suzhenjie2026/world-model/workspaces/wm-safetylab-team/repository
git status --short
git switch team/integration-20261027
git log --oneline team/integration-20261027..dev/a
git diff team/integration-20261027...dev/a
```

确认管理目录没有待提交修改，检查作者提供的测试证据后合并。A 示例：

```bash
git -c user.name="维护者姓名" -c user.email="维护者邮箱" \
  merge --no-ff --no-commit dev/a
git diff --cached --stat
git diff --cached
git -c user.name="维护者姓名" -c user.email="维护者邮箱" \
  commit -m "Merge A plugin work into team integration"
```

出现冲突时先和作者确认；若决定本次暂不合并，使用 `git merge --abort` 取消这次未完成的合并。B/C/D 同理。

维护者合入后，成员回到自己的干净工作区同步公共修改：

```bash
git status --short
git -c user.name="你的姓名" -c user.email="你的邮箱" \
  merge team/integration-20261027
```

先保存自己的未提交工作再同步。合并后的组合效果由指定开发工作区继续做集成验收，仍使用自己的运行目录和 GPU 排班。合入集成分支不等于已经完成 G1–G3。

## 10. 发布到 GitHub 和最终主仓库 PR

日常主仓库是 `https://github.com/hahahahaaaaaa/wm-safetylab-update.git`，remote 名为 `origin`。当前初始化分支尚未推送到 GitHub；本地合并流程现在就能使用。

如需 GitHub PR 评审，维护者先审核并发布共同集成基线：

```bash
git push -u origin team/integration-20261027
```

确认 GitHub 上这个目标分支已存在后，开发成员可在自己的工作区发布自己的分支，例如 A：

```bash
git push -u origin dev/a
```

随后在 GitHub 建 PR：**head 为 `dev/a`，base 为 `team/integration-20261027`**。B/C/D 替换个人分支。PR 中写清具体修改、适用模型/环节、测试和运行证据。服务端的仓库写入权限由维护者安排；没有 GitHub push 权限也不影响在服务器本地提交和交付评审。

若通过 GitHub 合并，维护者在管理目录更新集成分支：

```bash
git fetch origin
git -c user.name="维护者姓名" -c user.email="维护者邮箱" \
  merge --ff-only origin/team/integration-20261027
```

同一批改动选择本地合并或 GitHub PR 合并其中一种流程，避免重复合并和分支分叉。`--ff-only` 拒绝时先检查两边差异，不要强推覆盖。

团队集成验收完成后，由维护者把 `team/integration-20261027` 向日常仓库约定的主开发分支提出最终 PR。本轮源码起点是 `dual-entry`，可作为最终目标，但发布前维护者必须确认团队仍使用它；不要直接假定 GitHub 默认 `main` 就是本轮开发目标。首次集成 PR 会同时包含初始化时记录的当前平台源码快照，维护者应整体评审。

**Git 提交、合并或 push 都不会更新现有 8765/8766 部署。** 线上版本更新属于维护者另行安排的部署操作，不要进入现有主平台目录执行覆盖、同步或重启。

## 11. 常见问题

| 现象 | 处理 |
| --- | --- |
| SSH 超时 | 确认已接入可访问 `10.161.97.111` 的内网/VPN，并向维护者确认服务器可达性 |
| `Permission denied (publickey)` | 确认使用的是自己已授权的密钥，请维护者登记公钥 |
| `./dev.sh` 不存在 | 检查是否进入表格中自己的完整工作区路径 |
| `Connection refused` | Web/API 尚未启动；在自己工作区执行 `./dev.sh serve`。直接运行 `./dev.sh run` 不需要服务 |
| `Address already in use` | 查看自己已有的服务或隧道，不要结束他人的服务；必要时找维护者确认端口 |
| 运行编号已经有结果 | 换一个新的 `RUN_ID`，保留旧结果 |
| Web 看不到直接运行的任务 | 直接运行存到 `.workspace/runs/`；Web 只显示本人的 `.workspace/store/` 任务 |
| 缺包、模型或 GPU 显存不足 | 先发出 `run.log` 和配置给维护者；不要修改共享环境或结束别人的进程 |
| Git 身份未配置 | 提交时使用 `git -c user.name=... -c user.email=... commit ...` |
| GitHub 找不到集成分支 | 当前分支可能还未发布；先使用第 9 节的本地合并流程 |

共享 Python、模型、源码和资产作为固定输入使用；新依赖或新训练权重放到个人目录。所有实验记录保留在自己的 `.workspace/` 下，提交代码时附准确的运行编号，方便复现。
