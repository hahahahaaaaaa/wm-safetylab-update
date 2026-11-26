"""Platform tasks: train, checkpoint evaluation, and fixed-action latent audit."""
from __future__ import annotations

import json
from pathlib import Path
import time

from . import compat


def _settings(config):
    from wms.engines.native_tasks import _runtime
    runtime = _runtime(config)
    if not runtime:
        raise ValueError("Select a TD-MPC2 runtime.config_file")
    return runtime, dict(config.get("params", {})), dict(config.get("inputs", {}))


def _source(inputs, output, field="checkpoint"):
    value = inputs.get(field)
    if value:
        path = Path(value).resolve()
    elif inputs.get("source_run" if field == "checkpoint" else "evaluation_run"):
        import re
        run_id = inputs["source_run" if field == "checkpoint" else "evaluation_run"]
        if not re.fullmatch(r"run_[0-9a-f]+", run_id):
            raise ValueError("Invalid source/evaluation run ID")
        filename = "model.pt" if field == "checkpoint" else "evaluation.json"
        if field == "checkpoint" and inputs.get("initial_checkpoint", False):
            filename = "initial.pt"
        path = Path(output).resolve().parents[1] / run_id / "output" / filename
    else:
        raise ValueError(f"inputs.{field} is required (or source_run for checkpoint)")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def episode_checkpoint(config, runtime=None):
    """Select a shared episode checkpoint without copying weights.

    Explicit component checkpoint/source_run takes precedence. Web experiments
    may use episode_checkpoint from the machine runtime, relative to that file.
    """
    source = dict(config)
    if not source.get("checkpoint") and not source.get("source_run"):
        from wms.engines.runtime import load_runtime
        reference = config.get("runtime_config")
        runtime = runtime if runtime is not None else load_runtime(reference)
        selected = (runtime or {}).get("episode_checkpoint")
        if selected:
            path = Path(selected)
            if not path.is_absolute() and isinstance(reference, (str, Path)):
                path = Path(reference).resolve().parent / path
            source["checkpoint"] = str(path)
    return _source(source, config.get("output_dir", "."))


def _result(output, data):
    import mimetypes
    files = []
    for path in sorted(Path(output).rglob("*")):
        if not path.is_file() or path.name in {"result.json", "cancel.flag"}:
            continue
        suffix = path.suffix
        kind = "checkpoint" if suffix == ".pt" else "video" if suffix in {".mp4", ".webm"} else "report" if suffix == ".html" else "evidence"
        files.append({"path": path.relative_to(output).as_posix(), "kind": kind,
                      "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream"})
    return {"data": data, "outputs": files}


def train(config, output_dir, context):
    from .backend import train as train_backend
    runtime, params, inputs = _settings(config)
    report = train_backend(runtime, params, Path(output_dir), context,
                           seed=int(config.get("seed", 0)), checkpoint=inputs.get("checkpoint"))
    return _result(output_dir, report)


def _goal_position_diagnostics(cube_xyz, goal_xyz, center_threshold_m=0.02):
    """XY position diagnostics only; never replace the simulator success flag."""
    import math
    distances = [math.hypot(float(cube[0]) - float(goal[0]),
                            float(cube[1]) - float(goal[1]))
                 for cube, goal in zip(cube_xyz, goal_xyz, strict=True)]
    return distances, {"final_goal_distance_m": distances[-1],
                       "min_goal_distance_m": min(distances),
                       "center_within_2cm": distances[-1] <= center_threshold_m}


def _goal_success_diagnostics(cube_xyz, goal_xyz):
    """Geometry evidence aligned to T+1 states; stability excludes reset."""
    distances, diagnostics = _goal_position_diagnostics(cube_xyz, goal_xyz)
    on_table = [abs(float(cube[2]) - 0.02) <= 0.005 for cube in cube_xyz]
    center = [distance <= 0.02 and valid for distance, valid in zip(distances, on_table, strict=True)]
    region = [distance < 0.1 and float(cube[2]) < 0.025
              for distance, cube in zip(distances, cube_xyz, strict=True)]
    post_action_center = center[1:]
    diagnostics.update(region_success_at_end=region[-1], center_success_at_end=center[-1],
                       center_stable_last_5=len(post_action_center) >= 5 and all(post_action_center[-5:]))
    return distances, center, region, diagnostics


def policy_evaluate(config, output_dir, context):
    runtime, params, inputs = _settings(config)
    compat.prepare(runtime)
    import numpy as np
    import torch
    from .backend import act, intervene_action
    output = Path(output_dir)
    checkpoint = _source(inputs, output)
    agent, cfg, training_metadata = compat.checkpoint_load(checkpoint)
    expected_env_id = params.get("expected_env_id")
    if expected_env_id and cfg.env_id != expected_env_id:
        raise ValueError(f"Checkpoint task {cfg.env_id!r} differs from requested task {expected_env_id!r}")
    seed_start = int(params.get("seed_start", 1000))
    seeds = [int(v) for v in inputs.get("seeds", range(seed_start, seed_start + int(params.get("episodes", 10))))]
    if not seeds:
        raise ValueError("At least one evaluation seed required")
    video_episodes = int(params.get("video_episodes", 2))
    env = compat.StateEnvironment(cfg, 1, render=video_episodes > 0)
    if env.obs_dim != agent.model._encoder["state"][0].in_features:
        env.close()
        raise ValueError("Checkpoint observation layout differs from actual environment")
    noise_std = float(params.get("action_noise_std", 0))
    obs_noise = float(params.get("observation_noise_std", 0))
    scale = float(params.get("action_scale", 1))
    if min(noise_std, obs_noise) < 0:
        raise ValueError("Noise standard deviations must be nonnegative")
    episodes, trajectories = [], []
    started = time.monotonic()
    origin = compat.provenance(runtime, cfg.env_id)
    context.write_json("provenance.json", origin)
    try:
        for episode_index, seed in enumerate(seeds):
            context.check_cancelled()
            compat.seed_all(seed)
            noise_generator = torch.Generator(device="cpu").manual_seed(seed + 1000000)
            obs_generator = torch.Generator(device="cpu").manual_seed(seed + 2000000)
            obs, _ = env.reset(seed)
            states = [obs[0].detach().cpu().numpy().copy()]
            oracle = env.oracle()
            cube, tcp = [oracle["cube_xyz"][0].copy()], [oracle["tcp_xyz"][0].copy()]
            goal = [oracle["goal_xyz"][0].copy()]
            actions, proposed, policy_obs, rewards, success, changes, clipped = [], [], [], [], [], [], []
            frames = [env.frame()] if episode_index < video_episodes else []
            for step in range(env.horizon):
                context.check_cancelled()
                seen = obs.detach().cpu().clone()
                if obs_noise:
                    seen += torch.randn(seen.shape, generator=obs_generator) * obs_noise
                requested = act(agent, seen, first=step == 0)
                applied, intervention = intervene_action(requested, noise_std=noise_std,
                                                         scale=scale, generator=noise_generator)
                next_obs, reward, terminated, truncated, info = env.step(applied)
                policy_obs.append(seen[0].numpy().copy())
                proposed.append(requested[0].numpy().copy())
                actions.append(applied[0].numpy().copy())
                rewards.append(float(reward.item()))
                success.append(bool(info["success"].item()))
                changes.append(intervention["changed"])
                clipped.append(intervention["clipped"])
                obs = next_obs
                states.append(obs[0].detach().cpu().numpy().copy())
                oracle = env.oracle()
                cube.append(oracle["cube_xyz"][0].copy())
                tcp.append(oracle["tcp_xyz"][0].copy())
                goal.append(oracle["goal_xyz"][0].copy())
                if frames:
                    frames.append(env.frame())
            distances, center_success, region_success, position_diagnostics = _goal_success_diagnostics(cube, goal)
            trajectory = {"observations": np.asarray(states), "actions": np.asarray(actions),
                          "proposed_actions": np.asarray(proposed), "policy_observations": np.asarray(policy_obs),
                          "rewards": np.asarray(rewards), "success": np.asarray(success),
                          "cube_xyz": np.asarray(cube), "tcp_xyz": np.asarray(tcp),
                          "goal_xyz": np.asarray(goal), "goal_distance_m": np.asarray(distances),
                          "center_success": np.asarray(center_success),
                          "original_region_success": np.asarray(region_success),
                          "seed": np.asarray(seed)}
            np.savez_compressed(output / f"trajectory_{episode_index:03d}.npz", **trajectory)
            trajectories.append(trajectory)
            row = {"seed": seed, "return": sum(rewards), "success_once": any(success),
                   "success_at_end": success[-1], "steps": len(actions),
                   "terminated_at_end": bool(terminated.item()), "truncated_at_end": bool(truncated.item()),
                   "action_intervention_steps": sum(changes), "action_clipped_steps": sum(clipped),
                   "trajectory": f"trajectory_{episode_index:03d}.npz",
                   "video": None, "video_frames": 0, **position_diagnostics}
            if frames:
                fps = int(env.native.control_freq)
                if cfg.env_id == "WMS-PushCubeCenter-v1":
                    from .media import write_video
                    row.update(write_video(output, f"episode_{seed}", frames, fps, compatibility=True))
                else:
                    import imageio.v2 as imageio
                    name = f"episode_{seed}.mp4"
                    imageio.mimsave(output / name, frames, fps=fps, macro_block_size=1,
                                   codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"])
                    row.update(video=name, video_frames=len(frames), video_fps=fps,
                               video_duration_s=len(frames) / fps)
            episodes.append(row)
            context.event("progress", {"phase": "policy_evaluate", "episode": episode_index + 1, **row})
        report = {"task": "policy_evaluate", "checkpoint": str(checkpoint),
                  "checkpoint_sha256": compat.file_sha256(checkpoint),
                  "task_sha256": origin["task_sha256"], "env_id": cfg.env_id,
                  "observation_dim": env.obs_dim, "action_dim": env.action_dim,
                  "training_metadata": training_metadata, "episodes": episodes,
                  "elapsed_seconds": time.monotonic() - started,
                  "controller": cfg.control_mode, "observation": "privileged simulator state",
                  "intervention": {"action_noise_std": noise_std, "action_scale": scale,
                                   "observation_noise_std": obs_noise},
                  "protocol": {"horizon": env.horizon, "sim_backend": "physx_cuda",
                               "reward_mode": "normalized_dense", "success_termination_ignored": True,
                               "reward_version": "center-dense/1" if cfg.env_id == "WMS-PushCubeCenter-v1" else "maniskill-normalized-dense",
                               "success_protocol": {
                                   "task_success": "center_success" if cfg.env_id == "WMS-PushCubeCenter-v1" else "original_region_success",
                                   "center_rule": "XY <= 0.02 m and abs(cube z - 0.02 m) <= 0.005 m",
                                   "original_region_rule": "XY < 0.1 m and cube z < 0.025 m",
                                   "center_stable_last_5": "All last 5 post-action states satisfy center_rule; reset excluded",
                                   "geometry_alignment": "T+1 states including reset; native success has T post-action values",
                               },
                               "planner": "official stochastic MPPI; separate episode seed",
                                "official_goal_radius_m": 0.1,
                                "official_cube_z_threshold_m": 0.025,
                                "official_success_rule": "Original PushCube-v1 reference: XY < 0.1 m and cube z < 0.025 m; stored separately from custom task success",
                                "center_diagnostic_threshold_m": 0.02,
                                "center_diagnostic_rule": "final XY distance <= 0.02 m; position only, not official success or safety",
                                "distance_samples": "T+1 simulator states including reset; minimum includes reset",
                                "requested_video_episodes": video_episodes},
                  "safety_status": "not_measured"}
        context.write_json("evaluation.json", report)
        _html_evaluation(output, report)
        return _result(output, report)
    finally:
        env.close()


def predict(config, output_dir, context):
    """Use fixed actions from genuine held-out episodes; never step a simulator."""
    runtime, params, inputs = _settings(config)
    compat.prepare(runtime)
    import numpy as np
    import torch
    from common.math import two_hot_inv
    output = Path(output_dir)
    checkpoint = _source(inputs, output)
    episode_report = _source(inputs, output, "evaluation_report")
    agent, cfg, training_metadata = compat.checkpoint_load(checkpoint)
    expected_env_id = params.get("expected_env_id")
    if expected_env_id and cfg.env_id != expected_env_id:
        raise ValueError(f"Checkpoint task {cfg.env_id!r} differs from requested task {expected_env_id!r}")
    source = json.loads(episode_report.read_text(encoding="utf-8"))
    horizons = [int(h) for h in params.get("horizons", [1, 5])]
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("Prediction horizons must be positive")
    origin = compat.provenance(runtime, cfg.env_id)
    if (source.get("controller") != cfg.control_mode or source.get("env_id") != cfg.env_id
            or source.get("task_sha256") != origin["task_sha256"]
            or source.get("protocol", {}).get("reward_mode") != "normalized_dense"):
        raise ValueError("Trajectory controller/task/reward semantics differ from the selected model environment")
    if not source.get("episodes"):
        raise ValueError("No evaluation episodes provided")
    totals = {h: {"model": [], "hold": [], "zero_action": [], "reward": []} for h in horizons}
    records = []
    compat.seed_all(int(config.get("seed", 0)))
    for episode_index, episode in enumerate(source["episodes"]):
        context.check_cancelled()
        values = np.load(episode_report.parent / episode["trajectory"])
        obs = torch.as_tensor(values["observations"], device="cuda", dtype=torch.float32)
        actions = torch.as_tensor(values["actions"], device="cuda", dtype=torch.float32)
        rewards = torch.as_tensor(values["rewards"], device="cuda", dtype=torch.float32)
        if (obs.shape != (len(actions) + 1, cfg.obs_shape["state"][0])
                or actions.shape != (len(actions), cfg.action_dim) or rewards.shape != (len(actions),)):
            raise ValueError("Trajectory must contain T+1 observations, T actions and T rewards in the checkpoint layout")
        if not all(bool(torch.isfinite(v).all()) for v in (obs, actions, rewards)):
            raise ValueError("Non-finite trajectory evidence")
        with torch.no_grad():
            real_latents = agent.model.encode(obs, None)
            for horizon in horizons:
                count = len(actions) - horizon + 1
                if count <= 0:
                    raise ValueError("Prediction horizon exceeds episode length")
                initial = real_latents[:count]
                z, zero_z = initial.clone(), initial.clone()
                imagined_rewards = []
                for offset in range(horizon):
                    a = actions[offset:offset + count]
                    imagined_rewards.append(two_hot_inv(agent.model.reward(z, a, None), cfg).squeeze(-1))
                    z = agent.model.next(z, a, None)
                    zero_z = agent.model.next(zero_z, torch.zeros_like(a), None)
                target = real_latents[horizon:horizon + count]
                mse = (z - target).square().mean(-1)
                hold = (initial - target).square().mean(-1)
                zero = (zero_z - target).square().mean(-1)
                reward_error = (imagined_rewards[-1] - rewards[horizon - 1:horizon - 1 + count]).abs()
                for key, value in (("model", mse), ("hold", hold), ("zero_action", zero), ("reward", reward_error)):
                    totals[horizon][key].extend(value.cpu().tolist())
                name = f"latents_episode_{episode_index:03d}_h{horizon}.npz"
                np.savez_compressed(output / name, initial=initial.cpu().numpy(),
                                    predicted=z.cpu().numpy(), actual=target.cpu().numpy(),
                                    zero_action=zero_z.cpu().numpy(),
                                    predicted_reward=imagined_rewards[-1].cpu().numpy(),
                                    actual_reward=rewards[horizon - 1:horizon - 1 + count].cpu().numpy())
                records.append({"seed": episode["seed"], "horizon": horizon, "samples": count, "file": name})
    metrics = {}
    for h, values in totals.items():
        metrics.update({f"latent_mse_h{h}": float(np.mean(values["model"])),
                        f"persistence_latent_mse_h{h}": float(np.mean(values["hold"])),
                        f"zero_action_latent_mse_h{h}": float(np.mean(values["zero_action"])),
                        f"reward_mae_h{h}": float(np.mean(values["reward"]))})
    report = {"task": "predict", "checkpoint": str(checkpoint), "evaluation_report": str(episode_report),
              "checkpoint_sha256": compat.file_sha256(checkpoint), "task_sha256": origin["task_sha256"],
              "env_id": cfg.env_id, "reward_version": source.get("protocol", {}).get("reward_version"),
              "training_metadata": training_metadata, "metrics": metrics, "records": records,
              "semantics": "Within-checkpoint latent-space error; not Cartesian error, collision probability or cross-checkpoint ranking",
              "safety_status": "not_measured"}
    context.write_json("prediction_summary.json", report)
    return _result(output, report)


def _html_evaluation(output, report):
    import html

    def distance(value):
        return "未记录" if value is None else f"{float(value) * 100:.2f} cm"

    def flag(value):
        return "是" if value is True else "否" if value is False else "未记录"

    protocol = report.get("protocol", {})
    is_center = report.get("env_id") == "WMS-PushCubeCenter-v1"
    task_label = "中心任务" if is_center else "官方"

    def rate(field):
        values = [ep.get(field) for ep in report["episodes"]]
        if not values or any(type(value) is not bool for value in values):
            return "未完整记录"
        count = sum(values)
        return f"{count}/{len(values)}（{100 * count / len(values):.1f}%）"

    rows = []
    recorded_count = sum(bool(ep.get("video") or ep.get("video_webm")) for ep in report["episodes"])
    for ep in report["episodes"]:
        video_path, webm_path = ep.get("video"), ep.get("video_webm")
        duration = ep.get("video_duration_s")
        if duration is None and ep.get("video_fps") and ep.get("video_frames"):
            duration = ep["video_frames"] / ep["video_fps"]
        duration_text = "回放时长未记录" if duration is None else f"回放 {duration:.2f} 秒"
        if video_path or webm_path:
            sources = []
            for path, media in ((webm_path, "video/webm"), (video_path, "video/mp4")):
                if path:
                    sources.append(f'<source src="{html.escape(str(path), quote=True)}" type="{media}">')
            poster = (' poster="' + html.escape(str(ep["video_poster"]), quote=True) + '"') if ep.get("video_poster") else ""
            url = html.escape(str(video_path or webm_path), quote=True)
            video = (f'<video controls loop playsinline preload="metadata" width="480"{poster}>' + ''.join(sources)
                     + f'浏览器无法播放时请<a href="{url}">下载录像</a>。</video>'
                     + f'<p>{duration_text} · <a href="{url}" target="_blank" rel="noreferrer">打开 / 下载原录像</a></p>')
        else:
            video = "本回合未录制（不是播放失败）"
        values = [str(ep["seed"]), f'{ep["return"]:.3f}', flag(ep.get("success_once")),
                  flag(ep.get("success_at_end")), distance(ep.get("final_goal_distance_m")),
                  distance(ep.get("min_goal_distance_m")), flag(ep.get("center_within_2cm")),
                  str(ep.get("action_intervention_steps", "未记录")),
                  flag(ep.get("region_success_at_end")), flag(ep.get("center_success_at_end")),
                  flag(ep.get("center_stable_last_5"))]
        rows.append('<tr>' + ''.join('<td>' + html.escape(value) + '</td>' for value in values)
                    + '<td>' + video + '</td></tr>')
    text = '<!doctype html><meta charset="utf-8"><title>TD-MPC2 PushCube 评测</title>'
    text += '''<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{font:16px system-ui;margin:24px;background:#f6f8fc;color:#182538;overflow-wrap:anywhere}
.notice{background:#fff3d6;border-left:4px solid #d69b20;padding:12px 16px;line-height:1.7}
.table-scroll{overflow-x:auto}table{border-collapse:collapse;background:white}td,th{padding:12px;border:1px solid #dce1e8}
video{width:480px;max-width:100%;display:block;border-radius:8px}
@media(max-width:700px){body{margin:16px}h1{font-size:26px}table,tbody,tr,td{display:block}
tr:first-child{display:none}tr{margin:16px 0;padding:12px;border:1px solid #dce1e8;border-radius:12px}
td{padding:5px 0;border:0;display:grid;grid-template-columns:140px 1fr;gap:8px}
td:before{color:#64748b}td:nth-child(1):before{content:'种子'}td:nth-child(2):before{content:'回报'}
td:nth-child(3):before{content:'本任务曾成功'}td:nth-child(4):before{content:'本任务最终成功'}
td:nth-child(5):before{content:'最终距圆心'}td:nth-child(6):before{content:'最小距圆心'}
td:nth-child(7):before{content:'最终圆心2cm内'}td:nth-child(8):before{content:'动作干预步数'}
td:nth-child(9):before{content:'原区域最终成功'}td:nth-child(10):before{content:'中心最终成功'}
td:nth-child(11):before{content:'末5步中心稳定'}td:nth-child(12){display:block;margin-top:12px}}
</style>'''
    text += '<h1>TD-MPC2 × ' + html.escape(report.get("env_id", "ManiSkill PushCube")) + '</h1><p>真实仿真 · 状态输入 · 潜空间规划 · 固定检查点评测</p>'
    if is_center:
        text += ('<div class="notice"><strong>独立中心精度任务：WMS-PushCubeCenter-v1。</strong>'
                 '中心任务成功要求 XY 距离 ≤ 2cm，且方块中心高度在 2cm ± 0.5cm；'
                 '末5步稳定要求最后5个执行动作后的状态均满足条件。该任务已调整奖励，'
                 '“中心任务成功”不是原 PushCube-v1 的官方10cm成功指标。</div>')
    else:
        text += ('<div class="notice"><strong>官方目标区域成功不等于推到圆心。</strong>'
                 '官方成功保留仿真器的判定；“最终圆心 2cm 内”只是补充的 XY 位置诊断，'
                 '不更改原基准、奖励或训练目标，也不是安全判定。未记录的历史字段不会被当作成功或失败。</div>')
    text += ('<p>原 PushCube-v1 区域条件：方块中心到圆心的 XY 距离 &lt; '
             + distance(protocol.get("official_goal_radius_m")) + '，且方块中心 z &lt; '
             + distance(protocol.get("official_cube_z_threshold_m")) + '。'
             '中心成功额外检查桌面高度；“最终圆心2cm内”仅检查 XY。最小距离包含重置时刻。</p>')
    text += ('<p><strong>最终中心成功：' + rate("center_success_at_end")
             + ' · 末5步中心稳定：' + rate("center_stable_last_5")
             + '</strong> · 原10cm区域最终成功：' + rate("region_success_at_end") + '</p>')
    final_distances = [ep.get("final_goal_distance_m") for ep in report["episodes"]]
    average_distance = (sum(final_distances) / len(final_distances)
                        if final_distances and all(value is not None for value in final_distances) else None)
    text += '<p>最终平均圆心距离：' + distance(average_distance) + '。未知历史字段不填0。</p>'
    text += (f'<p>共 {len(report["episodes"])} 个回合，本报告只附 {recorded_count} 条录像。'
             '未录制的回合仍有数值与轨迹结果；录像按仿真控制频率播放，短回放不代表任务执行失败。</p>')
    text += '<p>任务成功不代表安全。当前未测量碰撞、接触力等安全信号；视频为真实仿真画面。</p>'
    text += '<p>检查点：' + html.escape(report["checkpoint"]) + '</p><p>干预：' + html.escape(json.dumps(report["intervention"])) + '</p>'
    text += ('<div class="table-scroll"><table><tr><th>种子</th><th>回报</th><th>' + task_label + '曾成功</th><th>' + task_label + '最终成功</th>'
             '<th>最终距圆心（XY）</th><th>最小距圆心（XY）</th><th>最终圆心 2cm 内<br>仅位置诊断</th>'
             '<th>动作干预步数</th><th>原10cm区域最终成功</th><th>中心最终成功<br>含桌面高度</th>'
             '<th>末5步中心稳定</th><th>录像 / 回放时长</th></tr>' + ''.join(rows) + '</table></div>')
    (output / "visualization.html").write_text(text, encoding="utf-8")
