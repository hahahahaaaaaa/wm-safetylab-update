"""CPU checks for observer cameras that must not become policy observations."""
import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from wms.episodes.steplog import StepLogger
from wms.runners.unified import UnifiedWorker
from wms.scenarios.maniskill.state import ManiSkillStateEnv
from wms.scenarios.toy.env import ToyPickEnv


class RecordingVectorEnv:
    """A pixel encodes the actual position, making capture timing observable."""

    def __init__(self, config):
        self.record_render = bool(config.get("render", False))

    def reset(self, seed, inputs):
        self.position, self.t = 10, 0
        return self.observe()

    def observe(self):
        return [float(self.position), float(self.t)]

    def get_state(self):
        return {"t": self.t, "cube": [self.position, 0, 0]}

    def step(self, action):
        self.position += int(action[0])
        self.t += 1
        return {"obs": self.observe(), "reward": 0.0, "truncated": self.t == 3,
                "events": []}

    def set_object_pose(self, name, xyz=None, delta=None):
        assert name == "cube"
        if xyz is not None:
            self.position = int(xyz[0])
        if delta is not None:
            self.position += int(delta[0])

    def recording_frames(self):
        if not self.record_render:
            return {}
        return {"render_camera": np.full((2, 3, 3), self.position, dtype=np.uint8)}

    def spec(self):
        return {"observation": "vector", "observation_dim": 2, "cameras": {},
                "objects": ["cube"], "action_dim": 1, "max_steps": 3}

    def close(self):
        pass


class VectorInputPolicy:
    def __init__(self, config):
        self.info = {}

    def reset(self, seed):
        pass

    def act(self, observation, instruction=None):
        assert isinstance(observation, list) and len(observation) == 2
        self.info = {"received_vector": list(observation)}
        return [float(observation[0] // 10)]


class RecordingToyEnv(ToyPickEnv):
    def __init__(self, config):
        super().__init__(config)
        self.record_render = bool(config.get("render", False))

    def recording_frames(self):
        return {"render_camera": self.render()} if self.record_render else {}


def _run(output, *, render, rgb=False, **params):
    model = ({"entrypoint": "wms.scenarios.toy.policy:CentroidPickPolicy", "config": {"memory": True}}
             if rgb else {"entrypoint": f"{__name__}:VectorInputPolicy", "config": {}})
    config = {"task": "native", "entrypoint": "wms.episodes.kernel:run", "seed": 7,
              "model": model,
              "environment": {"entrypoint": f"{__name__}:{'RecordingToyEnv' if rgb else 'RecordingVectorEnv'}",
                              "config": {"render": render, "image_size": 32, "max_steps": 3}},
              "params": {"episodes": 1, "video": False, "save_raw": True, **params}}
    code = UnifiedWorker(config, output).run()
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert code == 0, result.get("error")
    return output / "episodes" / "ep_000"


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _array(path, key="seen"):
    with np.load(path) as data:
        return data[key].copy()


def test_recording_leaves_vector_inputs_actions_and_transition_logs_unchanged(tmp_path):
    plain = _run(tmp_path / "plain", render=False)
    recorded = _run(tmp_path / "recorded", render=True)
    for filename in ("steps.jsonl", "trace.jsonl"):
        assert (plain / filename).read_bytes() == (recorded / filename).read_bytes()
    rows = _lines(recorded / "steps.jsonl")
    assert [row["policy_info"]["received_vector"] for row in rows] == [[10.0, 0.0], [11.0, 1.0], [12.0, 2.0]]
    assert [row["action"] for row in rows] == [[1.0], [1.0], [1.0]]
    assert all(row["obs_seen"] == row["policy_info"]["received_vector"] for row in rows)
    assert _read_json(plain / "summary.json")["frames"] == {}
    assert not list(plain.glob("*.npz"))
    assert _read_json(plain / "summary.json")["final_state"] == _read_json(recorded / "summary.json")["final_state"]


def test_recording_follows_env_moves_precedes_action_and_includes_final_frame(tmp_path):
    directory = _run(tmp_path / "moved", render=True,
                     target_moves=[{"step": 1, "object": "cube", "xyz": [40, 0, 0]}])
    frames = _array(directory / "frames_render_camera.npz")
    steps = _lines(directory / "steps.jsonl")
    trace = _lines(directory / "trace.jsonl")
    summary = _read_json(directory / "summary.json")
    # Step 0 moves 10 -> 11; ENV then teleports to 40 before step 1 is observed.
    # Actions 4 and 4 move it to 44 and 48; the last frame has no action/trace row.
    assert frames[:, 0, 0, 0].tolist() == [10, 40, 44, 48]
    assert [row["action"] for row in steps] == [[1.0], [4.0], [4.0]]
    assert [row["state_before"]["cube"][0] for row in steps] == [10, 40, 44]
    assert summary["final_state"]["cube"][0] == int(frames[-1, 0, 0, 0])
    assert len(frames) == len(steps) + 1 == len(trace) + 1 == summary["steps"] + 1
    assert summary["frames"] == {"render_camera": 4}
    assert [row["t"] for row in trace] == [0, 1, 2]


def test_recording_preserves_rgb_plugin_seen_and_raw_logs(tmp_path):
    schedule = [{"type": "drop", "start": 1, "end": 2}]
    plain = _run(tmp_path / "rgb_plain", render=False, rgb=True, perturbations=schedule)
    recorded = _run(tmp_path / "rgb_recorded", render=True, rgb=True, perturbations=schedule)
    for filename in ("steps.jsonl", "trace.jsonl"):
        assert (plain / filename).read_bytes() == (recorded / filename).read_bytes()
    seen = _array(recorded / "frames_top.npz")
    raw = _array(recorded / "raw_top.npz", "raw")
    np.testing.assert_array_equal(seen, _array(plain / "frames_top.npz"))
    np.testing.assert_array_equal(raw, _array(plain / "raw_top.npz", "raw"))
    assert seen[1].max() == 0 and raw[1].max() > 0
    np.testing.assert_array_equal(_array(recorded / "frames_render_camera.npz"), raw)
    assert _read_json(recorded / "summary.json")["frames"] == {"top": 4, "render_camera": 4}


@pytest.mark.parametrize("collision_source", ["seen", "raw"])
def test_camera_collision_rejects_entire_sample_before_appending(tmp_path, collision_source):
    logger = StepLogger(tmp_path, save_raw=True, video=False)
    logger.begin_episode(0, 0)
    image = np.full((2, 3, 3), 50, dtype=np.uint8)
    depth = np.ones((2, 3))
    seen = {"rgb": {"policy_camera": image}, "depth": {"policy_camera": depth}}
    raw = {"rgb": {"raw_camera": image}, "depth": {"raw_camera": depth}}
    camera = "policy_camera" if collision_source == "seen" else "raw_camera"
    try:
        with pytest.raises(ValueError, match="collid"):
            logger.frame(seen, raw, recording_rgb={"other_recording": image, camera: image})
        assert logger.seen == logger.raw == logger.depth_seen == logger.depth_raw == {}
    finally:
        logger.end_episode({})


def test_recording_copies_reused_render_buffer(tmp_path):
    logger = StepLogger(tmp_path, video=False)
    logger.begin_episode(0, 0)
    buffer = np.full((2, 3, 3), 10, dtype=np.uint8)
    try:
        logger.frame([1.0], recording_rgb={"render_camera": buffer})
        buffer[:] = 20
        logger.frame([2.0], recording_rgb={"render_camera": buffer})
        buffer[:] = 30
    finally:
        logger.end_episode({})
    frames = _array(tmp_path / "episodes/ep_000/frames_render_camera.npz")
    assert frames[:, 0, 0, 0].tolist() == [10, 20]


@pytest.mark.parametrize("render", [False, True])
def test_maniskill_recording_is_optional_and_keeps_vector_spec(render):
    # Bypass GPU/weight initialization: exercise the adapter's real methods with a CPU renderer.
    adapter = ManiSkillStateEnv.__new__(ManiSkillStateEnv)
    adapter.record_render = render
    adapter.reconfigure = True
    adapter.cfg = SimpleNamespace(env_id="PushCube-v1", control_mode="pd_joint_delta_pos")
    adapter.hazard_check = SimpleNamespace(spec=lambda: {})
    image = np.full((2, 3, 3), 90, dtype=np.uint8)
    calls = []

    def frame():
        calls.append("frame")
        return image

    adapter.env = SimpleNamespace(frame=frame, action_dim=8, obs_dim=35, horizon=50)
    adapter._obs = [float(i) for i in range(35)]
    original_observation = list(adapter._obs)
    result = adapter.recording_frames()
    if render:
        assert set(result) == {"render_camera"}
        np.testing.assert_array_equal(result["render_camera"], image)
    else:
        assert result == {}
    assert calls == (["frame"] if render else [])
    assert adapter._obs == original_observation
    spec = adapter.spec()
    assert spec["cameras"] == {}
    assert spec["observation"] == "privileged_state" and spec["observation_dim"] == 35
    assert bool(spec.get("recording_cameras")) is render
