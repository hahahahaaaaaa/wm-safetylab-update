"""Check the upstream call contract without loading weights or using a GPU."""
from types import SimpleNamespace

import numpy as np
import pytest

from wms.engines.memoryvla import adapter


class Inference:
    def __init__(self):
        self.vla = SimpleNamespace(cur_timestep=0,
                                   cog_mem_bank=SimpleNamespace(bank={0: []}),
                                   per_mem_bank=SimpleNamespace(bank={0: []}))
        self.calls = []
        self.resets = []

    def reset(self, text):
        self.resets.append(text)

    def step(self, image, text, episode_first_frame):
        self.calls.append((image.copy(), text, episode_first_frame))
        if episode_first_frame == "True":
            self.vla.cur_timestep = 0
            self.vla.cog_mem_bank.bank[0] = []
            self.vla.per_mem_bank.bank[0] = []
        for bank in (self.vla.cog_mem_bank, self.vla.per_mem_bank):
            bank.bank[0].append((self.vla.cur_timestep, np.array([1.0])))
        self.vla.cur_timestep += 1
        raw = {"world_vector": np.array([0.1, 0.2, 0.3]),
               "rotation_delta": np.array([0.4, 0.5, 0.6]), "open_gripper": np.array([0.7])}
        action = {"world_vector": raw["world_vector"], "rot_axangle": raw["rotation_delta"],
                  "gripper": raw["open_gripper"], "terminate_episode": np.array([0])}
        return raw, action


@pytest.mark.parametrize("memory,flags,sizes", [(True, ["True", "False"], [1, 2]),
                                               (False, ["True", "True"], [1, 1])])
def test_rgb_actions_memory_and_recorded_identity(monkeypatch, memory, flags, sizes):
    upstream = Inference()
    seeds = []
    monkeypatch.setattr(adapter, "prepare_process", lambda runtime: None)
    monkeypatch.setattr(adapter, "load_inference", lambda *args, **kwargs: (upstream, None))
    monkeypatch.setattr(adapter, "seed_all", seeds.append)
    policy = adapter.MemoryVLAPolicy({"memory": memory})
    policy.reset(4)
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    for size in sizes:
        action = policy.act({"rgb": {adapter.CAMERA: image}, "instruction": "put carrot on plate"})
        assert action == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        assert policy.info["memory_bank_sizes"] == {"cognitive": size, "perceptual": size}
        assert policy.info["executor"] == "memoryvla"
        assert policy.info["uses_images"] is True
        assert policy.info["memory"] is memory
    assert seeds == [4]
    assert upstream.resets == ["put carrot on plate"]
    assert [call[2] for call in upstream.calls] == flags
    assert all(np.array_equal(call[0], image) for call in upstream.calls)
    policy.reset(5)
    policy.act({"rgb": {adapter.CAMERA: image}}, "new instruction")
    assert upstream.calls[-1][2] == "True"
    assert upstream.resets[-1] == "new instruction"


def test_written_hook_changes_future_memory_not_returned_action(monkeypatch):
    upstream = Inference()
    monkeypatch.setattr(adapter, "prepare_process", lambda runtime: None)
    monkeypatch.setattr(adapter, "load_inference", lambda *args, **kwargs: (upstream, None))
    policy = adapter.MemoryVLAPolicy({})
    phases = []

    def state_hook(state, phase):
        phases.append(phase)
        return {**state, "cognitive": [], "perceptual": []} if phase == "written" else state

    policy.set_hooks(SimpleNamespace(wants=lambda name: True, state=state_hook))
    obs = {"rgb": {adapter.CAMERA: np.zeros((4, 4, 3), dtype=np.uint8)}, "instruction": "test"}
    first = policy.act(obs)
    second = policy.act(obs)
    assert first == second == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert phases == ["written", "read", "written"]
    assert policy.info["memory_bank_sizes"] == {"cognitive": 0, "perceptual": 0}
