"""OpenVLA 7B (openvla/openvla-7b) as a step-by-step policy for SimplerEnv-Bridge (path B).

Loading, prompt preparation and predict_action come from .common, the same code the
single-action inference (backend.py, path A) uses. The pinned SimplerEnv (eddb569) ships
RT-1 and Octo policies only, so the pre- and post-processing follow the OpenVLA policy
wrapper of the SimplerEnv-OpenVLA fork:

  * the camera image is resized to 224x224 (TensorFlow lanczos3 with antialias, as
    SimplerEnv's policies do);
  * prompt: "template" (default) = OpenVLA's documented prompt; "raw" = the bare
    instruction, which is what the SimplerEnv-OpenVLA fork passes;
  * predict_action(unnorm_key="bridge_orig", do_sample=False) gives 7 values
    [dx, dy, dz, roll, pitch, yaw, gripper] in Bridge units;
  * translation is used as is, roll/pitch/yaw become an axis-angle vector, and the
    gripper value g in [0, 1] becomes 2 * (g > 0.5) - 1 (WidowX: +1 open, -1 close).

OpenVLA has no memory: every step uses only the current frame. In this platform it is the
no-memory comparison for MemoryVLA; its clean success on Bridge is low (0/6 on carrot in the
sub-platform's smoke test). Ported from the sub-platform's sublab_ext.models.openvla.

config: runtime_config  "$runtime.config_file" (checkpoint_dir)
        unnorm_key      default "bridge_orig"
        camera          default 3rd_view_camera
        action_scale    default 1.0
        prompt          "template" (default) or "raw"
"""
from __future__ import annotations

import numpy as np

from wms.scenarios.simpler.common import CAMERA, load_runtime_file

from .common import PROMPT, load, predict_action, prepare_inputs


class OpenVLAPolicy:
    def __init__(self, config):
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")       # TensorFlow only resizes images; keep it off the GPU
        import torch
        self.torch = torch
        self.config = dict(config)
        runtime = load_runtime_file(self.config.get("runtime_config"))
        checkpoint = self.config.get("checkpoint_dir") or runtime["checkpoint_dir"]
        self.camera = self.config.get("camera", CAMERA)
        self.unnorm_key = self.config.get("unnorm_key", "bridge_orig")
        self.action_scale = float(self.config.get("action_scale", 1.0))
        self.prompt = self.config.get("prompt", "template")
        if self.prompt not in ("template", "raw"):
            raise ValueError("prompt must be 'template' or 'raw'")
        self.processor, self.model = load(checkpoint)
        self.info = {}

    def reset(self, seed=None):
        if seed is not None:
            self.torch.manual_seed(int(seed))
            self.torch.cuda.manual_seed_all(int(seed))

    @staticmethod
    def _resize(image):
        import tensorflow as tf
        resized = tf.image.resize(image, size=(224, 224), method="lanczos3", antialias=True)
        return tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.uint8).numpy()

    def act(self, observation, instruction=None):
        from PIL import Image
        from transforms3d.euler import euler2axangle
        text = observation.get("instruction") or instruction
        image = Image.fromarray(self._resize(np.asarray(observation["rgb"][self.camera], dtype=np.uint8)))
        prompt = PROMPT.format(instruction=text) if self.prompt == "template" else text
        inputs, _ = prepare_inputs(self.processor, image, prompt)
        raw = np.asarray(predict_action(self.model, inputs, self.unnorm_key), dtype=np.float64)
        roll, pitch, yaw = raw[3:6]
        axis, angle = euler2axangle(roll, pitch, yaw)
        world = raw[:3] * self.action_scale
        rotation = np.asarray(axis) * angle * self.action_scale
        gripper = 2.0 * float(raw[6] > 0.5) - 1.0
        self.info = {"raw_action": raw.tolist(), "prompt": self.prompt}
        return [*world.tolist(), *rotation.tolist(), gripper]

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()
