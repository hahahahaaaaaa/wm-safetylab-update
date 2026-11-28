"""Use installed Cosmos models in isolated WMS workers; never actuate hardware."""
import json
import math
from pathlib import Path
import sys
import time

from wms.engines.native_tasks import _runtime, _artifact


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def generate(config, output_dir, context):
    runtime, inputs = _runtime(config), config.get("inputs", {})
    model = Path(runtime["checkpoint_dir"])
    width = _integer(inputs.get("width", 320), "width", 128, 1920)
    height = _integer(inputs.get("height", 192), "height", 128, 1080)
    frames = _integer(inputs.get("frames", 53), "frames", 5, 241)
    steps = _integer(inputs.get("steps", 20), "steps", 1, 100)
    if width % 16 or height % 16 or (frames - 1) % 4:
        raise ValueError("Cosmos requires width/height multiples of 16 and frames = 4n+1")
    import torch
    import diffusers
    from diffusers import Cosmos3OmniPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video, load_image
    import imageio.v3 as iio
    start = time.monotonic()
    context.check_cancelled()
    pipe = Cosmos3OmniPipeline.from_pretrained(str(model), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=12.0, use_karras_sigmas=False)
    prompt = inputs.get("prompt") or json.loads((model / "assets/example_i2v_prompt.json").read_text())
    negative = json.loads((model / "assets/negative_prompt.json").read_text())
    image_path = inputs.get("image_path") or str(model / "assets/example_i2v_input.jpg")
    context.event("progress", {"phase": "cosmos_generate"})
    context.check_cancelled()
    out = pipe(prompt=prompt if isinstance(prompt, str) else json.dumps(prompt),
        negative_prompt=json.dumps(negative), image=load_image(image_path),
        num_frames=frames, height=height, width=width, fps=24., num_inference_steps=steps,
        guidance_scale=6., generator=torch.Generator(device="cuda").manual_seed(config.get("seed", 0)),
        add_resolution_template=False, add_duration_template=False)
    video = Path(output_dir) / "video.mp4"
    export_to_video(out.video, str(video), fps=24, macro_block_size=1)
    decoded = list(iio.imiter(video))
    if len(decoded) != frames:
        raise RuntimeError(f"Expected {frames} video frames, decoded {len(decoded)}")
    report = {"model": "Cosmos3-Edge", "scope": "image_to_video", "seed": config.get("seed", 0),
              "torch": torch.__version__, "diffusers": diffusers.__version__,
              "decoded_frames": len(decoded), "shape": list(decoded[0].shape),
              "seconds": time.monotonic() - start, "output_bytes": video.stat().st_size}
    report_path = context.write_json("runtime_report.json", report)
    return {"outputs": [_artifact(video, "video", "video/mp4"), _artifact(report_path, "report")], "data": report}


def policy_predict(config, output_dir, context):
    runtime, inputs = _runtime(config), config.get("inputs", {})
    import typing
    from typing_extensions import override
    if not hasattr(typing, "override"):
        typing.override = override
    sys.path.insert(0, runtime["source_dir"])
    import numpy as np
    from PIL import Image
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs
    import torch.distributed as dist

    steps = _integer(inputs.get("steps", 4), "steps", 1, 50)
    joint = np.asarray(inputs.get("joint_position", [0.] * 7), dtype=np.float32)
    gripper = inputs.get("gripper_position", 0.)
    if joint.shape != (7,) or not np.isfinite(joint).all() or not isinstance(gripper, (float, int)) or not math.isfinite(gripper):
        raise ValueError("Provide seven finite joint positions and a finite gripper position")
    image_path = inputs.get("image_path") or runtime.get("test_image")
    if not image_path:
        raise ValueError("An observation image_path is required")
    observation = {"prompt": inputs.get("prompt", "Put the banana in the bowl."),
                   "observation/image": np.asarray(Image.open(image_path).convert("RGB")),
                   "observation/joint_position": joint,
                   "observation/gripper_position": float(gripper)}
    start = time.monotonic()
    context.check_cancelled()
    args = RobolabServerArgs(checkpoint_path=runtime["checkpoint_dir"], output_dir=Path(output_dir) / "inference",
        seed=config.get("seed", 0), deterministic_seed=True, num_steps=steps, format_prompt_as_json=True)
    try:
        service = RobolabPolicyService(args)
        context.event("progress", {"phase": "cosmos_policy_inference"})
        context.check_cancelled()
        result = service.infer(observation)
        action = np.asarray(result["action"])
        if action.ndim != 2 or action.shape[0] == 0 or action.shape[1] != 8 or not np.isfinite(action).all():
            raise RuntimeError("Policy returned invalid or nonfinite actions")
        action_path = context.write_json("actions.json", {"action": action.tolist()})
        report = {"model": "Cosmos3-Edge-Policy-DROID", "scope": "offline_action_inference",
                  "observation_source": str(image_path), "observation_note": inputs.get("observation_note", "User supplied observation"),
                  "action_shape": list(action.shape), "finite": True, "seed": config.get("seed", 0),
                  "seconds": time.monotonic() - start, "robot_actuated": False, "simulation_validated": False}
        report_path = context.write_json("runtime_report.json", report)
        return {"outputs": [_artifact(action_path, "actions"), _artifact(report_path, "report")], "data": report}
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
