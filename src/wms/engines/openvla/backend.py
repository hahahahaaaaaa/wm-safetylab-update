"""Official HF predict_action integration. Imports ML libraries only inside the worker."""
import math
import sys
import time



def checked_action(value):
    """Return a JSON-safe 7-vector; malformed outputs must fail before sealing evidence."""
    action = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(action, list) or len(action) != 7:
        raise ValueError("OpenVLA output must have exactly seven action dimensions")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
           for item in action):
        raise ValueError("OpenVLA action must contain seven finite numbers")
    return [float(item) for item in action]


def align_prompt_tokens(inputs):
    """Mirror the pinned predict_action suffix while keeping an optional mask aligned.

    Upstream appends token 29871 when absent, but leaves a supplied attention_mask
    unchanged. Doing the identical token append here avoids an eager-attention
    length mismatch without modifying checkpoint code or the text prompt.
    """
    import torch
    token_id = 29871
    tokens = inputs["input_ids"]
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
        raise ValueError("OpenVLA requires a nonempty batch of one token sequence")
    mask = inputs.get("attention_mask")
    if mask is not None and tuple(mask.shape) != tuple(tokens.shape):
        raise ValueError("Processor attention_mask does not match input_ids before token alignment")
    before = int(tokens.shape[1])
    appended = int(tokens[0, -1].item()) != token_id
    if appended:
        inputs["input_ids"] = torch.cat((tokens, tokens.new_full((1, 1), token_id)), dim=1)
        if mask is not None:
            inputs["attention_mask"] = torch.cat((mask, mask.new_ones((1, 1))), dim=1)
    return {
        "token_id": token_id, "appended": appended,
        "input_tokens_before": before, "input_tokens_after": int(inputs["input_ids"].shape[1]),
        "attention_mask_present": mask is not None,
        "attention_mask_extended": appended and mask is not None,
        "reason": "Mirror pinned predict_action suffix; preserve token/mask length for eager attention",
    }


def infer(config, case, seed, image_path, context):
    import torch
    import transformers
    from PIL import Image
    from .common import PROMPT, load, predict_action, prepare_inputs
    context.check_cancelled()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    context.event("progress", {"phase": "loading_model"})
    processor, model = load(config["checkpoint_dir"])
    context.check_cancelled()
    prompt = PROMPT.format(instruction=case["instruction"])
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
    inputs, prompt_token_alignment = prepare_inputs(processor, image, prompt)
    context.event("progress", {"phase": "predicting_action", "step": 1, "total": 1})
    torch.cuda.synchronize()
    inference_started = time.monotonic()
    prediction = predict_action(model, inputs, case["unnorm_key"])
    torch.cuda.synchronize()
    latency = time.monotonic() - inference_started
    action = checked_action(prediction)
    context.check_cancelled()
    return {
        "action": action, "dimension_names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        "unnorm_key": case["unnorm_key"], "normalization": "dataset_statistics_inverse_applied",
        "action_semantics": case["action_semantics"], "executed": False,
        "step": 0, "environment_feedback": False,
    }, {
        "model_executed": True, "execution_mode": "native", "operation": "infer",
        "python": sys.executable, "seed": seed, "prompt": prompt,
        "prompt_token_alignment": prompt_token_alignment,
        "input_size": list(image.size), "input_image_sha256": case.get("image_sha256"),
        "inference_seconds": latency, "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "weights_repository": config.get("weights", {}).get("repository"),
        "weights_revision": config.get("weights", {}).get("revision"),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "device": "cuda:0", "precision": "bfloat16", "attention": "eager", "quantization": None,
        "executed_on_robot": False, "safety_evaluation": False, "task_success_evaluated": False,
    }
