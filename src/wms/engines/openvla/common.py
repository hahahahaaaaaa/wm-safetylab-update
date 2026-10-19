"""Shared by both OpenVLA entries: single-action inference (backend.py, path A) and the
step-by-step SimplerEnv policy (adapter.py, path B). Loading, prompt and input preparation
and the call into upstream predict_action live here so the two entries cannot drift apart."""
import os

PROMPT = "In: What action should the robot take to {instruction}?\nOut:"


def load(checkpoint_dir, device="cuda:0"):
    """(processor, model) as the pinned official integration loads them: bf16, eager attention, local files."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_dir, attn_implementation="eager", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True,
        use_safetensors=True).to(device).eval()
    return processor, model


def prepare_inputs(processor, image, prompt, device="cuda:0"):
    """Processor inputs for one PIL image, with the prompt-token alignment; returns (inputs, report)."""
    import torch
    from .backend import align_prompt_tokens
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    return inputs, align_prompt_tokens(inputs)


def predict_action(model, inputs, unnorm_key):
    """Upstream predict_action, deterministic decoding; returns the raw 7-vector."""
    import torch
    with torch.inference_mode():
        return model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
