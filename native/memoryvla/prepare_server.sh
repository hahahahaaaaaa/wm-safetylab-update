#!/usr/bin/env bash
# Prepare MemoryVLA (openvla-codebase) + SimplerEnv for the platform on a Linux GPU server.
#   source scripts/linux/use-linux.sh   (with WMS_ROOT=<world-model root>)
#   bash native/memoryvla/prepare_server.sh [sources|env|weights|runtime|check|all]
# Writes only under $WMS_ROOT (repos, envs, weights, cache, downloads) and this project's
# native/memoryvla/runtime.json + results/memoryvla-integration-20261015/. Upstream sources are
# unpacked from pinned GitHub archives and never edited. Steps can be rerun; finished ones are skipped.
# Hosts without direct GitHub/Hugging Face access: codeload.github.com and HF_ENDPOINT (default
# https://hf-mirror.com) are used; set HF_ENDPOINT=https://huggingface.co where that works.
# Python packages come from UV_INDEX_URL when set (e.g. https://pypi.tuna.tsinghua.edu.cn/simple when
# files.pythonhosted.org does not resolve); torch 2.2.0 from PyPI or a PyPI mirror is the CUDA 12.1 build.
#   bash native/memoryvla/prepare_server.sh checkpoint   # can run in parallel with "env"
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${WMS_ROOT:?Set WMS_ROOT to the absolute project storage root; source scripts/linux/use-linux.sh first.}"
case "$ROOT" in /*) ;; *) echo 'WMS_ROOT must be an absolute path.' >&2; exit 1 ;; esac

MEMVLA_COMMIT=d732ea9072bc063399ccc817aed74ab172eb50be
SIMPLER_COMMIT=eddb5690749a1e34267807c0f7b5eede323918d7
REAL2SIM_COMMIT=ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682   # the SimplerEnv submodule pointer at SIMPLER_COMMIT
DLIMP_COMMIT=040105d256bd28866cc6620621a3d5f7b6b91b46       # moojink/dlimp_openvla, MemoryVLA's git dependency
CKPT_REPO=shihao1895/memvla-bridge
CKPT_REVISION=82b2b6196e331270c80e2b2e853ecf18635b6b53
LLAMA_REPO=NousResearch/Llama-2-7b-hf                       # public copy of the gated meta-llama/Llama-2-7b-hf
LLAMA_REVISION="${WMS_LLAMA_REVISION:-main}"

SOURCE="$ROOT/repos/MemoryVLA-${MEMVLA_COMMIT:0:12}"
SIMPLER="$ROOT/repos/SimplerEnv-${SIMPLER_COMMIT:0:12}"
ENVIRONMENT="$ROOT/envs/memoryvla-simpler-20261015"
WEIGHTS="$ROOT/weights/memvla-bridge"
LLAMA="$ROOT/weights/llama2-7b-hf-config-tokenizer"
DOWNLOADS="$ROOT/downloads/memoryvla"
REPORT="$PROJECT/results/memoryvla-integration-20261015"
PYTHON_BASE="${WMS_MEMVLA_PYTHON:-/usr/bin/python3.10}"     # upstream requires Python 3.10
UV="${UV:-$(command -v uv || echo "$ROOT/runtime/uv-x86_64-unknown-linux-gnu/uv")}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$ROOT/cache/huggingface}" UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/cache/uv}"
export TMPDIR="${TMPDIR:-$ROOT/tmp}" PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/cache/pip}" PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH VIRTUAL_ENV || true
mkdir -p "$DOWNLOADS" "$REPORT" "$TMPDIR" "$ROOT/repos" "$ROOT/envs" "$ROOT/weights"

fetch_archive() {  # fetch_archive OWNER/REPO COMMIT TARGET_DIR
  local archive="$DOWNLOADS/$(basename "$1")-$2.tar.gz"
  if [ -d "$3" ]; then echo "keep $3"; return; fi
  if [ ! -s "$archive" ]; then
    curl -fL --retry 3 --connect-timeout 20 --max-time 1800 "https://codeload.github.com/$1/tar.gz/$2" -o "$archive.part"
    mv -f "$archive.part" "$archive"
  fi
  rm -rf "$3.partial"; mkdir -p "$3.partial"
  tar -xzf "$archive" -C "$3.partial" --strip-components=1
  mv "$3.partial" "$3"
  echo "unpacked $1@$2 -> $3"
}

sources() {
  fetch_archive shihao1895/MemoryVLA "$MEMVLA_COMMIT" "$SOURCE"
  fetch_archive simpler-env/SimplerEnv "$SIMPLER_COMMIT" "$SIMPLER"
  if [ ! -f "$SIMPLER/ManiSkill2_real2sim/setup.py" ]; then   # archive leaves the submodule directory empty
    rmdir "$SIMPLER/ManiSkill2_real2sim" 2>/dev/null || true
    fetch_archive simpler-env/ManiSkill2_real2sim "$REAL2SIM_COMMIT" "$SIMPLER/ManiSkill2_real2sim"
  fi
  ls "$SIMPLER/ManiSkill2_real2sim/data/real_inpainting" | grep -E "bridge_real_eval_1.png|bridge_sink.png"
}

env_install() {
  [ -x "$UV" ] || { echo "uv not found (set UV=...)" >&2; exit 1; }
  if [ ! -x "$ENVIRONMENT/bin/python" ]; then
    "$UV" venv --python "$PYTHON_BASE" "$ENVIRONMENT"
  fi
  "$UV" pip install --python "$ENVIRONMENT/bin/python" -r "$PROJECT/native/memoryvla/requirements-inference.txt"
  local dlimp="$DOWNLOADS/dlimp_openvla-$DLIMP_COMMIT.tar.gz"
  [ -s "$dlimp" ] || curl -fL --retry 3 --connect-timeout 20 "https://codeload.github.com/moojink/dlimp_openvla/tar.gz/$DLIMP_COMMIT" -o "$dlimp"
  "$UV" pip install --python "$ENVIRONMENT/bin/python" --no-deps "$dlimp"
  "$UV" pip freeze --python "$ENVIRONMENT/bin/python" > "$REPORT/requirements-effective.lock.txt"
  "$UV" pip check --python "$ENVIRONMENT/bin/python" || echo "WARN: pip check reported conflicts (kept in the log)"
}

checkpoint() {  # the 33.5 GB .pt with plain curl: resumable, needs no Python environment, can run in parallel
  local url="$HF_ENDPOINT/$CKPT_REPO/resolve/$CKPT_REVISION/checkpoints/memvla-bridge.pt"
  local target="$WEIGHTS/checkpoints/memvla-bridge.pt" size
  mkdir -p "$WEIGHTS/checkpoints"
  size=$(curl -sIL --connect-timeout 20 "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
  [ -n "$size" ] || { echo "cannot read the checkpoint size from $url" >&2; exit 1; }
  if [ -f "$target" ] && [ "$(stat -c %s "$target")" = "$size" ]; then echo "keep $target ($size bytes)"; return; fi
  until [ -f "$target.part" ] && [ "$(stat -c %s "$target.part")" = "$size" ]; do
    curl -fL -C - --retry 5 --retry-delay 10 --connect-timeout 20 "$url" -o "$target.part" || sleep 10
  done
  mv -f "$target.part" "$target"
  echo "checkpoint complete: $target ($size bytes)"
}

weights() {
  local cli="$ENVIRONMENT/bin/huggingface-cli"
  "$cli" download "$CKPT_REPO" config.json config.yaml dataset_statistics.json \
      --revision "$CKPT_REVISION" --local-dir "$WEIGHTS"
  checkpoint
  "$cli" download "$LLAMA_REPO" config.json generation_config.json special_tokens_map.json tokenizer.json \
      tokenizer.model tokenizer_config.json --revision "$LLAMA_REVISION" --local-dir "$LLAMA"
  # Vision backbones are built by timm with pretrained=True before the checkpoint overwrites them:
  # put them in the shared Hugging Face cache now so evaluation runs offline.
  "$ENVIRONMENT/bin/python" - <<'EOF'
import timm
for name in ("vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"):
    timm.create_model(name, pretrained=True, num_classes=0)
    print("cached", name)
EOF
  ls -la "$WEIGHTS" "$WEIGHTS/checkpoints" "$LLAMA"
}

runtime() {
  local target="$PROJECT/native/memoryvla/runtime.json"
  "$ENVIRONMENT/bin/python" - "$target" <<EOF
import json, sys
json.dump({
  "python": "$ENVIRONMENT/bin/python",
  "source_dir": "$SOURCE", "simpler_dir": "$SIMPLER",
  "checkpoint": "$WEIGHTS/checkpoints/memvla-bridge.pt", "llama2_dir": "$LLAMA",
  "source_commits": {"MemoryVLA": "$MEMVLA_COMMIT", "SimplerEnv": "$SIMPLER_COMMIT",
                     "ManiSkill2_real2sim": "$REAL2SIM_COMMIT", "dlimp_openvla": "$DLIMP_COMMIT",
                     "memvla-bridge": "$CKPT_REVISION", "llama2_config_tokenizer": "$LLAMA_REPO@$LLAMA_REVISION"},
  "gpu_device": "0",
  "env": {"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HOME": "$HF_HOME",
          "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json"},
  "timeout_s": 21600}, open(sys.argv[1], "w"), indent=2)
print("wrote", sys.argv[1])
EOF
}

check() {  # imports and the Llama redirect, without loading the 7B weights
  HF_HUB_OFFLINE=1 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json DISPLAY= \
  PYTHONPATH="$PROJECT/src:$SOURCE:$SIMPLER:$SIMPLER/ManiSkill2_real2sim" "$ENVIRONMENT/bin/python" - "$PROJECT/native/memoryvla/runtime.json" <<'EOF'
import json, sys, torch
from wms.engines.memoryvla.simpler import _prepare_process
runtime = json.load(open(sys.argv[1]))
_prepare_process(runtime)
from evaluation.simpler_env import VLAInference                      # noqa: F401
from evaluation.simpler_env.maniskill2_evaluator import run_maniskill2_eval_single_episode  # noqa: F401
from transformers import AutoConfig, AutoTokenizer
from prismatic.models.backbones.llm import llama2
path = llama2.LLAMA2_MODELS["llama2-7b-pure"]["hf_hub_path"]
print("llama2 config", AutoConfig.from_pretrained(path).model_type, "tokenizer", len(AutoTokenizer.from_pretrained(path)))
from simpler_env.utils.env.env_builder import build_maniskill2_env
env = build_maniskill2_env("PutCarrotOnPlateInScene-v0", obs_mode="rgbd", robot="widowx", sim_freq=500,
                           control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos", control_freq=5,
                           max_episode_steps=120, scene_name="bridge_table_1_v1", camera_cfgs={"add_segmentation": True},
                           rgb_overlay_path=runtime["simpler_dir"] + "/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png")
obs, _ = env.reset(options={"obj_init_options": {"episode_id": 0}})
print("simpler env ok:", env.get_language_instruction(), "torch", torch.__version__, "cuda", torch.cuda.is_available())
env.close()
EOF
}

case "${1:-all}" in
  sources) sources ;;
  env) env_install ;;
  checkpoint) checkpoint ;;
  weights) weights ;;
  runtime) runtime ;;
  check) check ;;
  all) sources; env_install; weights; runtime; check ;;
  *) echo "usage: $0 [sources|env|checkpoint|weights|runtime|check|all]" >&2; exit 2 ;;
esac
