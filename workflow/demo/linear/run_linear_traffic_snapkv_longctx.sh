#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

export MODEL_PATH="${MODEL_PATH:-/disk1/hf_cache/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba}"
export CONTEXT_TOKENS="${CONTEXT_TOKENS:-131072}"
export TOTAL_WORKFLOWS="${TOTAL_WORKFLOWS:-16}"
export TARGET_CONCURRENCY="${TARGET_CONCURRENCY:-16}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-132096}"
export QUESTION_PREFIX="${QUESTION_PREFIX:-Traffic demo request}"
export HYPER_PARAMS="${HYPER_PARAMS:-{\"gpu_memory_utilization\": 0.9, \"chunk_prefill_size\": 4096, \"num_sink_tokens\": 4, \"num_recent_tokens\": 32, \"num_top_tokens\": 39286}}"

python - <<'PY'
import json
import os
import sys

from transformers import AutoTokenizer

from workflow.demo.linear.main import main


def build_context(tokenizer, target_tokens: int) -> str:
    seed = (
        "Traffic report: lane occupancy, flow, speed, queue length, signal timing, "
        "merge pressure, incident handling, ramp behavior, and corridor travel time. "
    )
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("Failed to build seed token ids for context generation.")

    repeats = (target_tokens + len(seed_ids) - 1) // len(seed_ids)
    context_ids = (seed_ids * repeats)[:target_tokens]
    context = tokenizer.decode(context_ids, skip_special_tokens=True)

    # Decode/encode round-trips can drift slightly; fix once against the target.
    encoded = tokenizer.encode(context, add_special_tokens=False)
    if len(encoded) < target_tokens:
        deficit = target_tokens - len(encoded)
        encoded = (encoded + seed_ids * ((deficit + len(seed_ids) - 1) // len(seed_ids)))[:target_tokens]
        context = tokenizer.decode(encoded, skip_special_tokens=True)
        encoded = tokenizer.encode(context, add_special_tokens=False)
    elif len(encoded) > target_tokens:
        encoded = encoded[:target_tokens]
        context = tokenizer.decode(encoded, skip_special_tokens=True)
        encoded = tokenizer.encode(context, add_special_tokens=False)

    print(f"context_tokens={len(encoded)}", flush=True)
    return context


model_path = os.environ["MODEL_PATH"]
context_tokens = int(os.environ["CONTEXT_TOKENS"])
total_workflows = os.environ["TOTAL_WORKFLOWS"]
target_concurrency = os.environ["TARGET_CONCURRENCY"]
max_model_len = os.environ["MAX_MODEL_LEN"]
question_prefix = os.environ["QUESTION_PREFIX"]
hyper_params = json.loads(os.environ["HYPER_PARAMS"])

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
context = build_context(tokenizer, context_tokens)

sys.argv = [
    "workflow/demo/linear/main.py",
    "--mode", "traffic",
    "--model_path", model_path,
    "--vllm_sparse_method", "snapkv",
    "--max_model_len", str(max_model_len),
    "--total_workflows", str(total_workflows),
    "--target_concurrency", str(target_concurrency),
    "--question", "",
    "--question_prefix", question_prefix,
    "--context", context,
    "--hyper_params", json.dumps(hyper_params),
]
main()
PY
