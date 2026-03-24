#!/usr/bin/env python3
"""Compare runtime dump tensors against source Qwen3-Coder-Next BF16 weights.

This harness remains lazy: it reads only the tensors needed for the requested
layers and stages from local safetensors. It serves two roles:

1. Operator-local parity checks for the runtime dump checkpoints.
2. A thresholded "first failure" oracle suitable for regression gating.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.model_metadata import (  # noqa: E402
    MetadataError,
    build_tensor_records,
    load_config,
    load_index,
    resolve_model_source,
)
from tools.q4_affine import bf16_to_f32, require_numpy  # noqa: E402

RMS_NORM_EPS = 1e-6

THRESHOLD_PROFILES = {
    "q4_runtime": {
        "embedding": 0.002,
        "attn": 0.02,
        "router": 0.03,
        "shared": 0.02,
        "final": 0.03,
    }
}

STAGE_ORDER = [
    ("attn_norm_in", "attn"),
    ("post_attn_residual", "attn"),
    ("post_attn_norm", "attn"),
    ("router_logits", "router"),
    ("shared_gate", "shared"),
    ("shared_up", "shared"),
    ("shared_swiglu", "shared"),
    ("shared_down_pre_gate", "shared"),
    ("shared_gate_score", "shared"),
    ("moe_sum_pre_residual", "shared"),
    ("layer_out", "shared"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare runtime dumps against source BF16 tensors")
    parser.add_argument("--model", required=True, help="Local Qwen3-Coder-Next model directory")
    parser.add_argument("--dump-dir", required=True, help="Runtime dump directory from infer --dump-dir")
    parser.add_argument("--step-prefix", default="auto", help='Dump step prefix, e.g. "prefill_last_pos_000000_tok_42"')
    parser.add_argument("--layers", default="all", help='Layer spec: "all", "0-3", or "0,7,11"')
    parser.add_argument("--output", default=None, help="Optional summary JSON output path")
    parser.add_argument("--lm-head-chunk-rows", type=int, default=1024, help="Rows per chunk for lm_head comparison")
    parser.add_argument(
        "--fail-threshold-profile",
        default="q4_runtime",
        choices=sorted(THRESHOLD_PROFILES.keys()),
        help="Named mean-absolute-error threshold profile",
    )
    parser.add_argument(
        "--skip-tokenizer-parity",
        action="store_true",
        help="Skip prompt token / decode parity checks even when prompt_context.json is available",
    )
    return parser.parse_args()


def parse_layers(spec: str, num_layers: int) -> List[int]:
    if spec == "all":
        return list(range(num_layers))
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            out.extend(range(int(start_s), int(end_s) + 1))
        else:
            out.append(int(part))
    out = sorted(set(out))
    invalid = [x for x in out if x < 0 or x >= num_layers]
    if invalid:
        raise ValueError(f"Invalid layer ids: {invalid}")
    return out


def choose_step_prefix(dump_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    prefixes = sorted(
        path.name[: -len("__result.json")]
        for path in dump_dir.glob("*__result.json")
    )
    if not prefixes:
        raise FileNotFoundError(f"No step result files found in {dump_dir}")
    return prefixes[-1]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def try_load_json(path: Path):
    return load_json(path) if path.exists() else None


def load_dump_tensor_f32(dump_dir: Path, prefix: str, name: str):
    np = require_numpy()
    meta = load_json(dump_dir / f"{prefix}__{name}.json")
    if meta.get("dtype") != "f32":
        raise RuntimeError(f"Unsupported dump dtype for {name}: {meta.get('dtype')}")
    shape = tuple(meta["shape"])
    arr = np.frombuffer((dump_dir / f"{prefix}__{name}.bin").read_bytes(), dtype=np.float32)
    return arr.reshape(shape)


def load_dump_layer_tensor_f32(dump_dir: Path, prefix: str, layer_idx: int, names: Sequence[str]):
    if isinstance(names, str):
        names = [names]
    last_error = None
    for name in names:
        try:
            return load_dump_tensor_f32(dump_dir, prefix, f"layer_{layer_idx:02d}_{name}")
        except FileNotFoundError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"No dump tensor for layer {layer_idx} among {list(names)}")


def read_bf16_rows(model_root: Path, record: Dict, row_start: int, row_end: int):
    np = require_numpy()
    shape = tuple(record["shape"])
    if record["dtype"] != "BF16":
        raise MetadataError(f"Expected BF16 tensor, found {record['dtype']}")
    if len(shape) != 2:
        raise MetadataError(f"Expected rank-2 tensor, found shape {shape}")
    cols = shape[1]
    row_start = max(0, row_start)
    row_end = min(shape[0], row_end)
    if row_end <= row_start:
        return np.zeros((0, cols), dtype=np.float32)
    byte_offset = record["absolute_offset"] + row_start * cols * 2
    byte_len = (row_end - row_start) * cols * 2
    with (model_root / record["file"]).open("rb") as handle:
        handle.seek(byte_offset)
        data = handle.read(byte_len)
    arr = np.frombuffer(data, dtype=np.uint16).reshape(row_end - row_start, cols)
    return bf16_to_f32(arr)


def read_bf16_full(model_root: Path, record: Dict):
    shape = tuple(record["shape"])
    if len(shape) == 2:
        return read_bf16_rows(model_root, record, 0, shape[0])
    np = require_numpy()
    with (model_root / record["file"]).open("rb") as handle:
        handle.seek(record["absolute_offset"])
        data = handle.read(record["nbytes"])
    arr = np.frombuffer(data, dtype=np.uint16).reshape(shape)
    return bf16_to_f32(arr)


def linear_bf16(model_root: Path, tensor_records: Dict[str, Dict], tensor_name: str, x):
    w = read_bf16_full(model_root, tensor_records[tensor_name])
    return w @ x


def silu(x):
    np = require_numpy()
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))


def softmax(x):
    np = require_numpy()
    x = x.astype(np.float64, copy=False)
    x = x - np.max(x)
    ex = np.exp(x)
    return (ex / np.sum(ex)).astype(np.float32)


def topk_normalized(router_logits, k: int):
    np = require_numpy()
    probs = softmax(router_logits)
    idx = np.argpartition(-probs, k - 1)[:k]
    idx = idx[np.argsort(-probs[idx])]
    weights = probs[idx]
    weights = weights / np.sum(weights)
    return idx.astype(np.int32), weights.astype(np.float32)


def error_metrics(ref, got):
    np = require_numpy()
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)
    delta = got - ref
    return {
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "rmse": float(math.sqrt(float(np.mean(delta * delta)))),
    }


def attn_out_proj_name(config: Dict, layer_idx: int) -> str:
    layer_type = config["layer_types"][layer_idx]
    if layer_type == "full_attention":
        return f"model.layers.{layer_idx}.self_attn.o_proj.weight"
    return f"model.layers.{layer_idx}.linear_attn.out_proj.weight"


def ensure_layer_types(config: Dict, tensor_records: Dict[str, Dict]) -> None:
    if isinstance(config.get("layer_types"), list) and len(config["layer_types"]) == config["num_hidden_layers"]:
        return

    interval = config.get("full_attention_interval")
    if isinstance(interval, int) and interval > 0:
        config["layer_types"] = [
            "full_attention" if (layer_idx + 1) % interval == 0 else "linear_attention"
            for layer_idx in range(config["num_hidden_layers"])
        ]
        return

    layer_types: List[str] = []
    for layer_idx in range(config["num_hidden_layers"]):
        if f"model.layers.{layer_idx}.self_attn.o_proj.weight" in tensor_records:
            layer_types.append("full_attention")
        elif f"model.layers.{layer_idx}.linear_attn.out_proj.weight" in tensor_records:
            layer_types.append("linear_attention")
        else:
            layer_types.append("unknown")
    config["layer_types"] = layer_types


def rms_norm_bf16(model_root: Path, tensor_records: Dict[str, Dict], tensor_name: str, x):
    np = require_numpy()
    weight = read_bf16_full(model_root, tensor_records[tensor_name]).reshape(-1) + 1.0
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    inv_rms = 1.0 / math.sqrt(float(np.mean(x * x)) + RMS_NORM_EPS)
    return (x * inv_rms * weight).astype(np.float32)


def expert_tensor_name(layer_idx: int, expert_idx: int, proj_name: str) -> str:
    return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.weight"


def compute_expert_output(model_root: Path, tensor_records: Dict[str, Dict], layer_idx: int, expert_idx: int, h_post):
    gate_proj = linear_bf16(model_root, tensor_records, expert_tensor_name(layer_idx, expert_idx, "gate_proj"), h_post)
    up_proj = linear_bf16(model_root, tensor_records, expert_tensor_name(layer_idx, expert_idx, "up_proj"), h_post)
    act = silu(gate_proj) * up_proj
    return linear_bf16(model_root, tensor_records, expert_tensor_name(layer_idx, expert_idx, "down_proj"), act)


def stage_result(ref, got, stage_type: str, thresholds: Dict[str, float], **extra):
    out = error_metrics(ref, got)
    out["threshold_type"] = stage_type
    out["threshold_mean_abs"] = thresholds[stage_type]
    out["pass"] = out["mean_abs_error"] <= thresholds[stage_type]
    out.update(extra)
    return out


def compare_embedding(model_root: Path, tensor_records: Dict[str, Dict], dump_dir: Path, prefix: str, input_token_id: int,
                      thresholds: Dict[str, float]):
    record = tensor_records["model.embed_tokens.weight"]
    ref = read_bf16_rows(model_root, record, input_token_id, input_token_id + 1)[0]
    got = load_dump_tensor_f32(dump_dir, prefix, "embedding")
    return stage_result(ref, got, "embedding", thresholds)


def compare_attn_norm_layer(model_root: Path, tensor_records: Dict[str, Dict], dump_dir: Path, prefix: str,
                            layer_idx: int, thresholds: Dict[str, float]):
    layer_input = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "layer_input")
    got = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "attn_norm_in")
    ref = rms_norm_bf16(model_root, tensor_records,
                        f"model.layers.{layer_idx}.input_layernorm.weight",
                        layer_input)
    return stage_result(ref, got, "attn", thresholds)


def compare_post_attn_residual_layer(model_root: Path, config: Dict, tensor_records: Dict[str, Dict], dump_dir: Path,
                                     prefix: str, layer_idx: int, thresholds: Dict[str, float]):
    layer_input = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "layer_input")
    attn_out = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "attn_out")
    got = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "post_attn_residual")
    ref = layer_input + linear_bf16(model_root, tensor_records, attn_out_proj_name(config, layer_idx), attn_out)
    return stage_result(ref, got, "attn", thresholds)


def compare_post_attn_norm_layer(model_root: Path, tensor_records: Dict[str, Dict], dump_dir: Path, prefix: str,
                                 layer_idx: int, thresholds: Dict[str, float]):
    residual = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "post_attn_residual")
    got = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, ["post_attn_norm", "h_post"])
    ref = rms_norm_bf16(model_root, tensor_records,
                        f"model.layers.{layer_idx}.post_attention_layernorm.weight",
                        residual)
    return stage_result(ref, got, "attn", thresholds)


def compare_router_layer(model_root: Path, tensor_records: Dict[str, Dict], dump_dir: Path, prefix: str,
                         layer_idx: int, thresholds: Dict[str, float]):
    np = require_numpy()
    h_post = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, ["post_attn_norm", "h_post"])
    got_logits = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "router_logits")
    topk_info = load_json(dump_dir / f"{prefix}__layer_{layer_idx:02d}_topk.json")

    ref_logits = linear_bf16(model_root, tensor_records, f"model.layers.{layer_idx}.mlp.gate.weight", h_post)
    ref_idx, ref_w = topk_normalized(ref_logits, topk_info["k"])
    got_idx = np.array(topk_info["indices"], dtype=np.int32)
    got_w = np.array(topk_info["weights"], dtype=np.float32)
    topk_overlap = int(len(set(ref_idx.tolist()) & set(got_idx.tolist())))

    out = stage_result(ref_logits, got_logits, "router", thresholds)
    out.update(
        {
            "topk_exact_match": bool(np.array_equal(ref_idx, got_idx)),
            "topk_overlap": topk_overlap,
            "ref_topk": ref_idx.tolist(),
            "runtime_topk": got_idx.tolist(),
            "ref_topk_weights": [float(x) for x in ref_w],
            "runtime_topk_weights": [float(x) for x in got_w],
            "pass": bool(out["pass"] and topk_overlap == topk_info["k"]),
        }
    )
    return out


def compare_shared_path_layer(model_root: Path, tensor_records: Dict[str, Dict], dump_dir: Path, prefix: str,
                              layer_idx: int, thresholds: Dict[str, float]):
    h_post = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, ["post_attn_norm", "h_post"])
    gate_ref = linear_bf16(model_root, tensor_records,
                           f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight", h_post)
    up_ref = linear_bf16(model_root, tensor_records,
                         f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight", h_post)
    swiglu_ref = silu(gate_ref) * up_ref
    down_ref = linear_bf16(model_root, tensor_records,
                           f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight", swiglu_ref)
    gate_score_ref = float(linear_bf16(model_root, tensor_records,
                                       f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight",
                                       h_post).reshape(-1)[0])

    gate_got = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "shared_gate")
    up_got = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "shared_up")
    swiglu_got = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "shared_swiglu")
    down_got = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, ["shared_down_pre_gate", "shared_out_pre_gate"])
    gate_score_got = load_json(dump_dir / f"{prefix}__layer_{layer_idx:02d}_shared_gate_score.json")["value"]

    gate_score_out = {
        "abs_error": abs(gate_score_ref - gate_score_got),
        "threshold_abs": thresholds["shared"],
        "pass": abs(gate_score_ref - gate_score_got) <= thresholds["shared"],
        "ref": gate_score_ref,
        "runtime": gate_score_got,
    }
    return {
        "shared_gate": stage_result(gate_ref, gate_got, "shared", thresholds),
        "shared_up": stage_result(up_ref, up_got, "shared", thresholds),
        "shared_swiglu": stage_result(swiglu_ref, swiglu_got, "shared", thresholds),
        "shared_down_pre_gate": stage_result(down_ref, down_got, "shared", thresholds),
        "shared_gate_score": gate_score_out,
        "ref_shared_gate_score": gate_score_ref,
        "ref_shared_down_pre_gate": down_ref,
    }


def compare_moe_layer(model_root: Path, tensor_records: Dict[str, Dict], dump_dir: Path, prefix: str,
                      layer_idx: int, thresholds: Dict[str, float],
                      router_info: Dict, shared_info: Dict):
    np = require_numpy()
    residual = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "post_attn_residual")
    h_post = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, ["post_attn_norm", "h_post"])
    moe_sum_got = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "moe_sum_pre_residual")
    layer_out_got = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "layer_out")

    ref_idx = router_info["ref_topk"]
    ref_weights = router_info["ref_topk_weights"]
    ref_moe_sum = np.zeros_like(moe_sum_got)
    for expert_idx, weight in zip(ref_idx, ref_weights):
        ref_moe_sum += compute_expert_output(model_root, tensor_records, layer_idx, int(expert_idx), h_post) * float(weight)

    shared_weight = sigmoid(shared_info["ref_shared_gate_score"])
    ref_moe_sum += shared_info["ref_shared_down_pre_gate"] * shared_weight
    ref_layer_out = residual + ref_moe_sum

    return {
        "moe_sum_pre_residual": stage_result(ref_moe_sum, moe_sum_got, "shared", thresholds),
        "layer_out": stage_result(ref_layer_out, layer_out_got, "shared", thresholds),
    }


def compare_final_logits(model_root: Path, tensor_records: Dict[str, Dict], dump_dir: Path, prefix: str,
                         chunk_rows: int, thresholds: Dict[str, float]):
    np = require_numpy()
    hidden = load_dump_tensor_f32(dump_dir, prefix, "final_hidden").reshape(-1)
    got = load_dump_tensor_f32(dump_dir, prefix, "logits").reshape(-1)
    record = tensor_records["lm_head.weight"]
    rows, cols = record["shape"]
    if cols != hidden.shape[0]:
        raise RuntimeError(f"lm_head width mismatch: {cols} vs hidden {hidden.shape[0]}")

    ref = np.empty(rows, dtype=np.float32)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        chunk = read_bf16_rows(model_root, record, start, end)
        ref[start:end] = chunk @ hidden

    result_info = load_json(dump_dir / f"{prefix}__result.json")
    out = stage_result(ref, got, "final", thresholds)
    out.update(
        {
            "ref_argmax": int(np.argmax(ref)),
            "runtime_argmax": int(np.argmax(got)),
            "runtime_next_token": int(result_info["next_token_id"]),
        }
    )
    out["pass"] = bool(out["pass"] and out["ref_argmax"] == out["runtime_argmax"])
    return out


def load_vocab_bin(path: Path) -> List[str]:
    out: List[str] = []
    with path.open("rb") as f:
        count, _max_id = struct.unpack("<II", f.read(8))
        for _ in range(count):
            (length,) = struct.unpack("<H", f.read(2))
            out.append(f.read(length).decode("utf-8", errors="replace"))
    return out


def compare_tokenizer_parity(model_root: Path, dump_dir: Path, result_info: Dict,
                             prompt_info: Dict, skip: bool):
    if skip:
        return None
    prompt_context = try_load_json(dump_dir / "prompt_context.json")
    if not prompt_context or not prompt_context.get("full_text"):
        return None
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "reason": f"transformers unavailable: {exc}"}

    tokenizer = AutoTokenizer.from_pretrained(str(model_root), trust_remote_code=True)
    ref_prompt_ids = tokenizer.encode(prompt_context["full_text"], add_special_tokens=False)
    runtime_prompt_ids = [int(x) for x in prompt_info["ids"]]
    next_token_id = int(result_info["next_token_id"])
    vocab = load_vocab_bin(model_root / "vocab.bin")
    runtime_decode = vocab[next_token_id] if 0 <= next_token_id < len(vocab) else ""
    ref_decode = tokenizer.decode([next_token_id], clean_up_tokenization_spaces=False)
    return {
        "available": True,
        "prompt_token_ids_match": ref_prompt_ids == runtime_prompt_ids,
        "runtime_prompt_ids": runtime_prompt_ids,
        "ref_prompt_ids": ref_prompt_ids,
        "runtime_decode": runtime_decode,
        "ref_decode": ref_decode,
        "decoded_next_token_match": runtime_decode == ref_decode,
    }


def first_failure_for_layer(layer_summary: Dict) -> Optional[Dict]:
    for stage_name, _stage_type in STAGE_ORDER:
        stage_info = layer_summary.get(stage_name)
        if stage_info is None:
            continue
        if not stage_info.get("pass", True):
            return {
                "stage": stage_name,
                "details": stage_info,
            }
    return None


def first_failure_global(summary: Dict, layers: Sequence[int]) -> Optional[Dict]:
    if not summary["embedding"]["pass"]:
        return {"scope": "embedding", "details": summary["embedding"]}
    for layer_idx in layers:
        layer_failure = summary["layers"][str(layer_idx)]["first_failure"]
        if layer_failure:
            return {"scope": f"layer_{layer_idx:02d}", **layer_failure}
    if not summary["final_logits"]["pass"]:
        return {"scope": "final_logits", "details": summary["final_logits"]}
    tokenizer_info = summary.get("tokenizer_parity")
    if tokenizer_info and tokenizer_info.get("available"):
        if not tokenizer_info.get("prompt_token_ids_match", True):
            return {"scope": "tokenizer_prompt_ids", "details": tokenizer_info}
        if not tokenizer_info.get("decoded_next_token_match", True):
            return {"scope": "tokenizer_decode", "details": tokenizer_info}
    return None


def main() -> int:
    args = parse_args()
    dump_dir = Path(args.dump_dir).expanduser().resolve()
    step_prefix = choose_step_prefix(dump_dir, args.step_prefix)
    thresholds = THRESHOLD_PROFILES[args.fail_threshold_profile]

    model_source = resolve_model_source(args.model)
    if not model_source.is_local:
        print("ERROR: --model must be a local model directory", file=sys.stderr)
        return 1

    model_root = Path(model_source.ref).resolve()
    config = load_config(model_source)
    if config.get("model_type") != "qwen3_next":
        print(f"ERROR: expected qwen3_next model, found {config.get('model_type')!r}", file=sys.stderr)
        return 1

    prompt_info = load_json(dump_dir / "prompt_tokens.json")
    step_meta = load_json(dump_dir / f"{step_prefix}__meta.json")
    result_info = load_json(dump_dir / f"{step_prefix}__result.json")
    input_token_id = int(step_meta["input_token_id"])
    if prompt_info["ids"][-1] != input_token_id:
        print(
            f"WARNING: last prompt token {prompt_info['ids'][-1]} != dumped input token {input_token_id}",
            file=sys.stderr,
        )

    index_data = load_index(model_source)
    tensor_records = build_tensor_records(model_source, index_data)
    ensure_layer_types(config, tensor_records)
    layers = parse_layers(args.layers, config["num_hidden_layers"])

    summary = {
        "model": str(model_root),
        "dump_dir": str(dump_dir),
        "step_prefix": step_prefix,
        "threshold_profile": args.fail_threshold_profile,
        "thresholds": thresholds,
        "prompt_token_count": int(prompt_info["count"]),
        "input_token_id": input_token_id,
        "embedding": compare_embedding(model_root, tensor_records, dump_dir, step_prefix, input_token_id, thresholds),
        "layers": {},
        "final_logits": compare_final_logits(
            model_root, tensor_records, dump_dir, step_prefix, args.lm_head_chunk_rows, thresholds
        ),
        "tokenizer_parity": compare_tokenizer_parity(
            model_root, dump_dir, result_info, prompt_info, args.skip_tokenizer_parity
        ),
    }

    for layer_idx in layers:
        layer_summary = {
            "attn_norm_in": compare_attn_norm_layer(model_root, tensor_records, dump_dir, step_prefix, layer_idx, thresholds),
            "post_attn_residual": compare_post_attn_residual_layer(
                model_root, config, tensor_records, dump_dir, step_prefix, layer_idx, thresholds
            ),
            "post_attn_norm": compare_post_attn_norm_layer(
                model_root, tensor_records, dump_dir, step_prefix, layer_idx, thresholds
            ),
        }
        router_info = compare_router_layer(model_root, tensor_records, dump_dir, step_prefix, layer_idx, thresholds)
        layer_summary["router_logits"] = router_info

        shared_info = compare_shared_path_layer(model_root, tensor_records, dump_dir, step_prefix, layer_idx, thresholds)
        layer_summary.update(
            {
                "shared_gate": shared_info["shared_gate"],
                "shared_up": shared_info["shared_up"],
                "shared_swiglu": shared_info["shared_swiglu"],
                "shared_down_pre_gate": shared_info["shared_down_pre_gate"],
                "shared_gate_score": shared_info["shared_gate_score"],
            }
        )

        moe_info = compare_moe_layer(
            model_root, tensor_records, dump_dir, step_prefix, layer_idx, thresholds, router_info, shared_info
        )
        layer_summary["moe_sum_pre_residual"] = moe_info["moe_sum_pre_residual"]
        layer_summary["layer_out"] = moe_info["layer_out"]
        layer_summary["first_failure"] = first_failure_for_layer(layer_summary)
        summary["layers"][str(layer_idx)] = layer_summary

    summary["first_failure"] = first_failure_global(summary, layers)

    print(f"step_prefix={step_prefix}")
    print(
        "embedding "
        f"max_abs={summary['embedding']['max_abs_error']:.6f} "
        f"mean_abs={summary['embedding']['mean_abs_error']:.6f} "
        f"pass={summary['embedding']['pass']}"
    )
    for layer_idx in layers:
        layer = summary["layers"][str(layer_idx)]
        failure = layer["first_failure"]["stage"] if layer["first_failure"] else "none"
        print(
            f"layer={layer_idx:02d} first_failure={failure} "
            f"attn_norm_mean={layer['attn_norm_in']['mean_abs_error']:.6f} "
            f"router_mean={layer['router_logits']['mean_abs_error']:.6f} "
            f"topk_overlap={layer['router_logits']['topk_overlap']}/{len(layer['router_logits']['runtime_topk'])} "
            f"layer_out_mean={layer['layer_out']['mean_abs_error']:.6f}"
        )
    print(
        "final_logits "
        f"max_abs={summary['final_logits']['max_abs_error']:.6f} "
        f"mean_abs={summary['final_logits']['mean_abs_error']:.6f} "
        f"ref_argmax={summary['final_logits']['ref_argmax']} "
        f"runtime_argmax={summary['final_logits']['runtime_argmax']} "
        f"pass={summary['final_logits']['pass']}"
    )
    tokenizer_info = summary.get("tokenizer_parity")
    if tokenizer_info:
        if tokenizer_info.get("available"):
            print(
                "tokenizer_parity "
                f"prompt_ids_match={tokenizer_info['prompt_token_ids_match']} "
                f"decode_match={tokenizer_info['decoded_next_token_match']}"
            )
        else:
            print(f"tokenizer_parity skipped: {tokenizer_info.get('reason', 'unavailable')}")
    if summary["first_failure"]:
        print(f"first_failure={summary['first_failure']['scope']}")
    else:
        print("first_failure=none")

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")

    return 1 if summary["first_failure"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
