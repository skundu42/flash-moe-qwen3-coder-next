#!/usr/bin/env python3
"""Compare runtime dump tensors against source Qwen3-Coder-Next BF16 weights.

This harness is intentionally partial and lightweight. It does not require
loading the full Hugging Face model into memory. Instead it compares selected
runtime checkpoints against source tensors read lazily from local safetensors:

- input embedding for the last prompt token
- per-layer router logits and top-k routing
- per-layer shared expert pre-gate output
- final logits from the dumped final hidden state

Expected runtime dump source:
  metal_infer/infer --dump-dir <dir> ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare runtime dumps against source BF16 tensors")
    parser.add_argument("--model", required=True, help="Local Qwen3-Coder-Next model directory")
    parser.add_argument("--dump-dir", required=True, help="Runtime dump directory from infer --dump-dir")
    parser.add_argument("--step-prefix", default="auto", help='Dump step prefix, e.g. "prefill_last_pos_000000_tok_42"')
    parser.add_argument("--layers", default="all", help='Layer spec: "all", "0-3", or "0,7,11"')
    parser.add_argument("--output", default=None, help="Optional summary JSON output path")
    parser.add_argument("--lm-head-chunk-rows", type=int, default=1024, help="Rows per chunk for lm_head comparison")
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


def load_dump_tensor_f32(dump_dir: Path, prefix: str, name: str):
    np = require_numpy()
    meta = load_json(dump_dir / f"{prefix}__{name}.json")
    if meta.get("dtype") != "f32":
        raise RuntimeError(f"Unsupported dump dtype for {name}: {meta.get('dtype')}")
    shape = tuple(meta["shape"])
    arr = np.frombuffer((dump_dir / f"{prefix}__{name}.bin").read_bytes(), dtype=np.float32)
    return arr.reshape(shape)


def load_dump_layer_tensor_f32(dump_dir: Path, prefix: str, layer_idx: int, name: str):
    return load_dump_tensor_f32(dump_dir, prefix, f"layer_{layer_idx:02d}_{name}")


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
    delta = got - ref
    return {
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "rmse": float(math.sqrt(float(np.mean(delta * delta)))),
    }


def compare_embedding(model_root: Path, tensor_records: Dict[str, Dict], dump_dir: Path, prefix: str, input_token_id: int):
    record = tensor_records["model.embed_tokens.weight"]
    ref = read_bf16_rows(model_root, record, input_token_id, input_token_id + 1)[0]
    got = load_dump_tensor_f32(dump_dir, prefix, "embedding")
    return error_metrics(ref, got)


def compare_router_layer(model_root: Path, tensor_records: Dict[str, Dict], dump_dir: Path, prefix: str, layer_idx: int):
    np = require_numpy()
    h_post = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "h_post")
    got_logits = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "router_logits")
    topk_info = load_json(dump_dir / f"{prefix}__layer_{layer_idx:02d}_topk.json")

    tensor_name = f"model.layers.{layer_idx}.mlp.gate.weight"
    ref_logits = linear_bf16(model_root, tensor_records, tensor_name, h_post)

    ref_idx, ref_w = topk_normalized(ref_logits, topk_info["k"])
    got_idx = np.array(topk_info["indices"], dtype=np.int32)
    got_w = np.array(topk_info["weights"], dtype=np.float32)

    out = error_metrics(ref_logits, got_logits)
    out.update(
        {
            "topk_exact_match": bool(np.array_equal(ref_idx, got_idx)),
            "topk_overlap": int(len(set(ref_idx.tolist()) & set(got_idx.tolist()))),
            "ref_topk": ref_idx.tolist(),
            "runtime_topk": got_idx.tolist(),
            "ref_topk_weights": [float(x) for x in ref_w],
            "runtime_topk_weights": [float(x) for x in got_w],
        }
    )
    return out


def compare_shared_expert_layer(model_root: Path, tensor_records: Dict[str, Dict], dump_dir: Path, prefix: str, layer_idx: int):
    h_post = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "h_post")
    got_pre_gate = load_dump_layer_tensor_f32(dump_dir, prefix, layer_idx, "shared_out_pre_gate")
    gate_score = load_json(dump_dir / f"{prefix}__layer_{layer_idx:02d}_shared_gate_score.json")["value"]

    gate_proj = linear_bf16(model_root, tensor_records,
                            f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight", h_post)
    up_proj = linear_bf16(model_root, tensor_records,
                          f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight", h_post)
    ref_pre_gate = linear_bf16(model_root, tensor_records,
                               f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight",
                               silu(gate_proj) * up_proj)
    ref_gate_score = float(linear_bf16(model_root, tensor_records,
                                       f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight",
                                       h_post).reshape(-1)[0])

    out = error_metrics(ref_pre_gate, got_pre_gate)
    out["shared_gate_score_abs_error"] = abs(ref_gate_score - gate_score)
    return out


def compare_final_logits(model_root: Path, tensor_records: Dict[str, Dict], dump_dir: Path, prefix: str, chunk_rows: int):
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
    return {
        **error_metrics(ref, got),
        "ref_argmax": int(np.argmax(ref)),
        "runtime_argmax": int(np.argmax(got)),
        "runtime_next_token": int(result_info["next_token_id"]),
    }


def main() -> int:
    args = parse_args()
    dump_dir = Path(args.dump_dir).expanduser().resolve()
    step_prefix = choose_step_prefix(dump_dir, args.step_prefix)
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
    input_token_id = int(step_meta["input_token_id"])
    if prompt_info["ids"][-1] != input_token_id:
        print(
            f"WARNING: last prompt token {prompt_info['ids'][-1]} != dumped input token {input_token_id}",
            file=sys.stderr,
        )

    index_data = load_index(model_source)
    tensor_records = build_tensor_records(model_source, index_data)
    layers = parse_layers(args.layers, config["num_hidden_layers"])

    summary = {
        "model": str(model_root),
        "dump_dir": str(dump_dir),
        "step_prefix": step_prefix,
        "prompt_token_count": int(prompt_info["count"]),
        "input_token_id": input_token_id,
        "embedding": compare_embedding(model_root, tensor_records, dump_dir, step_prefix, input_token_id),
        "layers": {},
        "final_logits": compare_final_logits(
            model_root, tensor_records, dump_dir, step_prefix, args.lm_head_chunk_rows
        ),
    }

    for layer_idx in layers:
        summary["layers"][str(layer_idx)] = {
            "router": compare_router_layer(model_root, tensor_records, dump_dir, step_prefix, layer_idx),
            "shared_expert": compare_shared_expert_layer(model_root, tensor_records, dump_dir, step_prefix, layer_idx),
        }

    print(f"step_prefix={step_prefix}")
    print(
        "embedding "
        f"max_abs={summary['embedding']['max_abs_error']:.6f} "
        f"mean_abs={summary['embedding']['mean_abs_error']:.6f}"
    )
    print(
        "final_logits "
        f"max_abs={summary['final_logits']['max_abs_error']:.6f} "
        f"mean_abs={summary['final_logits']['mean_abs_error']:.6f} "
        f"ref_argmax={summary['final_logits']['ref_argmax']} "
        f"runtime_argmax={summary['final_logits']['runtime_argmax']}"
    )
    for layer_idx in layers:
        layer = summary["layers"][str(layer_idx)]
        print(
            f"layer={layer_idx:02d} router "
            f"max_abs={layer['router']['max_abs_error']:.6f} "
            f"mean_abs={layer['router']['mean_abs_error']:.6f} "
            f"topk_overlap={layer['router']['topk_overlap']}/{len(layer['router']['runtime_topk'])}"
        )
        print(
            f"layer={layer_idx:02d} shared_expert "
            f"max_abs={layer['shared_expert']['max_abs_error']:.6f} "
            f"mean_abs={layer['shared_expert']['mean_abs_error']:.6f} "
            f"gate_abs={layer['shared_expert']['shared_gate_score_abs_error']:.6f}"
        )

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
