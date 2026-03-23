#!/usr/bin/env python3
"""Extract non-routed weights into the runtime binary format.

Modes:
  - legacy: original MLX-style quantized extraction for Qwen3.5-397B-A17B
  - qwen3-next: local BF16 Qwen3-Coder-Next -> runtime q4/BF16/F32 layout

The qwen3-next path is additive. It emits runtime tensor names that mostly match
the current C engine's expectations, including splitting merged upstream tensors
such as `linear_attn.in_proj_qkvz.weight` into `in_proj_qkv` + `in_proj_z`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.model_metadata import (  # noqa: E402
    MetadataError,
    build_runtime_config_candidate,
    build_tensor_records,
    default_qwen35_a17b_runtime_config,
    load_config,
    load_index,
    load_runtime_config_from_summary,
    parse_safetensors_header,
    resolve_model_source,
    summarize_layers,
)
from tools.q4_affine import quantize_matrix_q4_affine, require_numpy  # noqa: E402
from tools.qwen3_next_adapter import build_qwen3_next_conversion_plan  # noqa: E402


LEGACY_DEFAULT_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit"
    "/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3"
)


def detect_mode(args: argparse.Namespace) -> str:
    if args.mode != "auto":
        return args.mode
    if args.summary:
        summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        if summary.get("config_subset", {}).get("model_type") == "qwen3_next":
            return "qwen3-next"
    cfg_path = Path(args.model).expanduser() / "config.json"
    if cfg_path.exists():
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        if config.get("model_type") == "qwen3_next":
            return "qwen3-next"
    return "legacy"


def parse_safetensors_header_local(filepath: Path):
    with filepath.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def write_aligned_tensor(out_f, manifest_tensors: Dict, name: str, payload: bytes, shape: Iterable[int], dtype: str, offset: int) -> int:
    align = 64
    if offset % align != 0:
        pad = align - (offset % align)
        out_f.write(b"\x00" * pad)
        offset += pad
    out_f.write(payload)
    manifest_tensors[name] = {
        "offset": offset,
        "size": len(payload),
        "shape": list(shape),
        "dtype": dtype,
    }
    return offset + len(payload)


def emit_quantized_matrix(out_f, manifest_tensors: Dict, base_name: str, matrix_f32, offset: int, group_size: int) -> int:
    packed, scales_bf16, biases_bf16, _stats = quantize_matrix_q4_affine(matrix_f32, group_size=group_size)
    offset = write_aligned_tensor(out_f, manifest_tensors, f"{base_name}.weight", packed.tobytes(), packed.shape, "U32", offset)
    offset = write_aligned_tensor(out_f, manifest_tensors, f"{base_name}.scales", scales_bf16.tobytes(), scales_bf16.shape, "BF16", offset)
    offset = write_aligned_tensor(out_f, manifest_tensors, f"{base_name}.biases", biases_bf16.tobytes(), biases_bf16.shape, "BF16", offset)
    return offset


def legacy_extract(args: argparse.Namespace) -> int:
    model_path = Path(args.model).expanduser()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        return 1

    idx = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = idx["weight_map"]

    expert_pattern = re.compile(r"\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$")
    vision_pattern = re.compile(r"^(vision_tower|model\.visual)")

    tensors_to_extract = {}
    skipped_expert = 0
    skipped_vision = 0
    for name, filename in weight_map.items():
        if vision_pattern.match(name):
            skipped_vision += 1
            continue
        if not args.include_experts and expert_pattern.search(name):
            skipped_expert += 1
            continue
        tensors_to_extract[name] = filename

    print(f"Model: {model_path}")
    print(f"Total weights in index: {len(weight_map)}")
    print(f"Skipped vision: {skipped_vision}")
    print(f"Skipped expert: {skipped_expert}")
    print(f"Extracting: {len(tensors_to_extract)} tensors")

    by_file = defaultdict(list)
    for name, filename in tensors_to_extract.items():
        by_file[filename].append(name)

    header_cache = {}
    for filename in sorted(by_file.keys()):
        header_cache[filename] = parse_safetensors_header_local(model_path / filename)

    def sanitize_name(name: str) -> str:
        return name[len("language_model.") :] if name.startswith("language_model.") else name

    all_tensors = [(sanitize_name(name), name, tensors_to_extract[name]) for name in sorted(tensors_to_extract.keys())]
    runtime_config = default_qwen35_a17b_runtime_config()
    if args.summary:
        runtime_config = load_runtime_config_from_summary(args.summary)

    bin_path = output_dir / "model_weights.bin"
    manifest = {"model": str(model_path), "num_tensors": len(all_tensors), "tensors": {}, "config": runtime_config}

    t0 = time.time()
    offset = 0
    total_bytes = 0
    with bin_path.open("wb") as out_f:
        for i, (san_name, orig_name, filename) in enumerate(all_tensors):
            header, data_start = header_cache[filename]
            meta = header.get(orig_name)
            if meta is None:
                continue
            tensor_offsets = meta["data_offsets"]
            byte_len = tensor_offsets[1] - tensor_offsets[0]
            with (model_path / filename).open("rb") as sf:
                sf.seek(data_start + tensor_offsets[0])
                data = sf.read(byte_len)
            offset = write_aligned_tensor(out_f, manifest["tensors"], san_name, data, meta["shape"], meta["dtype"], offset)
            total_bytes += byte_len
            if (i + 1) % 100 == 0 or i == len(all_tensors) - 1:
                print(f"  [{i+1}/{len(all_tensors)}] {total_bytes / 1e9:.2f} GB written")

    elapsed = time.time() - t0
    print(f"Done: {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s")
    (output_dir / "model_weights.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {output_dir / 'model_weights.json'}")
    return 0


def read_local_tensor_payload(model_root: Path, record: Dict) -> bytes:
    with (model_root / record["file"]).open("rb") as handle:
        handle.seek(record["absolute_offset"])
        return handle.read(record["nbytes"])


def qwen3_next_extract(args: argparse.Namespace) -> int:
    np = require_numpy()
    source = resolve_model_source(args.model)
    if not source.is_local:
        raise MetadataError("qwen3-next extraction requires a local model directory")

    model_root = Path(source.ref)
    config = load_config(source)
    if config.get("model_type") != "qwen3_next":
        raise MetadataError(f"Expected qwen3_next model, found {config.get('model_type')!r}")
    if args.include_experts:
        raise MetadataError("Routed experts are handled by repack_experts.py, not extract_weights.py")

    index_data = load_index(source)
    tensor_records = build_tensor_records(source, index_data)
    layer_summary = summarize_layers(tensor_records, config["num_hidden_layers"])
    runtime_config = (
        load_runtime_config_from_summary(args.summary)
        if args.summary
        else build_runtime_config_candidate(config, layer_summary)
    )
    runtime_config["model_type"] = "qwen3_next"
    runtime_config["source_model"] = str(model_root)
    runtime_config["quantization"] = {"bits": 4, "scheme": "affine", "group_size": args.group_size}
    runtime_config["high_precision_moe_layers"] = args.high_precision_moe_layers

    plan = build_qwen3_next_conversion_plan(runtime_config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / "model_weights.bin"
    manifest = {
        "model": str(model_root),
        "source_model_type": "qwen3_next",
        "num_tensors": 0,
        "tensors": {},
        "config": runtime_config,
    }

    print(f"Qwen3-Coder-Next extraction")
    print(f"  Model:  {model_root}")
    print(f"  Output: {output_dir}")
    print(f"  Plan entries: {len(plan)}")

    offset = 0
    quantized_targets = 0
    raw_targets = 0
    start = time.time()

    with bin_path.open("wb") as out_f:
        for idx, spec in enumerate(plan, start=1):
            record = tensor_records.get(spec.source_name)
            if record is None:
                raise MetadataError(f"Missing source tensor {spec.source_name}")
            payload = read_local_tensor_payload(model_root, record)

            if spec.kind == "quantize_matrix":
                matrix = np.frombuffer(payload, dtype=np.uint16).reshape(record["shape"])
                matrix = (matrix.astype(np.uint32) << 16).view(np.float32)
                if spec.row_range is not None:
                    start_row, end_row = spec.row_range
                    matrix = matrix[start_row:end_row, :]
                offset = emit_quantized_matrix(
                    out_f,
                    manifest["tensors"],
                    spec.target_name,
                    matrix.astype(np.float32, copy=False),
                    offset,
                    group_size=args.group_size,
                )
                quantized_targets += 1
            elif spec.kind == "copy_bf16":
                shape = record["shape"]
                offset = write_aligned_tensor(out_f, manifest["tensors"], spec.target_name, payload, shape, "BF16", offset)
                raw_targets += 1
            elif spec.kind == "convert_bf16_to_f32":
                arr = np.frombuffer(payload, dtype=np.uint16).reshape(record["shape"])
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
                offset = write_aligned_tensor(out_f, manifest["tensors"], spec.target_name, arr.tobytes(), arr.shape, "F32", offset)
                raw_targets += 1
            else:
                raise MetadataError(f"Unsupported conversion kind {spec.kind}")

            if idx % 64 == 0 or idx == len(plan):
                print(f"  [{idx}/{len(plan)}] tensors emitted")

    manifest["num_tensors"] = len(manifest["tensors"])
    (output_dir / "model_weights.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")
    print(f"  Quantized targets: {quantized_targets}")
    print(f"  Raw targets:       {raw_targets}")
    print(f"  Manifest tensors:  {manifest['num_tensors']}")
    print(f"  Binary size:       {bin_path.stat().st_size / (1024 ** 3):.2f} GiB")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract non-routed weights to runtime binary format")
    parser.add_argument("--mode", choices=["auto", "legacy", "qwen3-next"], default="auto")
    parser.add_argument("--model", type=str, default=LEGACY_DEFAULT_MODEL, help="Path to local model directory")
    parser.add_argument("--output", type=str, default=".", help="Output directory for model_weights.bin/.json")
    parser.add_argument("--include-experts", action="store_true", help="Legacy mode only: also extract expert weights")
    parser.add_argument("--summary", type=str, default=None, help="Optional inspection summary JSON with runtime_config_candidate")
    parser.add_argument("--group-size", type=int, default=64, help="qwen3-next q4 group size")
    parser.add_argument(
        "--high-precision-moe-layers",
        type=int,
        default=2,
        help="qwen3-next: keep the first N layers of MoE gate/shared tensors in BF16",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = detect_mode(args)
    print(f"Mode: {mode}")
    try:
        if mode == "legacy":
            return legacy_extract(args)
        if mode == "qwen3-next":
            return qwen3_next_extract(args)
        raise RuntimeError(f"Unsupported mode {mode}")
    except (MetadataError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
