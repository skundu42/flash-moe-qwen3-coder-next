#!/usr/bin/env python3
"""Pack routed expert weights into contiguous per-layer runtime blobs.

Modes:
  - legacy: original Qwen3.5-397B-A17B MLX repack path using expert_index.json
  - qwen3-next: local BF16 Qwen3-Coder-Next -> affine q4 runtime expert blobs

The new qwen3-next path is additive and writes to `packed_experts_q4/` by default
so the original repo's `packed_experts/` path is left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.model_metadata import (  # noqa: E402
    MetadataError,
    build_tensor_records,
    load_config,
    load_index,
    load_runtime_config_from_summary,
    resolve_model_source,
)
from tools.q4_affine import (  # noqa: E402
    bf16_to_f32,
    build_moe_expert_q4_layout,
    dequantize_matrix_q4_affine,
    quantize_matrix_q4_affine,
    require_numpy,
)


LEGACY_COMPONENTS = [
    {"name": "gate_proj.weight", "offset": 0, "size": 2097152, "dtype": "U32", "shape": [1024, 512]},
    {"name": "gate_proj.scales", "offset": 2097152, "size": 131072, "dtype": "BF16", "shape": [1024, 64]},
    {"name": "gate_proj.biases", "offset": 2228224, "size": 131072, "dtype": "BF16", "shape": [1024, 64]},
    {"name": "up_proj.weight", "offset": 2359296, "size": 2097152, "dtype": "U32", "shape": [1024, 512]},
    {"name": "up_proj.scales", "offset": 4456448, "size": 131072, "dtype": "BF16", "shape": [1024, 64]},
    {"name": "up_proj.biases", "offset": 4587520, "size": 131072, "dtype": "BF16", "shape": [1024, 64]},
    {"name": "down_proj.weight", "offset": 4718592, "size": 2097152, "dtype": "U32", "shape": [4096, 128]},
    {"name": "down_proj.scales", "offset": 6815744, "size": 131072, "dtype": "BF16", "shape": [4096, 16]},
    {"name": "down_proj.biases", "offset": 6946816, "size": 131072, "dtype": "BF16", "shape": [4096, 16]},
]
LEGACY_EXPERT_SIZE = 7077888
LEGACY_NUM_EXPERTS = 512
LEGACY_NUM_LAYERS = 60
LEGACY_LAYER_SIZE = LEGACY_EXPERT_SIZE * LEGACY_NUM_EXPERTS


def parse_layers(spec: str | None, num_layers: int) -> List[int]:
    if spec is None or spec == "all":
        return list(range(num_layers))
    layers: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            layers.extend(range(int(start_s), int(end_s) + 1))
        else:
            layers.append(int(part))
    out = sorted(set(layers))
    invalid = [layer for layer in out if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(f"Invalid layer indices {invalid} for num_layers={num_layers}")
    return out


def write_layout_json(output_dir: Path, layout: Dict) -> None:
    path = output_dir / "layout.json"
    path.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def detect_mode(args: argparse.Namespace) -> str:
    if args.mode != "auto":
        return args.mode
    if args.summary:
        summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        source_model = summary.get("config_subset", {}).get("model_type")
        if source_model == "qwen3_next":
            return "qwen3-next"
    if args.model:
        cfg_path = Path(args.model).expanduser() / "config.json"
        if cfg_path.exists():
            config = json.loads(cfg_path.read_text(encoding="utf-8"))
            if config.get("model_type") == "qwen3_next":
                return "qwen3-next"
    return "legacy"


def run_legacy(args: argparse.Namespace) -> int:
    def load_index_file(index_path: Path):
        with index_path.open() as handle:
            idx = json.load(handle)
        return idx["expert_reads"], idx["model_path"]

    def verify_component_sizes(expert_reads: Dict[str, Dict]) -> None:
        expected = {c["name"]: c["size"] for c in LEGACY_COMPONENTS}
        for layer_key, comps in expert_reads.items():
            for comp_name, info in comps.items():
                if comp_name not in expected:
                    raise RuntimeError(f"Unknown component {comp_name} in layer {layer_key}")
                if info["expert_size"] != expected[comp_name]:
                    raise RuntimeError(
                        f"Component mismatch: layer {layer_key} {comp_name} "
                        f"{info['expert_size']} != {expected[comp_name]}"
                    )

    index_path = Path(args.index).expanduser()
    expert_reads, model_path = load_index_file(index_path)
    verify_component_sizes(expert_reads)

    layers = parse_layers(args.layers, LEGACY_NUM_LAYERS)
    output_dir = Path(args.output) if args.output else Path(model_path) / "packed_experts"
    output_dir.mkdir(parents=True, exist_ok=True)

    needed_files = {
        info["file"]
        for layer_idx in layers
        for info in expert_reads[str(layer_idx)].values()
    }
    fds = {
        fname: os.open(os.path.join(model_path, fname), os.O_RDONLY)
        for fname in sorted(needed_files)
    }

    layout = {
        "version": 1,
        "mode": "legacy",
        "source_model": model_path,
        "num_layers": LEGACY_NUM_LAYERS,
        "num_experts": LEGACY_NUM_EXPERTS,
        "expert_size": LEGACY_EXPERT_SIZE,
        "components": LEGACY_COMPONENTS,
    }
    write_layout_json(output_dir, layout)

    def verify_layer(layer_idx: int) -> bool:
        out_path = output_dir / f"layer_{layer_idx:02d}.bin"
        if not out_path.exists():
            return False
        fd_packed = os.open(out_path, os.O_RDONLY)
        try:
            for expert_idx in [0, 1, 255, 511]:
                for comp in LEGACY_COMPONENTS:
                    info = expert_reads[str(layer_idx)][comp["name"]]
                    original = os.pread(
                        fds[info["file"]], comp["size"], info["abs_offset"] + expert_idx * info["expert_stride"]
                    )
                    packed = os.pread(
                        fd_packed, comp["size"], expert_idx * LEGACY_EXPERT_SIZE + comp["offset"]
                    )
                    if original != packed:
                        print(
                            f"  MISMATCH: layer {layer_idx}, expert {expert_idx}, {comp['name']}"
                        )
                        return False
            return True
        finally:
            os.close(fd_packed)

    total_written = 0
    start = time.monotonic()
    try:
        for layer_idx in layers:
            out_path = output_dir / f"layer_{layer_idx:02d}.bin"
            if args.dry_run:
                print(f"Layer {layer_idx:02d}: dry run ok -> {out_path}")
                continue

            fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            os.ftruncate(fd_out, LEGACY_LAYER_SIZE)
            try:
                read_plan = []
                layer_info = expert_reads[str(layer_idx)]
                for expert_idx in range(LEGACY_NUM_EXPERTS):
                    for comp in LEGACY_COMPONENTS:
                        info = layer_info[comp["name"]]
                        read_plan.append(
                            (
                                fds[info["file"]],
                                info["abs_offset"] + expert_idx * info["expert_stride"],
                                expert_idx * LEGACY_EXPERT_SIZE + comp["offset"],
                                comp["size"],
                            )
                        )
                read_plan.sort(key=lambda x: (x[0], x[1]))
                for src_fd, src_offset, dst_offset, size in read_plan:
                    data = os.pread(src_fd, size, src_offset)
                    if len(data) != size:
                        raise IOError(f"Short read at {src_offset}: {len(data)} != {size}")
                    os.pwrite(fd_out, data, dst_offset)
                    total_written += size
            finally:
                os.close(fd_out)

            ok = verify_layer(layer_idx)
            print(f"Layer {layer_idx:02d}: {'verified' if ok else 'FAILED'}")
            if not ok:
                return 1
    finally:
        for fd in fds.values():
            os.close(fd)

    elapsed = time.monotonic() - start
    if not args.dry_run:
        print(
            f"Legacy repack complete: {total_written / (1024 ** 3):.1f} GiB in {elapsed:.1f}s "
            f"({total_written / max(elapsed, 1e-9) / (1024 ** 3):.2f} GiB/s)"
        )
    return 0


def load_qwen3_next_context(args: argparse.Namespace) -> Tuple[Path, Dict, Dict, Dict, Dict]:
    model_arg = args.model
    if not model_arg:
        raise MetadataError("--model LOCAL_DIR is required for qwen3-next mode")
    source = resolve_model_source(model_arg)
    if not source.is_local:
        raise MetadataError("qwen3-next packing requires a local model directory")

    config = load_config(source)
    if config.get("model_type") != "qwen3_next":
        raise MetadataError(f"Expected local qwen3_next model, found {config.get('model_type')!r}")

    index_data = load_index(source)
    tensor_records = build_tensor_records(source, index_data)
    runtime_summary = load_runtime_config_from_summary(args.summary) if args.summary else None
    if runtime_summary and runtime_summary.get("num_hidden_layers") != config.get("num_hidden_layers"):
        raise MetadataError("Summary/runtime layer count does not match local model config")

    return Path(source.ref), config, index_data, tensor_records, runtime_summary or {}


def qwen3_next_tensor_name(layer_idx: int, expert_idx: int, proj_name: str) -> str:
    return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.weight"


def read_tensor_bf16_as_f32(model_root: Path, record: Dict):
    np = require_numpy()
    shape = tuple(record["shape"])
    if record["dtype"] != "BF16":
        raise MetadataError(f"Expected BF16 tensor, found {record['dtype']}")
    with (model_root / record["file"]).open("rb") as handle:
        handle.seek(record["absolute_offset"])
        data = handle.read(record["nbytes"])
    arr = np.frombuffer(data, dtype=np.uint16).reshape(shape)
    return bf16_to_f32(arr)


def quantize_qwen3_next_expert(model_root: Path, tensor_records: Dict, layer_idx: int, expert_idx: int, group_size: int):
    np = require_numpy()
    parts = []
    per_projection_stats = {}
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        tensor_name = qwen3_next_tensor_name(layer_idx, expert_idx, proj_name)
        record = tensor_records.get(tensor_name)
        if record is None:
            raise MetadataError(f"Missing tensor {tensor_name}")
        source_f32 = read_tensor_bf16_as_f32(model_root, record)
        packed, scales_bf16, biases_bf16, stats = quantize_matrix_q4_affine(source_f32, group_size=group_size)
        parts.extend([packed.tobytes(), scales_bf16.tobytes(), biases_bf16.tobytes()])
        per_projection_stats[proj_name] = stats
    blob = b"".join(parts)
    return blob, per_projection_stats


def compare_qwen3_next_expert(
    expert_blob: bytes,
    tensor_records: Dict,
    model_root: Path,
    layer_idx: int,
    expert_idx: int,
    layout: Dict,
    group_size: int,
) -> Dict:
    np = require_numpy()
    metrics = {}
    components = {entry["name"]: entry for entry in layout["expert_components"]}
    projection_shapes = layout["projection_shapes"]

    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        w_comp = components[f"{proj_name}.weight"]
        s_comp = components[f"{proj_name}.scales"]
        b_comp = components[f"{proj_name}.biases"]
        out_dim = projection_shapes[proj_name]["out_dim"]
        in_dim = projection_shapes[proj_name]["in_dim"]

        packed = np.frombuffer(
            expert_blob[w_comp["offset"] : w_comp["offset"] + w_comp["size"]],
            dtype=np.uint32,
        ).reshape(w_comp["shape"])
        scales = np.frombuffer(
            expert_blob[s_comp["offset"] : s_comp["offset"] + s_comp["size"]],
            dtype=np.uint16,
        ).reshape(s_comp["shape"])
        biases = np.frombuffer(
            expert_blob[b_comp["offset"] : b_comp["offset"] + b_comp["size"]],
            dtype=np.uint16,
        ).reshape(b_comp["shape"])

        recon = dequantize_matrix_q4_affine(packed, scales, biases, in_dim=in_dim, group_size=group_size)
        src = read_tensor_bf16_as_f32(
            model_root, tensor_records[qwen3_next_tensor_name(layer_idx, expert_idx, proj_name)]
        )
        delta = recon - src
        metrics[proj_name] = {
            "max_abs_error": float(np.max(np.abs(delta))),
            "mean_abs_error": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
        }

    return metrics


def run_qwen3_next(args: argparse.Namespace) -> int:
    np = require_numpy()
    model_root, config, _index, tensor_records, runtime_summary = load_qwen3_next_context(args)
    runtime_cfg = runtime_summary or {
        "hidden_size": config["hidden_size"],
        "moe_intermediate_size": config["moe_intermediate_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_experts": config["num_experts"],
    }
    group_size = args.group_size
    if runtime_cfg["hidden_size"] % group_size != 0 or runtime_cfg["moe_intermediate_size"] % group_size != 0:
        raise MetadataError(
            f"group_size={group_size} must divide hidden={runtime_cfg['hidden_size']} and "
            f"intermediate={runtime_cfg['moe_intermediate_size']}"
        )

    layout = build_moe_expert_q4_layout(
        hidden_size=runtime_cfg["hidden_size"],
        intermediate_size=runtime_cfg["moe_intermediate_size"],
        group_size=group_size,
    )
    layout.update(
        {
            "version": 2,
            "mode": "qwen3-next-q4",
            "source_model": str(model_root),
            "source_model_type": config["model_type"],
            "num_layers": runtime_cfg["num_hidden_layers"],
            "num_experts": runtime_cfg["num_experts"],
        }
    )

    layers = parse_layers(args.layers, runtime_cfg["num_hidden_layers"])
    output_dir = Path(args.output) if args.output else model_root / "packed_experts_q4"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_layout_json(output_dir, layout)

    expert_size = layout["expert_size"]
    num_experts = runtime_cfg["num_experts"]
    layer_size = expert_size * num_experts
    sample_experts = [int(x) for x in args.sample_experts.split(",") if x.strip()]

    print(f"Qwen3-Next expert pack")
    print(f"  Model:         {model_root}")
    print(f"  Layers:        {len(layers)} / {runtime_cfg['num_hidden_layers']}")
    print(f"  Experts/layer: {num_experts}")
    print(f"  Expert size:   {expert_size:,} bytes")
    print(f"  Layer size:    {layer_size / (1024 ** 3):.2f} GiB")
    print(f"  Output:        {output_dir}")

    total_bytes = 0
    total_stats = {name: [] for name in ["gate_proj", "up_proj", "down_proj"]}
    start = time.monotonic()

    for layer_idx in layers:
        out_path = output_dir / f"layer_{layer_idx:02d}.bin"
        if args.dry_run:
            print(f"Layer {layer_idx:02d}: dry run ok -> {out_path}")
            continue

        fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd_out, layer_size)
        layer_start = time.monotonic()
        try:
            for expert_idx in range(num_experts):
                blob, stats = quantize_qwen3_next_expert(
                    model_root=model_root,
                    tensor_records=tensor_records,
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    group_size=group_size,
                )
                if len(blob) != expert_size:
                    raise RuntimeError(f"Expert blob size mismatch {len(blob)} != {expert_size}")
                os.pwrite(fd_out, blob, expert_idx * expert_size)
                total_bytes += len(blob)
                for proj_name, proj_stats in stats.items():
                    total_stats[proj_name].append(proj_stats["rmse"])

                if expert_idx in sample_experts:
                    metrics = compare_qwen3_next_expert(
                        blob, tensor_records, model_root, layer_idx, expert_idx, layout, group_size
                    )
                    print(f"  Verify layer {layer_idx:02d} expert {expert_idx:03d}: {json.dumps(metrics)}")

                if (expert_idx + 1) % 32 == 0 or expert_idx + 1 == num_experts:
                    elapsed = time.monotonic() - layer_start
                    rate = (expert_idx + 1) / max(elapsed, 1e-9)
                    print(
                        f"  Layer {layer_idx:02d} [{expert_idx + 1:3d}/{num_experts}] "
                        f"{elapsed:.1f}s elapsed, {rate:.2f} experts/s"
                    )
        finally:
            os.close(fd_out)

        layer_elapsed = time.monotonic() - layer_start
        print(
            f"Layer {layer_idx:02d}: wrote {layer_size / (1024 ** 3):.2f} GiB in "
            f"{layer_elapsed:.1f}s"
        )

    total_elapsed = time.monotonic() - start
    if not args.dry_run:
        avg_rmse = {
            name: (sum(vals) / len(vals) if vals else 0.0)
            for name, vals in total_stats.items()
        }
        print(
            f"Qwen3-next repack complete: {total_bytes / (1024 ** 3):.2f} GiB in "
            f"{total_elapsed:.1f}s ({total_bytes / max(total_elapsed, 1e-9) / (1024 ** 3):.2f} GiB/s)"
        )
        print(f"Average projection RMSE: {json.dumps(avg_rmse)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack expert weights into runtime layer blobs")
    parser.add_argument("--mode", choices=["auto", "legacy", "qwen3-next"], default="auto")
    parser.add_argument(
        "--index",
        default=str(REPO_ROOT / "expert_index.json"),
        help="Legacy mode: path to expert_index.json",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="qwen3-next mode: local model directory",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional model summary JSON from tools/inspect_qwen3_coder_next.py",
    )
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--layers", default=None, help='Layer spec: "all", "0-4", "0,5,10"')
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size for qwen3-next mode")
    parser.add_argument(
        "--sample-experts",
        default="0,1",
        help="Comma-separated expert ids to verify after packing in qwen3-next mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = detect_mode(args)
    print(f"Mode: {mode}")
    try:
        if mode == "legacy":
            return run_legacy(args)
        if mode == "qwen3-next":
            return run_qwen3_next(args)
        raise RuntimeError(f"Unsupported mode {mode}")
    except (MetadataError, RuntimeError, ValueError, IOError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
