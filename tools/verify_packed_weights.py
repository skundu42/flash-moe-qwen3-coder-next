#!/usr/bin/env python3
"""Verify packed expert weight artifacts for legacy or qwen3-next q4 layouts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.model_metadata import MetadataError, build_tensor_records, load_index, resolve_model_source  # noqa: E402
from tools.q4_affine import dequantize_matrix_q4_affine, require_numpy  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Verify packed expert runtime artifacts")
    parser.add_argument("--packed-dir", required=True, help="Directory containing layer_XX.bin and layout.json")
    parser.add_argument("--model", default=None, help="Optional local source model directory for slice comparison")
    parser.add_argument("--layers", default="0", help='Layer spec, e.g. "0", "0-3", or "all"')
    parser.add_argument("--experts", default="0,1", help="Comma-separated expert ids to compare")
    return parser.parse_args()


def parse_layers(spec: str, num_layers: int):
    if spec == "all":
        return list(range(num_layers))
    out = []
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
        raise ValueError(f"Invalid layer ids {invalid}")
    return out


def read_layout(packed_dir: Path):
    path = packed_dir / "layout.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sizes(packed_dir: Path, layout: dict, layers: list[int]) -> None:
    expected_layer_size = layout["expert_size"] * layout["num_experts"]
    for layer_idx in layers:
        path = packed_dir / f"layer_{layer_idx:02d}.bin"
        if not path.exists():
            raise FileNotFoundError(f"Missing packed layer file {path}")
        actual = path.stat().st_size
        if actual != expected_layer_size:
            raise RuntimeError(f"{path} size mismatch: {actual} != {expected_layer_size}")
    print(f"Size verification passed for {len(layers)} layers")


def compare_qwen3_next(packed_dir: Path, layout: dict, model_dir: Path, layers: list[int], experts: list[int]) -> None:
    np = require_numpy()
    source = resolve_model_source(str(model_dir))
    index_data = load_index(source)
    tensor_records = build_tensor_records(source, index_data)

    components = {entry["name"]: entry for entry in layout["expert_components"]}
    shapes = layout["projection_shapes"]
    group_size = layout["quantization"]["group_size"]

    def src_name(layer_idx: int, expert_idx: int, proj_name: str) -> str:
        return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.weight"

    def read_src(record: dict):
        with (model_dir / record["file"]).open("rb") as handle:
            handle.seek(record["absolute_offset"])
            data = handle.read(record["nbytes"])
        arr = np.frombuffer(data, dtype=np.uint16).reshape(record["shape"])
        return (arr.astype(np.uint32) << 16).view(np.float32)

    for layer_idx in layers:
        blob = (packed_dir / f"layer_{layer_idx:02d}.bin").read_bytes()
        for expert_idx in experts:
            start = expert_idx * layout["expert_size"]
            expert_blob = memoryview(blob)[start : start + layout["expert_size"]]
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                w_comp = components[f"{proj_name}.weight"]
                s_comp = components[f"{proj_name}.scales"]
                b_comp = components[f"{proj_name}.biases"]
                packed = np.frombuffer(expert_blob[w_comp["offset"] : w_comp["offset"] + w_comp["size"]], dtype=np.uint32)
                packed = packed.reshape(w_comp["shape"])
                scales = np.frombuffer(expert_blob[s_comp["offset"] : s_comp["offset"] + s_comp["size"]], dtype=np.uint16)
                scales = scales.reshape(s_comp["shape"])
                biases = np.frombuffer(expert_blob[b_comp["offset"] : b_comp["offset"] + b_comp["size"]], dtype=np.uint16)
                biases = biases.reshape(b_comp["shape"])
                src = read_src(tensor_records[src_name(layer_idx, expert_idx, proj_name)])
                recon = dequantize_matrix_q4_affine(
                    packed,
                    scales,
                    biases,
                    in_dim=shapes[proj_name]["in_dim"],
                    group_size=group_size,
                )
                delta = recon - src
                print(
                    f"layer={layer_idx:02d} expert={expert_idx:03d} proj={proj_name} "
                    f"max_abs={float(np.max(np.abs(delta))):.6f} "
                    f"mean_abs={float(np.mean(np.abs(delta))):.6f}"
                )


def main() -> int:
    args = parse_args()
    packed_dir = Path(args.packed_dir).expanduser().resolve()
    layout = read_layout(packed_dir)
    layers = parse_layers(args.layers, layout["num_layers"])
    experts = [int(x) for x in args.experts.split(",") if x.strip()]

    try:
        verify_sizes(packed_dir, layout, layers)
        if args.model:
            if layout.get("mode") != "qwen3-next-q4":
                raise MetadataError("--model comparison is currently only implemented for qwen3-next-q4 layouts")
            compare_qwen3_next(packed_dir, layout, Path(args.model).expanduser().resolve(), layers, experts)
    except (MetadataError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
