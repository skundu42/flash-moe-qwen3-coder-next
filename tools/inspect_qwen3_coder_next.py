#!/usr/bin/env python3
"""Inspect Qwen/Qwen3-Coder-Next config and safetensors layout.

The script works against:
  - a local model directory containing config.json + model.safetensors.index.json
  - a Hugging Face repo id such as Qwen/Qwen3-Coder-Next

For remote repos it uses HTTP range requests to read only safetensors headers, so
the 159 GB model does not need to be downloaded just to inspect tensor keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.model_metadata import (
    MetadataError,
    build_runtime_config_candidate,
    build_tensor_records,
    extract_relevant_config,
    infer_expert_layout,
    load_config,
    load_index,
    resolve_model_source,
    summarize_layers,
    unique_shard_filenames,
    validate_qwen3_next_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Qwen3-Coder-Next config and safetensors layout"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Coder-Next",
        help="Local model directory or Hugging Face repo id",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision when --model is a Hugging Face repo id",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory for JSON inspection outputs",
    )
    parser.add_argument(
        "--print-layer-limit",
        type=int,
        default=8,
        help="How many layers to print in the console summary",
    )
    return parser.parse_args()


def build_tensor_map_payload(
    tensor_records: Dict[str, Dict],
    layer_summary: Dict[str, Dict],
) -> Dict[str, object]:
    return {
        "tensor_count": len(tensor_records),
        "layer_summary": layer_summary,
        "tensors": tensor_records,
    }


def build_summary_payload(
    source_display: str,
    source_type: str,
    revision: str,
    config: Dict,
    index_data: Dict,
    tensor_records: Dict[str, Dict],
    layer_summary: Dict[str, Dict],
    expert_layout: Dict,
) -> Dict[str, object]:
    attention_counts = Counter(
        layer["attention_variant"]
        for layer in layer_summary.values()
        if layer.get("attention_variant")
    )
    runtime_candidate = build_runtime_config_candidate(config, layer_summary)
    shard_files = unique_shard_filenames(index_data)
    return {
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "type": source_type,
            "model": source_display,
            "revision": revision,
        },
        "config_subset": extract_relevant_config(config),
        "runtime_config_candidate": runtime_candidate,
        "tensor_count": len(tensor_records),
        "safetensors_shard_count": len(shard_files),
        "safetensors_shards": shard_files,
        "attention_layer_counts": dict(attention_counts),
        "expert_layout": expert_layout,
        "shared_expert_components": sorted(
            {
                record["relative_name"]
                for record in tensor_records.values()
                if record.get("shared_expert")
            }
        ),
        "shared_expert_gate_tensors": sorted(
            name for name, record in tensor_records.items() if record.get("shared_expert_gate")
        ),
        "router_gate_tensors": sorted(
            name for name, record in tensor_records.items() if record.get("router_gate")
        ),
        "layers": layer_summary,
    }


def print_console_summary(
    config: Dict,
    layer_summary: Dict[str, Dict],
    expert_layout: Dict,
    shard_count: int,
    tensor_count: int,
    limit: int,
) -> None:
    print("Qwen3-Coder-Next summary")
    print(f"  Layers:              {config['num_hidden_layers']}")
    print(f"  Hidden size:         {config['hidden_size']}")
    print(f"  Attention heads:     {config['num_attention_heads']}")
    print(f"  KV heads:            {config['num_key_value_heads']}")
    print(f"  Experts:             {config['num_experts']}")
    print(f"  Active experts/tok:  {config['num_experts_per_tok']}")
    print(f"  MoE intermediate:    {config['moe_intermediate_size']}")
    print(f"  Shared intermediate: {config['shared_expert_intermediate_size']}")
    print(f"  Full-attn interval:  {config['full_attention_interval']}")
    print(f"  Shards:              {shard_count}")
    print(f"  Tensors:             {tensor_count}")
    print(f"  Expert tensor style: {expert_layout['style']}")
    print("  Expert components:")
    for component in expert_layout["expert_component_names"]:
        print(f"    - {component}")

    print(f"\nLayer sample (first {min(limit, len(layer_summary))})")
    for layer_idx in range(min(limit, len(layer_summary))):
        layer = layer_summary[str(layer_idx)]
        print(
            f"  Layer {layer_idx:02d}: attention={layer.get('attention_variant') or 'n/a'} "
            f"subsystems={sorted(layer['subsystems'].keys())}"
        )


def main() -> int:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        source = resolve_model_source(args.model, revision=args.revision)
        config = load_config(source)
        validate_qwen3_next_config(config)
        index_data = load_index(source)
        tensor_records = build_tensor_records(source, index_data)
        layer_summary = summarize_layers(tensor_records, config["num_hidden_layers"])
        expert_layout = infer_expert_layout(
            tensor_records,
            expected_experts=config.get("num_experts"),
        )
    except MetadataError as exc:
        print(f"ERROR: {exc}")
        return 1

    discovered_layers = {
        int(layer_idx) for layer_idx, layer in layer_summary.items() if layer["tensor_count"] > 0
    }
    if len(discovered_layers) != config["num_hidden_layers"]:
        print(
            "ERROR: layer count mismatch: "
            f"config says {config['num_hidden_layers']}, discovered {len(discovered_layers)}"
        )
        return 1

    if any(layer_summary[str(layer_idx)]["attention_variant"] is None for layer_idx in discovered_layers):
        print("ERROR: at least one transformer layer has no detectable attention variant")
        return 1

    tensor_map_path = artifacts_dir / "qwen3_coder_next_tensor_map.json"
    summary_path = artifacts_dir / "qwen3_coder_next_model_summary.json"

    tensor_map_payload = build_tensor_map_payload(tensor_records, layer_summary)
    summary_payload = build_summary_payload(
        source_display=source.display_name,
        source_type=source.source_type,
        revision=source.revision,
        config=config,
        index_data=index_data,
        tensor_records=tensor_records,
        layer_summary=layer_summary,
        expert_layout=expert_layout,
    )

    tensor_map_path.write_text(json.dumps(tensor_map_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print_console_summary(
        config=config,
        layer_summary=layer_summary,
        expert_layout=expert_layout,
        shard_count=len(summary_payload["safetensors_shards"]),
        tensor_count=len(tensor_records),
        limit=args.print_layer_limit,
    )
    print(f"\nWrote {tensor_map_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
