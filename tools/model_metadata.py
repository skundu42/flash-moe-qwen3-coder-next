#!/usr/bin/env python3
"""Helpers for inspecting local or Hugging Face safetensors model layouts.

This module stays dependency-light on purpose:
  - local inspection uses only the stdlib
  - remote inspection uses HTTP range requests, so large safetensors shards do
    not need to be downloaded just to read their JSON headers

The first consumer is the Qwen3-Coder-Next inspection utility, but the helpers
are kept generic because the conversion pipeline will reuse the same metadata.
"""

from __future__ import annotations

import json
import re
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


HF_RESOLVE_URL = "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
LAYER_RE = re.compile(r"^(?:language_model\.)?model\.layers\.(\d+)\.(.+)$")
PER_EXPERT_RE = re.compile(
    r"^(?:language_model\.)?model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$"
)
PACKED_EXPERT_RE = re.compile(
    r"^(?:language_model\.)?model\.layers\.(\d+)\.(?:mlp\.)?switch_mlp\.(.+)$"
)

DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "BF16": 2,
    "F16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


class MetadataError(RuntimeError):
    """Raised when model metadata is missing or inconsistent."""


@dataclass(frozen=True)
class ModelSource:
    source_type: str
    ref: str
    revision: str = "main"

    @property
    def is_local(self) -> bool:
        return self.source_type == "local"

    @property
    def display_name(self) -> str:
        if self.is_local:
            return str(Path(self.ref).resolve())
        return f"{self.ref}@{self.revision}"


def resolve_model_source(model: str, revision: str = "main") -> ModelSource:
    path = Path(model).expanduser()
    if path.exists():
        return ModelSource("local", str(path.resolve()), revision)
    if "/" in model and not model.startswith(("http://", "https://")):
        return ModelSource("huggingface_repo", model, revision)
    raise MetadataError(
        f"Model reference '{model}' is neither an existing local path nor a repo id"
    )


def _hf_url(source: ModelSource, filename: str) -> str:
    return HF_RESOLVE_URL.format(
        repo_id=source.ref,
        revision=source.revision,
        filename=filename,
    )


def _http_get(url: str, *, byte_range: Optional[Tuple[int, int]] = None) -> bytes:
    headers = {}
    if byte_range is not None:
        start, end = byte_range
        headers["Range"] = f"bytes={start}-{end}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise MetadataError(f"HTTP {exc.code} for {url}: {detail[:200]}") from exc
    except urllib.error.URLError as exc:
        raise MetadataError(f"Network error for {url}: {exc}") from exc


def read_text_asset(source: ModelSource, filename: str) -> str:
    if source.is_local:
        path = Path(source.ref) / filename
        if not path.exists():
            raise MetadataError(f"Missing local asset: {path}")
        return path.read_text(encoding="utf-8")
    return _http_get(_hf_url(source, filename)).decode("utf-8")


def read_json_asset(source: ModelSource, filename: str) -> Dict[str, Any]:
    return json.loads(read_text_asset(source, filename))


def parse_safetensors_header_local(path: Path) -> Tuple[Dict[str, Any], int]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return header, 8 + header_len


def parse_safetensors_header_remote(source: ModelSource, filename: str) -> Tuple[Dict[str, Any], int]:
    url = _hf_url(source, filename)
    prefix = _http_get(url, byte_range=(0, 7))
    if len(prefix) != 8:
        raise MetadataError(f"Short safetensors prefix for {filename}: got {len(prefix)} bytes")
    header_len = struct.unpack("<Q", prefix)[0]
    header_bytes = _http_get(url, byte_range=(8, 8 + header_len - 1))
    return json.loads(header_bytes), 8 + header_len


def parse_safetensors_header(source: ModelSource, filename: str) -> Tuple[Dict[str, Any], int]:
    if source.is_local:
        return parse_safetensors_header_local(Path(source.ref) / filename)
    return parse_safetensors_header_remote(source, filename)


def tensor_nbytes(dtype: str, shape: Iterable[int]) -> int:
    if dtype not in DTYPE_SIZES:
        raise MetadataError(f"Unsupported dtype '{dtype}'")
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * DTYPE_SIZES[dtype]


def load_index(source: ModelSource) -> Dict[str, Any]:
    return read_json_asset(source, "model.safetensors.index.json")


def load_config(source: ModelSource) -> Dict[str, Any]:
    return read_json_asset(source, "config.json")


def unique_shard_filenames(index_data: Dict[str, Any]) -> List[str]:
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise MetadataError("model.safetensors.index.json is missing a usable weight_map")
    return sorted(set(weight_map.values()))


def build_tensor_records(
    source: ModelSource,
    index_data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    weight_map = index_data["weight_map"]
    headers: Dict[str, Tuple[Dict[str, Any], int]] = {}
    for shard in unique_shard_filenames(index_data):
        headers[shard] = parse_safetensors_header(source, shard)

    records: Dict[str, Dict[str, Any]] = {}
    for tensor_name, shard in sorted(weight_map.items()):
        header, data_start = headers[shard]
        meta = header.get(tensor_name)
        if meta is None:
            raise MetadataError(f"Tensor '{tensor_name}' missing from shard header {shard}")
        offsets = meta.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise MetadataError(f"Tensor '{tensor_name}' in {shard} has malformed data_offsets")
        dtype = meta["dtype"]
        shape = meta["shape"]
        nbytes = offsets[1] - offsets[0]
        record = {
            "file": shard,
            "dtype": dtype,
            "shape": shape,
            "nbytes": nbytes,
            "data_offsets": offsets,
            "absolute_offset": data_start + offsets[0],
        }
        records[tensor_name] = annotate_tensor_record(tensor_name, record)
    return records


def annotate_tensor_record(name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(record)

    layer_match = LAYER_RE.match(name)
    if layer_match:
        layer_index = int(layer_match.group(1))
        relative_name = layer_match.group(2)
        subsystem = relative_name.split(".", 1)[0]
        record["layer_index"] = layer_index
        record["relative_name"] = relative_name
        record["subsystem"] = subsystem

    expert_match = PER_EXPERT_RE.match(name)
    if expert_match:
        record["expert_tensor_style"] = "per_expert_named"
        record["expert_layer_index"] = int(expert_match.group(1))
        record["expert_index"] = int(expert_match.group(2))
        record["expert_component"] = expert_match.group(3)
    else:
        packed_match = PACKED_EXPERT_RE.match(name)
        if packed_match:
            record["expert_tensor_style"] = "packed_tensor"
            record["expert_layer_index"] = int(packed_match.group(1))
            record["expert_component"] = packed_match.group(2)

    if ".shared_expert." in name:
        record["shared_expert"] = True
    if name.endswith(".mlp.gate.weight") or name.endswith(".mlp.gate"):
        record["router_gate"] = True
    if "shared_expert_gate" in name:
        record["shared_expert_gate"] = True
    if ".linear_attn." in name:
        record["attention_variant"] = "linear"
    if ".self_attn." in name:
        record["attention_variant"] = "full"
    return record


def summarize_layers(
    tensor_records: Dict[str, Dict[str, Any]],
    num_hidden_layers: int,
) -> Dict[str, Dict[str, Any]]:
    layers: Dict[str, Dict[str, Any]] = {}
    for layer_idx in range(num_hidden_layers):
        layers[str(layer_idx)] = {
            "tensor_count": 0,
            "subsystems": {},
            "attention_variant": None,
            "expert_tensor_style": None,
        }

    for tensor_name, record in tensor_records.items():
        layer_index = record.get("layer_index")
        if layer_index is None:
            continue
        layer = layers.setdefault(
            str(layer_index),
            {"tensor_count": 0, "subsystems": {}, "attention_variant": None, "expert_tensor_style": None},
        )
        layer["tensor_count"] += 1
        subsystem = record.get("subsystem", "other")
        layer["subsystems"].setdefault(subsystem, []).append(tensor_name)

        attention_variant = record.get("attention_variant")
        if attention_variant:
            prev = layer["attention_variant"]
            if prev and prev != attention_variant:
                raise MetadataError(
                    f"Layer {layer_index} mixes attention variants: {prev} and {attention_variant}"
                )
            layer["attention_variant"] = attention_variant

        expert_style = record.get("expert_tensor_style")
        if expert_style:
            prev_style = layer["expert_tensor_style"]
            if prev_style and prev_style != expert_style:
                raise MetadataError(
                    f"Layer {layer_index} mixes expert tensor styles: {prev_style} and {expert_style}"
                )
            layer["expert_tensor_style"] = expert_style

    return layers


def infer_expert_layout(
    tensor_records: Dict[str, Dict[str, Any]],
    expected_experts: Optional[int],
) -> Dict[str, Any]:
    per_expert_components: Dict[str, List[Dict[str, Any]]] = {}
    packed_components: Dict[str, List[Dict[str, Any]]] = {}

    for name, record in tensor_records.items():
        style = record.get("expert_tensor_style")
        if not style:
            continue
        component = record["expert_component"]
        if style == "per_expert_named":
            per_expert_components.setdefault(component, []).append({"name": name, **record})
        elif style == "packed_tensor":
            packed_components.setdefault(component, []).append({"name": name, **record})

    if per_expert_components:
        expert_indices = {
            record["expert_index"]
            for entries in per_expert_components.values()
            for record in entries
        }
        if expected_experts is not None and len(expert_indices) != expected_experts:
            raise MetadataError(
                f"Expected {expected_experts} experts, discovered {len(expert_indices)} per-expert tensors"
            )
        return {
            "style": "per_expert_named",
            "expert_component_names": sorted(per_expert_components.keys()),
            "discovered_expert_count": len(expert_indices),
            "component_examples": {
                comp: _component_example(entries[0]) for comp, entries in sorted(per_expert_components.items())
            },
        }

    if packed_components:
        inferred_expert_count = None
        for entries in packed_components.values():
            shape = entries[0]["shape"]
            if not shape:
                raise MetadataError("Packed expert tensor is missing shape metadata")
            first_dim = int(shape[0])
            if inferred_expert_count is None:
                inferred_expert_count = first_dim
            elif inferred_expert_count != first_dim:
                raise MetadataError("Packed expert tensors disagree on expert count")
        if expected_experts is not None and inferred_expert_count != expected_experts:
            raise MetadataError(
                f"Expected {expected_experts} experts, inferred {inferred_expert_count} from packed tensors"
            )
        return {
            "style": "packed_tensor",
            "expert_component_names": sorted(packed_components.keys()),
            "discovered_expert_count": inferred_expert_count,
            "component_examples": {
                comp: _component_example(entries[0]) for comp, entries in sorted(packed_components.items())
            },
        }

    raise MetadataError("No expert tensor pattern was detected in the safetensors index")


def _component_example(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": record["name"],
        "dtype": record["dtype"],
        "shape": record["shape"],
        "nbytes": record["nbytes"],
    }


def extract_relevant_config(config: Dict[str, Any]) -> Dict[str, Any]:
    interesting = {}
    exact_keys = [
        "architectures",
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "full_attention_interval",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "partial_rotary_factor",
        "rope_theta",
        "vocab_size",
        "rms_norm_eps",
        "decoder_sparse_step",
        "norm_topk_prob",
        "mlp_only_layers",
    ]
    for key in exact_keys:
        if key in config:
            interesting[key] = config[key]
    for key, value in config.items():
        if key in interesting:
            continue
        lower = key.lower()
        if "delta" in lower or "router" in lower or "attention" in lower:
            interesting[key] = value
    return interesting


def validate_qwen3_next_config(config: Dict[str, Any]) -> None:
    model_type = config.get("model_type")
    architectures = config.get("architectures") or []
    if model_type != "qwen3_next":
        raise MetadataError(f"Expected model_type='qwen3_next', found {model_type!r}")
    if "Qwen3NextForCausalLM" not in architectures:
        raise MetadataError(
            f"Expected architectures to include 'Qwen3NextForCausalLM', found {architectures!r}"
        )


def build_runtime_config_candidate(config: Dict[str, Any], layer_summary: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    candidate = {
        key: config[key]
        for key in [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "rms_norm_eps",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "full_attention_interval",
            "linear_num_value_heads",
            "linear_num_key_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "partial_rotary_factor",
            "rope_theta",
        ]
        if key in config
    }
    if "head_dim" in candidate and "partial_rotary_factor" in candidate:
        candidate["rotary_dim"] = int(candidate["head_dim"] * candidate["partial_rotary_factor"])
    if "linear_num_key_heads" in candidate and "linear_key_head_dim" in candidate:
        candidate["linear_total_key"] = (
            candidate["linear_num_key_heads"] * candidate["linear_key_head_dim"]
        )
    if "linear_num_value_heads" in candidate and "linear_value_head_dim" in candidate:
        candidate["linear_total_value"] = (
            candidate["linear_num_value_heads"] * candidate["linear_value_head_dim"]
        )

    layer_types = []
    for layer_idx in range(config["num_hidden_layers"]):
        layer = layer_summary.get(str(layer_idx), {})
        variant = layer.get("attention_variant")
        if variant == "full":
            layer_types.append("full_attention")
        elif variant == "linear":
            layer_types.append("linear_attention")
        else:
            layer_types.append("unknown")
    candidate["layer_types"] = layer_types
    return candidate


def default_qwen35_a17b_runtime_config() -> Dict[str, Any]:
    layer_types = []
    for layer_idx in range(60):
        layer_types.append("full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention")
    return {
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "rms_norm_eps": 1e-6,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "full_attention_interval": 4,
        "linear_num_value_heads": 64,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000.0,
        "rotary_dim": 64,
        "linear_total_key": 2048,
        "linear_total_value": 8192,
        "layer_types": layer_types,
    }


def load_runtime_config_from_summary(summary_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not summary_path:
        return None
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    candidate = summary.get("runtime_config_candidate")
    if not isinstance(candidate, dict):
        raise MetadataError(f"Summary file {summary_path} does not contain runtime_config_candidate")
    return candidate
