#!/usr/bin/env python3
"""Shared helpers for affine 4-bit quantization and runtime layout metadata."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


def require_numpy():
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "numpy is required for quantization/verification. Install it with "
            "`python3 -m pip install numpy`."
        ) from exc
    return np


def bf16_to_f32(bf16_u16):
    np = require_numpy()
    return (bf16_u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    np = require_numpy()
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


def pack_4bit(values):
    """Pack uint8 values in [0, 15] into uint32 words with 8 nibbles per word."""
    np = require_numpy()
    if values.shape[-1] % 8 != 0:
        raise ValueError(f"Last dimension {values.shape[-1]} is not divisible by 8")
    flat = values.reshape(-1, values.shape[-1])
    packed_cols = values.shape[-1] // 8
    out = np.zeros((flat.shape[0], packed_cols), dtype=np.uint32)
    for i in range(8):
        out |= flat[:, i::8].astype(np.uint32) << (i * 4)
    return out.reshape(values.shape[:-1] + (packed_cols,))


def unpack_4bit(packed):
    """Unpack uint32 words into uint8 values in [0, 15]."""
    np = require_numpy()
    flat = packed.reshape(-1, packed.shape[-1])
    out = np.empty((flat.shape[0], flat.shape[1] * 8), dtype=np.uint8)
    for i in range(8):
        out[:, i::8] = ((flat >> (i * 4)) & 0xF).astype(np.uint8)
    return out.reshape(packed.shape[:-1] + (packed.shape[-1] * 8,))


def quantize_matrix_q4_affine(weights_f32, group_size: int = 64):
    """Quantize [out_dim, in_dim] float32 matrix to affine q4 with bf16 scales/biases."""
    np = require_numpy()
    if weights_f32.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {weights_f32.shape}")
    out_dim, in_dim = weights_f32.shape
    if in_dim % group_size != 0:
        raise ValueError(f"in_dim={in_dim} must be divisible by group_size={group_size}")

    num_groups = in_dim // group_size
    grouped = weights_f32.reshape(out_dim, num_groups, group_size).astype(np.float32, copy=False)
    minv = grouped.min(axis=2, keepdims=True)
    maxv = grouped.max(axis=2, keepdims=True)
    scales = (maxv - minv) / 15.0
    biases = minv
    degenerate = scales == 0.0
    safe_scales = np.where(degenerate, 1.0, scales)

    q = np.rint((grouped - biases) / safe_scales)
    q = np.clip(q, 0, 15).astype(np.uint8)
    q = np.where(degenerate, 0, q).astype(np.uint8)

    reconstructed = q.astype(np.float32) * safe_scales + biases
    reconstructed = np.where(degenerate, minv, reconstructed)
    error = reconstructed - grouped
    stats = {
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_abs_error": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(float(np.mean(error * error)))),
    }

    packed = pack_4bit(q.reshape(out_dim, in_dim))
    scales_bf16 = f32_to_bf16(scales.squeeze(axis=2).astype(np.float32))
    biases_bf16 = f32_to_bf16(biases.squeeze(axis=2).astype(np.float32))
    return packed, scales_bf16, biases_bf16, stats


def dequantize_matrix_q4_affine(packed, scales_bf16, biases_bf16, in_dim: int, group_size: int = 64):
    """Dequantize q4 affine matrix back to float32."""
    np = require_numpy()
    values = unpack_4bit(packed)
    out_dim = values.shape[0]
    num_groups = in_dim // group_size
    grouped = values.reshape(out_dim, num_groups, group_size).astype(np.float32)
    scales = bf16_to_f32(scales_bf16)[:, :, np.newaxis]
    biases = bf16_to_f32(biases_bf16)[:, :, np.newaxis]
    return (grouped * scales + biases).reshape(out_dim, in_dim)


def matrix_component_layout(name: str, out_dim: int, in_dim: int, group_size: int, start_offset: int) -> Tuple[List[Dict], int]:
    if in_dim % group_size != 0:
        raise ValueError(f"in_dim={in_dim} must be divisible by group_size={group_size}")
    num_groups = in_dim // group_size
    packed_cols = in_dim // 8
    weight_size = out_dim * packed_cols * 4
    sb_size = out_dim * num_groups * 2

    components = [
        {
            "name": f"{name}.weight",
            "offset": start_offset,
            "size": weight_size,
            "dtype": "U32",
            "shape": [out_dim, packed_cols],
            "logical_shape": [out_dim, in_dim],
        },
        {
            "name": f"{name}.scales",
            "offset": start_offset + weight_size,
            "size": sb_size,
            "dtype": "BF16",
            "shape": [out_dim, num_groups],
            "logical_shape": [out_dim, num_groups],
        },
        {
            "name": f"{name}.biases",
            "offset": start_offset + weight_size + sb_size,
            "size": sb_size,
            "dtype": "BF16",
            "shape": [out_dim, num_groups],
            "logical_shape": [out_dim, num_groups],
        },
    ]
    return components, start_offset + weight_size + sb_size + sb_size


def build_moe_expert_q4_layout(hidden_size: int, intermediate_size: int, group_size: int = 64) -> Dict:
    offset = 0
    components: List[Dict] = []
    for name, out_dim, in_dim in [
        ("gate_proj", intermediate_size, hidden_size),
        ("up_proj", intermediate_size, hidden_size),
        ("down_proj", hidden_size, intermediate_size),
    ]:
        subcomponents, offset = matrix_component_layout(name, out_dim, in_dim, group_size, offset)
        components.extend(subcomponents)

    return {
        "quantization": {"bits": 4, "scheme": "affine", "group_size": group_size},
        "expert_components": components,
        "expert_size": offset,
        "projection_shapes": {
            "gate_proj": {"out_dim": intermediate_size, "in_dim": hidden_size},
            "up_proj": {"out_dim": intermediate_size, "in_dim": hidden_size},
            "down_proj": {"out_dim": hidden_size, "in_dim": intermediate_size},
        },
    }
