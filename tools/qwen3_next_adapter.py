#!/usr/bin/env python3
"""Qwen3-Coder-Next tensor mapping for Flash-MoE runtime conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class TensorConversionSpec:
    source_name: str
    target_name: str
    kind: str  # quantize_matrix | copy_bf16 | convert_bf16_to_f32
    row_range: Optional[Tuple[int, int]] = None
    row_indices: Optional[Tuple[int, ...]] = None


def build_qwen3_next_conversion_plan(runtime_config: dict) -> List[TensorConversionSpec]:
    plan: List[TensorConversionSpec] = []
    high_precision_moe_layers = int(runtime_config.get("high_precision_moe_layers", 0))

    def q(
        src: str,
        dst: str,
        row_range: Optional[Tuple[int, int]] = None,
        row_indices: Optional[Tuple[int, ...]] = None,
    ) -> None:
        plan.append(TensorConversionSpec(src, dst, "quantize_matrix", row_range, row_indices))

    def bf16(src: str, dst: str) -> None:
        plan.append(TensorConversionSpec(src, dst, "copy_bf16"))

    def f32(src: str, dst: str) -> None:
        plan.append(TensorConversionSpec(src, dst, "convert_bf16_to_f32"))

    hidden = runtime_config["hidden_size"]
    linear_total_key = runtime_config["linear_total_key"]
    linear_total_value = runtime_config["linear_total_value"]
    linear_num_key_heads = runtime_config["linear_num_key_heads"]
    linear_num_value_heads = runtime_config["linear_num_value_heads"]
    linear_key_head_dim = runtime_config["linear_key_head_dim"]
    linear_value_head_dim = runtime_config["linear_value_head_dim"]
    qkv_rows = linear_total_key * 2 + linear_total_value
    value_heads_per_key = linear_num_value_heads // linear_num_key_heads

    q_rows_per_group = linear_key_head_dim
    k_rows_per_group = linear_key_head_dim
    v_rows_per_group = value_heads_per_key * linear_value_head_dim
    z_rows_per_group = v_rows_per_group
    qkvz_rows_per_group = q_rows_per_group + k_rows_per_group + v_rows_per_group + z_rows_per_group

    ba_b_rows_per_group = value_heads_per_key
    ba_a_rows_per_group = value_heads_per_key
    ba_rows_per_group = ba_b_rows_per_group + ba_a_rows_per_group

    def grouped_row_indices(num_groups: int, block_size: int, block_offset: int, block_rows: int) -> List[int]:
        rows: List[int] = []
        for group_idx in range(num_groups):
            base = group_idx * block_size + block_offset
            rows.extend(range(base, base + block_rows))
        return rows

    q_rows = grouped_row_indices(linear_num_key_heads, qkvz_rows_per_group, 0, q_rows_per_group)
    k_rows = grouped_row_indices(
        linear_num_key_heads,
        qkvz_rows_per_group,
        q_rows_per_group,
        k_rows_per_group,
    )
    v_rows = grouped_row_indices(
        linear_num_key_heads,
        qkvz_rows_per_group,
        q_rows_per_group + k_rows_per_group,
        v_rows_per_group,
    )
    z_rows = grouped_row_indices(
        linear_num_key_heads,
        qkvz_rows_per_group,
        q_rows_per_group + k_rows_per_group + v_rows_per_group,
        z_rows_per_group,
    )
    ba_b_rows = grouped_row_indices(linear_num_key_heads, ba_rows_per_group, 0, ba_b_rows_per_group)
    ba_a_rows = grouped_row_indices(
        linear_num_key_heads,
        ba_rows_per_group,
        ba_b_rows_per_group,
        ba_a_rows_per_group,
    )

    q("model.embed_tokens.weight", "model.embed_tokens")
    bf16("lm_head.weight", "lm_head.weight")
    bf16("model.norm.weight", "model.norm.weight")

    for layer_idx, layer_type in enumerate(runtime_config["layer_types"]):
        prefix = f"model.layers.{layer_idx}"
        bf16(f"{prefix}.input_layernorm.weight", f"{prefix}.input_layernorm.weight")
        bf16(
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
        )

        if layer_type == "full_attention":
            q(f"{prefix}.self_attn.q_proj.weight", f"{prefix}.self_attn.q_proj")
            q(f"{prefix}.self_attn.k_proj.weight", f"{prefix}.self_attn.k_proj")
            q(f"{prefix}.self_attn.v_proj.weight", f"{prefix}.self_attn.v_proj")
            q(f"{prefix}.self_attn.o_proj.weight", f"{prefix}.self_attn.o_proj")
            bf16(f"{prefix}.self_attn.q_norm.weight", f"{prefix}.self_attn.q_norm.weight")
            bf16(f"{prefix}.self_attn.k_norm.weight", f"{prefix}.self_attn.k_norm.weight")
        else:
            q(
                f"{prefix}.linear_attn.in_proj_qkvz.weight",
                f"{prefix}.linear_attn.in_proj_qkv",
                row_indices=tuple(q_rows + k_rows + v_rows),
            )
            q(
                f"{prefix}.linear_attn.in_proj_qkvz.weight",
                f"{prefix}.linear_attn.in_proj_z",
                row_indices=tuple(z_rows),
            )
            q(
                f"{prefix}.linear_attn.in_proj_ba.weight",
                f"{prefix}.linear_attn.in_proj_b",
                row_indices=tuple(ba_b_rows),
            )
            q(
                f"{prefix}.linear_attn.in_proj_ba.weight",
                f"{prefix}.linear_attn.in_proj_a",
                row_indices=tuple(ba_a_rows),
            )
            bf16(f"{prefix}.linear_attn.conv1d.weight", f"{prefix}.linear_attn.conv1d.weight")
            f32(f"{prefix}.linear_attn.A_log", f"{prefix}.linear_attn.A_log")
            bf16(f"{prefix}.linear_attn.dt_bias", f"{prefix}.linear_attn.dt_bias")
            bf16(f"{prefix}.linear_attn.norm.weight", f"{prefix}.linear_attn.norm.weight")
            q(f"{prefix}.linear_attn.out_proj.weight", f"{prefix}.linear_attn.out_proj")

        if layer_idx < high_precision_moe_layers:
            bf16(f"{prefix}.mlp.gate.weight", f"{prefix}.mlp.gate.weight")
            bf16(f"{prefix}.mlp.shared_expert.gate_proj.weight", f"{prefix}.mlp.shared_expert.gate_proj.weight")
            bf16(f"{prefix}.mlp.shared_expert.up_proj.weight", f"{prefix}.mlp.shared_expert.up_proj.weight")
            bf16(f"{prefix}.mlp.shared_expert.down_proj.weight", f"{prefix}.mlp.shared_expert.down_proj.weight")
            bf16(f"{prefix}.mlp.shared_expert_gate.weight", f"{prefix}.mlp.shared_expert_gate.weight")
        else:
            q(f"{prefix}.mlp.gate.weight", f"{prefix}.mlp.gate")
            q(f"{prefix}.mlp.shared_expert.gate_proj.weight", f"{prefix}.mlp.shared_expert.gate_proj")
            q(f"{prefix}.mlp.shared_expert.up_proj.weight", f"{prefix}.mlp.shared_expert.up_proj")
            q(f"{prefix}.mlp.shared_expert.down_proj.weight", f"{prefix}.mlp.shared_expert.down_proj")
            q(f"{prefix}.mlp.shared_expert_gate.weight", f"{prefix}.mlp.shared_expert_gate")

    return plan
