# Qwen3-Coder-Next Binary Layout

This document describes the additive runtime weight layout introduced for `Qwen/Qwen3-Coder-Next` expert packing.

## Scope

Current scope in this repo:
- Routed expert weights only
- Source format: upstream Hugging Face BF16 safetensors
- Runtime format: affine 4-bit grouped quantization compatible with the existing Flash-MoE expert kernels

Still pending:
- Non-expert weight conversion
- Runtime host-side adapter changes in `metal_infer/infer.m`
- Any Qwen3-Coder-Next-specific token ids / prompt formatting updates in the runtime

## Source layout

The upstream model stores routed experts as separate BF16 tensors:

```text
model.layers.<L>.mlp.experts.<E>.gate_proj.weight   [512, 2048] BF16
model.layers.<L>.mlp.experts.<E>.up_proj.weight     [512, 2048] BF16
model.layers.<L>.mlp.experts.<E>.down_proj.weight   [2048, 512] BF16
```

Key verified model dimensions:
- `num_hidden_layers = 48`
- `hidden_size = 2048`
- `num_experts = 512`
- `moe_intermediate_size = 512`
- `shared_expert_intermediate_size = 512`

## Runtime q4 expert blob

The new packer writes one file per layer:

```text
packed_experts_q4/layer_00.bin
packed_experts_q4/layer_01.bin
...
packed_experts_q4/layer_47.bin
```

Each file contains `512` experts back-to-back.

Each expert is quantized independently using:
- affine 4-bit quantization
- group size `64`
- one `scale` and `bias` pair per row-group
- weights packed as `uint32`, with `8` 4-bit values per word

## Per-expert layout

Component order is fixed:

```text
gate_proj.weight
gate_proj.scales
gate_proj.biases
up_proj.weight
up_proj.scales
up_proj.biases
down_proj.weight
down_proj.scales
down_proj.biases
```

### Shapes

`gate_proj`:
- logical shape: `[512, 2048]`
- packed weight shape: `[512, 256]` `U32`
- scales shape: `[512, 32]` `BF16`
- biases shape: `[512, 32]` `BF16`

`up_proj`:
- logical shape: `[512, 2048]`
- packed weight shape: `[512, 256]` `U32`
- scales shape: `[512, 32]` `BF16`
- biases shape: `[512, 32]` `BF16`

`down_proj`:
- logical shape: `[2048, 512]`
- packed weight shape: `[2048, 64]` `U32`
- scales shape: `[2048, 8]` `BF16`
- biases shape: `[2048, 8]` `BF16`

### Byte sizes

`gate_proj`:
- weights: `512 * 256 * 4 = 524,288`
- scales: `512 * 32 * 2 = 32,768`
- biases: `512 * 32 * 2 = 32,768`
- total: `589,824`

`up_proj`:
- total: `589,824`

`down_proj`:
- weights: `2048 * 64 * 4 = 524,288`
- scales: `2048 * 8 * 2 = 32,768`
- biases: `2048 * 8 * 2 = 32,768`
- total: `589,824`

Per expert total:

```text
1,769,472 bytes
```

Per layer total:

```text
512 * 1,769,472 = 905,969,664 bytes
```

## Quantization math

For each contiguous group of `64` source values:

```text
scale = (max(group) - min(group)) / 15
bias  = min(group)
q[i]  = clamp(round((x[i] - bias) / scale), 0, 15)
```

Degenerate groups where `max == min` are encoded with:
- `q[i] = 0`
- `bias = min`
- a safe internal scale value during quantization

Runtime dequantization is:

```text
x_hat[i] = q[i] * scale + bias
```

## Metadata file

Each packed output directory also contains `layout.json` with:
- quantization metadata
- `num_layers`
- `num_experts`
- `expert_size`
- component offsets
- logical and packed shapes

This file is the contract for:
- [repack_experts.py](/Users/sk/dev/flash-moe/repack_experts.py)
- [tools/verify_packed_weights.py](/Users/sk/dev/flash-moe/tools/verify_packed_weights.py)
- the upcoming runtime-side adapter work

## Verification path

Use:

```bash
python3 tools/verify_packed_weights.py \
  --packed-dir /path/to/model/packed_experts_q4 \
  --model /path/to/model \
  --layers 0 \
  --experts 0,1
```

This verifies:
- expected file sizes
- layer count / expert count against `layout.json`
- sampled dequantized slices against the original BF16 source tensors
