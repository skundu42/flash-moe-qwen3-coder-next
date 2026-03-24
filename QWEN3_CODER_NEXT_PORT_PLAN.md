# Qwen3-Coder-Next Port Plan

## Goal
Port `flash-moe` from the current Qwen3.5-397B-A17B-specific path to an additive, model-config-driven path that can ingest `Qwen/Qwen3-Coder-Next`, convert weights into the runtime format, and run Apple-Silicon-first inference without breaking the existing model flow.

## Phase 0 Audit

### Verified upstream facts
- `Qwen/Qwen3-Coder-Next` `config.json` reports:
  - `model_type = "qwen3_next"`
  - `architectures = ["Qwen3NextForCausalLM"]`
  - `hidden_size = 2048`
  - `num_hidden_layers = 48`
  - `num_attention_heads = 16`
  - `num_key_value_heads = 2`
  - `head_dim = 256`
  - `num_experts = 512`
  - `num_experts_per_tok = 10`
  - `moe_intermediate_size = 512`
  - `shared_expert_intermediate_size = 512`
  - `full_attention_interval = 4`
  - `linear_num_key_heads = 16`
  - `linear_key_head_dim = 128`
  - `linear_num_value_heads = 32`
  - `linear_value_head_dim = 128`
  - `linear_conv_kernel_dim = 4`
  - `partial_rotary_factor = 0.25`
  - `rope_theta = 5000000`
  - `vocab_size = 151936`
- The Hugging Face repo currently exposes `40` safetensors shards and `model.safetensors.index.json`.
- Remote header inspection succeeded against the real repo and found:
  - `74391` tensors
  - `36` linear-attention layers and `12` full-attention layers
  - routed expert tensors are stored as per-expert BF16 weights:
    - `model.layers.<L>.mlp.experts.<E>.gate_proj.weight`
    - `model.layers.<L>.mlp.experts.<E>.up_proj.weight`
    - `model.layers.<L>.mlp.experts.<E>.down_proj.weight`
  - linear-attention projections are fused differently from the current runtime assumptions:
    - `model.layers.<L>.linear_attn.in_proj_qkvz.weight`
    - `model.layers.<L>.linear_attn.in_proj_ba.weight`
  - shared MoE tensors are:
    - `model.layers.<L>.mlp.gate.weight`
    - `model.layers.<L>.mlp.shared_expert.{gate_proj,up_proj,down_proj}.weight`
    - `model.layers.<L>.mlp.shared_expert_gate.weight`
- Important consequence: the source model repo is BF16 safetensors, not the already-quantized MLX 4-bit format used by the current repo. A BF16 -> runtime quantization/packing step is required.

Sources:
- [config.json](https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/faa3cdf286453e2a5f837611dfde6268b7b3070b/config.json)
- [repo file tree](https://huggingface.co/Qwen/Qwen3-Coder-Next/tree/main)

### Repo files with model-specific assumptions
- `metal_infer/infer.m`
  - Hardcodes model dimensions, layer count, expert count, top-k defaults, attention schedule, rotary dims, delta-net dims, EOS/think token ids, and expert binary offsets.
  - Hardcodes tensor name strings for every layer subsystem.
  - Assumes the current packed-expert format and current linear-attention/full-attention layer pattern.
- `metal_infer/main.m`
  - Hardcodes expert layout offsets, dimensions, layer count, expert count, and benchmark routing assumptions.
- `metal_infer/shaders.metal`
  - Assumes max input size `4096`, fixed expert intermediate sizes in comments/launch expectations, and fixed delta-net dimensions (`64 x 128 x 128` state layout).
- `metal_infer/extract_weights.py`
  - Assumes pre-quantized MLX tensor names and emits a hardcoded 397B runtime config.
- `repack_experts.py`
  - Assumes one specific packed-expert binary layout:
    - 60 layers
    - 512 experts/layer
    - fixed component set and byte sizes
    - expert tensors stored aggregated per layer in safetensors
- `metal_infer/export_tokenizer.py`
  - Hardcodes the old model path and assumes the tokenizer export source already exists locally.
- `metal_infer/chat.m`
  - Mostly model-agnostic, but the banner is model-specific.
- `expert_index.json`
  - Encodes the old model path and old packed expert offsets only.

### Hard assumptions that block Qwen3-Coder-Next
- The runtime currently consumes already-quantized MLX tensors (`weight/scales/biases`) for non-experts and experts. Qwen3-Coder-Next upstream is BF16.
- Layer count is fixed at `60`; Qwen3-Coder-Next has `48`.
- Hidden size is fixed at `4096`; Qwen3-Coder-Next uses `2048`.
- Expert intermediate size is fixed at `1024`; Qwen3-Coder-Next uses `512`.
- Shared expert intermediate size is fixed at `1024`; Qwen3-Coder-Next uses `512`.
- Linear attention value head count is fixed at `64`; Qwen3-Coder-Next uses `32`.
- Vocabulary size and special token ids are model-specific.
- The runtime tensor loader is hardwired to old tensor names.
- Expert packing currently assumes an aggregated expert tensor layout from the MLX quantized model; upstream Qwen3-Coder-Next likely needs a different extraction path from BF16 tensors.

### Exact files to change first
- [tools/inspect_qwen3_coder_next.py](tools/inspect_qwen3_coder_next.py)
- [tools/model_metadata.py](tools/model_metadata.py)
- [metal_infer/extract_weights.py](metal_infer/extract_weights.py)
- [repack_experts.py](repack_experts.py)
- [metal_infer/export_tokenizer.py](metal_infer/export_tokenizer.py)
- [metal_infer/infer.m](metal_infer/infer.m)
- [metal_infer/main.m](metal_infer/main.m)
- [metal_infer/shaders.metal](metal_infer/shaders.metal)

## Migration Log

### Step 1: Inspection and config metadata seam
What changed:
- Added [tools/model_metadata.py](tools/model_metadata.py).
  - Provides local/remote metadata loading.
  - Uses HTTP range requests for safetensors header inspection, so the full Qwen3-Coder-Next weights do not need to be downloaded just to inspect keys.
  - Builds layer summaries and expert-layout candidates.
- Added [tools/inspect_qwen3_coder_next.py](tools/inspect_qwen3_coder_next.py).
  - Validates `qwen3_next` config metadata.
  - Enumerates tensor keys from local or remote safetensors headers.
  - Writes:
    - `artifacts/qwen3_coder_next_tensor_map.json`
    - `artifacts/qwen3_coder_next_model_summary.json`
- Updated [metal_infer/extract_weights.py](metal_infer/extract_weights.py).
  - Added `--summary`.
  - Stops hardcoding the emitted runtime config when a model summary is available.
  - Keeps the original Qwen3.5-397B-A17B config as the fallback path, preserving existing behavior.
- Updated [.gitignore](.gitignore) to ignore generated `artifacts/`.
- Ran the inspector against the real Hugging Face repo and generated:
  - [artifacts/qwen3_coder_next_tensor_map.json](artifacts/qwen3_coder_next_tensor_map.json)
  - [artifacts/qwen3_coder_next_model_summary.json](artifacts/qwen3_coder_next_model_summary.json)

What still blocks correctness:
- The repo still lacks the BF16 -> runtime quantization path for Qwen3-Coder-Next.
- `repack_experts.py` still assumes the old aggregated-expert layout.
- `extract_weights.py` still assumes the current already-quantized non-expert tensor organization and does not yet quantize BF16 source weights.
- The Objective-C runtime still hardcodes old dimensions, layer count, token ids, and tensor names.
- No intermediate/reference correctness harness exists yet.

How to test this milestone:
1. Run:
   ```bash
   python3 tools/inspect_qwen3_coder_next.py --model Qwen/Qwen3-Coder-Next
   ```
2. Check that:
   - `artifacts/qwen3_coder_next_tensor_map.json` exists
   - `artifacts/qwen3_coder_next_model_summary.json` exists
3. Spot-check the generated summary:
   - `expert_layout.style == "per_expert_named"`
   - `runtime_config_candidate.num_hidden_layers == 48`
   - `runtime_config_candidate.linear_total_value == 4096`
4. Re-run the old extractor help path to confirm backward compatibility:
   ```bash
   python3 metal_infer/extract_weights.py --help
   ```

## Next milestone
- Use the generated inspection summary to:
  - build an explicit Qwen3-Coder-Next tensor adapter
  - define a binary layout spec for converted non-expert/expert weights
  - refactor `repack_experts.py` into a layout-driven packer
- Then start the runtime-side config refactor so `infer.m` reads model metadata instead of compile-time constants wherever possible.

### Step 2: Qwen3-Coder-Next expert packer, verifier, and layout spec
What changed:
- Added [tools/q4_affine.py](tools/q4_affine.py).
  - Shared affine q4 quantization helpers.
  - Shared expert runtime layout builder.
- Replaced [repack_experts.py](repack_experts.py) with a dual-path packer:
  - `legacy` mode preserves the original repack flow for the existing model path.
  - `qwen3-next` mode consumes a local BF16 Qwen3-Coder-Next model directory and writes quantized routed expert blobs to `packed_experts_q4/`.
  - Output layout is derived from model dimensions instead of fixed 397B constants.
- Added [tools/verify_packed_weights.py](tools/verify_packed_weights.py).
  - Verifies file sizes and `layout.json`.
  - For qwen3-next, can dequantize sampled packed experts and compare against source BF16 tensors.
- Added [docs/qwen3_coder_next_binary_layout.md](docs/qwen3_coder_next_binary_layout.md).
  - Documents the routed-expert q4 runtime format.

What still blocks correctness:
- Non-expert BF16 -> runtime conversion is not implemented yet.
- The runtime still expects old non-expert tensor names and old model dimensions.
- `infer.m` and `main.m` still hardcode the old expert blob offsets and old hidden/intermediate dimensions.
- Shared expert and router gate conversion are not packed into the new runtime format yet; only routed experts are covered by the new qwen3-next repacker.
- No end-to-end generation path exists yet for Qwen3-Coder-Next.

How to test this milestone:
1. CLI smoke tests:
   ```bash
   python3 repack_experts.py --help
   python3 tools/verify_packed_weights.py --help
   ```
2. Layout sanity:
   ```bash
   python3 - <<'PY'
   from tools.q4_affine import build_moe_expert_q4_layout
   layout = build_moe_expert_q4_layout(hidden_size=2048, intermediate_size=512, group_size=64)
   print(layout["expert_size"])
   for comp in layout["expert_components"]:
       print(comp["name"], comp["offset"], comp["size"], comp["shape"])
   PY
   ```
3. On a local downloaded Qwen3-Coder-Next directory, run:
   ```bash
   python3 repack_experts.py \
     --mode qwen3-next \
     --model /path/to/Qwen3-Coder-Next \
     --summary artifacts/qwen3_coder_next_model_summary.json \
     --layers 0 \
     --sample-experts 0,1
   ```
4. Then verify:
   ```bash
   python3 tools/verify_packed_weights.py \
     --packed-dir /path/to/Qwen3-Coder-Next/packed_experts_q4 \
     --model /path/to/Qwen3-Coder-Next \
     --layers 0 \
     --experts 0,1
   ```

### Step 3: Non-expert q4 converter and first runtime config seam
What changed:
- Added [tools/qwen3_next_adapter.py](tools/qwen3_next_adapter.py).
  - Defines the explicit Qwen3-Coder-Next source -> runtime tensor mapping used during non-expert conversion.
  - Splits merged upstream tensors:
    - `linear_attn.in_proj_qkvz.weight` -> `in_proj_qkv` + `in_proj_z`
    - `linear_attn.in_proj_ba.weight` -> `in_proj_b` + `in_proj_a`
- Replaced [metal_infer/extract_weights.py](metal_infer/extract_weights.py) with a dual-path extractor:
  - `legacy` mode preserves the current model path.
  - `qwen3-next` mode reads a local BF16 model, quantizes matrix weights to affine q4, preserves BF16 tensors where the runtime expects BF16, and converts `A_log` to F32 because the runtime reads it as float.
- Updated [metal_infer/infer.m](metal_infer/infer.m):
  - Loads runtime config from `model_weights.json`.
  - Validates loaded config against compiled maxima.
  - Derives layer scheduling from manifest `layer_types` instead of hardcoding `(layer + 1) % 4 == 0` in several high-level execution loops.
  - Uses loaded layer count for major outer loops, layer cache construction, expert file opening, and cleanup.
  - Keeps the rest of the engine additive and backward-compatible by falling back to the legacy defaults when no config is present.
- Verified that [metal_infer/infer.m](metal_infer/infer.m) still compiles:
  - `make infer`

What still blocks correctness:
- The runtime still computes with compiled maxima for core tensor dimensions:
  - hidden size
  - linear attention buffer sizes
  - attention output dimensions
  - expert MoE intermediate sizes
- That means the config seam is real, but the full compute path is not yet dimension-generic enough to run Qwen3-Coder-Next correctly.
- Special token ids and prompt-format assumptions are still hardcoded for the old model family.
- The new qwen3-next extraction path has not been exercised end-to-end in this workspace because the local downloaded weights and `numpy` are not present here.
- Shared expert/routing execution uses converted weights, but the runtime still needs shape-driven allocation and launch dimension updates before that path is trustworthy for Qwen3-Coder-Next.

How to test this milestone:
1. Python smoke tests:
   ```bash
   python3 -m py_compile tools/qwen3_next_adapter.py metal_infer/extract_weights.py
   python3 metal_infer/extract_weights.py --help
   ```
2. C build smoke test:
   ```bash
   cd metal_infer
   make infer
   ```
3. On a local Qwen3-Coder-Next checkout, run the non-expert converter:
   ```bash
   python3 metal_infer/extract_weights.py \
     --mode qwen3-next \
     --model /path/to/Qwen3-Coder-Next \
     --output /path/to/runtime-out \
     --summary artifacts/qwen3_coder_next_model_summary.json
   ```
4. Inspect `/path/to/runtime-out/model_weights.json` and confirm:
   - `config.model_type == "qwen3_next"`
   - tensor names include split linear-attention entries like:
     - `model.layers.0.linear_attn.in_proj_qkv.weight`
     - `model.layers.0.linear_attn.in_proj_z.weight`
     - `model.layers.0.linear_attn.in_proj_b.weight`
     - `model.layers.0.linear_attn.in_proj_a.weight`

### Step 4: Runtime dimension/config refactor inside the fused execution path
What changed:
- Updated [metal_infer/infer.m](metal_infer/infer.m) to derive more of the hot inference path from the loaded manifest config instead of the legacy compile-time dimensions.
- Added a runtime-derived expert layout helper used by both GPU and CPU expert execution.
  - The 4-bit expert blob size and offsets now derive from:
    - `hidden_size`
    - `moe_intermediate_size`
    - `GROUP_SIZE`
  - This matches the new qwen3-next packer layout instead of assuming the old `4096 x 1024` expert shape.
- Refactored the main fused layer path `fused_layer_forward(...)` so the following are now config-driven:
  - full-attention projection/output dimensions
  - RoPE application dimensions
  - GQA routing dimensions
  - linear-attention split sizes
  - delta-net head counts and state math dimensions
  - MoE routing vector sizes
  - shared expert intermediate size
  - deferred combine buffers and final hidden-width combines
- Refactored deferred expert completion so CPU/GPU final combine now uses runtime hidden size rather than always iterating `4096`.
- Verified the updated runtime still builds:
  - `cd metal_infer && make infer`

What still blocks correctness:
- Other older code paths in [metal_infer/infer.m](metal_infer/infer.m) still contain legacy constants outside the main fused path.
  - These include older debug/fallback helper paths and some prefill/generation helpers.
  - They may not all be exercised by the intended fused runtime path, but they still need a cleanup pass before the port is complete.
- [metal_infer/main.m](metal_infer/main.m) still assumes the old expert blob layout and old dimensions.
- Special token ids and prompt-format assumptions are still tied to the old model family.
- No end-to-end Qwen3-Coder-Next generation run has been executed in this workspace because the local weights are not available here.
- The runtime still needed the dead 2-bit path deleted so the qwen3-next port had one consistent expert format.

How to test this milestone:
1. Rebuild the runtime:
   ```bash
   cd metal_infer
   make infer
   ```
2. Confirm the main fused path no longer contains the old hardcoded dimensions:
   ```bash
   rg -n "HIDDEN_DIM|NUM_EXPERTS|SHARED_INTERMEDIATE|MOE_INTERMEDIATE|LINEAR_TOTAL_VALUE|LINEAR_TOTAL_KEY|NUM_ATTN_HEADS|NUM_KV_HEADS|HEAD_DIM|ROTARY_DIM" metal_infer/infer.m
   ```
   Then verify the remaining matches are outside the refactored fused path or are legacy defaults/limits.
3. Once local converted weights exist, use the qwen3-next output manifest and ensure startup prints the loaded runtime config instead of only the legacy defaults.

## Next milestone
- Finish the remaining runtime generalization outside the fused path:
  - remaining fallback/helper execution paths in [metal_infer/infer.m](metal_infer/infer.m)
  - [metal_infer/main.m](metal_infer/main.m)
- Add tokenizer export/model-profile support for Qwen3-Coder-Next.
- Add validation tooling:
  - [tools/reference_compare.py](tools/reference_compare.py)
  - runtime activation dump hooks
- Then attempt the first real end-to-end Qwen3-Coder-Next prompt run.

### Step 5: Qwen3-Coder-Next-only defaults and top-k=10 runtime path
What changed:
- Switched the runtime defaults in [metal_infer/infer.m](metal_infer/infer.m) from the old 397B profile to Qwen3-Coder-Next values:
  - `hidden_size = 2048`
  - `num_hidden_layers = 48`
  - `num_attention_heads = 16`
  - `num_key_value_heads = 2`
  - `vocab_size = 151936`
  - `moe_intermediate_size = 512`
  - `shared_expert_intermediate_size = 512`
  - `linear_num_value_heads = 32`
  - `rope_theta = 5000000`
- Switched CLI/runtime defaults to Qwen3-Coder-Next behavior:
  - default `--k` is now `10`
  - default model path now resolves from `QWEN3_CODER_NEXT_MODEL_PATH` or `./Qwen3-Coder-Next`
  - runtime banners/labels now refer to Qwen3-Coder-Next
- Removed the old host-side `MAX_K=8` limitation in the inference engine:
  - [metal_infer/infer.m](metal_infer/infer.m) now allocates/binds `MAX_K=10`
  - the deferred combine parameter buffer now holds 10 expert weights plus the shared gate score
  - speculative routing and prediction state arrays now size off `NUM_LAYERS` and `MAX_K`
- Updated [metal_infer/shaders.metal](metal_infer/shaders.metal):
  - `moe_combine_residual` now accepts and combines 10 expert outputs instead of 8
- Updated [metal_infer/main.m](metal_infer/main.m) to the Qwen3-Coder-Next packed expert layout:
  - expert offsets/sizes now match the new `1769472`-byte q4 expert blob
  - benchmark defaults now assume 48 layers and `K=10`
- Updated tooling/defaults:
  - [metal_infer/export_tokenizer.py](metal_infer/export_tokenizer.py)
  - [metal_infer/chat.m](metal_infer/chat.m)
  - [metal_infer/Makefile](metal_infer/Makefile)

What still blocks correctness:
- I could not validate the edited Metal shader offline because this environment lacks the Metal toolchain component.
  - Objective-C builds pass.
  - Runtime shader compilation could not be exercised here because the sandbox exposes no Metal device.
- Some fallback/helper paths in [metal_infer/infer.m](metal_infer/infer.m) still contain old hardcoded dimensions outside the main fused path.
- Special token ids are still hardcoded in the runtime and should ideally be exported from tokenizer/model metadata instead of remaining literals.
- No end-to-end converted local Qwen3-Coder-Next run has been executed yet.
- The runtime still needed the dead 2-bit path removed entirely so the Qwen3-Coder-Next port had one maintained expert path.

How to test this milestone:
1. Build the binaries:
   ```bash
   cd metal_infer
   make infer metal_infer chat
   ```

### Step 9: Remove dead 2-bit runtime path
What changed:
- Deleted the obsolete [metal_infer/repack_experts_2bit.py](metal_infer/repack_experts_2bit.py) script.
- Removed the 2-bit runtime state, CLI flag, auto-detection, and Metal pipeline from [metal_infer/infer.m](metal_infer/infer.m).
- Removed the unused `dequant_matvec_2bit` kernel from [metal_infer/shaders.metal](metal_infer/shaders.metal).
- Simplified expert-file loading so the runtime only opens `packed_experts/layer_XX.bin`.
- Switched the CPU fallback in [metal_infer/infer.m](metal_infer/infer.m) to use the shared runtime expert-layout helper instead of stale hardcoded offsets.
- Updated [README.md](README.md) and [RUN_QWEN3_CODER_NEXT.md](RUN_QWEN3_CODER_NEXT.md) to describe the runtime as q4-only instead of “2-bit unsupported”.

What still blocks correctness:
- No end-to-end Qwen3-Coder-Next prompt run has been validated on Apple Silicon in this workspace.
- [tools/reference_compare.py](tools/reference_compare.py) is partial; it does not yet prove full-model equivalence to Hugging Face.
- [metal_infer/main.m](metal_infer/main.m) still needs the same cleanup standard as [metal_infer/infer.m](metal_infer/infer.m) if it will remain part of the supported surface area.

How to test this milestone:
1. Rebuild the runtime:
   ```bash
   cd metal_infer
   make infer
   ```
2. Confirm the dead 2-bit path is gone from the active Qwen docs and runtime:
   ```bash
   rg -n "2bit|packed_experts_2bit|dequant_matvec_2bit|matvec_2bit|g_use_2bit" metal_infer README.md RUN_QWEN3_CODER_NEXT.md
   ```
3. Check the CLI no longer advertises the removed flag:
   ```bash
   cd metal_infer
   ./infer --help
   ```
2. Check the benchmark CLI defaults:
   ```bash
   ./metal_infer --help
   ```
   Confirm it reports:
   - `full 48-layer`
   - default `--k 10`
3. Build smoke test for the inference engine:
   ```bash
   cd metal_infer
   make infer
   ```
4. On a machine with a Metal device and a local converted model, start the runtime and confirm:
   - it prints `Qwen3-Coder-Next`
   - it reports `K: 10 experts/layer`
   - it opens `48` packed expert layer files

### Step 6: Runtime cleanup around the fused Qwen3-Coder-Next path
What changed:
- Isolated the stale CPU/hybrid fallback implementations in [metal_infer/infer.m](metal_infer/infer.m) behind `#if 0`.
  - `linear_attention_forward(...)`
  - `moe_forward(...)`
- This removes the old 397B-shaped helper paths from the active build without touching the fused Qwen3-Coder-Next execution path.
- Cleaned stale layout comments in:
  - [metal_infer/shaders.metal](metal_infer/shaders.metal)
  - [metal_infer/main.m](metal_infer/main.m)
  so the documented expert dimensions now match Qwen3-Coder-Next (`2048 -> 512 -> 2048`, `48` layers).

What still blocks correctness:
- The Metal shader source still contains hardcoded shared-memory sizing and launch-shape assumptions tuned around the old larger shapes.
  - These may still be valid upper bounds for Qwen3-Coder-Next, but they have not been validated on-device.
- Several inactive helper paths and benchmarking utilities still carry compile-time shapes even though the main runtime path is now Qwen3-Coder-Next-specific.
- No real converted Qwen3-Coder-Next weights have been exercised through `./infer` on a Metal device yet.
- [tools/reference_compare.py](tools/reference_compare.py) and runtime activation dumps are still missing, so numerical correctness is not yet checked against a Hugging Face reference.

How to test this milestone:
1. Rebuild the inference engine:
   ```bash
   cd metal_infer
   make infer
   ```
2. Confirm the old helper paths are no longer active build code:
   ```bash
   rg -n "#if 0|linear_attention_forward\\(|moe_forward\\(" metal_infer/infer.m
   ```
3. Re-scan for stale old-model labels:
   ```bash
   rg -n "Qwen3\\.5|397B|A17B|all 60|4096 -> 1024|1024 -> 4096" metal_infer/main.m metal_infer/shaders.metal metal_infer/infer.m
   ```
   Remaining hits should be either valid Qwen3-Coder-Next dimensions, inactive code, or general buffer-size limits rather than active legacy model descriptions.

### Step 7: Runnable operator guide and asset-path cleanup
What changed:
- Added [RUN_QWEN3_CODER_NEXT.md](RUN_QWEN3_CODER_NEXT.md) with a concrete operator flow:
  - install dependencies
  - download the model
  - inspect tensors
  - export `tokenizer.bin` and `vocab.bin`
  - convert non-expert weights
  - pack experts into `packed_experts/`
  - verify packed experts
  - build and run `metal_infer/infer`
- Fixed [metal_infer/export_tokenizer.py](metal_infer/export_tokenizer.py):
  - no longer crashes on `--help`
  - now emits both `tokenizer.bin` and `vocab.bin`
  - now has explicit CLI flags instead of positional-only behavior
- Improved [metal_infer/infer.m](metal_infer/infer.m) default asset discovery:
  - searches the selected model directory first for `model_weights.bin`, `model_weights.json`, and `vocab.bin`
  - searches the model directory and `QWEN3_CODER_NEXT_MODEL_PATH` for `tokenizer.bin`
  - `--prompt` help text now reflects tokenizer-based prompt encoding rather than the old Python helper note

What still blocks correctness:
- The guide documents the current intended run path, but I still have not executed a full Qwen3-Coder-Next prompt on a real Metal device in this environment.
- Runtime performance/correctness for the fused linear-attention path is still not cross-checked layer-by-layer against Hugging Face.

How to test this milestone:
1. Validate the tokenizer exporter CLI:
   ```bash
   python3 metal_infer/export_tokenizer.py --help
   ```
2. Rebuild the runtime:
   ```bash
   cd metal_infer
   make infer
   ```
3. Follow [RUN_QWEN3_CODER_NEXT.md](RUN_QWEN3_CODER_NEXT.md) from top to bottom on a real Apple Silicon machine with the downloaded model.

### Step 8: Partial reference comparison harness and runtime dump hooks
What changed:
- Added [tools/reference_compare.py](tools/reference_compare.py).
  - It compares runtime dumps against source BF16 tensors without loading the full model.
  - Current comparisons:
    - last prompt-token embedding
    - per-layer router logits
    - per-layer top-k routing overlap
    - per-layer shared expert pre-gate output
    - final logits from the dumped final hidden state
- Added runtime dump support in [metal_infer/infer.m](metal_infer/infer.m):
  - new flag: `--dump-dir PATH`
  - dumps the last prompt-position checkpoints needed by the reference harness
- Updated [RUN_QWEN3_CODER_NEXT.md](RUN_QWEN3_CODER_NEXT.md) to document:
  - runtime dump generation
  - running the comparison harness

What still blocks correctness:
- The comparison harness is partial. It does not yet run a full Hugging Face forward pass for end-to-end token-by-token comparison.
- No real Apple Silicon runtime dump has been produced in this environment yet.
- Metal-kernel correctness is still not proven until the dumped runtime tensors are compared on a real converted model run.

How to test this milestone:
1. Validate the new comparison tool:
   ```bash
   python3 tools/reference_compare.py --help
   ```
2. Rebuild the runtime and confirm the dump flag exists:
   ```bash
   cd metal_infer
   make infer
   ./infer --help
   ```
3. On a real Apple Silicon machine with the converted model, run:
   ```bash
   ./infer --model "$QWEN3_CODER_NEXT_MODEL_PATH" --prompt "Hello" --tokens 1 --k 10 --dump-dir /tmp/qwen3-next-dump
   ```
4. Then compare:
   ```bash
   python3 tools/reference_compare.py --model "$QWEN3_CODER_NEXT_MODEL_PATH" --dump-dir /tmp/qwen3-next-dump --layers 0,1,47
   ```

### Step 10: Fix tokenizer/vocab decode table for added tokens
What changed:
- Fixed [metal_infer/export_tokenizer.py](metal_infer/export_tokenizer.py) so `vocab.bin` now includes the full contiguous decode table:
  - base tokenizer vocab entries
  - added special tokens such as `<|im_end|>`, `<|endoftext|>`, `<think>`, and `</think>`
- Added validation in the exporter to reject conflicting base-vocab vs added-token ids.
- Hardened [metal_infer/infer.m](metal_infer/infer.m) vocabulary loading:
  - it now respects the full decode-table `max_id`
  - it prints the loaded decode range explicitly
  - it warns if configured special token ids still exceed the loaded decode table

Why this changed:
- Real Qwen3-Coder-Next runs showed `vocab.bin` only contained `151643` base vocab entries while the tokenizer's special tokens live at ids `151643..151668`.
- That meant runtime decoding and special-token handling could be wrong even when the model forward pass itself was otherwise functioning.

What still blocks correctness:
- The runtime can now load the full decode table, but generation quality is still not yet acceptable.
- Early-layer routing/shared-expert drift remains visible in [tools/reference_compare.py](tools/reference_compare.py) output and still needs investigation in the fused runtime path.
- The custom BPE tokenizer still has not been compared side-by-side against Hugging Face tokenization for representative prompts.

How to test this milestone:
1. Re-export tokenizer assets:
   ```bash
   python3 metal_infer/export_tokenizer.py --model-dir "$QWEN3_CODER_NEXT_MODEL_PATH"
   ```
2. Confirm `vocab.bin` now exposes the full decode range:
   ```bash
   python3 - <<'PY'
   import os, struct
   path = os.path.join(os.environ["QWEN3_CODER_NEXT_MODEL_PATH"], "vocab.bin")
   with open(path, "rb") as f:
       print(struct.unpack("<II", f.read(8)))
   PY
   ```
   Expected: `(151669, 151668)`
3. Rebuild and rerun `infer`, then confirm startup reports the full decode table and no longer uses out-of-range special token ids.

### Step 11: Add targeted dump filters, operator-local parity checks, and canned validation
What changed:
- Expanded [metal_infer/infer.m](metal_infer/infer.m) runtime dumps:
  - new flags: `--dump-layers SPEC` and `--dump-stages LIST`
  - writes prompt context metadata for tokenizer-parity checks
  - can dump selected checkpoints for selected layers only
  - added checkpoints around:
    - layer input and attention input norm
    - post-attention residual and post-attention norm
    - shared gate / up / SwiGLU / down-proj path
    - MoE sum before residual
    - layer output after residual
- Reworked [tools/reference_compare.py](tools/reference_compare.py):
  - threshold profile support via `--fail-threshold-profile`
  - per-layer first-failure reporting
  - nonzero exit on the earliest failed checkpoint
  - tokenizer prompt/decode parity checks when `prompt_context.json` is available
  - compares:
    - embedding
    - attention input norm
    - post-attention residual and norm
    - router logits + exact top-k
    - shared expert gate / up / SwiGLU / down path
    - routed MoE sum and final layer output
    - final logits
- Added [tools/validate_qwen3_next_runtime.py](tools/validate_qwen3_next_runtime.py):
  - canned plain-text, tool-envelope, and JSON validation prompts
  - runs the compare harness on the first pass of each prompt

Why this changed:
- The earlier comparison flow was enough to prove that the runtime was alive, but not enough to isolate the first bad operator.
- The remaining correctness gap is now in early-layer math and tool/chat behavior, so the debugging flow needs narrower checkpoints and regression gating.

What still blocks correctness:
- The compare harness is still operator-local around the attention output itself; it does not yet reconstruct the full reference attention output for each dumped token position.
- Real on-device validation still has to happen on Apple Silicon with Metal enabled.
- Output quality and tool-calling parity are still not proven until the new validation suite passes on the converted model.

How to test this milestone:
1. Rebuild and inspect the new CLI:
   ```bash
   cd metal_infer
   make infer
   ./infer --help
   ```
2. Generate a filtered dump:
   ```bash
   ./infer \
     --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
     --prompt "Hello" \
     --tokens 1 \
     --k 10 \
     --dump-dir /tmp/qwen3-next-dump \
     --dump-layers 0,1,47 \
     --dump-stages embedding,attn,router,shared,final
   ```
3. Run the first-failure oracle:
   ```bash
   python3 tools/reference_compare.py \
     --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
     --dump-dir /tmp/qwen3-next-dump \
     --layers 0,1,47
   ```
4. Run the canned runtime suite:
   ```bash
   python3 tools/validate_qwen3_next_runtime.py \
     --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
     --output /tmp/qwen3-next-validation.json
   ```
