# Flash-MoE

Apple-Silicon-first C/Metal inference runtime and weight-conversion pipeline for `Qwen/Qwen3-Coder-Next`.

This repository originally focused on a 397B SSD-streamed MoE experiment. The current implementation has been moved toward `Qwen3-Coder-Next` and now contains:
- a C/Metal runtime in `metal_infer/`
- Python conversion and inspection tools for Hugging Face weights
- q4 expert packing for SSD-streamed MoE execution
- tokenizer and vocab export for local prompt encoding/decoding

## Current Status

Implemented:
- source-model inspection for `Qwen3-Coder-Next`
- non-expert BF16 -> runtime conversion
- expert BF16 -> packed q4 conversion
- runtime defaults for:
  - `48` layers
  - `2048` hidden size
  - `512` experts
  - top-k `10`
- buildable `infer`, `metal_infer`, and `chat` binaries

Not yet fully proven:
- a fully validated end-to-end generation run on a real Apple Silicon machine within this environment
- reference-vs-Hugging-Face numerical comparison
- final correctness/performance claims for the Qwen3-Coder-Next port

Do not treat this repo as a finished, benchmarked release for Qwen3-Coder-Next yet. Treat it as an active bring-up port with a concrete conversion path and an adapting runtime.

## Read This First

- [RUN_QWEN3_CODER_NEXT.md](RUN_QWEN3_CODER_NEXT.md): exact download, conversion, build, and run steps
- [QWEN3_CODER_NEXT_PORT_PLAN.md](QWEN3_CODER_NEXT_PORT_PLAN.md): migration log, blockers, and validation status
- [docs/qwen3_coder_next_binary_layout.md](docs/qwen3_coder_next_binary_layout.md): packed expert binary layout

Historical background:
- [paper/flash_moe.pdf](paper/flash_moe.pdf)

The paper documents the original 397B experiment and remains useful for runtime design context, but the repo root now reflects the Qwen3-Coder-Next porting effort rather than the original published configuration.

## Quick Start

Install Python dependencies:

```bash
python3 -m pip install -U "huggingface_hub[cli]" numpy
```

Download the model locally:

```bash
cd /Users/sk/dev/flash-moe
hf download Qwen/Qwen3-Coder-Next --local-dir ./Qwen3-Coder-Next
export QWEN3_CODER_NEXT_MODEL_PATH="$PWD/Qwen3-Coder-Next"
```

Inspect the source model:

```bash
python3 tools/inspect_qwen3_coder_next.py \
  --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --artifacts-dir ./artifacts
```

Export tokenizer assets:

```bash
python3 metal_infer/export_tokenizer.py \
  --model-dir "$QWEN3_CODER_NEXT_MODEL_PATH"
```

Convert non-expert weights:

```bash
python3 metal_infer/extract_weights.py \
  --mode qwen3-next \
  --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --output "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --summary ./artifacts/qwen3_coder_next_model_summary.json
```

Pack experts:

```bash
python3 repack_experts.py \
  --mode qwen3-next \
  --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --summary ./artifacts/qwen3_coder_next_model_summary.json \
  --output "$QWEN3_CODER_NEXT_MODEL_PATH/packed_experts" \
  --layers all \
  --sample-experts 0,1
```

Build and run:

```bash
cd metal_infer
make infer
./infer --model "$QWEN3_CODER_NEXT_MODEL_PATH" --prompt "Hello" --tokens 16 --k 10
```

For the full operator flow, use [RUN_QWEN3_CODER_NEXT.md](RUN_QWEN3_CODER_NEXT.md).

## Project Structure

```text
metal_infer/
  infer.m                  # Main C/Metal inference runtime
  shaders.metal            # Metal kernels
  main.m                   # MoE benchmark / lower-level runtime harness
  chat.m                   # Chat / server entrypoint
  extract_weights.py       # Non-expert weight conversion
  export_tokenizer.py      # tokenizer.json -> tokenizer.bin + vocab.bin
  tokenizer.h              # C tokenizer implementation
  Makefile                 # Build targets

tools/
  inspect_qwen3_coder_next.py
  model_metadata.py
  q4_affine.py
  qwen3_next_adapter.py
  verify_packed_weights.py

docs/
  qwen3_coder_next_binary_layout.md

repack_experts.py          # Expert packing for qwen3-next q4 blobs
QWEN3_CODER_NEXT_PORT_PLAN.md
RUN_QWEN3_CODER_NEXT.md
```

## Runtime Notes

- The runtime is currently Qwen3-Coder-Next-focused.
- The runtime is 4-bit-expert-only for this port.
- The runtime expects packed experts under:
  - `$QWEN3_CODER_NEXT_MODEL_PATH/packed_experts/`
- The runtime now looks in the selected model directory first for:
  - `model_weights.bin`
  - `model_weights.json`
  - `vocab.bin`
  - `tokenizer.bin`

## Validation Notes

What has been validated locally in this repo:
- Python conversion scripts parse and build
- Objective-C runtime binaries build
- stale old-model labels and inactive helper paths were cleaned up

What has not been validated here:
- a real on-device full prompt run with converted Qwen3-Coder-Next weights
- reference tensor comparisons against Hugging Face

Until that is done, claims like “fully compatible” or “production ready” would be inaccurate.
