# Run Qwen3-Coder-Next

This document shows the current end-to-end path for:

1. downloading `Qwen/Qwen3-Coder-Next`
2. converting it into this runtime's expected files
3. building the Metal runtime
4. running a first prompt on Apple Silicon

This repo is now oriented around `Qwen3-Coder-Next`. It does not require backward compatibility with the old 397B path.

## Status

What is implemented:
- model inspection
- non-expert weight conversion into `model_weights.bin` + `model_weights.json`
- routed expert packing into runtime q4 blobs
- tokenizer export into `tokenizer.bin`
- vocab export into `vocab.bin`
- runtime defaults for `48` layers and `K=10`
- runtime dump hooks for last-prompt-position comparison
- partial reference comparison against source BF16 tensors

What is still not fully validated:
- a full on-device correctness comparison against Hugging Face
- a complete end-to-end generation run in this sandbox
- final quality and performance measurements on real Apple Silicon hardware

Treat this as a working bring-up path, not a final polished release.

## Requirements

- Apple Silicon Mac with Metal support
- macOS with Xcode Command Line Tools
- Python 3.10+
- enough free SSD space

Recommended free space:
- source model: about `159 GB`
- packed q4 experts: about `40.5 GB`
- non-expert runtime weights and manifests: several more GB
- working headroom: at least `220 GB` free is a reasonable minimum

## 1. Install Python Dependencies

The repo conversion scripts need `numpy`. For model download, use the Hugging Face Hub CLI.

```bash
python3 -m pip install -U "huggingface_hub[cli]" numpy
```

Official references:
- [Hugging Face Hub download docs](https://huggingface.co/docs/huggingface_hub/en/guides/download)
- [Qwen/Qwen3-Coder-Next model page](https://huggingface.co/Qwen/Qwen3-Coder-Next/tree/main)

If your Hugging Face setup needs authentication:

```bash
hf auth login
```

## 2. Download The Model

From the repo root:

```bash
cd /Users/sk/dev/flash-moe
hf download Qwen/Qwen3-Coder-Next --local-dir ./Qwen3-Coder-Next
```

This should leave you with a local directory like:

```text
Qwen3-Coder-Next/
  config.json
  tokenizer.json
  model.safetensors.index.json
  model-00001-of-00040.safetensors
  ...
```

Set the model path for later commands:

```bash
export QWEN3_CODER_NEXT_MODEL_PATH="$PWD/Qwen3-Coder-Next"
```

## 3. Inspect The Source Model

This step records the tensor layout and runtime metadata under `artifacts/`.

```bash
python3 tools/inspect_qwen3_coder_next.py \
  --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --artifacts-dir ./artifacts \
  --print-layer-limit 4
```

Expected outputs:

```text
artifacts/qwen3_coder_next_model_summary.json
artifacts/qwen3_coder_next_tensor_map.json
```

## 4. Export Tokenizer Assets

The runtime uses two local binary assets:
- `tokenizer.bin` for prompt encoding
- `vocab.bin` for token decoding and special-token lookup

Generate both directly into the model directory:

```bash
python3 metal_infer/export_tokenizer.py \
  --model-dir "$QWEN3_CODER_NEXT_MODEL_PATH"
```

Expected outputs:

```text
$QWEN3_CODER_NEXT_MODEL_PATH/tokenizer.bin
$QWEN3_CODER_NEXT_MODEL_PATH/vocab.bin
```

## 5. Convert Non-Expert Weights

This writes:
- `model_weights.bin`
- `model_weights.json`

directly into the model directory.

```bash
python3 metal_infer/extract_weights.py \
  --mode qwen3-next \
  --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --output "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --summary ./artifacts/qwen3_coder_next_model_summary.json
```

Notes:
- this is the non-routed path only
- routed experts are handled separately by `repack_experts.py`

## 6. Pack Routed Experts

Important: the runtime opens `packed_experts/`, not `packed_experts_q4/`.

So write the expert blobs into `packed_experts/` directly:

```bash
python3 repack_experts.py \
  --mode qwen3-next \
  --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --summary ./artifacts/qwen3_coder_next_model_summary.json \
  --output "$QWEN3_CODER_NEXT_MODEL_PATH/packed_experts" \
  --layers all \
  --sample-experts 0,1
```

Expected output:

```text
$QWEN3_CODER_NEXT_MODEL_PATH/packed_experts/layout.json
$QWEN3_CODER_NEXT_MODEL_PATH/packed_experts/layer_00.bin
...
$QWEN3_CODER_NEXT_MODEL_PATH/packed_experts/layer_47.bin
```

## 7. Verify Packed Experts

Do at least one spot check before trying inference:

```bash
python3 tools/verify_packed_weights.py \
  --packed-dir "$QWEN3_CODER_NEXT_MODEL_PATH/packed_experts" \
  --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --layers 0 \
  --experts 0,1
```

This verifies:
- layer file sizes
- expert blob layout
- sampled q4 dequant slices against the original BF16 tensors

## 8. Build The Runtime

```bash
cd /Users/sk/dev/flash-moe/metal_infer
make infer chat
```

If the build succeeds, the main runtime binary is:

```text
/Users/sk/dev/flash-moe/metal_infer/infer
```

## 9. Run A First Prompt

The runtime now looks in the model directory first for:
- `model_weights.bin`
- `model_weights.json`
- `vocab.bin`
- `tokenizer.bin`

So the shortest working invocation is:

```bash
cd /Users/sk/dev/flash-moe/metal_infer
export QWEN3_CODER_NEXT_MODEL_PATH=/Users/sk/dev/flash-moe/Qwen3-Coder-Next

./infer \
  --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --prompt "Write a Python function that reverses a linked list." \
  --tokens 64 \
  --k 10
```

If you want to be fully explicit:

```bash
./infer \
  --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --weights "$QWEN3_CODER_NEXT_MODEL_PATH/model_weights.bin" \
  --manifest "$QWEN3_CODER_NEXT_MODEL_PATH/model_weights.json" \
  --vocab "$QWEN3_CODER_NEXT_MODEL_PATH/vocab.bin" \
  --prompt "Write a Python function that reverses a linked list." \
  --tokens 64 \
  --k 10
```

Useful debug flags:

```bash
./infer \
  --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --prompt "Hello" \
  --tokens 8 \
  --k 10 \
  --timing \
  --cache-telemetry
```

## 10. Dump Comparison Tensors

The runtime can now dump tensors for the last prompt position. This is intended
for bring-up and accuracy debugging.

It writes:
- prompt token ids
- the last prompt-token embedding
- per-layer `h_post`
- per-layer raw router logits
- per-layer top-k routing
- per-layer shared expert pre-gate output
- final hidden state after final norm
- final logits

Example:

```bash
cd /Users/sk/dev/flash-moe/metal_infer
export QWEN3_CODER_NEXT_MODEL_PATH=/Users/sk/dev/flash-moe/Qwen3-Coder-Next

./infer \
  --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --prompt "Write a Python function that reverses a linked list." \
  --tokens 1 \
  --k 10 \
  --dump-dir /tmp/qwen3-next-dump
```

Expected output files include names like:

```text
/tmp/qwen3-next-dump/prompt_tokens.json
/tmp/qwen3-next-dump/prefill_last_pos_000000_tok_...__embedding.bin
/tmp/qwen3-next-dump/prefill_last_pos_000000_tok_...__final_hidden.bin
/tmp/qwen3-next-dump/prefill_last_pos_000000_tok_...__logits.bin
/tmp/qwen3-next-dump/prefill_last_pos_000000_tok_...__layer_00_router_logits.bin
/tmp/qwen3-next-dump/prefill_last_pos_000000_tok_...__layer_00_topk.json
```

## 11. Compare Runtime Dumps Against Source BF16 Tensors

The comparison harness does not need to load the full model into RAM. It reads
only the source tensors needed for the dumped checkpoints.

Current comparisons:
- last-token embedding
- per-layer router logits
- per-layer router top-k overlap
- per-layer shared expert pre-gate output
- final logits from dumped final hidden state

Example:

```bash
python3 /Users/sk/dev/flash-moe/tools/reference_compare.py \
  --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --dump-dir /tmp/qwen3-next-dump \
  --layers 0,1,2,47 \
  --output /tmp/qwen3-next-compare.json
```

This prints per-checkpoint error summaries and writes a JSON report if
`--output` is provided.

## 12. Run OpenAI-Compatible Server Mode

```bash
cd /Users/sk/dev/flash-moe/metal_infer
export QWEN3_CODER_NEXT_MODEL_PATH=/Users/sk/dev/flash-moe/Qwen3-Coder-Next

./infer \
  --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --serve 8080 \
  --k 10
```

Then test:

```bash
curl http://127.0.0.1:8080/health
```

## Expected Runtime Behavior

At startup, you should see lines like:
- `Qwen3-Coder-Next Metal Inference Engine`
- `K: 10 experts/layer`
- `[manifest] Model config: type=qwen3_next layers=48 hidden=2048 experts=512 ...`
- `[experts] 48/48 packed layer files available`
- `[vocab] Loaded ... tokens`

## Troubleshooting

### `ERROR: No Metal device`

You are not running on a machine where Metal is available to this process.

### `ERROR: Cannot open vocab ...`

You did not generate `vocab.bin`, or the runtime is not looking in the right directory.

Fix:

```bash
python3 /Users/sk/dev/flash-moe/metal_infer/export_tokenizer.py \
  --model-dir "$QWEN3_CODER_NEXT_MODEL_PATH"
```

### `WARNING: tokenizer.bin not found, tokenization will fail`

Prompt encoding with `--prompt` requires `tokenizer.bin`.

Fix:

```bash
python3 /Users/sk/dev/flash-moe/metal_infer/export_tokenizer.py \
  --model-dir "$QWEN3_CODER_NEXT_MODEL_PATH"
```

### `[experts] 0/48 packed layer files available`

You packed experts into the wrong directory.

The runtime currently expects:

```text
$QWEN3_CODER_NEXT_MODEL_PATH/packed_experts/layer_00.bin
...
```

If you wrote to `packed_experts_q4/`, rerun:

```bash
python3 /Users/sk/dev/flash-moe/repack_experts.py \
  --mode qwen3-next \
  --model "$QWEN3_CODER_NEXT_MODEL_PATH" \
  --summary /Users/sk/dev/flash-moe/artifacts/qwen3_coder_next_model_summary.json \
  --output "$QWEN3_CODER_NEXT_MODEL_PATH/packed_experts" \
  --layers all
```

### `ERROR: Failed to encode prompt`

The runtime could not load `tokenizer.bin`. Generate it and retry.

### `ERROR: manifest config exceeds compiled runtime maxima`

The manifest does not match the compiled runtime assumptions. Recreate `model_weights.json` from the inspected Qwen3-Coder-Next model with the current scripts.

## Known Limitations

- This port only supports the q4 packed-expert path under `packed_experts/`.
- The reference comparison path is partial. It compares dumped checkpoints against source BF16 tensors, not a full end-to-end Hugging Face forward pass.
- This document reflects the current bring-up state of the port, not a fully performance-tuned or correctness-proven release.
