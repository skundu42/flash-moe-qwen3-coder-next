# Qwen3-Coder-Next on Flash-MoE: A Porting and Runtime Recovery Report

## Abstract

This document summarizes the port of `Qwen3-Coder-Next` into the Flash-MoE runtime and the engineering work required to make the model run end to end in a Flash-MoE-style deployment. The work progressed from initial structural compatibility, through asset conversion and expert packing, to runtime correctness recovery, chat integration, and tool-calling support. The final outcome is a working Metal-based runtime that can load the converted model, generate coherent responses, participate in multi-turn chat, and execute tool calls through the local chat client. The remaining work is primarily performance recovery and broader regression hardening rather than first-principles model bring-up.

## 1. Motivation

Flash-MoE was originally built around a different large MoE architecture and a specific runtime/data layout. Porting `Qwen3-Coder-Next` required answering two questions:

1. Can the model be expressed in the same streamed-expert inference architecture?
2. Can the runtime reproduce the source model accurately enough for chat and tool use?

The first question is structural. The second is numerical and behavioral. This project succeeded only once both were addressed.

## 2. Objectives

The target was not merely "the model loads" or "it emits tokens." The goal was to make `Qwen3-Coder-Next` fit the Flash-MoE workflow accurately enough to support:

- model inspection and tensor mapping
- tokenizer export and decoding parity
- non-expert weight extraction
- expert repacking into streamed layer files
- runtime execution on Metal
- direct prompt inference
- multi-turn chat
- tool-calling through the local chat client

## 3. System Overview

The final working path consists of the following stages:

1. Inspect the Hugging Face model and produce a structural summary.
2. Export tokenizer assets into a runtime-friendly binary format.
3. Extract non-expert tensors into `model_weights.bin` plus `model_weights.json`.
4. Repack experts into per-layer streamed binary files.
5. Load the converted assets in the Metal runtime.
6. Run direct prompt inference or the HTTP chat server.
7. Use the local `chat` client for plain chat or tool-enabled chat.

The runtime remains Flash-MoE in style:

- non-expert tensors are memory-mapped
- experts are stored in packed per-layer files
- only selected experts are loaded during inference
- Metal handles GPU compute while host code orchestrates routing and I/O

## 4. Major Technical Challenges

### 4.1 Tokenizer and decoding mismatch

The initial tokenizer export path was incomplete. The runtime decode table did not include the full added-token range, and merge export was not fully faithful to `tokenizer.json`. This created:

- missing special-token coverage
- mismatched visible vocab size
- incorrect decoding artifacts
- prompt/token mismatch against the source model

This was fixed by exporting a contiguous decode table, preserving added-token ids, and correcting merge serialization.

### 4.2 Prompt formatting mismatch

Early testing fed plain raw prompt text into the model. `Qwen3-Coder-Next` is chat-oriented, so this was a distribution mismatch. The CLI and chat flows were updated to use the Qwen-style chat wrapper, which immediately improved prompt tokenization and next-token behavior.

### 4.3 Linear-attention projection layout

One of the most important correctness bugs came from the conversion of linear-attention projection weights. The Hugging Face source layout for key projection tensors was not equivalent to the runtime’s assumed contiguous row order. This caused the runtime to look superficially alive while still producing bad generations.

The fix was to reorder:

- `in_proj_qkvz.weight`
- `in_proj_ba.weight`

from source grouped/interleaved layout into the runtime layout expected by Flash-MoE.

### 4.4 RMSNorm semantics

A later major issue was a runtime-side mismatch in Qwen3-Next normalization behavior. Even after conversion fixes, the model could still produce coherent-looking but wrong outputs until the Qwen3-Next RMSNorm behavior in the runtime was corrected.

### 4.5 Chat client overhead and rendering bugs

The chat client originally always sent a `bash` tool schema on every request. That forced the server into the heavier tool-capable prompt path even for simple prompts like `hi`, inflating first-turn prompt length and TTFT. The client also used a stateful ANSI markdown renderer that could emit corrupted output when streaming partial UTF-8 or formatting markers.

These were fixed by:

- making tools opt-in
- defaulting the client to plain UTF-8-safe output
- retaining markdown rendering only as an explicit option

## 5. Instrumentation and Validation

The recovery work depended on significantly better observability.

### 5.1 Runtime dump controls

The runtime was extended to dump selected layers and selected stages rather than large undifferentiated dumps. This made it practical to compare:

- embeddings
- attention inputs/outputs
- router logits
- shared-expert intermediates
- layer outputs
- final logits

### 5.2 Reference comparison tooling

`reference_compare.py` was extended into a first-failure oracle. It now reports:

- earliest failing stage per layer
- max absolute error
- mean absolute error
- RMSE
- top-k routing overlap
- final-logit argmax agreement

This changed debugging from guesswork into a bounded search problem.

### 5.3 Mixed-precision bring-up

For early debugging, sensitive MoE tensors in the first layers were kept at higher precision. This reduced early-layer drift and confirmed that some of the initial instability came from the interaction between quantization-sensitive paths and still-incorrect runtime math.

## 6. What Was Achieved

The following milestones were completed.

### 6.1 Structural port

`Qwen3-Coder-Next` can now be:

- inspected correctly
- mapped into runtime tensor categories
- converted into Flash-MoE non-expert assets
- repacked into streamed expert layer files

### 6.2 End-to-end runtime bring-up

The model now:

- loads successfully in the Metal runtime
- executes end-to-end without crashing
- produces direct prompt completions
- supports dump/compare validation

### 6.3 Correctness recovery

The main correctness blockers identified during the port were fixed:

- tokenizer/decode table mismatch
- merge export bug
- prompt-format mismatch
- linear-attention projection row ordering
- Qwen3-Next RMSNorm semantics

After these fixes, sampled runtime traces matched the reference model on tested steps, including exact final-logit parity on traced prefill and generated-token comparisons.

### 6.4 Chat support

The HTTP server path is working, and the local `chat` client can:

- maintain sessions
- send plain chat requests
- stream assistant output
- resume saved sessions

### 6.5 Tool-calling support

The local chat client can now:

- receive a model-emitted tool call
- parse the tool request
- prompt the user for confirmation
- execute the local shell command
- feed the tool response back to the model
- continue the assistant turn coherently

This is a strong qualitative milestone because tool use is more sensitive to formatting and control-token behavior than plain text generation.

## 7. Experimental Findings

Several practical lessons emerged during the port.

### 7.1 Numerical parity matters more than "plausible output"

At multiple points the runtime produced text that looked alive but was still wrong. Useful debugging only began once layer- and stage-level comparisons were added.

### 7.2 Tokenizer parity is not optional

Incorrect decode tables or merge handling can make a runtime appear badly broken even when most model math is correct.

### 7.3 Chat-template parity strongly affects apparent quality

A correct model executed under the wrong prompt template can still look unusable. Prompt construction had to be aligned with Qwen conventions before generation behavior became interpretable.

### 7.4 Chat UX can hide system problems

The original chat client made ordinary requests pay the cost of the tool path. This was not a model-quality problem; it was a client protocol problem. Fixing client defaults immediately improved perceived responsiveness.

## 8. Current Status

The project has crossed the most important threshold: it is no longer a broken experimental port. It is now a working Flash-MoE-style runtime path for `Qwen3-Coder-Next`.

Current state:

- conversion pipeline: working
- expert repack pipeline: working
- runtime loading: working
- direct prompt inference: working
- chat server: working
- local chat client: working
- tool-calling loop: working
- major numerical correctness blockers: fixed

## 9. Remaining Work

Although the core port works, two areas remain open.

### 9.1 Performance recovery

Some correctness-first choices are slower than the desired production target. In particular, the `lm_head` path and portions of serve-mode latency still need optimization without regressing parity.

### 9.2 Broader regression coverage

The port should still be hardened with a wider validation suite covering:

- longer plain-text generations
- structured JSON outputs
- repeated tool calls
- longer multi-turn sessions
- edge-case rendering and stop-token behavior

## 10. Conclusion

This work demonstrates that `Qwen3-Coder-Next` can be adapted into the Flash-MoE execution style with working chat and tool use. The hardest part of the effort was not simple file conversion but correctness recovery across tokenizer export, prompt formatting, projection layout, normalization semantics, and client/server behavior. Once the right instrumentation existed, the port became tractable.

The resulting system is best described as a successful functional port with working interaction paths and remaining optimization work, rather than an unfinished bring-up. The next phase is no longer "make it run." The next phase is "make it fast and harden it."

## 11. Suggested Paper Extensions

If this draft is expanded into a full paper or public technical note, the next useful additions would be:

- exact benchmark tables before and after each major fix
- trace visualizations of first-failure movement during recovery
- a tokenizer parity appendix
- a section comparing tool-calling reliability before and after the normalization and projection fixes
- a production-readiness checklist for future model ports into Flash-MoE
