# wwama Backend Sprint Progress

## Current State

- A task-specific `jj` workspace named `wwama-bankai-plan` was created from
  `libs/rust/wwama` at the current `master` parent.
- The workspace is isolated at `libs/rust/wwama-bankai`.
- No source code has been changed yet.
- The sprint plan is saved in
  [`WWAMA-MIYAGI-BANKAI-SPRINT-PLAN.md`](WWAMA-MIYAGI-BANKAI-SPRINT-PLAN.md).

## Evidence Collected

- Bankai's backend abstraction requires row discovery, row-scale retrieval,
  tokenization, logit-gap measurement, row flipping, and optional generation.
- `wwama` already provides session loading, tokenization, generation, streaming,
  embeddings, and raw llama.cpp bindings.
- `wwama` does not currently expose model tensor enumeration, tensor metadata,
  tensor byte transfer, or weight mutation.
- The checked-out llama.cpp fork has an internal model tensor map, public GGML
  tensor metadata, backend-aware tensor get/set operations, and a `Q1_0` block
  layout suitable for validating row-level packed-bit XOR.
- `wwama` currently lacks a public logits accessor and probe-oriented evaluation
  helper, so deterministic logit-gap measurement is also a required backend
  capability.
- The preferred tensor-access path is a `wwama`-owned C/C++ bridge using the
  existing llama.cpp internals and GGML transfer APIs; llama.cpp source changes
  are an escalation only if bridge validation fails.
- `miyagi` is currently only an empty README placeholder.

## Implemented Capability

- Added the native `wwama` tensor bridge without changing the llama.cpp source
  tree. It enumerates owned tensor descriptors and performs backend-aware byte
  reads/writes through `ggml_backend_tensor_get/set`.
- Added `Model::tensors`, `Model::tensor`, `Session::read_tensor_range`, and
  `Session::write_tensor_range` with owned metadata and deliberate error cases.
- Added `Session::evaluate_selected_logits` and `Session::logit_gap` with
  context reset, final-position output selection, token validation, context
  overflow handling, and synchronization.
- Added opt-in mutable loading via `SessionOptions::mutable_tensors`; ordinary
  mmap-backed sessions reject writes instead of risking a native fault.
- Added Q1_0 descriptor validation, FP16 scale aggregation, row-scale access,
  stride-aware row XOR, and `RowXorResult` reporting.
- Added pure layout/idempotence tests, fixture-gated model tests, tensor
  inventory and adapter examples, and native feature validation.
- Fixed the wasm CMake invocation to disable the excluded llama.cpp app target;
  CPU-only wasm32 compilation now passes.

## Validation Results

- `cargo test --no-default-features --all-targets`: passed.
- Bonsai 8B Q1_0 CPU model test: passed with 1.07 GiB fixture, including row
  scales, packed-byte mutation, double-XOR restoration, and logits restoration.
- Bonsai 8B Q1_0 CUDA test: passed with all 37 layers on NVIDIA GeForce RTX 4050.
- Bonsai 8B Q1_0 Vulkan test: passed with all 37 layers on Vulkan0 on the same GPU.
- `WWAMA_EMSDK=/home/bitnom/emsdk cargo check --no-default-features --target
  wasm32-unknown-unknown`: passed. Mutable tensor runtime remains unsupported.

## Handoff State

The wwama blocker for Miyagi is resolved for native CPU, CUDA, and Vulkan
paths. Miyagi still needs to own architecture mapping, Bankai patch format,
probe/search policy, and behavioral evaluation. No llama.cpp source change is
currently justified.

## Baseline Verification

- `cargo fmt --check` passed.
- `cargo test --no-default-features` passed: 2 unit tests, 0 failures.

## Next Work Item

Implement the Miyagi crate against the owned wwama capability. Keep WebAssembly
mutation behind an explicit capability check until a runtime fixture exists.
