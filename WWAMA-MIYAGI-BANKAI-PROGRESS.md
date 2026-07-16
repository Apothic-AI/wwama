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

## Baseline Verification

- `cargo fmt --check` passed.
- `cargo test --no-default-features` passed: 2 unit tests, 0 failures.

## Next Work Item

Execute Sprint 0 and Sprint 1 from the plan: verify the bridge approach and
produce a model tensor inventory before adding mutation APIs.
