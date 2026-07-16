# wwama Backend Sprint Plan for a Rust Bankai Port

## Objective

Extend `wwama` with the smallest durable backend capability needed for a future
Rust crate named `miyagi`, which will port the behavior of Python `bankai` and
use `wwama` for model execution. The immediate target is sparse XOR patching of
true binary `Q1_0` model weights, while preserving `wwama`'s existing text,
embedding, native, and WebAssembly responsibilities.

The plan deliberately puts tensor-access validation before public API design.
The current `llama.cpp` integration has enough low-level primitives to make the
feature plausible, but it does not expose model tensor lookup through the
public `llama.h` API.

## Evidence Baseline

The following observations were verified in the checked-out sources:

| Area | Evidence | Consequence |
| --- | --- | --- |
| Bankai backend contract | [`bankai/backends/base.py`](../../python/bankai/bankai/backends/base.py) defines loading, layer/row discovery, row-scale retrieval, tokenization, logit-gap evaluation, row flipping, and optional generation. | `miyagi` needs these behaviors; `wwama` must provide runtime primitives without owning search policy. |
| Bankai search and patches | [`bankai/search.py`](../../python/bankai/bankai/search.py) pre-tokenizes probes, evaluates logit gaps, samples rows using scale magnitudes, screens candidates, and reverts with XOR. [`bankai/patch.py`](../../python/bankai/bankai/patch.py) stores `(layer, projection, row)` records. | `wwama` should expose reversible row mutation and row metadata; search, patch serialization, and probe policy belong in `miyagi`. |
| Existing `wwama` runtime | [`wwama/src/lib.rs`](src/lib.rs) already provides `Session`, tokenization, detokenization, generation/streaming, embeddings, chat templates, and raw FFI. | Most Bankai runtime work can build on existing session primitives. |
| Missing logits readout | `Session::evaluate_tokens` is private, `Context` exposes decode but no logits accessor, and `raw` does not bind `llama_get_logits_ith`. | `wwama` needs deterministic prompt evaluation and selected-logit readout so `miyagi` does not duplicate batch, cache-reset, and synchronization logic. |
| Missing tensor access | `wwama::Model` exposes vocabulary and model metadata, but no tensor enumeration, name lookup, tensor descriptor, read, or write methods. | A new capability is required before `miyagi` can inspect or modify weights. |
| `llama.cpp` model lookup | [`llama-model.h`](../../cpp/llama.cpp/src/llama-model.h) declares the internal `llama_internal_get_tensor_map`, while public [`llama.h`](../../cpp/llama.cpp/include/llama.h) does not expose equivalent model-tensor access. | The default path is a narrow C/C++ bridge built and owned by `wwama`; changing `llama.cpp` is an escalation only if that bridge fails validation. |
| Device-resident tensor transfer | [`ggml-backend.h`](../../cpp/llama.cpp/ggml/include/ggml-backend.h) exposes `ggml_backend_tensor_get` and `ggml_backend_tensor_set`. | A host-side read-modify-write path may support CPU and GPU-resident tensors without exposing raw device pointers. |
| Binary quantization layout | [`ggml-common.h`](../../cpp/llama.cpp/ggml/src/ggml-common.h) defines `QK1_0 = 128` and `block_q1_0` as a 2-byte `d` scale plus 16 packed-bit bytes. [`ggml.h`](../../cpp/llama.cpp/ggml/include/ggml.h) exposes tensor shape and byte strides. | A row XOR must modify only packed `qs` bytes, preserve scales, and honor tensor strides and orientation. |
| Current Rust FFI coverage | `wwama::raw::ggml_type` includes `Tq1_0` but not `Q1_0`; there are no raw bindings for `ggml_tensor`, tensor metadata helpers, backend tensor transfer, or the internal model tensor map. | FFI additions and safety checks are necessary; hard-coding model dimensions would be brittle. |
| Build surface | [`wwama/build.rs`](build.rs) builds static `llama`, `ggml`, and optional backend libraries, while excluding llama.cpp common/tools/server targets. | Any bridge must fit the existing static-link/build-feature model and must not pull in excluded application targets. |
| Bankai GGUF status | [`bankai/backends/gguf_backend.py`](../../python/bankai/bankai/backends/gguf_backend.py) leaves both `get_row_scales` and `flip_row` as explicit `NotImplementedError` paths. | The Rust port must validate these operations against the actual GGUF tensor layout rather than treating the Python TODOs as a specification. |
| Repository state | `miyagi` currently contains only an empty [`README.md`](../../rust/miyagi/README.md). `wwama` has a narrow unit-test module and no model-backed tensor integration tests. | The first implementation should establish a small fixture/validation harness before expanding the public API. |

## Scope Boundary

### `wwama` owns

- capability-gated model tensor metadata access;
- backend-aware byte reads and writes for supported tensors;
- a reversible row-level XOR primitive for a supported binary quantization;
- row-scale extraction needed for candidate weighting;
- deterministic prompt evaluation and selected-logit readout for probe gaps;
- model and tensor validation errors;
- tests for byte-level reversibility, row bounds, layout handling, and device
  transfer where the environment supports it.

### `miyagi` will own later

- Bankai-compatible patch serialization;
- probe definitions, fitness calculations, and control penalties;
- candidate sampling, screening, hill climbing, and patch composition policy;
- architecture-specific mapping from `(layer, proj)` to GGUF tensor names;
- CLI and generated-patch metadata;
- behavioral evaluation and generalization experiments.

### Explicit non-goals

- modifying arbitrary floating-point or ternary tensors;
- exposing raw GPU pointers as the primary public API;
- embedding Bankai's search algorithm into `wwama`;
- hard-coding Bonsai/Qwen dimensions into `wwama`;
- promising WebAssembly support for mutable model tensors before runtime
  behavior is demonstrated;
- changing the `llama.cpp` source tree during the initial implementation;
- changing the upstream remote or canonical monorepo integration.

## Sprint Sequence

### Sprint 0: Freeze the capability contract

**Work**

- Define the supported operation in terms of tensor name, row index, row
  metadata, and reversible XOR mutation.
- Decide whether the first public surface is a generic tensor capability with a
  `Q1_0` implementation, or a narrower Bankai-specific capability. Prefer the
  generic capability if it does not leak unstable `llama.cpp` internals.
- Define feature/target policy: native CPU first, native GPU where transfer
  semantics pass validation, and explicit treatment of wasm targets.
- Define errors for unknown tensors, unsupported types, non-matrix shapes,
  invalid rows, unsupported strides, and unavailable device transfer.

**Exit criteria**

- A short API contract exists in Rust doc comments or design tests.
- Supported tensor layouts and the target matrix are explicit.
- The local C/C++ bridge is selected as the default implementation path.
- Any need for a public `llama.cpp` API is recorded as an explicit escalation
  backed by a reproducible bridge failure.

### Sprint 1: Build a tensor inventory probe

**Work**

- Add the minimum raw bindings or bridge needed to enumerate model tensor names,
  type, dimensions, strides, and byte size.
- Inspect a representative GGUF model used by Bankai without hard-coding its
  dimensions into `wwama`.
- Confirm MLP projection naming and orientation, including how a logical output
  row maps to `ne`, `nb`, and byte offsets.
- Record which tensors are CPU, CUDA, Metal, Vulkan, or other backend-resident
  under the current `SessionOptions`.

**Exit criteria**

- A repeatable inventory command/test identifies candidate MLP tensors and
  reports their exact layout.
- Tensor lookup works against the checked-out `llama.cpp` commit.
- No mutation is attempted until the row mapping is verified.

### Sprint 2: Add deterministic probe evaluation

**Work**

- Bind the required logits accessor and expose a controlled way to evaluate a
  tokenized prompt with the final position marked for logits.
- Reuse `Session` cache reset, batching, decoder/encoder selection, and
  synchronization rules instead of making `miyagi` reconstruct them.
- Add a logit-gap helper or equivalent logits view with clear token-position
  semantics. Keep probe definitions and fitness calculations in `miyagi`.
- Define behavior for empty prompts, multi-token correct/wrong strings, context
  overflow, and models that do not produce decoder logits.

**Exit criteria**

- A deterministic prompt can return selected logits through safe high-level
  `wwama` methods.
- Repeated evaluation after a mutation uses a clean context and does not retain
  rejected-candidate state.
- Tests cover final-position selection, token IDs, context reset, and expected
  error cases.

### Sprint 3: Implement backend-aware tensor byte access

**Work**

- Add a narrow transfer abstraction around `ggml_backend_tensor_get/set`.
- Make synchronization and cache/device visibility explicit; callers should not
  need to know whether `tensor->data` is host-accessible.
- Keep raw pointers and FFI types behind `raw` or a private implementation
  boundary.
- Verify build/link behavior for the existing native feature matrix and avoid
  regressions to CPU, CUDA, Vulkan, and wasm compilation.

**Exit criteria**

- A tensor can be read and written through the wrapper on the supported native
  path.
- A read-write-read test proves exact byte preservation when no mutation occurs.
- Unsupported targets return a deliberate error.

### Sprint 4: Add Q1_0 row metadata and XOR mutation

**Work**

- Add the missing `Q1_0` type representation and validate block size, block
  width, row size, and tensor shape at runtime.
- Implement row-scale extraction from the `d` fields without changing scales.
  Define whether the returned value is per-block, mean absolute scale, or
  another documented aggregate; match Bankai's scale-guided intent.
- Implement row XOR by flipping only packed `qs` bytes in every block of the
  selected row. Preserve padding and scale bytes.
- Handle tensor orientation and stride from the descriptor rather than assuming
  row-major contiguous storage.
- Make the same call twice restore the original bytes and model behavior.

**Exit criteria**

- Tests cover one-block and multi-block rows, first/middle/last rows, invalid
  rows, unsupported types, and unsupported strides.
- A byte-level test proves scales are unchanged and every packed bit in the
  selected row is inverted.
- A model-backed test proves `xor_row(row); xor_row(row)` restores deterministic
  logits or generated output.

### Sprint 5: Expose the Miyagi-facing capability

**Work**

- Add a stable high-level `wwama` API for deterministic probe evaluation,
  listing/querying supported model tensors, retrieving row scales, and
  applying/reverting row XOR operations.
- Return owned metadata instead of exposing borrowed internal `ggml_tensor`
  layouts to ordinary callers.
- Define mutation lifetime and concurrency rules. A session with mutable model
  weights must not allow concurrent inference or unsynchronized mutation.
- Decide whether the capability is opt-in behind a feature flag and document
  its native-only or wasm-supported status.

**Exit criteria**

- An example or integration test performs load -> identify tensor -> read row
  metadata -> flip -> evaluate -> flip again -> evaluate.
- The public API has no model-specific Bonsai dimensions or tensor names.
- Safety and synchronization requirements are documented next to the API.

### Sprint 6: Validate the Bankai runtime contract

**Work**

- Build a temporary or test-only Miyagi adapter implementing Bankai's backend
  contract with `wwama::Session`.
- Reproduce tokenization, final-token selection, deterministic logit-gap
  measurement, scale-weighted candidate enumeration, row flip, and revert.
- Compare baseline and twice-reverted outputs to catch context-cache or backend
  synchronization errors.
- Test CPU first, then each available native accelerator. Treat each backend as
  a separate capability result.

**Exit criteria**

- The adapter executes one complete candidate trial without copying the entire
  model.
- The candidate operation is reversible and repeatable on supported backends.
- Unsupported backends are clearly reported and excluded from Miyagi's matrix.

### Sprint 7: Harden and hand off to Miyagi

**Work**

- Add regression tests for tensor naming, model reload, repeated mutation,
  context reset, and failure recovery after a rejected candidate.
- Document Miyagi integration points and patch/search ownership.
- Measure transfer and mutation costs so Bankai's "in-place" and "microsecond"
  claims are not repeated for a host/device copy path without evidence.
- Review for accidental exposure of `llama.cpp` internals and unnecessary
  compatibility shims.

**Exit criteria**

- `wwama` has a tested, documented mutation capability suitable for Miyagi.
- Remaining work is in Miyagi rather than an unverified wwama blocker.
- This plan is updated with actual supported model formats and backend results.

## Decision Gates and Risks

### Tensor lookup implementation

The public C API does not expose model tensors, but this fork has an internal
tensor map. The preferred implementation is a narrow bridge compiled as part
of `wwama`; it should return stable metadata and perform validated operations
without modifying the `llama.cpp` source tree. If the bridge cannot be linked,
cannot support the required native backends, or cannot remain stable across the
supported fork, stop and document the failure before considering a deliberate
public `llama.cpp` API. Do not bind safe Rust code directly to private C++
object layout.

### GPU mutation semantics

`ggml_backend_tensor_get/set` can transfer tensor bytes, but that does not prove
that mutation is cheap or safe during active inference. Serialize mutation with
evaluation, synchronize before and after transfer, and measure the actual path.

### Quantization compatibility

`Q1_0` is a true binary representation only if the packed-bit interpretation is
the one expected by the target model and kernel. The row XOR test must validate
the dequantized sign change or an equivalent model-level effect. Do not extend
the operation to `TQ1_0` or ternary formats merely because their names contain
"1".

### Model architecture mapping

Bankai's `(layer, proj, row)` records are logical coordinates. `wwama` should
not encode Qwen/Bonsai tensor names. Miyagi must resolve architecture-specific
names after inspecting `wwama`'s generic tensor inventory.

### WebAssembly

The crate supports wasm32/wasm64 builds, but mutable model tensor access has
separate Emscripten/WebGPU memory semantics. Keep it unavailable or experimental
until a real wasm model fixture proves read/write behavior. Compile-only support
is insufficient.

## Verification Matrix

| Layer | Required verification |
| --- | --- |
| Pure Rust | Descriptor validation, row bounds, checked arithmetic, error mapping, and operation idempotence. |
| FFI/bridge | Symbol/link validation, tensor enumeration, type/shape/stride reporting, and no direct C++ layout dependence in safe code. |
| Probe runtime | Deterministic tokenization, final-position logits, logit-gap calculation, context reset, and multi-token probe handling. |
| CPU model | Read/write round trip, Q1_0 row mutation, scale preservation, deterministic logit-gap change, and double-XOR restoration. |
| Native accelerator | Device transfer round trip, synchronization, serialized mutation, and cost measurement for each enabled backend. |
| WebAssembly | Compile checks remain green; runtime mutation is tested with a fixture or explicitly unsupported. |
| Miyagi handoff | Temporary adapter covers every Bankai backend method supported by the port. |

## First Implementation Slice

1. Add native tensor inventory/descriptor support through a `wwama`-owned
   bridge, without changing `llama.cpp` sources.
2. Add deterministic final-position logits evaluation.
3. Add a Q1_0-only row read-modify-write operation with explicit validation.
4. Add byte-level and model-backed reversibility tests.
5. Only then settle final public names and let Miyagi implement patch/search
   orchestration.

This order keeps the highest-risk assumptions observable and prevents a large
Bankai port from hiding an unresolved tensor-access or device-synchronization
problem.

## Implementation Results

The first implementation slice and the planned validation gates are complete
in this workspace.

| Capability | Result | Evidence |
| --- | --- | --- |
| llama.cpp source changes | Not required | `src/bridge/wwama_tensor_bridge.cpp` uses the checked-out internal tensor map plus public GGML transfer APIs; no file under `libs/cpp/llama.cpp` changed. |
| Native tensor inventory | Passed | The Bonsai fixture reports 399 tensors, including `blk.0.ffn_gate.weight`, `ffn_up`, and `ffn_down` as Q1_0 matrices. |
| Bankai row orientation | Confirmed | Gate/up are `[4096, 12288]` with strides `[18, 576]`; down is `[12288, 4096]` with strides `[18, 1728]`. The logical row is GGML dimension 1. |
| Deterministic selected logits | Passed | `Session::evaluate_selected_logits` clears memory, marks only the final position, synchronizes, and repeated evaluations match exactly. |
| CPU Q1_0 mutation | Passed | Bonsai 8B row mutation preserves scales, changes packed bytes, and double XOR restores bytes and logits. |
| CUDA Q1_0 mutation | Passed | Bonsai 8B with all 37 layers on an RTX 4050 passed the same transfer, scale, mutation, and restoration assertions. |
| Vulkan Q1_0 mutation | Passed | Bonsai 8B with all 37 layers on Vulkan0 on the same RTX 4050 passed the same assertions. |
| wasm32 build | Passed | CPU-only `wasm32-unknown-unknown` compilation passes; the bridge is excluded from wasm builds. |
| wasm tensor mutation runtime | Deliberately unsupported | `UnsupportedTarget` is returned until a model-backed Emscripten/WebGPU fixture proves read/write visibility and synchronization. |

The public mutation path is opt-in through `SessionOptions::mutable_tensors`.
It disables read-only mmap because attempting `ggml_backend_tensor_set` on a
`CPU_Mapped` model caused a reproducible SIGSEGV; mutable loading was then
validated with `mmap = false` on CPU, CUDA, and Vulkan. `&mut Session` and
explicit context synchronization serialize inference and mutation. Row XOR
copies only one row through the backend, not the whole model, but it is not a
zero-copy operation on device-resident tensors.

The temporary Bankai-facing adapter is in `examples/miyagi_backend.rs`. It
accepts an external `(layer, projection) -> tensor name` mapping, so
architecture-specific Bonsai/Qwen naming remains outside `wwama` as planned.
