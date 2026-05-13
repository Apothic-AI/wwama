# wwama

`wwama` is a small Rust wrapper crate around a local `llama.cpp` checkout. It is intended to make the `llama.cpp` C API available to Rust code on native targets and browser-oriented WebAssembly targets, including both `wasm32-unknown-unknown` and `wasm64-unknown-unknown`.

The crate currently focuses on the lower-level pieces needed for embedding inference work:

- building `llama.cpp` as static libraries from Cargo;
- exposing raw FFI bindings for the relevant `llama.h` APIs;
- providing lightweight RAII wrappers for backend, model, context, and batch lifetimes;
- supporting WebGPU-enabled Emscripten builds for WebAssembly;
- avoiding `llama.cpp` examples, tools, and server targets.

## Status

This is an early integration crate. It is useful as a buildable wrapper layer, but it is not yet a high-level embedding API. Callers are still responsible for model selection, tokenization/batching strategy, pooling configuration, and copying embedding vectors out of the returned `llama.cpp` buffers.

## Repository Layout

By default, `wwama` expects this layout:

```text
wwama-projects/
  llama.cpp/
  wwama/
```

The default `build.rs` lookup uses `../llama.cpp`. Override it with:

```sh
WWAMA_LLAMA_CPP_DIR=/path/to/llama.cpp cargo build
```

## Build Requirements

Native builds require:

- Rust 1.95 or newer;
- CMake;
- a C++ toolchain compatible with `llama.cpp`.

WebAssembly builds require:

- the `wasm32-unknown-unknown` Rust target for wasm32 builds;
- `rust-src` plus `cargo -Zbuild-std=core` for wasm64 builds;
- Emscripten SDK for building the `llama.cpp` C/C++ libraries;
- EMDawnWebGPU when building with the default `webgpu` feature.

The build script searches for Emscripten in this order:

- `WWAMA_EMSCRIPTEN_TOOLCHAIN_FILE`;
- `WWAMA_EMSDK`;
- `EMSDK`;
- `$HOME/emsdk`;
- `$HOME/Code/emsdk`.

The build script searches for EMDawnWebGPU using `WWAMA_EMDAWNWEBGPU_DIR` first. Set it explicitly when building outside this workstation.

## Build Commands

Native:

```sh
cargo build
```

WASM32 with WebGPU, using the default features:

```sh
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown
```

WASM64 with WebGPU:

```sh
rustup component add rust-src
RUSTC_BOOTSTRAP=1 cargo build -Zbuild-std=core --target wasm64-unknown-unknown
```

The wasm64 target currently needs `build-std` because the toolchain does not ship prebuilt `core` for `wasm64-unknown-unknown`.

CPU-only WASM builds are available by disabling default features:

```sh
cargo build --no-default-features --target wasm32-unknown-unknown
RUSTC_BOOTSTRAP=1 cargo build -Zbuild-std=core --no-default-features --target wasm64-unknown-unknown
```

## Features

- `webgpu` is enabled by default. For WebAssembly targets, it enables `GGML_WEBGPU=ON` in the `llama.cpp` CMake build.
- Native builds currently compile the CPU path even when the feature is enabled.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
