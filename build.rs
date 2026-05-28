use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=WWAMA_LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=WWAMA_EMSCRIPTEN_TOOLCHAIN_FILE");
    println!("cargo:rerun-if-env-changed=WWAMA_EMSDK");
    println!("cargo:rerun-if-env-changed=WWAMA_EMDAWNWEBGPU_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDAToolkit_ROOT");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=WWAMA_CMAKE_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=WWAMA_CUDA_HOST_COMPILER");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-changed=build.rs");

    let llama_dir = find_llama_cpp_dir();
    println!("cargo:rerun-if-changed={}", llama_dir.display());

    let target = env::var("TARGET").expect("TARGET is set by cargo");
    let target_arch =
        env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH is set by cargo");
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS is set by cargo");
    let enable_webgpu = env::var_os("CARGO_FEATURE_WEBGPU").is_some();
    let requested_cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let requested_vulkan = env::var_os("CARGO_FEATURE_VULKAN").is_some();

    let is_wasm = target.starts_with("wasm32") || target.starts_with("wasm64");
    if is_wasm && requested_cuda {
        panic!("wwama feature `cuda` is only supported for native targets");
    }
    if is_wasm && requested_vulkan {
        panic!("wwama feature `vulkan` is only supported for native targets");
    }
    let enable_cuda = requested_cuda && !is_wasm;
    let enable_vulkan = requested_vulkan && !is_wasm;
    let dst = if is_wasm {
        build_wasm(&llama_dir, &target_arch, enable_webgpu)
    } else {
        let mut cfg = cmake::Config::new(&llama_dir);
        cfg.profile("Release");
        cfg.define("BUILD_SHARED_LIBS", "OFF");
        cfg.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
        cfg.define("LLAMA_BUILD_COMMON", "OFF");
        cfg.define("LLAMA_BUILD_TESTS", "OFF");
        cfg.define("LLAMA_BUILD_TOOLS", "OFF");
        cfg.define("LLAMA_BUILD_EXAMPLES", "OFF");
        cfg.define("LLAMA_BUILD_SERVER", "OFF");
        cfg.define("LLAMA_BUILD_WEBUI", "OFF");
        cfg.define("GGML_WEBGPU", "OFF");
        cfg.define("GGML_CUDA", if enable_cuda { "ON" } else { "OFF" });
        cfg.define("GGML_VULKAN", if enable_vulkan { "ON" } else { "OFF" });
        if let Some(archs) = env::var_os("WWAMA_CMAKE_CUDA_ARCHITECTURES") {
            cfg.define("CMAKE_CUDA_ARCHITECTURES", archs);
        }
        if enable_cuda {
            cfg.define("CMAKE_CUDA_FLAGS", "-Xcompiler=-fPIC");
            if let Some(compiler) = cuda_compiler() {
                cfg.define("CMAKE_CUDA_COMPILER", compiler);
            }
            if let Some(host_compiler) = cuda_host_compiler() {
                cfg.define("CMAKE_CUDA_HOST_COMPILER", host_compiler);
            }
        }
        cfg.build()
    };
    let native_openmp_enabled = !is_wasm && cmake_cache_bool(&dst, "GGML_OPENMP_ENABLED");
    let native_cuda_enabled = !is_wasm && cmake_cache_bool(&dst, "GGML_CUDA");
    let native_vulkan_enabled = !is_wasm && cmake_cache_bool(&dst, "GGML_VULKAN");
    let lib_dir = dst.join("lib");
    assert!(
        lib_dir.exists(),
        "expected cmake install lib dir at {}",
        lib_dir.display()
    );

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    if native_cuda_enabled {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    if native_vulkan_enabled {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
    }
    println!("cargo:rustc-link-lib=static=ggml-base");

    if enable_webgpu && (target.starts_with("wasm32") || target.starts_with("wasm64")) {
        println!("cargo:rustc-link-lib=static=ggml-webgpu");
    }

    match target_os.as_str() {
        "linux" | "freebsd" | "openbsd" | "netbsd" => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=pthread");
            if native_openmp_enabled {
                println!("cargo:rustc-link-lib=dylib=gomp");
            }
            if native_cuda_enabled {
                emit_cuda_link_flags(&dst);
            }
            if native_vulkan_enabled {
                println!("cargo:rustc-link-lib=dylib=vulkan");
            }
        }
        "android" => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=log");
            if native_openmp_enabled {
                if let Some(dir) = android_openmp_lib_dir(&target_arch) {
                    println!("cargo:rustc-link-search=native={}", dir.display());
                }
                println!("cargo:rustc-link-lib=dylib=omp");
            }
            if native_vulkan_enabled {
                println!("cargo:rustc-link-lib=dylib=vulkan");
            }
        }
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=dylib=c++");
            if native_openmp_enabled {
                println!("cargo:rustc-link-lib=dylib=omp");
            }
        }
        _ => {}
    }
}

fn cmake_cache_bool(install_dir: &Path, key: &str) -> bool {
    let cache_path = install_dir.join("build/CMakeCache.txt");
    let Ok(cache) = std::fs::read_to_string(cache_path) else {
        return false;
    };
    let prefix = format!("{key}:");
    cache.lines().any(|line| {
        line.starts_with(&prefix)
            && line
                .split_once('=')
                .is_some_and(|(_, value)| matches!(value, "ON" | "TRUE" | "1"))
    })
}

fn android_openmp_lib_dir(target_arch: &str) -> Option<PathBuf> {
    let ndk_root = env::var_os("ANDROID_NDK_ROOT")
        .or_else(|| env::var_os("ANDROID_NDK"))
        .map(PathBuf::from)?;
    let clang_root = ndk_root.join("toolchains/llvm/prebuilt/linux-x86_64/lib/clang");
    let arch_dir = match target_arch {
        "aarch64" => "aarch64",
        "arm" => "arm",
        "x86" => "i386",
        "x86_64" => "x86_64",
        _ => return None,
    };
    let mut versions = std::fs::read_dir(clang_root)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join("lib/linux").exists())
        .collect::<Vec<_>>();
    versions.sort();
    versions
        .into_iter()
        .rev()
        .map(|path| path.join("lib/linux").join(arch_dir))
        .find(|path| path.join("libomp.so").exists() || path.join("libomp.a").exists())
}

fn emit_cuda_link_flags(install_dir: &Path) {
    if let Some(dir) = cuda_library_dir(install_dir) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    // llama.cpp links the CUDA backend against the static CUDA runtime and
    // cuBLAS libraries on Unix static builds. These transitive native
    // dependencies are not carried through Rust's direct static archive links.
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=static=cublas_static");
    println!("cargo:rustc-link-lib=static=cublasLt_static");
    println!("cargo:rustc-link-lib=static=culibos");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=pthread");
}

fn cuda_library_dir(install_dir: &Path) -> Option<PathBuf> {
    if let Some(dir) = cmake_cache_path(install_dir, "CUDAToolkit_LIBRARY_DIR") {
        return Some(dir);
    }
    for env_key in ["CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = env::var_os(env_key).map(PathBuf::from) {
            for candidate in [root.join("lib64"), root.join("lib"), root.join("lib/x64")] {
                if candidate.exists() {
                    return Some(candidate);
                }
            }
            for candidate in [
                root.join("targets/x86_64-linux/lib"),
                root.join("targets/sbsa-linux/lib"),
                root.join("targets/aarch64-linux/lib"),
            ] {
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
    }
    for candidate in ["/usr/local/cuda/lib64", "/usr/local/cuda/lib"] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn cuda_compiler() -> Option<PathBuf> {
    for env_key in ["CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = env::var_os(env_key).map(PathBuf::from) {
            let candidate = root.join("bin/nvcc");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

fn cuda_host_compiler() -> Option<PathBuf> {
    if let Some(path) = env::var_os("WWAMA_CUDA_HOST_COMPILER").map(PathBuf::from) {
        return Some(path);
    }
    for candidate in ["/usr/bin/g++-13", "/usr/bin/g++-12", "/usr/bin/g++"] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn cmake_cache_path(install_dir: &Path, key: &str) -> Option<PathBuf> {
    let cache_path = install_dir.join("build/CMakeCache.txt");
    let cache = std::fs::read_to_string(cache_path).ok()?;
    let prefix = format!("{key}:");
    cache.lines().find_map(|line| {
        if !line.starts_with(&prefix) {
            return None;
        }
        let (_, value) = line.split_once('=')?;
        if value.is_empty() || value.ends_with("-NOTFOUND") {
            return None;
        }
        Some(PathBuf::from(value))
    })
}

fn build_wasm(llama_dir: &Path, target_arch: &str, enable_webgpu: bool) -> PathBuf {
    let toolchain = find_emscripten_toolchain();
    let emscripten_dir = emscripten_dir_from_toolchain(&toolchain);
    let emsdk_root = emsdk_root_from_toolchain(&toolchain);
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let build_dir = out_dir.join("build");
    let install_dir = out_dir;
    if build_dir.exists() {
        std::fs::remove_dir_all(&build_dir).expect("failed to clean wasm build dir");
    }
    std::fs::create_dir_all(&build_dir).expect("failed to create wasm build dir");
    let emsdk_env = emsdk_environment(&emsdk_root);

    let mut configure = Command::new(emscripten_dir.join("emcmake"));
    configure.current_dir(&build_dir);
    configure.envs(emsdk_env.iter().cloned());
    configure.arg("cmake");
    configure.arg(llama_dir);
    configure.arg("-B");
    configure.arg(&build_dir);
    configure.arg("-DBUILD_SHARED_LIBS=OFF");
    configure.arg("-DLLAMA_BUILD_COMMON=OFF");
    configure.arg("-DLLAMA_BUILD_TESTS=OFF");
    configure.arg("-DLLAMA_BUILD_TOOLS=OFF");
    configure.arg("-DLLAMA_BUILD_EXAMPLES=OFF");
    configure.arg("-DLLAMA_BUILD_SERVER=OFF");
    configure.arg("-DLLAMA_BUILD_WEBUI=OFF");
    configure.arg(format!(
        "-DLLAMA_WASM_MEM64={}",
        if target_arch == "wasm64" { "ON" } else { "OFF" }
    ));
    configure.arg(format!(
        "-DGGML_WEBGPU={}",
        if enable_webgpu { "ON" } else { "OFF" }
    ));
    configure.arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_dir.display()));
    if enable_webgpu {
        if let Some(dir) = find_emdawnwebgpu_dir() {
            configure.arg(format!("-DEMDAWNWEBGPU_DIR={}", dir.display()));
        }
    }
    run(&mut configure, "failed to configure llama.cpp for wasm");

    let mut build = Command::new("cmake");
    build.current_dir(&build_dir);
    build.envs(emsdk_env);
    build.arg("--build");
    build.arg(&build_dir);
    build.arg("--config");
    build.arg("Release");
    build.arg("--target");
    build.arg("install");
    build.arg("-j4");
    run(&mut build, "failed to build llama.cpp for wasm");

    install_dir
}

fn find_llama_cpp_dir() -> PathBuf {
    if let Some(dir) = env::var_os("WWAMA_LLAMA_CPP_DIR") {
        return canonicalize_llama_cpp_dir(PathBuf::from(dir));
    }

    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo"),
    );
    let candidates = [
        manifest_dir.join("../../cpp/llama.cpp"),
        manifest_dir.join("../llama.cpp"),
    ];

    for candidate in &candidates {
        if candidate.join("CMakeLists.txt").exists() && candidate.join("include/llama.h").exists() {
            return canonicalize_llama_cpp_dir(candidate.clone());
        }
    }

    let checked = candidates
        .iter()
        .map(|candidate| candidate.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");

    panic!(
        "failed to locate llama.cpp; set WWAMA_LLAMA_CPP_DIR or use one of these layouts: {checked}"
    );
}

fn canonicalize_llama_cpp_dir(dir: PathBuf) -> PathBuf {
    dir.canonicalize()
        .unwrap_or_else(|err| panic!("failed to resolve llama.cpp dir {}: {err}", dir.display()))
}

fn find_emscripten_toolchain() -> PathBuf {
    if let Some(path) = env::var_os("WWAMA_EMSCRIPTEN_TOOLCHAIN_FILE") {
        return PathBuf::from(path);
    }

    let candidates = [
        env::var_os("WWAMA_EMSDK").map(PathBuf::from),
        env::var_os("EMSDK").map(PathBuf::from),
        home_dir().map(|home| home.join("emsdk")),
        home_dir().map(|home| home.join("Code/emsdk")),
    ];

    for root in candidates.into_iter().flatten() {
        let toolchain = root.join("upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake");
        if toolchain.exists() {
            return toolchain;
        }
    }

    panic!("failed to locate Emscripten.cmake; set WWAMA_EMSCRIPTEN_TOOLCHAIN_FILE or WWAMA_EMSDK");
}

fn emscripten_dir_from_toolchain(toolchain: &Path) -> PathBuf {
    toolchain
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .expect("expected standard emscripten toolchain layout")
}

fn emsdk_root_from_toolchain(toolchain: &Path) -> PathBuf {
    toolchain
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .expect("expected standard emsdk layout")
}

fn find_emdawnwebgpu_dir() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("WWAMA_EMDAWNWEBGPU_DIR") {
        return Some(PathBuf::from(dir));
    }

    let candidates = [
        home_dir().map(|home| home.join("Code/linux-wasm-server/llama.cpp/.deps/emdawnwebgpu_pkg")),
        home_dir().map(|home| {
            home.join(
                "Code/apothic-monorepo/libs/cpp/onnxruntime/build/wasm_inferencing_webgpu/Release/_deps/dawn-src/src/emdawnwebgpu/pkg",
            )
        }),
    ];

    candidates
        .into_iter()
        .flatten()
        .find(|candidate| candidate.join("emdawnwebgpu.port.py").exists())
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from)
}

fn run(cmd: &mut Command, context: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|err| panic!("{context}: {err}"));
    if !status.success() {
        panic!("{context}: command exited with {status}");
    }
}

fn emsdk_environment(emsdk_root: &Path) -> Vec<(OsString, OsString)> {
    let script = emsdk_root.join("emsdk_env.sh");
    let output = Command::new("bash")
        .arg("-lc")
        .arg(format!("source {} >/dev/null && env -0", script.display()))
        .output()
        .unwrap_or_else(|err| panic!("failed to source {}: {err}", script.display()));

    if !output.status.success() {
        panic!("failed to source {}", script.display());
    }

    output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty())
        .filter_map(|entry| {
            let split = entry.iter().position(|byte| *byte == b'=')?;
            Some((
                OsString::from(String::from_utf8_lossy(&entry[..split]).into_owned()),
                OsString::from(String::from_utf8_lossy(&entry[split + 1..]).into_owned()),
            ))
        })
        .collect()
}
