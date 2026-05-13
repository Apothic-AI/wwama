use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=WWAMA_LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=WWAMA_EMSCRIPTEN_TOOLCHAIN_FILE");
    println!("cargo:rerun-if-env-changed=WWAMA_EMSDK");
    println!("cargo:rerun-if-env-changed=WWAMA_EMDAWNWEBGPU_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    let llama_dir = find_llama_cpp_dir();
    println!("cargo:rerun-if-changed={}", llama_dir.display());

    let target = env::var("TARGET").expect("TARGET is set by cargo");
    let target_arch =
        env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH is set by cargo");
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS is set by cargo");
    let enable_webgpu = env::var_os("CARGO_FEATURE_WEBGPU").is_some();

    let dst = if target.starts_with("wasm32") || target.starts_with("wasm64") {
        build_wasm(&llama_dir, &target_arch, enable_webgpu)
    } else {
        let mut cfg = cmake::Config::new(&llama_dir);
        cfg.profile("Release");
        cfg.define("BUILD_SHARED_LIBS", "OFF");
        cfg.define("LLAMA_BUILD_COMMON", "OFF");
        cfg.define("LLAMA_BUILD_TESTS", "OFF");
        cfg.define("LLAMA_BUILD_TOOLS", "OFF");
        cfg.define("LLAMA_BUILD_EXAMPLES", "OFF");
        cfg.define("LLAMA_BUILD_SERVER", "OFF");
        cfg.define("LLAMA_BUILD_WEBUI", "OFF");
        cfg.define("GGML_WEBGPU", "OFF");
        cfg.build()
    };
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
    println!("cargo:rustc-link-lib=static=ggml-base");

    if enable_webgpu && (target.starts_with("wasm32") || target.starts_with("wasm64")) {
        println!("cargo:rustc-link-lib=static=ggml-webgpu");
    }

    match target_os.as_str() {
        "linux" | "android" | "freebsd" | "openbsd" | "netbsd" => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=pthread");
        }
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=dylib=c++");
        }
        _ => {}
    }
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
