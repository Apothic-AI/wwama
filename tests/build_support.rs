#[path = "../build_support.rs"]
mod build_support;

use std::path::Path;

#[test]
fn prefers_mainline_candidates() {
    let candidates = build_support::llama_cpp_candidates(Path::new("/workspace/libs/rust/wwama"));

    assert_eq!(
        candidates,
        [
            Path::new("/workspace/libs/rust/wwama/../../cpp/llama.cpp-mainline").to_path_buf(),
            Path::new("/workspace/libs/rust/wwama/../llama.cpp-mainline").to_path_buf(),
            Path::new("/workspace/libs/rust/wwama/../../cpp/llama.cpp").to_path_buf(),
            Path::new("/workspace/libs/rust/wwama/../llama.cpp").to_path_buf(),
        ]
    );
}

#[test]
fn strips_windows_verbatim_drive_and_unc_prefixes() {
    assert_eq!(
        build_support::strip_windows_verbatim_prefix(r"\\?\C:\src\llama.cpp".into()),
        Path::new(r"C:\src\llama.cpp")
    );
    assert_eq!(
        build_support::strip_windows_verbatim_prefix(r"\\?\UNC\server\share\llama.cpp".into()),
        Path::new(r"\\server\share\llama.cpp")
    );
}

#[test]
fn selects_windows_cuda_import_directory_for_target_architecture() {
    let temp = std::env::temp_dir().join(format!("wwama-build-support-{}", std::process::id()));
    std::fs::create_dir_all(temp.join("lib/x64")).unwrap();
    std::fs::create_dir_all(temp.join("lib/arm64")).unwrap();

    assert_eq!(
        build_support::windows_cuda_arch_lib_dir(&temp.join("lib"), "x86_64"),
        Some(temp.join("lib/x64"))
    );
    assert_eq!(
        build_support::windows_cuda_arch_lib_dir(&temp.join("lib"), "aarch64"),
        Some(temp.join("lib/arm64"))
    );
    assert_eq!(
        build_support::windows_cuda_arch_lib_dir(&temp.join("lib/x64"), "x86_64"),
        None
    );

    std::fs::remove_dir_all(temp).unwrap();
}
