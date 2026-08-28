use std::path::{Path, PathBuf};

pub(crate) fn strip_windows_verbatim_prefix(dir: PathBuf) -> PathBuf {
    let display = dir.to_string_lossy();
    if let Some(rest) = display.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{rest}"))
    } else if let Some(rest) = display.strip_prefix(r"\\?\") {
        PathBuf::from(rest)
    } else {
        dir
    }
}

pub(crate) fn windows_cuda_arch_lib_dir(dir: &Path, target_arch: &str) -> Option<PathBuf> {
    let directory_name = match target_arch {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        _ => return None,
    };

    if dir.file_name().is_some_and(|name| name == directory_name) {
        return None;
    }

    let candidate = dir.join(directory_name);
    candidate.exists().then_some(candidate)
}
