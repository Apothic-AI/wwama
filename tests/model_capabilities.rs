use std::env;

use wwama::{Error, Session, SessionOptions};

fn load_model(variable: &str, mutable_tensors: bool) -> Option<Session> {
    let path = env::var(variable).ok()?;
    let n_gpu_layers = env::var("WWAMA_TEST_GPU_LAYERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let options = SessionOptions {
        n_ctx: 256,
        n_batch: 64,
        n_ubatch: 64,
        n_gpu_layers,
        mutable_tensors,
        ..SessionOptions::default()
    };
    Some(Session::load_from_path(&path, options).unwrap())
}

#[test]
fn native_inventory_and_noop_transfer_round_trip() {
    let Some(mut session) = load_model("WWAMA_TEST_MODEL", true) else {
        return;
    };
    let tensors = session.model().tensors().unwrap();
    assert!(!tensors.is_empty());
    let tensor = tensors
        .iter()
        .filter(|tensor| tensor.nbytes > 0)
        .min_by_key(|tensor| tensor.nbytes)
        .unwrap()
        .clone();
    let sample_size = tensor.nbytes.min(4096);
    let before = session
        .read_tensor_range(&tensor.name, 0, sample_size)
        .unwrap();
    session
        .write_tensor_range(&tensor.name, 0, &before)
        .unwrap();
    let after = session
        .read_tensor_range(&tensor.name, 0, sample_size)
        .unwrap();
    assert_eq!(after, before);
}

#[test]
fn deterministic_selected_logits_reset_context() {
    let Some(mut session) = load_model("WWAMA_TEST_DECODER_MODEL", false) else {
        return;
    };
    let prompt = session.tokenize_text("2 + 2 =", true, true).unwrap();
    let candidates = session.tokenize_text(" 4 5", false, true).unwrap();
    assert!(candidates.len() >= 2);
    let selected = [candidates[0], candidates[1]];
    let first = session
        .evaluate_selected_logits(&prompt, &selected)
        .unwrap();
    let second = session
        .evaluate_selected_logits(&prompt, &selected)
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(
        session.evaluate_selected_logits(&[], &selected),
        Err(Error::InvalidInput)
    );
}

#[test]
fn q1_0_model_row_xor_restores_bytes_and_logits() {
    let Some(mut session) = load_model("WWAMA_TEST_Q1_MODEL", true) else {
        return;
    };
    let tensor_name = env::var("WWAMA_TEST_Q1_TENSOR").unwrap();
    let row = env::var("WWAMA_TEST_Q1_ROW")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let descriptor = session.model().tensor(&tensor_name).unwrap();
    let rows = descriptor.row_count().unwrap();
    assert!(row < rows);
    let scales_before = session.q1_0_row_scales(&tensor_name).unwrap();
    assert_eq!(scales_before.len(), rows);
    let row_offset = row * descriptor.strides[1];
    let row_bytes = descriptor.strides[0] * (descriptor.dimensions[0] as usize / 128);
    let before = session
        .read_tensor_range(&tensor_name, row_offset, row_bytes)
        .unwrap();

    let prompt = session.tokenize_text("2 + 2 =", true, true).unwrap();
    let candidates = session.tokenize_text(" 4 5", false, true).unwrap();
    let baseline = session
        .evaluate_selected_logits(&prompt, &candidates[..2])
        .unwrap();
    session.xor_q1_0_row(&tensor_name, row).unwrap();
    let scales_after_flip = session.q1_0_row_scales(&tensor_name).unwrap();
    assert_eq!(scales_after_flip, scales_before);
    let mutated = session
        .read_tensor_range(&tensor_name, row_offset, row_bytes)
        .unwrap();
    assert_ne!(mutated, before);
    session.xor_q1_0_row(&tensor_name, row).unwrap();
    let restored = session
        .read_tensor_range(&tensor_name, row_offset, row_bytes)
        .unwrap();
    assert_eq!(restored, before);
    let restored_logits = session
        .evaluate_selected_logits(&prompt, &candidates[..2])
        .unwrap();
    assert_eq!(restored_logits, baseline);
}
