#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use alloc::borrow::ToOwned;
use alloc::ffi::CString;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::CStr;
use core::fmt;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use core::slice;
use core::str;

pub mod raw {
    #![allow(non_camel_case_types)]

    use core::ffi::{c_char, c_float, c_void};

    pub const LLAMA_DEFAULT_SEED: u32 = 0xFFFF_FFFF;
    pub const LLAMA_TOKEN_NULL: llama_token = -1;

    pub enum llama_vocab {}
    pub enum llama_model {}
    pub enum llama_context {}
    pub enum llama_sampler {}
    pub enum llama_memory_i {}
    pub enum llama_model_kv_override {}
    pub enum llama_model_tensor_buft_override {}
    pub enum llama_sampler_seq_config {}

    pub type llama_memory_t = *mut llama_memory_i;
    pub type llama_pos = i32;
    pub type llama_token = i32;
    pub type llama_seq_id = i32;
    pub type ggml_backend_dev_t = *mut c_void;
    pub type ggml_backend_buffer_type_t = *mut c_void;
    pub type ggml_abort_callback = Option<unsafe extern "C" fn(data: *mut c_void) -> bool>;
    pub type ggml_backend_sched_eval_callback = Option<
        unsafe extern "C" fn(tensor: *mut c_void, ask: bool, user_data: *mut c_void) -> bool,
    >;
    pub type llama_progress_callback =
        Option<unsafe extern "C" fn(progress: c_float, user_data: *mut c_void) -> bool>;

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum llama_rope_scaling_type {
        Unspecified = -1,
        None = 0,
        Linear = 1,
        Yarn = 2,
        LongRope = 3,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum llama_pooling_type {
        Unspecified = -1,
        None = 0,
        Mean = 1,
        Cls = 2,
        Last = 3,
        Rank = 4,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum llama_attention_type {
        Unspecified = -1,
        Causal = 0,
        NonCausal = 1,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum llama_flash_attn_type {
        Auto = -1,
        Disabled = 0,
        Enabled = 1,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum llama_context_type {
        Default = 0,
        Mtp = 1,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum llama_split_mode {
        None = 0,
        Layer = 1,
        Row = 2,
        Tensor = 3,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum ggml_type {
        F32 = 0,
        F16 = 1,
        Q4_0 = 2,
        Q4_1 = 3,
        Q5_0 = 6,
        Q5_1 = 7,
        Q8_0 = 8,
        Q8_1 = 9,
        Q2K = 10,
        Q3K = 11,
        Q4K = 12,
        Q5K = 13,
        Q6K = 14,
        Q8K = 15,
        Iq2Xxs = 16,
        Iq2Xs = 17,
        Iq3Xxs = 18,
        Iq1S = 19,
        Iq4Nl = 20,
        Iq3S = 21,
        Iq2S = 22,
        Iq4Xs = 23,
        I8 = 24,
        I16 = 25,
        I32 = 26,
        I64 = 27,
        F64 = 28,
        Iq1M = 29,
        Bf16 = 30,
        Tq1_0 = 34,
        Tq2_0 = 35,
        Q1_0 = 41,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_model_tensor_override {
        pub pattern: *const c_char,
        pub type_: ggml_type,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_model_imatrix_data {
        pub name: *const c_char,
        pub data: *const c_float,
        pub size: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_model_params {
        pub devices: *mut ggml_backend_dev_t,
        pub tensor_buft_overrides: *const llama_model_tensor_buft_override,
        pub n_gpu_layers: i32,
        pub split_mode: llama_split_mode,
        pub main_gpu: i32,
        pub tensor_split: *const c_float,
        pub progress_callback: llama_progress_callback,
        pub progress_callback_user_data: *mut c_void,
        pub kv_overrides: *const llama_model_kv_override,
        pub vocab_only: bool,
        pub use_mmap: bool,
        pub use_direct_io: bool,
        pub use_mlock: bool,
        pub check_tensors: bool,
        pub use_extra_bufts: bool,
        pub no_host: bool,
        pub no_alloc: bool,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_context_params {
        pub n_ctx: u32,
        pub n_batch: u32,
        pub n_ubatch: u32,
        pub n_seq_max: u32,
        pub n_rs_seq: u32,
        pub n_outputs_max: u32,
        pub n_threads: i32,
        pub n_threads_batch: i32,
        pub ctx_type: llama_context_type,
        pub rope_scaling_type: llama_rope_scaling_type,
        pub pooling_type: llama_pooling_type,
        pub attention_type: llama_attention_type,
        pub flash_attn_type: llama_flash_attn_type,
        pub rope_freq_base: c_float,
        pub rope_freq_scale: c_float,
        pub yarn_ext_factor: c_float,
        pub yarn_attn_factor: c_float,
        pub yarn_beta_fast: c_float,
        pub yarn_beta_slow: c_float,
        pub yarn_orig_ctx: u32,
        pub defrag_thold: c_float,
        pub cb_eval: ggml_backend_sched_eval_callback,
        pub cb_eval_user_data: *mut c_void,
        pub type_k: ggml_type,
        pub type_v: ggml_type,
        pub abort_callback: ggml_abort_callback,
        pub abort_callback_data: *mut c_void,
        pub embeddings: bool,
        pub offload_kqv: bool,
        pub no_perf: bool,
        pub op_offload: bool,
        pub swa_full: bool,
        pub kv_unified: bool,
        pub samplers: *mut llama_sampler_seq_config,
        pub n_samplers: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_batch {
        pub n_tokens: i32,
        pub token: *mut llama_token,
        pub embd: *mut c_float,
        pub pos: *mut llama_pos,
        pub n_seq_id: *mut i32,
        pub seq_id: *mut *mut llama_seq_id,
        pub logits: *mut i8,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_sampler_chain_params {
        pub no_perf: bool,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_chat_message {
        pub role: *const c_char,
        pub content: *const c_char,
    }

    #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct wwama_tensor_descriptor {
        pub name: *const c_char,
        pub type_name: *const c_char,
        pub backend_name: *const c_char,
        pub type_id: i32,
        pub n_dims: i32,
        pub ne: [i64; 4],
        pub nb: [usize; 4],
        pub nbytes: usize,
    }

    unsafe extern "C" {
        pub fn llama_model_default_params() -> llama_model_params;
        pub fn llama_context_default_params() -> llama_context_params;
        pub fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;

        pub fn llama_backend_init();
        pub fn llama_backend_free();

        pub fn llama_model_load_from_file(
            path_model: *const c_char,
            params: llama_model_params,
        ) -> *mut llama_model;
        pub fn llama_model_free(model: *mut llama_model);
        pub fn llama_init_from_model(
            model: *mut llama_model,
            params: llama_context_params,
        ) -> *mut llama_context;
        pub fn llama_free(ctx: *mut llama_context);

        pub fn llama_get_memory(ctx: *const llama_context) -> llama_memory_t;
        pub fn llama_n_ctx(ctx: *const llama_context) -> u32;
        pub fn llama_n_batch(ctx: *const llama_context) -> u32;
        pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
        pub fn llama_model_has_encoder(model: *const llama_model) -> bool;
        pub fn llama_model_has_decoder(model: *const llama_model) -> bool;
        pub fn llama_model_n_embd_out(model: *const llama_model) -> i32;
        pub fn llama_model_n_cls_out(model: *const llama_model) -> u32;
        pub fn llama_model_cls_label(model: *const llama_model, i: u32) -> *const c_char;
        pub fn llama_model_chat_template(
            model: *const llama_model,
            name: *const c_char,
        ) -> *const c_char;
        pub fn llama_vocab_n_tokens(vocab: *const llama_vocab) -> i32;
        pub fn llama_vocab_is_eog(vocab: *const llama_vocab, token: llama_token) -> bool;
        pub fn llama_vocab_bos(vocab: *const llama_vocab) -> llama_token;
        pub fn llama_vocab_eos(vocab: *const llama_vocab) -> llama_token;
        pub fn llama_vocab_sep(vocab: *const llama_vocab) -> llama_token;
        pub fn llama_vocab_get_add_bos(vocab: *const llama_vocab) -> bool;
        pub fn llama_vocab_get_add_eos(vocab: *const llama_vocab) -> bool;
        pub fn llama_vocab_get_add_sep(vocab: *const llama_vocab) -> bool;

        pub fn llama_pooling_type(ctx: *const llama_context) -> llama_pooling_type;
        pub fn llama_set_embeddings(ctx: *mut llama_context, embeddings: bool);
        pub fn llama_set_causal_attn(ctx: *mut llama_context, causal_attn: bool);
        pub fn llama_synchronize(ctx: *mut llama_context);

        pub fn llama_memory_clear(mem: llama_memory_t, data: bool);
        pub fn llama_memory_seq_rm(
            mem: llama_memory_t,
            seq_id: llama_seq_id,
            p0: llama_pos,
            p1: llama_pos,
        ) -> bool;

        pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch;
        pub fn llama_batch_free(batch: llama_batch);

        pub fn llama_encode(ctx: *mut llama_context, batch: llama_batch) -> i32;
        pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> i32;

        pub fn llama_get_embeddings_ith(ctx: *mut llama_context, i: i32) -> *mut c_float;
        pub fn llama_get_embeddings_seq(
            ctx: *mut llama_context,
            seq_id: llama_seq_id,
        ) -> *mut c_float;
        pub fn llama_get_logits_ith(ctx: *mut llama_context, i: i32) -> *mut c_float;

        pub fn llama_tokenize(
            vocab: *const llama_vocab,
            text: *const c_char,
            text_len: i32,
            tokens: *mut llama_token,
            n_tokens_max: i32,
            add_special: bool,
            parse_special: bool,
        ) -> i32;
        pub fn llama_token_to_piece(
            vocab: *const llama_vocab,
            token: llama_token,
            buf: *mut c_char,
            length: i32,
            lstrip: i32,
            special: bool,
        ) -> i32;
        pub fn llama_detokenize(
            vocab: *const llama_vocab,
            tokens: *const llama_token,
            n_tokens: i32,
            text: *mut c_char,
            text_len_max: i32,
            remove_special: bool,
            unparse_special: bool,
        ) -> i32;
        pub fn llama_chat_apply_template(
            tmpl: *const c_char,
            chat: *const llama_chat_message,
            n_msg: usize,
            add_ass: bool,
            buf: *mut c_char,
            length: i32,
        ) -> i32;

        pub fn llama_sampler_chain_init(params: llama_sampler_chain_params) -> *mut llama_sampler;
        pub fn llama_sampler_chain_add(chain: *mut llama_sampler, smpl: *mut llama_sampler);
        pub fn llama_sampler_free(smpl: *mut llama_sampler);
        pub fn llama_sampler_init_greedy() -> *mut llama_sampler;
        pub fn llama_sampler_init_dist(seed: u32) -> *mut llama_sampler;
        pub fn llama_sampler_init_top_k(k: i32) -> *mut llama_sampler;
        pub fn llama_sampler_init_top_p(p: c_float, min_keep: usize) -> *mut llama_sampler;
        pub fn llama_sampler_init_temp(t: c_float) -> *mut llama_sampler;
        pub fn llama_sampler_sample(
            smpl: *mut llama_sampler,
            ctx: *mut llama_context,
            idx: i32,
        ) -> llama_token;

        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        pub fn wwama_tensor_count(model: *const llama_model) -> usize;
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        pub fn wwama_tensor_descriptor_at(
            model: *const llama_model,
            index: usize,
            descriptor: *mut wwama_tensor_descriptor,
        ) -> i32;
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        pub fn wwama_tensor_descriptor_named(
            model: *const llama_model,
            name: *const c_char,
            descriptor: *mut wwama_tensor_descriptor,
        ) -> i32;
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        pub fn wwama_tensor_read(
            model: *const llama_model,
            name: *const c_char,
            offset: usize,
            destination: *mut c_void,
            size: usize,
        ) -> i32;
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        pub fn wwama_tensor_write(
            model: *mut llama_model,
            name: *const c_char,
            offset: usize,
            source: *const c_void,
            size: usize,
        ) -> i32;
    }
}

pub use raw::{
    LLAMA_DEFAULT_SEED, LLAMA_TOKEN_NULL, ggml_type, llama_attention_type, llama_batch,
    llama_chat_message, llama_context_params, llama_flash_attn_type, llama_model_params,
    llama_pooling_type, llama_pos, llama_rope_scaling_type, llama_sampler_chain_params,
    llama_seq_id, llama_split_mode, llama_token,
};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    ModelLoadFailed,
    ContextInitFailed,
    SamplerInitFailed,
    InvalidCString,
    InvalidInput,
    TokenizationFailed,
    DetokenizationFailed,
    ChatTemplateFailed,
    DecodeFailed(i32),
    EncodeFailed(i32),
    EmbeddingUnavailable,
    RerankUnavailable,
    ProjectorInvalid,
    DecoderLogitsUnavailable,
    InvalidToken,
    ContextOverflow,
    TensorNotFound,
    TensorTransferUnavailable,
    TensorTransferOutOfBounds,
    TensorMutationDisabled,
    UnsupportedTensorType,
    UnsupportedTensorShape,
    UnsupportedTensorStride,
    InvalidTensorRow,
    UnsupportedTarget,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelLoadFailed => f.write_str("failed to load llama.cpp model"),
            Self::ContextInitFailed => f.write_str("failed to initialize llama.cpp context"),
            Self::SamplerInitFailed => f.write_str("failed to initialize llama.cpp sampler"),
            Self::InvalidCString => f.write_str("input contains an interior nul byte"),
            Self::InvalidInput => f.write_str("invalid wwama input"),
            Self::TokenizationFailed => f.write_str("llama.cpp tokenization failed"),
            Self::DetokenizationFailed => f.write_str("llama.cpp detokenization failed"),
            Self::ChatTemplateFailed => f.write_str("llama.cpp chat template application failed"),
            Self::DecodeFailed(code) => write!(f, "llama.cpp decode failed with status {code}"),
            Self::EncodeFailed(code) => write!(f, "llama.cpp encode failed with status {code}"),
            Self::EmbeddingUnavailable => {
                f.write_str("llama.cpp did not return an embedding vector")
            }
            Self::RerankUnavailable => {
                f.write_str("llama.cpp reranking requires embeddings enabled with rank pooling")
            }
            Self::ProjectorInvalid => f.write_str("invalid rerank projector"),
            Self::DecoderLogitsUnavailable => f.write_str("model did not provide decoder logits"),
            Self::InvalidToken => f.write_str("token ID is outside the model vocabulary"),
            Self::ContextOverflow => f.write_str("token sequence exceeds the context window"),
            Self::TensorNotFound => f.write_str("model tensor was not found"),
            Self::TensorTransferUnavailable => {
                f.write_str("model tensor cannot be transferred through its backend")
            }
            Self::TensorTransferOutOfBounds => {
                f.write_str("model tensor transfer is out of bounds")
            }
            Self::TensorMutationDisabled => {
                f.write_str("model was not loaded with mutable tensors enabled")
            }
            Self::UnsupportedTensorType => f.write_str("unsupported model tensor type"),
            Self::UnsupportedTensorShape => f.write_str("unsupported model tensor shape"),
            Self::UnsupportedTensorStride => f.write_str("unsupported model tensor stride"),
            Self::InvalidTensorRow => f.write_str("model tensor row is out of bounds"),
            Self::UnsupportedTarget => {
                f.write_str("mutable model tensor access is unsupported on this target")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Owned model-tensor metadata copied from llama.cpp.
///
/// Dimensions and byte strides use GGML ordering. For an ordinary matrix,
/// `dimensions[0]` is the input width and `dimensions[1]` is the logical row
/// count. Higher dimensions must be one for Q1_0 row mutation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorDescriptor {
    pub name: String,
    pub type_id: i32,
    pub type_name: String,
    pub dimensions: [u64; 4],
    pub strides: [usize; 4],
    pub n_dims: usize,
    pub nbytes: usize,
    pub backend: String,
}

impl TensorDescriptor {
    pub fn row_count(&self) -> Result<usize> {
        if self.n_dims != 2 || self.dimensions[2] != 1 || self.dimensions[3] != 1 {
            return Err(Error::UnsupportedTensorShape);
        }
        usize::try_from(self.dimensions[1]).map_err(|_| Error::UnsupportedTensorShape)
    }
}

/// Summary of a completed Q1_0 row XOR operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowXorResult {
    pub row: usize,
    pub blocks: usize,
    pub packed_bytes_flipped: usize,
}

const Q1_0_TYPE_ID: i32 = 41;
const Q1_0_BLOCK_VALUES: usize = 128;
const Q1_0_SCALE_BYTES: usize = 2;
const Q1_0_PACKED_BYTES: usize = Q1_0_BLOCK_VALUES / 8;
const Q1_0_BLOCK_BYTES: usize = Q1_0_SCALE_BYTES + Q1_0_PACKED_BYTES;

#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
fn tensor_bridge_error(status: i32) -> Error {
    match status {
        2 => Error::TensorNotFound,
        3 => Error::TensorTransferOutOfBounds,
        4 => Error::TensorTransferUnavailable,
        _ => Error::InvalidInput,
    }
}

#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
unsafe fn owned_tensor_descriptor(raw: raw::wwama_tensor_descriptor) -> Result<TensorDescriptor> {
    if raw.name.is_null() || raw.type_name.is_null() || raw.n_dims < 0 || raw.n_dims > 4 {
        return Err(Error::InvalidInput);
    }
    let mut dimensions = [0_u64; 4];
    for (destination, source) in dimensions.iter_mut().zip(raw.ne) {
        *destination = u64::try_from(source).map_err(|_| Error::UnsupportedTensorShape)?;
    }
    Ok(TensorDescriptor {
        name: unsafe { CStr::from_ptr(raw.name) }
            .to_string_lossy()
            .into_owned(),
        type_id: raw.type_id,
        type_name: unsafe { CStr::from_ptr(raw.type_name) }
            .to_string_lossy()
            .into_owned(),
        dimensions,
        strides: raw.nb,
        n_dims: raw.n_dims as usize,
        nbytes: raw.nbytes,
        backend: if raw.backend_name.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(raw.backend_name) }
                .to_string_lossy()
                .into_owned()
        },
    })
}

pub struct Backend;

impl Backend {
    pub fn init() {
        unsafe { raw::llama_backend_init() }
    }

    pub fn free() {
        unsafe { raw::llama_backend_free() }
    }
}

pub struct Model {
    ptr: NonNull<raw::llama_model>,
}

impl Model {
    pub fn default_params() -> raw::llama_model_params {
        unsafe { raw::llama_model_default_params() }
    }

    pub fn load_from_file(path: &CStr, params: raw::llama_model_params) -> Result<Self> {
        let ptr = unsafe { raw::llama_model_load_from_file(path.as_ptr(), params) };
        NonNull::new(ptr)
            .map(|ptr| Self { ptr })
            .ok_or(Error::ModelLoadFailed)
    }

    pub fn load_from_path(path: &str, params: raw::llama_model_params) -> Result<Self> {
        let path = CString::new(path).map_err(|_| Error::InvalidCString)?;
        Self::load_from_file(&path, params)
    }

    pub fn as_ptr(&self) -> *mut raw::llama_model {
        self.ptr.as_ptr()
    }

    pub fn vocab(&self) -> *const raw::llama_vocab {
        unsafe { raw::llama_model_get_vocab(self.ptr.as_ptr()) }
    }

    pub fn has_encoder(&self) -> bool {
        unsafe { raw::llama_model_has_encoder(self.ptr.as_ptr()) }
    }

    pub fn has_decoder(&self) -> bool {
        unsafe { raw::llama_model_has_decoder(self.ptr.as_ptr()) }
    }

    pub fn n_embd_out(&self) -> i32 {
        unsafe { raw::llama_model_n_embd_out(self.ptr.as_ptr()) }
    }

    pub fn n_cls_out(&self) -> u32 {
        unsafe { raw::llama_model_n_cls_out(self.ptr.as_ptr()) }
    }

    pub fn cls_label(&self, index: u32) -> Option<String> {
        let ptr = unsafe { raw::llama_model_cls_label(self.ptr.as_ptr(), index) };
        if ptr.is_null() {
            return None;
        }
        Some(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    pub fn chat_template(&self, name: &str) -> Result<Option<String>> {
        let name = CString::new(name).map_err(|_| Error::InvalidCString)?;
        let ptr = unsafe { raw::llama_model_chat_template(self.ptr.as_ptr(), name.as_ptr()) };
        if ptr.is_null() {
            return Ok(None);
        }
        Ok(Some(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        ))
    }

    pub fn n_vocab(&self) -> i32 {
        unsafe { raw::llama_vocab_n_tokens(self.vocab()) }
    }

    pub fn bos(&self) -> raw::llama_token {
        unsafe { raw::llama_vocab_bos(self.vocab()) }
    }

    pub fn eos(&self) -> raw::llama_token {
        unsafe { raw::llama_vocab_eos(self.vocab()) }
    }

    pub fn sep(&self) -> raw::llama_token {
        unsafe { raw::llama_vocab_sep(self.vocab()) }
    }

    pub fn add_bos(&self) -> bool {
        unsafe { raw::llama_vocab_get_add_bos(self.vocab()) }
    }

    pub fn add_eos(&self) -> bool {
        unsafe { raw::llama_vocab_get_add_eos(self.vocab()) }
    }

    pub fn add_sep(&self) -> bool {
        unsafe { raw::llama_vocab_get_add_sep(self.vocab()) }
    }

    pub fn is_eog(&self, token: raw::llama_token) -> bool {
        unsafe { raw::llama_vocab_is_eog(self.vocab(), token) }
    }

    /// Returns owned metadata for every tensor loaded by llama.cpp.
    ///
    /// This inventory is available on native targets. Mutable tensor access is
    /// intentionally unavailable on WebAssembly until runtime transfer behavior
    /// is validated there.
    pub fn tensors(&self) -> Result<Vec<TensorDescriptor>> {
        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        {
            Err(Error::UnsupportedTarget)
        }
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        {
            let count = unsafe { raw::wwama_tensor_count(self.ptr.as_ptr()) };
            let mut tensors = Vec::with_capacity(count);
            for index in 0..count {
                let mut descriptor = MaybeUninit::uninit();
                let status = unsafe {
                    raw::wwama_tensor_descriptor_at(
                        self.ptr.as_ptr(),
                        index,
                        descriptor.as_mut_ptr(),
                    )
                };
                if status != 0 {
                    return Err(tensor_bridge_error(status));
                }
                tensors.push(unsafe { owned_tensor_descriptor(descriptor.assume_init()) }?);
            }
            Ok(tensors)
        }
    }

    pub fn tensor(&self, name: &str) -> Result<TensorDescriptor> {
        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        {
            let _ = name;
            Err(Error::UnsupportedTarget)
        }
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        {
            let name = CString::new(name).map_err(|_| Error::InvalidCString)?;
            let mut descriptor = MaybeUninit::uninit();
            let status = unsafe {
                raw::wwama_tensor_descriptor_named(
                    self.ptr.as_ptr(),
                    name.as_ptr(),
                    descriptor.as_mut_ptr(),
                )
            };
            if status != 0 {
                return Err(tensor_bridge_error(status));
            }
            unsafe { owned_tensor_descriptor(descriptor.assume_init()) }
        }
    }

    #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
    fn read_tensor_range(&self, name: &CStr, offset: usize, destination: &mut [u8]) -> Result<()> {
        let status = unsafe {
            raw::wwama_tensor_read(
                self.ptr.as_ptr(),
                name.as_ptr(),
                offset,
                destination.as_mut_ptr().cast(),
                destination.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(tensor_bridge_error(status))
        }
    }

    #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
    fn write_tensor_range(&mut self, name: &CStr, offset: usize, source: &[u8]) -> Result<()> {
        let status = unsafe {
            raw::wwama_tensor_write(
                self.ptr.as_ptr(),
                name.as_ptr(),
                offset,
                source.as_ptr().cast(),
                source.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(tensor_bridge_error(status))
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { raw::llama_model_free(self.ptr.as_ptr()) }
    }
}

pub struct Context {
    ptr: NonNull<raw::llama_context>,
}

impl Context {
    pub fn default_params() -> raw::llama_context_params {
        unsafe { raw::llama_context_default_params() }
    }

    pub fn new(model: &Model, params: raw::llama_context_params) -> Result<Self> {
        let ptr = unsafe { raw::llama_init_from_model(model.as_ptr(), params) };
        NonNull::new(ptr)
            .map(|ptr| Self { ptr })
            .ok_or(Error::ContextInitFailed)
    }

    pub fn as_ptr(&self) -> *mut raw::llama_context {
        self.ptr.as_ptr()
    }

    pub fn n_ctx(&self) -> u32 {
        unsafe { raw::llama_n_ctx(self.ptr.as_ptr()) }
    }

    pub fn n_batch(&self) -> u32 {
        unsafe { raw::llama_n_batch(self.ptr.as_ptr()) }
    }

    pub fn pooling_type(&self) -> raw::llama_pooling_type {
        unsafe { raw::llama_pooling_type(self.ptr.as_ptr()) }
    }

    pub fn set_embeddings(&mut self, enabled: bool) {
        unsafe { raw::llama_set_embeddings(self.ptr.as_ptr(), enabled) }
    }

    pub fn set_causal_attn(&mut self, causal_attn: bool) {
        unsafe { raw::llama_set_causal_attn(self.ptr.as_ptr(), causal_attn) }
    }

    pub fn synchronize(&mut self) {
        unsafe { raw::llama_synchronize(self.ptr.as_ptr()) }
    }

    pub fn clear_memory(&mut self, data: bool) {
        unsafe { raw::llama_memory_clear(raw::llama_get_memory(self.ptr.as_ptr()), data) }
    }

    pub fn remove_sequence(&mut self, seq_id: raw::llama_seq_id) -> bool {
        unsafe {
            raw::llama_memory_seq_rm(raw::llama_get_memory(self.ptr.as_ptr()), seq_id, -1, -1)
        }
    }

    pub fn encode(&mut self, batch: &Batch) -> i32 {
        unsafe { raw::llama_encode(self.ptr.as_ptr(), batch.raw) }
    }

    pub fn decode(&mut self, batch: &Batch) -> i32 {
        unsafe { raw::llama_decode(self.ptr.as_ptr(), batch.raw) }
    }

    pub fn embeddings_ith(&mut self, index: i32) -> *mut f32 {
        unsafe { raw::llama_get_embeddings_ith(self.ptr.as_ptr(), index) }
    }

    pub fn embeddings_seq(&mut self, seq_id: raw::llama_seq_id) -> *mut f32 {
        unsafe { raw::llama_get_embeddings_seq(self.ptr.as_ptr(), seq_id) }
    }

    pub fn logits_ith(&mut self, index: i32) -> *mut f32 {
        unsafe { raw::llama_get_logits_ith(self.ptr.as_ptr(), index) }
    }

    pub fn tokenize(
        &self,
        vocab: *const raw::llama_vocab,
        text: &CStr,
        tokens: &mut [raw::llama_token],
        add_special: bool,
        parse_special: bool,
    ) -> i32 {
        unsafe {
            raw::llama_tokenize(
                vocab,
                text.as_ptr(),
                text.to_bytes().len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                add_special,
                parse_special,
            )
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { raw::llama_free(self.ptr.as_ptr()) }
    }
}

pub struct Batch {
    raw: raw::llama_batch,
}

impl Batch {
    pub fn new(n_tokens: i32, embd: i32, n_seq_max: i32) -> Self {
        let raw = unsafe { raw::llama_batch_init(n_tokens, embd, n_seq_max) };
        Self { raw }
    }

    pub fn as_raw(&self) -> &raw::llama_batch {
        &self.raw
    }

    pub fn as_raw_mut(&mut self) -> &mut raw::llama_batch {
        &mut self.raw
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { raw::llama_batch_free(self.raw) }
    }
}

#[derive(Clone, Debug)]
pub struct SessionOptions {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub n_gpu_layers: i32,
    /// Loads weights into writable backend storage instead of a read-only mmap.
    /// This increases load time and resident memory and is required for writes
    /// and Q1_0 XOR mutation.
    pub mutable_tensors: bool,
    pub embeddings: bool,
    pub pooling_type: raw::llama_pooling_type,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 512,
            n_ubatch: 512,
            n_seq_max: 1,
            n_threads: 0,
            n_threads_batch: 0,
            n_gpu_layers: 999,
            mutable_tensors: false,
            embeddings: false,
            pooling_type: raw::llama_pooling_type::Unspecified,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerationOptions {
    /// Maximum number of generated tokens. `0` means no explicit generation
    /// limit; generation then stops on EOG or the model/context backend's own
    /// limit.
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub seed: u32,
    pub add_special: bool,
    pub parse_special: bool,
    pub emit_special: bool,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 0,
            temperature: 0.0,
            top_k: 40,
            top_p: 0.95,
            seed: raw::LLAMA_DEFAULT_SEED,
            add_special: true,
            parse_special: true,
            emit_special: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EmbeddingOptions {
    pub add_special: bool,
    pub parse_special: bool,
    pub normalize: bool,
}

impl Default for EmbeddingOptions {
    fn default() -> Self {
        Self {
            add_special: true,
            parse_special: true,
            normalize: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RerankOptions {
    pub instruction: Option<String>,
    pub prompt_template: Option<String>,
    pub add_special: bool,
    pub parse_special: bool,
}

impl Default for RerankOptions {
    fn default() -> Self {
        Self {
            instruction: None,
            prompt_template: None,
            add_special: true,
            parse_special: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GenerateOutput {
    pub text: String,
    pub token_count: usize,
}

#[derive(Clone, Debug, Default)]
pub struct RerankOutput {
    pub index: usize,
    pub score: f32,
    pub rank: usize,
    pub token_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttentionOperation {
    Generation,
    Embedding,
    Rerank,
}

fn causal_attention_for_operation(operation: AttentionOperation) -> bool {
    match operation {
        AttentionOperation::Generation => true,
        AttentionOperation::Embedding | AttentionOperation::Rerank => false,
    }
}

#[derive(Clone, Debug)]
pub struct JinaRerankProjector {
    linear1_weight: Vec<f32>,
    linear1_out: usize,
    linear1_in: usize,
    linear2_weight: Vec<f32>,
    linear2_out: usize,
    linear2_in: usize,
}

impl JinaRerankProjector {
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self> {
        let (linear1_shape, linear1_weight) = read_safetensor_f32(bytes, "projector.0.weight")?;
        let (linear2_shape, linear2_weight) = read_safetensor_f32(bytes, "projector.2.weight")?;
        if linear1_shape.len() != 2 || linear2_shape.len() != 2 {
            return Err(Error::ProjectorInvalid);
        }
        let linear1_out = linear1_shape[0];
        let linear1_in = linear1_shape[1];
        let linear2_out = linear2_shape[0];
        let linear2_in = linear2_shape[1];
        if linear1_out == 0
            || linear1_in == 0
            || linear2_out == 0
            || linear2_in == 0
            || linear1_weight.len() != linear1_out.saturating_mul(linear1_in)
            || linear2_weight.len() != linear2_out.saturating_mul(linear2_in)
            || linear2_in != linear1_out
        {
            return Err(Error::ProjectorInvalid);
        }
        Ok(Self {
            linear1_weight,
            linear1_out,
            linear1_in,
            linear2_weight,
            linear2_out,
            linear2_in,
        })
    }

    pub fn input_dim(&self) -> usize {
        self.linear1_in
    }

    pub fn output_dim(&self) -> usize {
        self.linear2_out
    }

    pub fn project(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.linear1_in {
            return Err(Error::ProjectorInvalid);
        }
        let mut hidden = vec![0.0_f32; self.linear1_out];
        for row in 0..self.linear1_out {
            let weight_offset = row * self.linear1_in;
            let mut sum = 0.0_f32;
            for col in 0..self.linear1_in {
                sum += input[col] * self.linear1_weight[weight_offset + col];
            }
            hidden[row] = sum.max(0.0);
        }

        let mut output = vec![0.0_f32; self.linear2_out];
        for row in 0..self.linear2_out {
            let weight_offset = row * self.linear2_in;
            let mut sum = 0.0_f32;
            for col in 0..self.linear2_in {
                sum += hidden[col] * self.linear2_weight[weight_offset + col];
            }
            output[row] = sum;
        }
        Ok(output)
    }
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

pub struct Session {
    context: Context,
    model: Model,
    n_batch: usize,
    mutable_tensors: bool,
}

impl Session {
    pub fn load_from_path(path: &str, options: SessionOptions) -> Result<Self> {
        Backend::init();

        let mut model_params = Model::default_params();
        model_params.n_gpu_layers = options.n_gpu_layers;
        model_params.use_mmap = !options.mutable_tensors;
        let model = Model::load_from_path(path, model_params)?;

        let mut context_params = Context::default_params();
        context_params.n_ctx = options.n_ctx;
        context_params.n_batch = options.n_batch;
        context_params.n_ubatch = options.n_ubatch;
        context_params.n_seq_max = options.n_seq_max;
        context_params.n_threads = options.n_threads;
        context_params.n_threads_batch = options.n_threads_batch;
        context_params.embeddings = options.embeddings;
        context_params.pooling_type = options.pooling_type;
        if options.embeddings {
            // Embedding/rerank models (Qwen3-Embedding, etc.) require bidirectional attention.
            context_params.attention_type = raw::llama_attention_type::NonCausal;
            context_params.flash_attn_type = raw::llama_flash_attn_type::Disabled;
        }
        let mut context = Context::new(&model, context_params)?;
        if options.embeddings {
            context.set_causal_attn(false);
        }

        Ok(Self {
            context,
            model,
            n_batch: options.n_batch.max(1) as usize,
            mutable_tensors: options.mutable_tensors,
        })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    pub fn tokenize_text(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<raw::llama_token>> {
        let text = CString::new(text).map_err(|_| Error::InvalidCString)?;
        let mut capacity = text.to_bytes().len().saturating_add(8).max(8);
        loop {
            let mut tokens = vec![0; capacity];
            let written = self.context.tokenize(
                self.model.vocab(),
                &text,
                &mut tokens,
                add_special,
                parse_special,
            );
            if written >= 0 {
                tokens.truncate(written as usize);
                return Ok(tokens);
            }
            let needed = written.checked_neg().ok_or(Error::TokenizationFailed)? as usize;
            if needed <= capacity {
                return Err(Error::TokenizationFailed);
            }
            capacity = needed;
        }
    }

    pub fn detokenize_tokens(
        &self,
        tokens: &[raw::llama_token],
        remove_special: bool,
        unparse_special: bool,
    ) -> Result<String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }
        let mut capacity = tokens.len().saturating_mul(8).max(32);
        loop {
            let mut bytes = vec![0_u8; capacity];
            let written = unsafe {
                raw::llama_detokenize(
                    self.model.vocab(),
                    tokens.as_ptr(),
                    tokens.len() as i32,
                    bytes.as_mut_ptr().cast(),
                    bytes.len() as i32,
                    remove_special,
                    unparse_special,
                )
            };
            if written >= 0 {
                bytes.truncate(written as usize);
                return Ok(String::from_utf8_lossy(&bytes).into_owned());
            }
            let needed = written.checked_neg().ok_or(Error::DetokenizationFailed)? as usize;
            if needed <= capacity {
                return Err(Error::DetokenizationFailed);
            }
            capacity = needed;
        }
    }

    pub fn token_to_piece(&self, token: raw::llama_token, special: bool) -> Result<String> {
        let mut capacity = 32_usize;
        loop {
            let mut bytes = vec![0_u8; capacity];
            let written = unsafe {
                raw::llama_token_to_piece(
                    self.model.vocab(),
                    token,
                    bytes.as_mut_ptr().cast(),
                    bytes.len() as i32,
                    0,
                    special,
                )
            };
            if written >= 0 {
                bytes.truncate(written as usize);
                return Ok(String::from_utf8_lossy(&bytes).into_owned());
            }
            let needed = written.checked_neg().ok_or(Error::DetokenizationFailed)? as usize;
            if needed <= capacity {
                return Err(Error::DetokenizationFailed);
            }
            capacity = needed;
        }
    }

    /// Evaluates a tokenized prompt with a clean context and returns selected
    /// logits at the final prompt position.
    pub fn evaluate_selected_logits(
        &mut self,
        prompt_tokens: &[raw::llama_token],
        selected_tokens: &[raw::llama_token],
    ) -> Result<Vec<f32>> {
        if prompt_tokens.is_empty() || selected_tokens.is_empty() {
            return Err(Error::InvalidInput);
        }
        if prompt_tokens.len() > self.context.n_ctx() as usize {
            return Err(Error::ContextOverflow);
        }
        let n_vocab = self.model.n_vocab();
        if selected_tokens
            .iter()
            .any(|token| *token < 0 || *token >= n_vocab)
        {
            return Err(Error::InvalidToken);
        }
        if !self.model.has_decoder() {
            return Err(Error::DecoderLogitsUnavailable);
        }

        self.context.set_embeddings(false);
        self.context.set_causal_attn(causal_attention_for_operation(
            AttentionOperation::Generation,
        ));
        self.context.clear_memory(true);
        self.evaluate_tokens(prompt_tokens, 0, true)?;
        self.context.synchronize();

        let logits = self.context.logits_ith(-1);
        if logits.is_null() {
            return Err(Error::DecoderLogitsUnavailable);
        }
        Ok(selected_tokens
            .iter()
            .map(|token| unsafe { *logits.add(*token as usize) })
            .collect())
    }

    /// Returns `logit(correct) - logit(wrong)` after a clean prompt evaluation.
    pub fn logit_gap(
        &mut self,
        prompt_tokens: &[raw::llama_token],
        correct: raw::llama_token,
        wrong: raw::llama_token,
    ) -> Result<f32> {
        let logits = self.evaluate_selected_logits(prompt_tokens, &[correct, wrong])?;
        Ok(logits[0] - logits[1])
    }

    /// Reads an entire tensor through its GGML backend transfer implementation.
    /// Inference and mutation are serialized by the required `&mut Session`.
    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<u8>> {
        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        {
            let _ = name;
            Err(Error::UnsupportedTarget)
        }
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        {
            let descriptor = self.model.tensor(name)?;
            self.read_tensor_range(name, 0, descriptor.nbytes)
        }
    }

    /// Reads a validated byte range through the tensor's owning GGML backend.
    pub fn read_tensor_range(&mut self, name: &str, offset: usize, size: usize) -> Result<Vec<u8>> {
        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        {
            let _ = (name, offset, size);
            Err(Error::UnsupportedTarget)
        }
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        {
            let name = CString::new(name).map_err(|_| Error::InvalidCString)?;
            let mut bytes = vec![0_u8; size];
            self.context.synchronize();
            self.model.read_tensor_range(&name, offset, &mut bytes)?;
            Ok(bytes)
        }
    }

    /// Writes an entire tensor through its GGML backend transfer implementation.
    /// The byte length must exactly match the loaded tensor.
    pub fn write_tensor(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        {
            let _ = (name, bytes);
            Err(Error::UnsupportedTarget)
        }
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        {
            let descriptor = self.model.tensor(name)?;
            if bytes.len() != descriptor.nbytes {
                return Err(Error::TensorTransferOutOfBounds);
            }
            self.write_tensor_range(name, 0, bytes)
        }
    }

    /// Writes a validated byte range through the tensor's owning GGML backend.
    /// Callers are responsible for preserving the tensor's encoding invariants.
    pub fn write_tensor_range(&mut self, name: &str, offset: usize, bytes: &[u8]) -> Result<()> {
        if !self.mutable_tensors {
            return Err(Error::TensorMutationDisabled);
        }
        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        {
            let _ = (name, offset, bytes);
            Err(Error::UnsupportedTarget)
        }
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        {
            let name = CString::new(name).map_err(|_| Error::InvalidCString)?;
            self.context.synchronize();
            self.model.write_tensor_range(&name, offset, bytes)?;
            self.context.synchronize();
            Ok(())
        }
    }

    /// Computes the mean absolute FP16 block scale for each logical Q1_0 row.
    pub fn q1_0_row_scales(&mut self, name: &str) -> Result<Vec<f32>> {
        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        {
            let _ = name;
            Err(Error::UnsupportedTarget)
        }
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        {
            let descriptor = self.model.tensor(name)?;
            let layout = q1_0_layout(&descriptor)?;
            let name = CString::new(name).map_err(|_| Error::InvalidCString)?;
            self.context.synchronize();
            let mut scales = Vec::with_capacity(layout.rows);
            let mut row_bytes = vec![0_u8; layout.payload_bytes];
            for row in 0..layout.rows {
                let offset = row
                    .checked_mul(layout.row_stride)
                    .ok_or(Error::UnsupportedTensorStride)?;
                self.model
                    .read_tensor_range(&name, offset, &mut row_bytes)?;
                scales.push(mean_q1_0_scale(&row_bytes, layout.blocks)?);
            }
            Ok(scales)
        }
    }

    /// XORs every packed binary weight in one logical Q1_0 matrix row.
    /// Calling this method twice with the same tensor and row restores the bytes.
    pub fn xor_q1_0_row(&mut self, name: &str, row: usize) -> Result<RowXorResult> {
        if !self.mutable_tensors {
            return Err(Error::TensorMutationDisabled);
        }
        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        {
            let _ = (name, row);
            Err(Error::UnsupportedTarget)
        }
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        {
            let descriptor = self.model.tensor(name)?;
            let layout = q1_0_layout(&descriptor)?;
            if row >= layout.rows {
                return Err(Error::InvalidTensorRow);
            }
            let offset = row
                .checked_mul(layout.row_stride)
                .ok_or(Error::UnsupportedTensorStride)?;
            let name = CString::new(name).map_err(|_| Error::InvalidCString)?;
            let mut row_bytes = vec![0_u8; layout.payload_bytes];
            self.context.synchronize();
            self.model
                .read_tensor_range(&name, offset, &mut row_bytes)?;
            xor_q1_0_payload(&mut row_bytes, layout.blocks)?;
            self.model.write_tensor_range(&name, offset, &row_bytes)?;
            self.context.synchronize();
            Ok(RowXorResult {
                row,
                blocks: layout.blocks,
                packed_bytes_flipped: layout
                    .blocks
                    .checked_mul(Q1_0_PACKED_BYTES)
                    .ok_or(Error::UnsupportedTensorShape)?,
            })
        }
    }

    pub fn generate_text(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<GenerateOutput> {
        let mut output = GenerateOutput::default();
        let streamed = self.stream_text(prompt, options, |piece, _token| {
            output.text.push_str(piece);
            Ok(())
        })?;
        output.token_count = streamed.token_count;
        Ok(output)
    }

    pub fn stream_text<F>(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        mut on_token: F,
    ) -> Result<GenerateOutput>
    where
        F: FnMut(&str, raw::llama_token) -> Result<()>,
    {
        let prompt_tokens =
            self.tokenize_text(prompt, options.add_special, options.parse_special)?;
        if prompt_tokens.is_empty() {
            return Err(Error::InvalidInput);
        }

        self.context.set_embeddings(false);
        self.context.set_causal_attn(causal_attention_for_operation(
            AttentionOperation::Generation,
        ));
        self.context.clear_memory(true);
        self.evaluate_tokens(&prompt_tokens, 0, true)?;

        let mut sampler = Sampler::new(options)?;
        let mut output = GenerateOutput::default();
        let mut position = prompt_tokens.len() as raw::llama_pos;

        let mut emitted_tokens = 0_usize;
        loop {
            if options.max_new_tokens > 0 && emitted_tokens >= options.max_new_tokens {
                break;
            }
            let token = sampler.sample(&mut self.context);
            if self.model.is_eog(token) {
                break;
            }

            let piece = self.token_to_piece(token, options.emit_special)?;
            on_token(&piece, token)?;
            output.text.push_str(&piece);
            output.token_count += 1;
            emitted_tokens += 1;

            self.context.set_causal_attn(causal_attention_for_operation(
                AttentionOperation::Generation,
            ));
            self.evaluate_tokens(&[token], position, true)?;
            position += 1;
        }

        Ok(output)
    }

    pub fn embed_text(&mut self, text: &str, options: &EmbeddingOptions) -> Result<Vec<f32>> {
        let tokens = self.tokenize_text(text, options.add_special, options.parse_special)?;
        if tokens.is_empty() {
            return Err(Error::InvalidInput);
        }

        self.context.set_embeddings(true);
        self.context.set_causal_attn(causal_attention_for_operation(
            AttentionOperation::Embedding,
        ));
        self.context.clear_memory(true);
        self.evaluate_tokens(&tokens, 0, false)?;
        self.context.synchronize();

        let dim = self.model.n_embd_out();
        if dim <= 0 {
            return Err(Error::EmbeddingUnavailable);
        }
        let ptr = if self.context.pooling_type() == raw::llama_pooling_type::None {
            self.context.embeddings_ith(-1)
        } else {
            self.context.embeddings_seq(0)
        };
        if ptr.is_null() {
            return Err(Error::EmbeddingUnavailable);
        }
        let mut vector = unsafe { slice::from_raw_parts(ptr, dim as usize) }.to_vec();
        if options.normalize {
            normalize_l2(&mut vector);
        }
        Ok(vector)
    }

    pub fn rerank_documents(
        &mut self,
        query: &str,
        documents: &[String],
        options: &RerankOptions,
    ) -> Result<Vec<RerankOutput>> {
        if self.context.pooling_type() != raw::llama_pooling_type::Rank {
            return Err(Error::RerankUnavailable);
        }
        if query.trim().is_empty() || documents.is_empty() {
            return Err(Error::InvalidInput);
        }

        let mut outputs = documents
            .iter()
            .enumerate()
            .map(|(index, document)| {
                self.rerank_score(query, document, options)
                    .map(|(score, token_count)| RerankOutput {
                        index,
                        score,
                        rank: 0,
                        token_count,
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        outputs.sort_by(|left, right| right.score.total_cmp(&left.score));
        for (rank, output) in outputs.iter_mut().enumerate() {
            output.rank = rank;
        }
        Ok(outputs)
    }

    pub fn rerank_documents_jina_v3(
        &mut self,
        query: &str,
        documents: &[String],
        projector: &JinaRerankProjector,
        options: &RerankOptions,
    ) -> Result<Vec<RerankOutput>> {
        if self.context.pooling_type() != raw::llama_pooling_type::None {
            return Err(Error::RerankUnavailable);
        }
        if query.trim().is_empty() || documents.is_empty() {
            return Err(Error::InvalidInput);
        }

        let prompt = jina_v3_rerank_prompt(query, documents, options.instruction.as_deref());
        let tokens = self.tokenize_text(&prompt, options.add_special, true)?;
        if tokens.is_empty() || tokens.len() > self.context.n_ctx() as usize {
            return Err(Error::InvalidInput);
        }

        let query_positions = token_positions(&tokens, JINA_V3_QUERY_EMBED_TOKEN_ID);
        let document_positions = token_positions(&tokens, JINA_V3_DOC_EMBED_TOKEN_ID);
        if query_positions.is_empty() || document_positions.len() != documents.len() {
            return Err(Error::RerankUnavailable);
        }

        self.context
            .set_causal_attn(causal_attention_for_operation(AttentionOperation::Rerank));
        let hidden_states = self.embed_tokens_unpooled(&tokens)?;
        let query_hidden = hidden_states
            .get(query_positions[0])
            .ok_or(Error::EmbeddingUnavailable)?;
        let query_embedding = projector.project(query_hidden)?;

        let mut outputs = Vec::with_capacity(documents.len());
        for (index, position) in document_positions.iter().copied().enumerate() {
            let document_hidden = hidden_states
                .get(position)
                .ok_or(Error::EmbeddingUnavailable)?;
            let document_embedding = projector.project(document_hidden)?;
            outputs.push(RerankOutput {
                index,
                score: cosine_similarity(&query_embedding, &document_embedding),
                rank: 0,
                token_count: tokens.len(),
            });
        }

        outputs.sort_by(|left, right| right.score.total_cmp(&left.score));
        for (rank, output) in outputs.iter_mut().enumerate() {
            output.rank = rank;
        }
        Ok(outputs)
    }

    pub fn rerank_score(
        &mut self,
        query: &str,
        document: &str,
        options: &RerankOptions,
    ) -> Result<(f32, usize)> {
        let tokens = self.rerank_tokens(query, document, options)?;
        if tokens.is_empty() {
            return Err(Error::InvalidInput);
        }

        self.context.set_embeddings(true);
        self.context
            .set_causal_attn(causal_attention_for_operation(AttentionOperation::Rerank));
        self.context.clear_memory(true);
        self.evaluate_tokens(&tokens, 0, false)?;
        self.context.synchronize();

        let dim = self.model.n_embd_out();
        if dim <= 0 {
            return Err(Error::RerankUnavailable);
        }
        let ptr = self.context.embeddings_seq(0);
        let ptr = if ptr.is_null() {
            self.context.embeddings_ith(-1)
        } else {
            ptr
        };
        if ptr.is_null() {
            return Err(Error::RerankUnavailable);
        }
        Ok((unsafe { *ptr }, tokens.len()))
    }

    fn rerank_tokens(
        &self,
        query: &str,
        document: &str,
        options: &RerankOptions,
    ) -> Result<Vec<raw::llama_token>> {
        if let Some(prompt) = self.rerank_prompt(query, document, options)? {
            return self.tokenize_text(&prompt, options.add_special, options.parse_special);
        }
        self.rerank_fallback_tokens(query, document, options)
    }

    fn rerank_prompt(
        &self,
        query: &str,
        document: &str,
        options: &RerankOptions,
    ) -> Result<Option<String>> {
        let template = options
            .prompt_template
            .as_deref()
            .filter(|template| !template.trim().is_empty())
            .map(String::from)
            .or(self.model.chat_template("rerank")?);
        Ok(template.map(|template| {
            let instruction = options.instruction.as_deref().unwrap_or("");
            template
                .replace("{instruction}", instruction)
                .replace("{query}", query)
                .replace("{document}", document)
        }))
    }

    fn rerank_fallback_tokens(
        &self,
        query: &str,
        document: &str,
        options: &RerankOptions,
    ) -> Result<Vec<raw::llama_token>> {
        let mut tokens = Vec::new();
        if self.model.add_bos() {
            push_if_token(&mut tokens, self.model.bos());
        }
        tokens.extend(self.tokenize_text(query, false, options.parse_special)?);
        if self.model.add_eos() {
            push_if_token(&mut tokens, self.model.eos());
        }
        if self.model.add_sep() {
            push_if_token(&mut tokens, self.model.sep());
        } else if !self.model.add_eos() {
            push_if_token(&mut tokens, self.model.eos());
        }
        tokens.extend(self.tokenize_text(document, false, options.parse_special)?);
        if self.model.add_eos() {
            push_if_token(&mut tokens, self.model.eos());
        }
        Ok(tokens)
    }

    fn embed_tokens_unpooled(&mut self, tokens: &[raw::llama_token]) -> Result<Vec<Vec<f32>>> {
        self.context.set_embeddings(true);
        self.context
            .set_causal_attn(causal_attention_for_operation(AttentionOperation::Rerank));
        self.context.clear_memory(true);

        let dim = self.model.n_embd_out();
        if dim <= 0 {
            return Err(Error::EmbeddingUnavailable);
        }
        let dim = dim as usize;
        let mut embeddings = Vec::with_capacity(tokens.len());

        for (chunk_index, chunk) in tokens.chunks(self.n_batch).enumerate() {
            let n_tokens = i32::try_from(chunk.len()).map_err(|_| Error::InvalidInput)?;
            let chunk_start_pos = (chunk_index * self.n_batch) as raw::llama_pos;
            let mut batch = Batch::new(n_tokens, 0, 1);
            fill_batch(&mut batch, chunk, chunk_start_pos, BatchOutput::All);

            let status = if self.model.has_encoder() && !self.model.has_decoder() {
                self.context.encode(&batch)
            } else {
                self.context.decode(&batch)
            };
            match status {
                0 => {}
                code if self.model.has_encoder() && !self.model.has_decoder() => {
                    return Err(Error::EncodeFailed(code));
                }
                code => return Err(Error::DecodeFailed(code)),
            }
            self.context.synchronize();

            for index in 0..chunk.len() {
                let ptr = self.context.embeddings_ith(index as i32);
                if ptr.is_null() {
                    return Err(Error::EmbeddingUnavailable);
                }
                embeddings.push(unsafe { slice::from_raw_parts(ptr, dim) }.to_vec());
            }
        }

        Ok(embeddings)
    }

    fn evaluate_tokens(
        &mut self,
        tokens: &[raw::llama_token],
        start_pos: raw::llama_pos,
        output_last_only: bool,
    ) -> Result<()> {
        let chunk_count = tokens.len().div_ceil(self.n_batch);
        for (chunk_index, chunk) in tokens.chunks(self.n_batch).enumerate() {
            let n_tokens = i32::try_from(chunk.len()).map_err(|_| Error::InvalidInput)?;
            let chunk_start_pos = start_pos + (chunk_index * self.n_batch) as raw::llama_pos;
            let mut batch = Batch::new(n_tokens, 0, 1);
            let output = if !output_last_only {
                BatchOutput::All
            } else if chunk_index + 1 == chunk_count {
                BatchOutput::Last
            } else {
                BatchOutput::None
            };
            fill_batch(&mut batch, chunk, chunk_start_pos, output);

            let status = if self.model.has_encoder() && !self.model.has_decoder() {
                self.context.encode(&batch)
            } else {
                self.context.decode(&batch)
            };
            match status {
                0 => {}
                code if self.model.has_encoder() && !self.model.has_decoder() => {
                    return Err(Error::EncodeFailed(code));
                }
                code => return Err(Error::DecodeFailed(code)),
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Q1_0Layout {
    rows: usize,
    blocks: usize,
    payload_bytes: usize,
    row_stride: usize,
}

fn q1_0_layout(descriptor: &TensorDescriptor) -> Result<Q1_0Layout> {
    if descriptor.type_id != Q1_0_TYPE_ID {
        return Err(Error::UnsupportedTensorType);
    }
    let rows = descriptor.row_count()?;
    let width =
        usize::try_from(descriptor.dimensions[0]).map_err(|_| Error::UnsupportedTensorShape)?;
    if width == 0 || width % Q1_0_BLOCK_VALUES != 0 || rows == 0 {
        return Err(Error::UnsupportedTensorShape);
    }
    let blocks = width / Q1_0_BLOCK_VALUES;
    let payload_bytes = blocks
        .checked_mul(Q1_0_BLOCK_BYTES)
        .ok_or(Error::UnsupportedTensorShape)?;
    if descriptor.strides[0] != Q1_0_BLOCK_BYTES
        || descriptor.strides[1] < payload_bytes
        || descriptor.strides[2]
            != descriptor.strides[1]
                .checked_mul(rows)
                .ok_or(Error::UnsupportedTensorStride)?
    {
        return Err(Error::UnsupportedTensorStride);
    }
    let last_end = rows
        .checked_sub(1)
        .and_then(|last| last.checked_mul(descriptor.strides[1]))
        .and_then(|offset| offset.checked_add(payload_bytes))
        .ok_or(Error::UnsupportedTensorStride)?;
    if last_end > descriptor.nbytes {
        return Err(Error::UnsupportedTensorStride);
    }
    Ok(Q1_0Layout {
        rows,
        blocks,
        payload_bytes,
        row_stride: descriptor.strides[1],
    })
}

fn xor_q1_0_payload(bytes: &mut [u8], blocks: usize) -> Result<()> {
    let expected = blocks
        .checked_mul(Q1_0_BLOCK_BYTES)
        .ok_or(Error::UnsupportedTensorShape)?;
    if blocks == 0 || bytes.len() != expected {
        return Err(Error::UnsupportedTensorShape);
    }
    for block in bytes.chunks_exact_mut(Q1_0_BLOCK_BYTES) {
        for byte in &mut block[Q1_0_SCALE_BYTES..] {
            *byte ^= u8::MAX;
        }
    }
    Ok(())
}

fn mean_q1_0_scale(bytes: &[u8], blocks: usize) -> Result<f32> {
    let expected = blocks
        .checked_mul(Q1_0_BLOCK_BYTES)
        .ok_or(Error::UnsupportedTensorShape)?;
    if blocks == 0 || bytes.len() != expected {
        return Err(Error::UnsupportedTensorShape);
    }
    let sum = bytes
        .chunks_exact(Q1_0_BLOCK_BYTES)
        .map(|block| f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]])).abs())
        .sum::<f32>();
    Ok(sum / blocks as f32)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = (bits & 0x03ff) as u32;
    let f32_bits = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = (fraction << shift) & 0x03ff;
            let adjusted_exponent = 113_u32 - shift;
            sign | (adjusted_exponent << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(f32_bits)
}

const JINA_V3_DOC_EMBED_TOKEN_ID: raw::llama_token = 151_670;
const JINA_V3_QUERY_EMBED_TOKEN_ID: raw::llama_token = 151_671;
const JINA_V3_DOC_EMBED_TOKEN: &str = "<|embed_token|>";
const JINA_V3_QUERY_EMBED_TOKEN: &str = "<|rerank_token|>";

fn jina_v3_rerank_prompt(query: &str, documents: &[String], instruction: Option<&str>) -> String {
    let special_tokens = [JINA_V3_DOC_EMBED_TOKEN, JINA_V3_QUERY_EMBED_TOKEN];
    let query = sanitize_jina_v3_input(query, &special_tokens);
    let mut prompt = String::from(
        "<|im_start|>system\n\
You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. \
If the query is a question, how relevant a passage is depends on how well it answers the question. \
If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. \
If an instruction is provided, you should follow the instruction when determining the ranking.\
<|im_end|>\n<|im_start|>user\n",
    );
    prompt.push_str("I will provide you with ");
    prompt.push_str(&documents.len().to_string());
    prompt.push_str(" passages, each indicated by a numerical identifier. Rank the passages based on their relevance to query: ");
    prompt.push_str(&query);
    prompt.push('\n');

    if let Some(instruction) = instruction.map(str::trim).filter(|value| !value.is_empty()) {
        prompt.push_str("<instruct>\n");
        prompt.push_str(&sanitize_jina_v3_input(instruction, &special_tokens));
        prompt.push_str("\n</instruct>\n");
    }

    for (index, document) in documents.iter().enumerate() {
        if index > 0 {
            prompt.push('\n');
        }
        prompt.push_str("<passage id=\"");
        prompt.push_str(&index.to_string());
        prompt.push_str("\">\n");
        prompt.push_str(&sanitize_jina_v3_input(document, &special_tokens));
        prompt.push_str(JINA_V3_DOC_EMBED_TOKEN);
        prompt.push_str("\n</passage>");
    }

    prompt.push_str("\n<query>\n");
    prompt.push_str(&query);
    prompt.push_str(JINA_V3_QUERY_EMBED_TOKEN);
    prompt.push_str("\n</query><|im_end|>\n<|im_start|>assistant\n");
    prompt
}

fn sanitize_jina_v3_input(input: &str, special_tokens: &[&str]) -> String {
    let mut output = input.to_owned();
    for token in special_tokens {
        output = output.replace(token, "");
    }
    output
}

fn token_positions(tokens: &[raw::llama_token], needle: raw::llama_token) -> Vec<usize> {
    tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| (*token == needle).then_some(index))
        .collect()
}

struct Sampler {
    ptr: NonNull<raw::llama_sampler>,
}

impl Sampler {
    fn new(options: &GenerationOptions) -> Result<Self> {
        let mut params = unsafe { raw::llama_sampler_chain_default_params() };
        params.no_perf = true;
        let chain = NonNull::new(unsafe { raw::llama_sampler_chain_init(params) })
            .ok_or(Error::SamplerInitFailed)?;

        if options.temperature > 0.0 {
            add_sampler(chain, unsafe {
                raw::llama_sampler_init_top_k(options.top_k)
            })?;
            add_sampler(chain, unsafe {
                raw::llama_sampler_init_top_p(options.top_p, 1)
            })?;
            add_sampler(chain, unsafe {
                raw::llama_sampler_init_temp(options.temperature)
            })?;
            add_sampler(chain, unsafe { raw::llama_sampler_init_dist(options.seed) })?;
        } else {
            add_sampler(chain, unsafe { raw::llama_sampler_init_greedy() })?;
        }

        Ok(Self { ptr: chain })
    }

    fn sample(&mut self, context: &mut Context) -> raw::llama_token {
        unsafe { raw::llama_sampler_sample(self.ptr.as_ptr(), context.as_ptr(), -1) }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { raw::llama_sampler_free(self.ptr.as_ptr()) }
    }
}

fn add_sampler(chain: NonNull<raw::llama_sampler>, sampler: *mut raw::llama_sampler) -> Result<()> {
    let sampler = NonNull::new(sampler).ok_or(Error::SamplerInitFailed)?;
    unsafe { raw::llama_sampler_chain_add(chain.as_ptr(), sampler.as_ptr()) };
    Ok(())
}

fn fill_batch(
    batch: &mut Batch,
    tokens: &[raw::llama_token],
    start_pos: raw::llama_pos,
    output: BatchOutput,
) {
    let raw = batch.as_raw_mut();
    raw.n_tokens = tokens.len() as i32;
    for (index, token) in tokens.iter().copied().enumerate() {
        unsafe {
            *raw.token.add(index) = token;
            *raw.pos.add(index) = start_pos + index as raw::llama_pos;
            *raw.n_seq_id.add(index) = 1;
            **raw.seq_id.add(index) = 0;
            *raw.logits.add(index) = match output {
                BatchOutput::All => 1,
                BatchOutput::None => 0,
                BatchOutput::Last if index + 1 == tokens.len() => 1,
                BatchOutput::Last => 0,
            };
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchOutput {
    All,
    None,
    Last,
}

fn push_if_token(tokens: &mut Vec<raw::llama_token>, token: raw::llama_token) {
    if token != raw::LLAMA_TOKEN_NULL {
        tokens.push(token);
    }
}

fn read_safetensor_f32(bytes: &[u8], name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    if bytes.len() < 8 {
        return Err(Error::ProjectorInvalid);
    }
    let header_len = u64::from_le_bytes(
        bytes[0..8]
            .try_into()
            .map_err(|_| Error::ProjectorInvalid)?,
    ) as usize;
    let header_start = 8_usize;
    let header_end = header_start
        .checked_add(header_len)
        .ok_or(Error::ProjectorInvalid)?;
    if header_end > bytes.len() {
        return Err(Error::ProjectorInvalid);
    }
    let header =
        str::from_utf8(&bytes[header_start..header_end]).map_err(|_| Error::ProjectorInvalid)?;
    let section = safetensor_tensor_section(header, name)?;
    let dtype = json_string_field(section, "dtype").ok_or(Error::ProjectorInvalid)?;
    if dtype != "F32" {
        return Err(Error::ProjectorInvalid);
    }
    let shape = json_usize_array_field(section, "shape")?;
    let offsets = json_usize_array_field(section, "data_offsets")?;
    if offsets.len() != 2 || shape.is_empty() {
        return Err(Error::ProjectorInvalid);
    }
    let item_count = shape
        .iter()
        .try_fold(1_usize, |acc, value| acc.checked_mul(*value))
        .ok_or(Error::ProjectorInvalid)?;
    let data_start = header_end
        .checked_add(offsets[0])
        .ok_or(Error::ProjectorInvalid)?;
    let data_end = header_end
        .checked_add(offsets[1])
        .ok_or(Error::ProjectorInvalid)?;
    if data_start > data_end || data_end > bytes.len() || data_end - data_start != item_count * 4 {
        return Err(Error::ProjectorInvalid);
    }

    let mut data = Vec::with_capacity(item_count);
    for chunk in bytes[data_start..data_end].chunks_exact(4) {
        data.push(f32::from_le_bytes(
            chunk.try_into().map_err(|_| Error::ProjectorInvalid)?,
        ));
    }
    Ok((shape, data))
}

fn safetensor_tensor_section<'a>(header: &'a str, name: &str) -> Result<&'a str> {
    let key = {
        let mut key = String::from("\"");
        key.push_str(name);
        key.push('"');
        key
    };
    let start = header.find(&key).ok_or(Error::ProjectorInvalid)? + key.len();
    let object_start = header[start..]
        .find('{')
        .map(|index| start + index)
        .ok_or(Error::ProjectorInvalid)?;
    let mut depth = 0_i32;
    for (offset, byte) in header.as_bytes()[object_start..]
        .iter()
        .copied()
        .enumerate()
    {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    let end = object_start + offset + 1;
                    return Ok(&header[object_start..end]);
                }
            }
            _ => {}
        }
    }
    Err(Error::ProjectorInvalid)
}

fn json_string_field<'a>(section: &'a str, field: &str) -> Option<&'a str> {
    let key = {
        let mut key = String::from("\"");
        key.push_str(field);
        key.push('"');
        key
    };
    let key_start = section.find(&key)? + key.len();
    let colon = section[key_start..]
        .find(':')
        .map(|index| key_start + index)?;
    let value_start = section[colon + 1..]
        .find('"')
        .map(|index| colon + 1 + index + 1)?;
    let value_end = section[value_start..]
        .find('"')
        .map(|index| value_start + index)?;
    Some(&section[value_start..value_end])
}

fn json_usize_array_field(section: &str, field: &str) -> Result<Vec<usize>> {
    let key = {
        let mut key = String::from("\"");
        key.push_str(field);
        key.push('"');
        key
    };
    let key_start = section.find(&key).ok_or(Error::ProjectorInvalid)? + key.len();
    let array_start = section[key_start..]
        .find('[')
        .map(|index| key_start + index + 1)
        .ok_or(Error::ProjectorInvalid)?;
    let array_end = section[array_start..]
        .find(']')
        .map(|index| array_start + index)
        .ok_or(Error::ProjectorInvalid)?;
    let mut values = Vec::new();
    for part in section[array_start..array_end].split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        values.push(part.parse().map_err(|_| Error::ProjectorInvalid)?);
    }
    Ok(values)
}

pub fn apply_chat_template(
    template: Option<&str>,
    messages: &[ChatMessage],
    add_assistant_marker: bool,
) -> Result<String> {
    let template = template
        .map(CString::new)
        .transpose()
        .map_err(|_| Error::InvalidCString)?;
    let mut role_storage = Vec::with_capacity(messages.len());
    let mut content_storage = Vec::with_capacity(messages.len());
    let mut raw_messages = Vec::with_capacity(messages.len());

    for message in messages {
        let role = CString::new(message.role.as_str()).map_err(|_| Error::InvalidCString)?;
        let content = CString::new(message.content.as_str()).map_err(|_| Error::InvalidCString)?;
        raw_messages.push(raw::llama_chat_message {
            role: role.as_ptr(),
            content: content.as_ptr(),
        });
        role_storage.push(role);
        content_storage.push(content);
    }

    let mut capacity = messages
        .iter()
        .map(|message| message.role.len() + message.content.len() + 16)
        .sum::<usize>()
        .saturating_mul(2)
        .max(256);
    loop {
        let mut bytes = vec![0_u8; capacity];
        let written = unsafe {
            raw::llama_chat_apply_template(
                template
                    .as_ref()
                    .map_or(core::ptr::null(), |value| value.as_ptr()),
                raw_messages.as_ptr(),
                raw_messages.len(),
                add_assistant_marker,
                bytes.as_mut_ptr().cast(),
                bytes.len() as i32,
            )
        };
        if written >= 0 && written as usize <= bytes.len() {
            bytes.truncate(written as usize);
            return Ok(String::from_utf8_lossy(&bytes).into_owned());
        }
        if written <= 0 {
            return Err(Error::ChatTemplateFailed);
        }
        capacity = written as usize;
    }
}

fn normalize_l2(vector: &mut [f32]) {
    let norm = libm::sqrtf(vector.iter().map(|value| value * value).sum::<f32>());
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut left_norm = 0.0_f32;
    let mut right_norm = 0.0_f32;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }
    let denom = libm::sqrtf(left_norm) * libm::sqrtf(right_norm);
    if denom > 0.0 && denom.is_finite() {
        dot / denom
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q1_descriptor(width: u64, rows: u64, row_padding: usize) -> TensorDescriptor {
        let blocks = width as usize / Q1_0_BLOCK_VALUES;
        let payload = blocks * Q1_0_BLOCK_BYTES;
        let row_stride = payload + row_padding;
        TensorDescriptor {
            name: "blk.0.ffn_gate.weight".into(),
            type_id: Q1_0_TYPE_ID,
            type_name: "q1_0".into(),
            dimensions: [width, rows, 1, 1],
            strides: [
                Q1_0_BLOCK_BYTES,
                row_stride,
                row_stride * rows as usize,
                row_stride,
            ],
            n_dims: 2,
            nbytes: row_stride * (rows as usize - 1) + payload,
            backend: "CPU".into(),
        }
    }

    #[test]
    fn generation_options_default_has_no_explicit_token_cap() {
        assert_eq!(GenerationOptions::default().max_new_tokens, 0);
    }

    /// Documents the causal-attention contract enforced before each evaluate/decode path.
    /// Full model sessions are not required for this contract test.
    #[test]
    fn attention_mode_contract_per_operation() {
        assert!(causal_attention_for_operation(
            AttentionOperation::Generation
        ));
        assert!(!causal_attention_for_operation(
            AttentionOperation::Embedding
        ));
        assert!(!causal_attention_for_operation(AttentionOperation::Rerank));
    }

    #[test]
    fn q1_0_layout_accepts_contiguous_and_padded_rows() {
        assert_eq!(
            q1_0_layout(&q1_descriptor(256, 3, 0)).unwrap(),
            Q1_0Layout {
                rows: 3,
                blocks: 2,
                payload_bytes: 36,
                row_stride: 36,
            }
        );
        assert_eq!(
            q1_0_layout(&q1_descriptor(128, 2, 14)).unwrap().row_stride,
            32
        );
    }

    #[test]
    fn q1_0_layout_rejects_wrong_type_shape_rows_and_stride() {
        let mut descriptor = q1_descriptor(128, 3, 0);
        descriptor.type_id = raw::ggml_type::Tq1_0 as i32;
        assert_eq!(q1_0_layout(&descriptor), Err(Error::UnsupportedTensorType));

        let mut descriptor = q1_descriptor(128, 3, 0);
        descriptor.n_dims = 3;
        assert_eq!(q1_0_layout(&descriptor), Err(Error::UnsupportedTensorShape));

        let descriptor = q1_descriptor(127, 3, 0);
        assert_eq!(q1_0_layout(&descriptor), Err(Error::UnsupportedTensorShape));

        let mut descriptor = q1_descriptor(128, 3, 0);
        descriptor.strides[1] = Q1_0_BLOCK_BYTES - 1;
        assert_eq!(
            q1_0_layout(&descriptor),
            Err(Error::UnsupportedTensorStride)
        );
    }

    #[test]
    fn q1_0_xor_preserves_scales_flips_payload_and_is_self_inverse() {
        let mut bytes = Vec::new();
        for block_index in 0..3_u8 {
            bytes.extend_from_slice(&[block_index, block_index + 1]);
            bytes.extend((0..Q1_0_PACKED_BYTES).map(|value| value as u8 + block_index));
        }
        let original = bytes.clone();
        xor_q1_0_payload(&mut bytes, 3).unwrap();
        for block_index in 0..3 {
            let offset = block_index * Q1_0_BLOCK_BYTES;
            assert_eq!(
                &bytes[offset..offset + Q1_0_SCALE_BYTES],
                &original[offset..offset + Q1_0_SCALE_BYTES]
            );
            for byte_index in Q1_0_SCALE_BYTES..Q1_0_BLOCK_BYTES {
                assert_eq!(bytes[offset + byte_index], !original[offset + byte_index]);
            }
        }
        xor_q1_0_payload(&mut bytes, 3).unwrap();
        assert_eq!(bytes, original);
    }

    #[test]
    fn q1_0_scale_aggregation_uses_mean_absolute_fp16_scale() {
        let mut bytes = vec![0_u8; 2 * Q1_0_BLOCK_BYTES];
        bytes[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes()); // 1.0
        bytes[Q1_0_BLOCK_BYTES..Q1_0_BLOCK_BYTES + 2].copy_from_slice(&0xc000_u16.to_le_bytes()); // -2.0
        assert_eq!(mean_q1_0_scale(&bytes, 2).unwrap(), 1.5);
    }

    #[test]
    fn f16_conversion_covers_zero_subnormal_normal_infinity_and_nan() {
        assert_eq!(f16_bits_to_f32(0), 0.0);
        assert_eq!(f16_bits_to_f32(0x8000).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0x0001), 2.0_f32.powi(-24));
        assert!(f16_bits_to_f32(0x7c00).is_infinite());
        assert!(f16_bits_to_f32(0x7e00).is_nan());
    }
}
