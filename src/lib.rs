#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use alloc::ffi::CString;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::CStr;
use core::fmt;
use core::ptr::NonNull;
use core::slice;

pub mod raw {
    #![allow(non_camel_case_types)]

    use core::ffi::{c_char, c_float, c_void};

    pub const LLAMA_DEFAULT_SEED: u32 = 0xFFFF_FFFF;

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
        pub n_threads: i32,
        pub n_threads_batch: i32,
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
        pub fn llama_vocab_n_tokens(vocab: *const llama_vocab) -> i32;
        pub fn llama_vocab_is_eog(vocab: *const llama_vocab, token: llama_token) -> bool;
        pub fn llama_vocab_get_add_bos(vocab: *const llama_vocab) -> bool;
        pub fn llama_vocab_get_add_eos(vocab: *const llama_vocab) -> bool;

        pub fn llama_pooling_type(ctx: *const llama_context) -> llama_pooling_type;
        pub fn llama_set_embeddings(ctx: *mut llama_context, embeddings: bool);
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
    }
}

pub use raw::{
    LLAMA_DEFAULT_SEED, ggml_type, llama_attention_type, llama_batch, llama_chat_message,
    llama_context_params, llama_flash_attn_type, llama_model_params, llama_pooling_type, llama_pos,
    llama_rope_scaling_type, llama_sampler_chain_params, llama_seq_id, llama_split_mode,
    llama_token,
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
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

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

    pub fn n_vocab(&self) -> i32 {
        unsafe { raw::llama_vocab_n_tokens(self.vocab()) }
    }

    pub fn add_bos(&self) -> bool {
        unsafe { raw::llama_vocab_get_add_bos(self.vocab()) }
    }

    pub fn add_eos(&self) -> bool {
        unsafe { raw::llama_vocab_get_add_eos(self.vocab()) }
    }

    pub fn is_eog(&self, token: raw::llama_token) -> bool {
        unsafe { raw::llama_vocab_is_eog(self.vocab(), token) }
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
            embeddings: false,
            pooling_type: raw::llama_pooling_type::Unspecified,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerationOptions {
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
            max_new_tokens: 256,
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

#[derive(Clone, Debug, Default)]
pub struct GenerateOutput {
    pub text: String,
    pub token_count: usize,
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
}

impl Session {
    pub fn load_from_path(path: &str, options: SessionOptions) -> Result<Self> {
        Backend::init();

        let mut model_params = Model::default_params();
        model_params.n_gpu_layers = options.n_gpu_layers;
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
        let context = Context::new(&model, context_params)?;

        Ok(Self {
            context,
            model,
            n_batch: options.n_batch.max(1) as usize,
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
        self.context.clear_memory(true);
        self.evaluate_tokens(&prompt_tokens, 0, true)?;

        let mut sampler = Sampler::new(options)?;
        let mut output = GenerateOutput::default();
        let mut position = prompt_tokens.len() as raw::llama_pos;

        for _ in 0..options.max_new_tokens {
            let token = sampler.sample(&mut self.context);
            if self.model.is_eog(token) {
                break;
            }

            let piece = self.token_to_piece(token, options.emit_special)?;
            on_token(&piece, token)?;
            output.text.push_str(&piece);
            output.token_count += 1;

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

    fn evaluate_tokens(
        &mut self,
        tokens: &[raw::llama_token],
        start_pos: raw::llama_pos,
        output_last_only: bool,
    ) -> Result<()> {
        for (chunk_index, chunk) in tokens.chunks(self.n_batch).enumerate() {
            let n_tokens = i32::try_from(chunk.len()).map_err(|_| Error::InvalidInput)?;
            let chunk_start_pos = start_pos + (chunk_index * self.n_batch) as raw::llama_pos;
            let mut batch = Batch::new(n_tokens, 0, 1);
            fill_batch(&mut batch, chunk, chunk_start_pos, output_last_only);

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
    output_last_only: bool,
) {
    let raw = batch.as_raw_mut();
    raw.n_tokens = tokens.len() as i32;
    for (index, token) in tokens.iter().copied().enumerate() {
        unsafe {
            *raw.token.add(index) = token;
            *raw.pos.add(index) = start_pos + index as raw::llama_pos;
            *raw.n_seq_id.add(index) = 1;
            **raw.seq_id.add(index) = 0;
            *raw.logits.add(index) = if !output_last_only || index + 1 == tokens.len() {
                1
            } else {
                0
            };
        }
    }
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
