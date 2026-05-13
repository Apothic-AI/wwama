#![no_std]

use core::ffi::CStr;
use core::ptr::NonNull;

pub mod raw {
    #![allow(non_camel_case_types)]

    use core::ffi::{c_char, c_float, c_void};

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

    unsafe extern "C" {
        pub fn llama_model_default_params() -> llama_model_params;
        pub fn llama_context_default_params() -> llama_context_params;

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

        pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
        pub fn llama_model_has_encoder(model: *const llama_model) -> bool;
        pub fn llama_model_has_decoder(model: *const llama_model) -> bool;
        pub fn llama_model_n_embd_out(model: *const llama_model) -> i32;

        pub fn llama_pooling_type(ctx: *const llama_context) -> llama_pooling_type;
        pub fn llama_set_embeddings(ctx: *mut llama_context, embeddings: bool);
        pub fn llama_synchronize(ctx: *mut llama_context);

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
    }
}

pub use raw::{
    ggml_type, llama_attention_type, llama_batch, llama_context_params, llama_flash_attn_type,
    llama_model_params, llama_pooling_type, llama_pos, llama_rope_scaling_type, llama_seq_id,
    llama_split_mode, llama_token,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    ModelLoadFailed,
    ContextInitFailed,
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

    pub fn load_from_file(path: &CStr, params: raw::llama_model_params) -> Result<Self, Error> {
        let ptr = unsafe { raw::llama_model_load_from_file(path.as_ptr(), params) };
        NonNull::new(ptr)
            .map(|ptr| Self { ptr })
            .ok_or(Error::ModelLoadFailed)
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

    pub fn new(model: &Model, params: raw::llama_context_params) -> Result<Self, Error> {
        let ptr = unsafe { raw::llama_init_from_model(model.as_ptr(), params) };
        NonNull::new(ptr)
            .map(|ptr| Self { ptr })
            .ok_or(Error::ContextInitFailed)
    }

    pub fn as_ptr(&self) -> *mut raw::llama_context {
        self.ptr.as_ptr()
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
