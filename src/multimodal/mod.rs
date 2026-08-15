//! Optional runtime bridge to the pinned Furnace `libmtmd` frontend.
//!
//! The Rust engine intentionally never links llama.cpp at build time.  When
//! present, the bridge only produces text tokens and copied image embeddings;
//! ReInstinct remains responsible for every transformer evaluation.

use std::ffi::{CStr, CString, c_char, c_int};
use std::path::Path;
use std::ptr::NonNull;

use libloading::Library;

const ABI_VERSION: u32 = 1;
const ERROR_CAP: usize = 512;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DecoderPos { pub t: u32, pub x: u32, pub y: u32, pub z: u32 }

#[derive(Debug, Clone)]
pub enum Chunk {
    Text { tokens: Vec<u32>, n_pos: usize },
    Image { embeddings: Vec<f32>, embedding_dim: usize, positions: Vec<DecoderPos>, n_pos: usize },
}

#[derive(Debug, Clone)]
pub struct ProcessedImage { pub uses_mrope: bool, pub chunks: Vec<Chunk> }

#[repr(C)]
struct ContextOpaque { _private: [u8; 0] }
#[repr(C)]
struct ResultOpaque { _private: [u8; 0] }

type Create = unsafe extern "C" fn(*const c_char, *const c_char, c_int, c_int, *mut c_char, usize) -> *mut ContextOpaque;
type Destroy = unsafe extern "C" fn(*mut ContextOpaque);
type Process = unsafe extern "C" fn(*mut ContextOpaque, *const c_char, *const u8, usize, *mut *mut ResultOpaque, *mut c_char, usize) -> c_int;
type ResultDestroy = unsafe extern "C" fn(*mut ResultOpaque);
type Count = unsafe extern "C" fn(*const ResultOpaque) -> usize;
type ChunkType = unsafe extern "C" fn(*const ResultOpaque, usize) -> c_int;
type ChunkSize = unsafe extern "C" fn(*const ResultOpaque, usize) -> usize;
type Tokens = unsafe extern "C" fn(*const ResultOpaque, usize) -> *const u32;
type Embeddings = unsafe extern "C" fn(*const ResultOpaque, usize) -> *const f32;
type Positions = unsafe extern "C" fn(*const ResultOpaque, usize) -> *const DecoderPos;
type UsesMrope = unsafe extern "C" fn(*const ResultOpaque) -> c_int;
type AbiVersion = unsafe extern "C" fn() -> u32;

/// A serialized bridge context. `libmtmd` reuses output storage per encode,
/// so callers must hold `&mut self` for every operation.
pub struct MtmdProcessor {
    _library: Library,
    context: NonNull<ContextOpaque>,
    destroy: Destroy,
    process: Process,
    result_destroy: ResultDestroy,
    chunk_count: Count,
    chunk_type: ChunkType,
    chunk_n_tokens: ChunkSize,
    chunk_n_pos: ChunkSize,
    chunk_tokens: Tokens,
    chunk_embeddings: Embeddings,
    chunk_embedding_dim: ChunkSize,
    chunk_positions: Positions,
    result_uses_mrope: UsesMrope,
}

impl MtmdProcessor {
    pub fn load(library_path: &Path, model: &Path, mmproj: &Path, use_gpu: bool, threads: i32) -> Result<Self, String> {
        if threads < 1 { return Err("mtmd thread count must be positive".into()); }
        let library = unsafe { Library::new(library_path) }.map_err(|e| format!("load mtmd bridge: {e}"))?;
        unsafe {
            let abi: AbiVersion = *library.get(b"ri_mtmd_abi_version\0").map_err(|e| e.to_string())?;
            if abi() != ABI_VERSION { return Err(format!("mtmd bridge ABI {} is incompatible with required ABI {ABI_VERSION}", abi())); }
            let create: Create = *library.get(b"ri_mtmd_create\0").map_err(|e| e.to_string())?;
            let model = path_cstring(model)?;
            let mmproj = path_cstring(mmproj)?;
            let mut error = [0 as c_char; ERROR_CAP];
            let raw = create(model.as_ptr(), mmproj.as_ptr(), if use_gpu { 1 } else { 0 }, threads, error.as_mut_ptr(), error.len());
            let context = NonNull::new(raw).ok_or_else(|| bridge_error("create mtmd context", &error))?;
            Ok(Self {
                destroy: *library.get(b"ri_mtmd_destroy\0").map_err(|e| e.to_string())?,
                process: *library.get(b"ri_mtmd_process\0").map_err(|e| e.to_string())?,
                result_destroy: *library.get(b"ri_mtmd_result_destroy\0").map_err(|e| e.to_string())?,
                chunk_count: *library.get(b"ri_mtmd_result_chunk_count\0").map_err(|e| e.to_string())?,
                chunk_type: *library.get(b"ri_mtmd_result_chunk_type\0").map_err(|e| e.to_string())?,
                chunk_n_tokens: *library.get(b"ri_mtmd_result_chunk_n_tokens\0").map_err(|e| e.to_string())?,
                chunk_n_pos: *library.get(b"ri_mtmd_result_chunk_n_pos\0").map_err(|e| e.to_string())?,
                chunk_tokens: *library.get(b"ri_mtmd_result_chunk_tokens\0").map_err(|e| e.to_string())?,
                chunk_embeddings: *library.get(b"ri_mtmd_result_chunk_embeddings\0").map_err(|e| e.to_string())?,
                chunk_embedding_dim: *library.get(b"ri_mtmd_result_chunk_embedding_dim\0").map_err(|e| e.to_string())?,
                chunk_positions: *library.get(b"ri_mtmd_result_chunk_positions\0").map_err(|e| e.to_string())?,
                result_uses_mrope: *library.get(b"ri_mtmd_result_uses_mrope\0").map_err(|e| e.to_string())?,
                _library: library,
                context,
            })
        }
    }

    pub fn process(&mut self, prompt: &str, image: &[u8]) -> Result<ProcessedImage, String> {
        if image.is_empty() { return Err("image input is empty".into()); }
        let prompt = CString::new(prompt).map_err(|_| "prompt contains an interior NUL byte")?;
        let mut error = [0 as c_char; ERROR_CAP];
        let mut raw = std::ptr::null_mut();
        if unsafe { (self.process)(self.context.as_ptr(), prompt.as_ptr(), image.as_ptr(), image.len(), &mut raw, error.as_mut_ptr(), error.len()) } != 0 {
            return Err(bridge_error("process image", &error));
        }
        let result = NonNull::new(raw).ok_or_else(|| "mtmd bridge returned success with no result".to_string())?;
        let output = unsafe { self.copy_result(result.as_ptr()) };
        unsafe { (self.result_destroy)(result.as_ptr()) };
        output
    }

    unsafe fn copy_result(&self, result: *const ResultOpaque) -> Result<ProcessedImage, String> {
        // Every FFI function pointer and raw pointer dereference is confined
        // to this block. The bridge contract guarantees result ownership until
        // `ri_mtmd_result_destroy`, which the caller performs immediately
        // after this method copies all storage into Rust-owned vectors.
        unsafe {
        let uses_mrope = (self.result_uses_mrope)(result) != 0;
        let mut chunks = Vec::with_capacity((self.chunk_count)(result));
        for index in 0..(self.chunk_count)(result) {
            let n_tokens = (self.chunk_n_tokens)(result, index);
            let n_pos = (self.chunk_n_pos)(result, index);
            match (self.chunk_type)(result, index) {
                0 => {
                    let ptr = (self.chunk_tokens)(result, index);
                    if n_tokens != 0 && ptr.is_null() { return Err("mtmd bridge returned null text token storage".into()); }
                    let tokens = if n_tokens == 0 { Vec::new() } else { std::slice::from_raw_parts(ptr, n_tokens).to_vec() };
                    chunks.push(Chunk::Text { tokens, n_pos });
                }
                1 => {
                    let dim = (self.chunk_embedding_dim)(result, index);
                    let count = n_tokens.checked_mul(dim).ok_or("image embedding size overflow")?;
                    let ptr = (self.chunk_embeddings)(result, index);
                    if count != 0 && ptr.is_null() { return Err("mtmd bridge returned null image embeddings".into()); }
                    let embeddings = if count == 0 { Vec::new() } else { std::slice::from_raw_parts(ptr, count).to_vec() };
                    if !embeddings.iter().all(|v| v.is_finite()) { return Err("mtmd bridge returned non-finite image embeddings".into()); }
                    let positions = if uses_mrope {
                        let ptr = (self.chunk_positions)(result, index);
                        if n_tokens != 0 && ptr.is_null() { return Err("mtmd bridge returned null M-RoPE positions".into()); }
                        if n_tokens == 0 { Vec::new() } else { std::slice::from_raw_parts(ptr, n_tokens).to_vec() }
                    } else { Vec::new() };
                    chunks.push(Chunk::Image { embeddings, embedding_dim: dim, positions, n_pos });
                }
                other => return Err(format!("mtmd bridge returned unknown chunk type {other}")),
            }
        }
        Ok(ProcessedImage { uses_mrope, chunks })
        }
    }
}

impl Drop for MtmdProcessor {
    fn drop(&mut self) { unsafe { (self.destroy)(self.context.as_ptr()) }; }
}

fn path_cstring(path: &Path) -> Result<CString, String> {
    CString::new(path.to_string_lossy().as_bytes()).map_err(|_| format!("path contains an interior NUL: {}", path.display()))
}

fn bridge_error(operation: &str, error: &[c_char]) -> String {
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    if message.is_empty() { format!("mtmd bridge failed to {operation}") } else { format!("mtmd bridge failed to {operation}: {message}") }
}
