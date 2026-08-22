//! Multi-model HTTP server: one GPU, several models resident, requests
//! serialised through a single FIFO queue.
//!
//! Three ports — Big LLM, Small LLM, Embedder — each its own listener.
//! Every request is pushed onto one shared channel; a single worker
//! thread owns the GPU, pulls jobs in order, runs the target model, and
//! sends the response back to the waiting connection. Models never run
//! concurrently (one GPU), so the worker simply blocks per job.
//!
//! API is OpenAI-shaped: `POST /v1/completions` on the LLM ports,
//! `POST /v1/embeddings` on the embedder port. The embedder is a
//! follow-up (nomic-bert is a new encoder architecture) — its port
//! answers 503 until then.

mod http;
mod json;
mod api;
pub mod config;

pub use api::DashboardLogWriter;

pub fn dashboard_log_writer() -> DashboardLogWriter { DashboardLogWriter }

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use json::Json;
use tracing::{error, info, warn};

use crate::gguf::GgufFile;
use crate::runtime::KernelCache;

/// Server-wide counters, shared between worker + acceptors. Atomic
/// so the `/metrics` endpoint can read them without locking.
#[derive(Default)]
struct Metrics {
    requests_total:    AtomicU64,    // every HTTP request reaching a route
    inference_runs_total: AtomicU64, // POST /v1 requests admitted to run history
    next_run_id: AtomicU64,          // dense inference-only flight-recorder IDs
    requests_ok:       AtomicU64,    // 2xx replies
    requests_4xx:      AtomicU64,
    requests_5xx:      AtomicU64,
    prompt_tokens:     AtomicU64,    // total across all completed requests
    completion_tokens: AtomicU64,
    prefill_us_total:  AtomicU64,    // sum of model/image prefill time
    decode_us_total:   AtomicU64,    // sum of generation-only wall time
    ttft_us_total:     AtomicU64,
    requests_eos:      AtomicU64,    // finish_reason == stop
    requests_length:   AtomicU64,    // finish_reason == length
    panics_recovered:  AtomicU64,    // catch_unwind hits in the worker
    start_unix:        AtomicU64,    // server up-time anchor
}

impl Metrics {
    fn new() -> Self {
        let m = Self::default();
        m.start_unix.store(unix_now(), Ordering::Relaxed);
        m
    }

    /// Prometheus-style text exposition. Cheap — read each counter once.
    fn render_prometheus(&self) -> String {
        use std::fmt::Write;
        let mut s = String::with_capacity(2048);
        let metric = |s: &mut String, name: &str, help: &str, value: u64| {
            let _ = writeln!(s, "# HELP reinstinct_{name} {help}");
            let _ = writeln!(s, "# TYPE reinstinct_{name} counter");
            let _ = writeln!(s, "reinstinct_{name} {value}");
        };
        metric(&mut s, "requests_total", "total HTTP requests reaching a route",
               self.requests_total.load(Ordering::Relaxed));
        metric(&mut s, "inference_runs_total", "POST /v1 requests admitted to inference history",
               self.inference_runs_total.load(Ordering::Relaxed));
        metric(&mut s, "requests_ok_total", "requests with a 2xx reply",
               self.requests_ok.load(Ordering::Relaxed));
        metric(&mut s, "requests_4xx_total", "requests with a 4xx reply",
               self.requests_4xx.load(Ordering::Relaxed));
        metric(&mut s, "requests_5xx_total", "requests with a 5xx reply",
               self.requests_5xx.load(Ordering::Relaxed));
        metric(&mut s, "prompt_tokens_total", "sum of prompt_tokens across completed requests",
               self.prompt_tokens.load(Ordering::Relaxed));
        metric(&mut s, "completion_tokens_total", "sum of completion_tokens",
               self.completion_tokens.load(Ordering::Relaxed));
        metric(&mut s, "prefill_us_total", "sum of prefill wall time (microseconds)",
               self.prefill_us_total.load(Ordering::Relaxed));
        metric(&mut s, "decode_us_total", "sum of decode wall time (microseconds)",
               self.decode_us_total.load(Ordering::Relaxed));
        metric(&mut s, "ttft_us_total", "sum of time-to-first-token wall time (microseconds)",
               self.ttft_us_total.load(Ordering::Relaxed));
        metric(&mut s, "requests_eos_total", "requests that ended at EOS",
               self.requests_eos.load(Ordering::Relaxed));
        metric(&mut s, "requests_length_total", "requests stopped by max_tokens or timeout",
               self.requests_length.load(Ordering::Relaxed));
        metric(&mut s, "panics_recovered_total", "panics caught by the worker's catch_unwind",
               self.panics_recovered.load(Ordering::Relaxed));
        let _ = writeln!(s, "# HELP reinstinct_start_unix_seconds server start time");
        let _ = writeln!(s, "# TYPE reinstinct_start_unix_seconds gauge");
        let _ = writeln!(s, "reinstinct_start_unix_seconds {}",
                         self.start_unix.load(Ordering::Relaxed));
        s
    }
}

/// Which resident model a request targets.
#[derive(Clone, Copy, PartialEq)]
enum Target { Big, Small, Embed }

impl Target {
    fn label(self) -> &'static str {
        match self { Target::Big => "big", Target::Small => "small", Target::Embed => "embed" }
    }
}

/// What the client sent as the prompt — either a raw text completion
/// (`/v1/completions` POST `prompt`) or a chat-completions message
/// array (`/v1/chat/completions` POST `messages`). The worker uses
/// this variant to pick the response shape (`text_completion` vs
/// `chat.completion`) AND, for the chat path, to apply the right
/// per-architecture chat template before tokenization.
enum PromptInput {
    Raw(String),
    Chat(Vec<crate::chat::ChatMessage>),
    /// One OpenAI structured-content image, retained as decoded bytes until
    /// the serialized GPU worker passes it to the mtmd bridge.
    ChatVision { messages: Vec<crate::chat::ChatMessage>, image: Vec<u8> },
}

/// A parsed `/v1/completions` or `/v1/chat/completions` request.
struct GenReq {
    prompt: PromptInput,
    /// Optional client-supplied model ID. The connection handler validates it
    /// against the one model advertised by this port before queueing work.
    model: Option<String>,
    max_tokens: usize,
    /// Up to four OpenAI stop strings. Matching is done against the decoded
    /// UTF-8 stream so a sequence split across token boundaries is handled.
    stop: Vec<String>,
    sampler: crate::sampling::SamplerParams,
    /// MTP spec-decode opt-in/opt-out. `None` ⇒ use the server default
    /// (true if the target has a drafter loaded, false otherwise).
    /// `Some(false)` lets a per-turn classifier disable the drafter on
    /// creative work where it would just waste verify cycles.
    use_speculative: Option<bool>,
    /// Per-request K override for spec-decode. `None` ⇒ server default
    /// (currently 3). Ignored when spec-decode is off.
    speculative_k: Option<usize>,
    /// Drafter confidence early-stop threshold. After each AR draft step
    /// the drafter's chosen-token probability gets checked; if below
    /// `speculative_p_min`, the K-round terminates early. `0.0` (default)
    /// disables. Lets K=4 default safely (creative prompts truncate the
    /// draft instead of paying for 4 verify positions at 30% accept).
    speculative_p_min: f32,
    /// Wall-clock cap on the generation. If decode runs past it, the
    /// request stops early with `finish_reason: "length"`. None disables.
    request_timeout: Option<std::time::Duration>,
    /// OpenAI `stream: true` — Server-Sent Events response. The worker
    /// emits one `Chunk` per decoded token; the connection writes them
    /// as `data: {json}\n\n`. Implicit cancellation: if the client closes
    /// the socket, the next Chunk send fails (Receiver dropped) and the
    /// worker stops generating.
    stream: bool,
    /// OpenAI `stream_options.include_usage`. When set with `stream:true`,
    /// emit one extra SSE chunk at end-of-stream with the `usage` object
    /// (prompt/completion/total token counts) — clients like Open WebUI
    /// use this to compute decode tok/s for the display. No-op for
    /// non-streaming responses (usage is always in the body there).
    stream_include_usage: bool,
    /// OpenAI `logprobs`. `0` ⇒ omit logprobs entirely (the common case;
    /// no extra cost). `1..=N` ⇒ report the chosen token's logprob plus
    /// the top-(N-1) alternatives with theirs. Capped at 20 server-side.
    ///
    /// Not supported on the MTP spec-decode path — the verify-step
    /// softmax probabilities aren't currently routed back from
    /// `spec_decode_generate`. Spec-decode responses always emit
    /// `logprobs: null`. Disable spec-decode (`use_speculative: false`)
    /// if you need logprobs.
    top_logprobs_n: usize,
}

/// Per-token logprob diagnostic for the OpenAI `logprobs:true` field.
/// `token` is the decoded UTF-8 byte sequence the token produced
/// (rendered exactly as a chat client would display it). `top_alts`
/// is the alternatives list — same shape: each alternative's text
/// and its log-probability under the post-filter, post-temperature
/// distribution. Includes the chosen token's own entry so a client
/// can rank-find it without a separate field.
#[derive(Clone, Debug)]
struct TokenLogprob {
    token: String,
    logprob: f32,
    top_alts: Vec<(String, f32)>,
}

/// Messages from the GPU worker to the connection-handler thread.
/// Non-streaming requests send exactly one `Done(reply)`. Streaming
/// requests send a sequence of `Chunk` (one per decoded token) then a
/// final `Done` carrying the trailing usage stats.
enum StreamMsg {
    /// SSE-style chunk — the connection writes `data: {payload}\n\n`.
    Chunk(String),
    /// Final reply for non-streaming, or stream-terminator for streaming.
    Done(HttpReply),
}

impl GenReq {
    fn is_chat(&self) -> bool { matches!(self.prompt, PromptInput::Chat(_) | PromptInput::ChatVision { .. }) }
}

struct GenerationOutput {
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    hit_stop: bool,
    logprobs: Vec<TokenLogprob>,
    prefill_ms: f64,
    ttft_ms: f64,
    generation_ms: f64,
}

/// A unit of work handed from a connection thread to the GPU worker.
struct Job {
    /// Monotonic id for log + metric correlation.
    request_id: u64,
    target: Target,
    /// `Err` carries an already-formed client error (bad request / wrong route).
    req: Result<GenReq, (u16, &'static str, String)>,
    reply: mpsc::Sender<StreamMsg>,
}

struct HttpReply {
    status: u16,
    status_text: &'static str,
    body: String,
}

// --- request parsing ---------------------------------------------------

/// Server-wide cap on a single request's wall-clock generation. Bigger
/// than this and the worker can't service incoming requests; clients
/// can ask for less via `request_timeout_seconds`.
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 120;

/// Common decode + sampling fields shared by both endpoint parsers.
/// Returns the parsed (GenReq fields). Accepts every OpenAI sampler
/// knob plus a few extensions (min_p, repetition_penalty, mirostat).
/// Per-endpoint default sampler settings. `/v1/chat/completions` gets
/// loop-resistant defaults aimed at the "drop a model in OWUI and chat
/// with it" path. Multiple layered defenses are needed because chat
/// clients sometimes send `temperature: 0` explicitly (greedy), at
/// which point only the rep/freq penalties stand between a confused
/// prompt and an `own own own own` collapse:
///
///   * temperature 0.7  — fallback when client omits temp
///   * top_p 0.95       — nucleus filter
///   * min_p 0.05       — floors out the long tail loops dip into
///   * rep_penalty 1.1  — divides logits of recently-seen tokens
///   * freq_penalty 0.1 — adds a per-occurrence linear penalty
///                        (compounds: 5× repeats = 5× penalty,
///                        crushes greedy loops even at temp=0)
///
/// `/v1/completions` keeps leaner power-user defaults. Per-request
/// fields always override — these only fill in when missing.
#[derive(Copy, Clone)]
struct SamplerDefaults {
    temperature: f32,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    frequency_penalty: f32,
}

const COMPLETION_DEFAULTS: SamplerDefaults = SamplerDefaults {
    temperature: 0.8, top_p: 1.0, min_p: 0.0,
    repetition_penalty: 1.0, frequency_penalty: 0.0,
};
const CHAT_DEFAULTS: SamplerDefaults = SamplerDefaults {
    temperature: 0.7, top_p: 0.95, min_p: 0.05,
    repetition_penalty: 1.1, frequency_penalty: 0.1,
};

fn parse_common_fields(j: &Json, defaults: SamplerDefaults)
    -> Result<(usize, Vec<String>, crate::sampling::SamplerParams, Option<bool>, Option<usize>, f32,
        Option<std::time::Duration>, bool, bool, usize), String>
{
    use crate::sampling::{SamplerParams, MirostatV2};
    let parse_integer = |key: &str| -> Result<Option<usize>, String> {
        match j.get(key) {
            None => Ok(None),
            Some(Json::Num(n)) if n.is_finite() && n.fract() == 0.0 && *n >= 0.0 =>
                Ok(Some(*n as usize)),
            Some(_) => Err(format!("field '{key}' must be a non-negative integer")),
        }
    };
    let max_tokens_value = parse_integer("max_tokens")?;
    let max_completion_value = parse_integer("max_completion_tokens")?;
    if max_tokens_value.is_some() && max_completion_value.is_some() {
        return Err("provide only one of 'max_tokens' and 'max_completion_tokens'".into());
    }
    let max_tokens = max_tokens_value.or(max_completion_value).unwrap_or(256).clamp(1, 4096);

    let stop = match j.get("stop") {
        None => Vec::new(),
        Some(Json::Str(s)) if !s.is_empty() => vec![s.clone()],
        Some(Json::Arr(a)) => {
            if a.len() > 4 { return Err("'stop' supports at most four strings".into()); }
            let mut values = Vec::with_capacity(a.len());
            for (i, item) in a.iter().enumerate() {
                match item {
                    Json::Str(s) if !s.is_empty() => values.push(s.clone()),
                    Json::Str(_) => return Err(format!("stop[{i}] must not be empty")),
                    _ => return Err(format!("stop[{i}] must be a string")),
                }
            }
            values
        }
        Some(_) => return Err("'stop' must be a string or an array of up to four strings".into()),
    };
    if let Some(Json::Num(n)) = j.get("n") {
        if !n.is_finite() || n.fract() != 0.0 || *n != 1.0 {
            return Err("only n=1 is supported; multiple choices are not implemented".into());
        }
    } else if j.get("n").is_some() {
        return Err("field 'n' must be the integer 1".into());
    }
    for key in ["tools", "tool_choice", "functions", "function_call",
                "parallel_tool_calls", "response_format"] {
        if j.get(key).is_some() {
            return Err(format!("field '{key}' is not supported yet"));
        }
    }

    let mut sp = SamplerParams::default();
    sp.temperature = j.get("temperature").and_then(Json::as_f64)
        .map(|n| n as f32).unwrap_or(defaults.temperature).max(0.0);
    sp.top_k = j.get("top_k").and_then(Json::as_f64)
        .map(|n| n as usize).unwrap_or(40);
    sp.top_p = j.get("top_p").and_then(Json::as_f64)
        .map(|n| n as f32).unwrap_or(defaults.top_p).clamp(0.0, 1.0);
    sp.min_p = j.get("min_p").and_then(Json::as_f64)
        .map(|n| n as f32).unwrap_or(defaults.min_p).clamp(0.0, 1.0);
    sp.repetition_penalty = j.get("repetition_penalty").and_then(Json::as_f64)
        .map(|n| n as f32).unwrap_or(defaults.repetition_penalty).max(0.0);
    sp.repetition_window = j.get("repetition_window").and_then(Json::as_f64)
        .map(|n| n as usize).unwrap_or(64);
    sp.frequency_penalty = j.get("frequency_penalty").and_then(Json::as_f64)
        .map(|n| n as f32).unwrap_or(defaults.frequency_penalty);
    sp.presence_penalty = j.get("presence_penalty").and_then(Json::as_f64)
        .map(|n| n as f32).unwrap_or(0.0);
    sp.seed = j.get("seed").and_then(Json::as_f64).map(|n| n as u64).unwrap_or(0);

    // Mirostat v2: opt-in via `mirostat: 2`. tau + eta override defaults.
    if j.get("mirostat").and_then(Json::as_f64).map(|n| n as i64) == Some(2) {
        let tau = j.get("mirostat_tau").and_then(Json::as_f64)
            .map(|n| n as f32).unwrap_or(5.0);
        let eta = j.get("mirostat_eta").and_then(Json::as_f64)
            .map(|n| n as f32).unwrap_or(0.1);
        sp.mirostat = Some(MirostatV2::new(tau, eta));
    }

    let use_speculative = j.get("use_speculative").and_then(Json::as_bool);
    let speculative_k = j.get("speculative_k").and_then(Json::as_f64)
        .map(|n| (n as usize).clamp(1, 4));
    let speculative_p_min = j.get("speculative_p_min").and_then(Json::as_f64)
        .map(|n| n as f32).unwrap_or(0.0).clamp(0.0, 1.0);
    let request_timeout = j.get("request_timeout_seconds").and_then(Json::as_f64)
        .map(|n| std::time::Duration::from_secs_f64(n.max(0.1).min(600.0)))
        .or(Some(std::time::Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS)));
    let stream = j.get("stream").and_then(Json::as_bool).unwrap_or(false);
    // OpenAI streaming-usage extension. Either:
    //   "stream_options": { "include_usage": true }
    // or some clients shorthand the field at top level. Default off.
    let stream_include_usage = j.get("stream_options")
        .and_then(|s| s.get("include_usage"))
        .and_then(Json::as_bool)
        .or(j.get("include_usage").and_then(Json::as_bool))
        .unwrap_or(false);
    // OpenAI `logprobs`: bool turns it on with a default top-N of 5;
    // an integer specifies the top-N directly. Capped at 20 so a
    // request can't ask us to sort the full vocab for diagnostics.
    let top_logprobs_n = match j.get("logprobs") {
        Some(Json::Bool(true))  => 5,
        Some(Json::Bool(false)) => 0,
        Some(Json::Num(n))      => (*n as usize).min(20),
        // OpenAI chat-completions uses `top_logprobs` for the count,
        // gated by a separate `logprobs: true`. Accept both shapes.
        _ => match (j.get("logprobs").and_then(Json::as_bool),
                    j.get("top_logprobs").and_then(Json::as_f64)) {
            (Some(true), Some(n)) => (n as usize).min(20),
            (Some(true), None)    => 5,
            _ => 0,
        },
    };
    Ok((max_tokens, stop, sp, use_speculative, speculative_k, speculative_p_min,
        request_timeout, stream, stream_include_usage, top_logprobs_n))
}

struct ModelSwitch {
    id: u64,
    path: PathBuf,
    projector: Option<PathBuf>,
}

struct ConfigReload {
    id: u64,
    config: config::ServeConfig,
}

enum WorkerCommand {
    Generate(Job),
    Switch(ModelSwitch),
    ReloadConfig(ConfigReload),
}

// --- streaming helpers (SSE) --------------------------------------------

/// One streamed text-completion chunk in OpenAI shape. Each event is a
/// separate `data: ...` line; the client SDK concatenates `text` fields.
fn completion_stream_chunk(id: &str, model: &str, text: &str,
                            finish: Option<&str>,
                            logprob: Option<&TokenLogprob>) -> String {
    let mut choice = vec![
        ("text".into(),     Json::Str(text.to_string())),
        ("index".into(),    Json::Num(0.0)),
        ("logprobs".into(), match logprob {
            Some(t) => render_text_logprobs(std::slice::from_ref(t)),
            None    => Json::Null,
        }),
    ];
    choice.push(("finish_reason".into(),
                 finish.map(|f| Json::Str(f.to_string())).unwrap_or(Json::Null)));
    Json::Obj(vec![
        ("id".into(),      Json::Str(id.to_string())),
        ("object".into(),  Json::Str("text_completion".into())),
        ("created".into(), Json::Num(unix_now() as f64)),
        ("model".into(),   Json::Str(model.to_string())),
        ("choices".into(), Json::Arr(vec![Json::Obj(choice)])),
    ]).to_string()
}

/// One streamed chat-completion chunk. First chunk carries `role`;
/// subsequent chunks carry just `content`; final chunk has `finish_reason`.
fn chat_stream_chunk(id: &str, model: &str, delta: ChatDelta,
                      finish: Option<&str>,
                      logprob: Option<&TokenLogprob>) -> String {
    let mut d = Vec::with_capacity(2);
    if let Some(r) = delta.role    { d.push(("role".into(),    Json::Str(r.to_string()))); }
    if let Some(c) = delta.content { d.push(("content".into(), Json::Str(c.to_string()))); }
    let mut choice = vec![
        ("index".into(), Json::Num(0.0)),
        ("delta".into(), Json::Obj(d)),
    ];
    choice.push(("logprobs".into(), match logprob {
        Some(t) => render_chat_logprobs(std::slice::from_ref(t)),
        None    => Json::Null,
    }));
    choice.push(("finish_reason".into(),
                 finish.map(|f| Json::Str(f.to_string())).unwrap_or(Json::Null)));
    Json::Obj(vec![
        ("id".into(),      Json::Str(id.to_string())),
        ("object".into(),  Json::Str("chat.completion.chunk".into())),
        ("created".into(), Json::Num(unix_now() as f64)),
        ("model".into(),   Json::Str(model.to_string())),
        ("choices".into(), Json::Arr(vec![Json::Obj(choice)])),
    ]).to_string()
}

struct ChatDelta<'a> { role: Option<&'a str>, content: Option<&'a str> }

/// Convert a `SampleResult` into a `TokenLogprob` by decoding each token
/// id through the caller-provided decoder. The "delta" text we surface
/// for an alternative is the standalone decode of that token id — which
/// can differ from the multi-token UTF-8 reassembly that the streamed
/// `content` uses (a single byte-fragment token won't be a valid UTF-8
/// boundary on its own). Callers display the alternative text raw
/// since OpenAI clients treat it as opaque-but-displayable.
///
/// The decode closure is taken instead of a concrete tokenizer type
/// because qwen + gemma use different tokenizers (`tokenizer::Tokenizer`
/// vs `gemma4::GemmaTokenizer`) — both expose a `decode(&[u32]) -> String`
/// method but neither implements a shared trait.
fn decode_token_logprob(decode: impl Fn(&[u32]) -> String,
                        chosen: u32,
                        res: &crate::sampling::SampleResult) -> TokenLogprob
{
    let token = decode(&[chosen]);
    let logprob = res.logprob.unwrap_or(f32::NEG_INFINITY);
    let top_alts = res.top_logprobs.iter()
        .map(|(id, lp)| (decode(&[*id]), *lp))
        .collect();
    TokenLogprob { token, logprob, top_alts }
}

/// Render a vector of TokenLogprob into the OpenAI logprobs object for
/// chat completions (`logprobs.content`). When empty, returns `Json::Null`
/// so the surrounding response stays valid for "logprobs not available
/// here" (spec-decode path, or feature not requested).
fn render_chat_logprobs(lp: &[TokenLogprob]) -> Json {
    if lp.is_empty() { return Json::Null; }
    let content = lp.iter().map(|t| {
        let alts: Vec<Json> = t.top_alts.iter().map(|(tk, l)|
            Json::Obj(vec![
                ("token".into(),   Json::Str(tk.clone())),
                ("logprob".into(), Json::Num(*l as f64)),
            ])).collect();
        Json::Obj(vec![
            ("token".into(),        Json::Str(t.token.clone())),
            ("logprob".into(),      Json::Num(t.logprob as f64)),
            ("top_logprobs".into(), Json::Arr(alts)),
        ])
    }).collect();
    Json::Obj(vec![("content".into(), Json::Arr(content))])
}

/// `text_completions`-shape logprobs object (the older /v1/completions
/// shape, which exposes parallel arrays rather than per-token objects).
fn render_text_logprobs(lp: &[TokenLogprob]) -> Json {
    if lp.is_empty() { return Json::Null; }
    let tokens: Vec<Json> = lp.iter().map(|t| Json::Str(t.token.clone())).collect();
    let token_logprobs: Vec<Json> = lp.iter()
        .map(|t| Json::Num(t.logprob as f64)).collect();
    let top_lps: Vec<Json> = lp.iter().map(|t| {
        let obj: Vec<(String, Json)> = t.top_alts.iter()
            .map(|(tk, l)| (tk.clone(), Json::Num(*l as f64)))
            .collect();
        Json::Obj(obj)
    }).collect();
    Json::Obj(vec![
        ("tokens".into(),         Json::Arr(tokens)),
        ("token_logprobs".into(), Json::Arr(token_logprobs)),
        ("top_logprobs".into(),   Json::Arr(top_lps)),
    ])
}

/// Parse an OpenAI `/v1/completions` body into a `GenReq`. Raw-prompt
/// path; no chat template is applied server-side.
fn parse_completions(body: &str) -> Result<GenReq, (u16, &'static str, String)> {
    let bad = |m: String| (400u16, "Bad Request", m);
    let j = Json::parse(body).map_err(|e| bad(format!("invalid JSON: {e}")))?;
    let model = parse_model(&j).map_err(&bad)?;
    let prompt = j.get("prompt").and_then(Json::as_str)
        .ok_or_else(|| bad("missing string field 'prompt'".into()))?
        .to_string();
    let (max_tokens, stop, sampler, use_speculative, speculative_k, speculative_p_min,
         request_timeout, stream, stream_include_usage, top_logprobs_n) =
        parse_common_fields(&j, COMPLETION_DEFAULTS).map_err(&bad)?;
    Ok(GenReq { prompt: PromptInput::Raw(prompt), model, max_tokens, stop, sampler,
                use_speculative, speculative_k, speculative_p_min,
                request_timeout, stream, stream_include_usage, top_logprobs_n })
}

fn parse_model(j: &Json) -> Result<Option<String>, String> {
    match j.get("model") {
        None => Ok(None),
        Some(Json::Str(s)) if !s.is_empty() => Ok(Some(s.clone())),
        Some(_) => Err("field 'model' must be a non-empty string".into()),
    }
}

/// Parse an OpenAI `/v1/chat/completions` body into a `GenReq`. The
/// `messages` array becomes a `PromptInput::Chat`; the worker's
/// model knows which per-architecture chat template to apply.
fn parse_chat_completions(body: &str) -> Result<GenReq, (u16, &'static str, String)> {
    use crate::chat::{ChatMessage, Role};
    let bad = |m: String| (400u16, "Bad Request", m);
    let j = Json::parse(body).map_err(|e| bad(format!("invalid JSON: {e}")))?;
    let model = parse_model(&j).map_err(&bad)?;
    let messages_arr = j.get("messages")
        .ok_or_else(|| bad("missing array field 'messages'".into()))?;
    let arr = match messages_arr {
        Json::Arr(a) => a,
        _ => return Err(bad("'messages' must be an array".into())),
    };
    if arr.is_empty() {
        return Err(bad("'messages' must contain at least one message".into()));
    }
    let mut messages: Vec<ChatMessage> = Vec::with_capacity(arr.len());
    let mut image: Option<Vec<u8>> = None;
    for (i, m) in arr.iter().enumerate() {
        let role_s = m.get("role").and_then(Json::as_str)
            .ok_or_else(|| bad(format!("messages[{i}]: missing string 'role'")))?;
        let role = match role_s {
            "system"    => Role::System,
            "user"      => Role::User,
            "assistant" => Role::Assistant,
            other => return Err(bad(format!(
                "messages[{i}]: unknown role '{other}' (want system|user|assistant)"))),
        };
        let content = match m.get("content") {
            Some(Json::Str(text)) => text.clone(),
            Some(Json::Arr(parts)) => {
                if role != Role::User {
                    return Err(bad(format!("messages[{i}]: structured content is supported only for user messages")));
                }
                let mut text = String::new();
                for (part_i, part) in parts.iter().enumerate() {
                    let kind = part.get("type").and_then(Json::as_str)
                        .ok_or_else(|| bad(format!("messages[{i}].content[{part_i}]: missing string 'type'")))?;
                    match kind {
                        "text" => text.push_str(part.get("text").and_then(Json::as_str)
                            .ok_or_else(|| bad(format!("messages[{i}].content[{part_i}]: text part needs string 'text'")))?),
                        "image_url" => {
                            if image.is_some() { return Err(bad("only one image is supported per request".into())); }
                            let url = part.get("image_url").and_then(|v| v.get("url")).and_then(Json::as_str)
                                .ok_or_else(|| bad(format!("messages[{i}].content[{part_i}]: image_url needs object field 'url'")))?;
                            image = Some(decode_data_image(url).map_err(bad)?);
                            text.push_str("<__media__>");
                        }
                        other => return Err(bad(format!("messages[{i}].content[{part_i}]: unsupported type '{other}' (want text|image_url)"))),
                    }
                }
                text
            }
            _ => return Err(bad(format!("messages[{i}]: content must be a string or array"))),
        };
        messages.push(ChatMessage { role, content });
    }
    let media_markers: usize = messages.iter().map(|m| m.content.matches("<__media__>").count()).sum();
    if media_markers != usize::from(image.is_some()) {
        return Err(bad("reserved media marker is not allowed in text".into()));
    }
    let (max_tokens, stop, sampler, use_speculative, speculative_k, speculative_p_min,
         request_timeout, stream, stream_include_usage, top_logprobs_n) =
        parse_common_fields(&j, CHAT_DEFAULTS).map_err(&bad)?;
    let prompt = match image { Some(image) => PromptInput::ChatVision { messages, image }, None => PromptInput::Chat(messages) };
    Ok(GenReq { prompt, model, max_tokens, stop, sampler,
                use_speculative, speculative_k, speculative_p_min,
                request_timeout, stream, stream_include_usage, top_logprobs_n })
}

/// Return the visible byte end and whether a stop string has completed.
/// Before a match exists, retain a trailing prefix of any stop string so a
/// sequence split across decoded token callbacks is never leaked to a stream.
fn stop_visible_end(text: &str, stops: &[String]) -> (usize, bool) {
    if stops.is_empty() { return (text.len(), false); }
    if let Some(end) = stops.iter().filter_map(|s| text.find(s)).min() {
        return (end, true);
    }
    let mut pending = 0;
    for stop in stops {
        let bytes = stop.as_bytes();
        for len in 1..bytes.len() {
            if stop.is_char_boundary(len)
                && text.as_bytes().ends_with(&bytes[..len])
                && text.is_char_boundary(text.len() - len)
            {
                pending = pending.max(len);
            }
        }
    }
    (text.len().saturating_sub(pending), false)
}

/// Decode every complete UTF-8 sequence while retaining an incomplete
/// trailing sequence for the next token. Definite invalid sequences are
/// replaced exactly as `String::from_utf8_lossy` would replace them, so the
/// returned text is an append-only stable prefix of the eventual decode.
fn stable_utf8_prefix(bytes: &[u8]) -> String {
    let mut out = String::new();
    let mut rest = bytes;
    loop {
        match std::str::from_utf8(rest) {
            Ok(text) => {
                out.push_str(text);
                break;
            }
            Err(error) => {
                let valid = error.valid_up_to();
                // SAFETY: `Utf8Error::valid_up_to` guarantees this prefix.
                out.push_str(unsafe { std::str::from_utf8_unchecked(&rest[..valid]) });
                match error.error_len() {
                    Some(invalid) => {
                        out.push('\u{fffd}');
                        rest = &rest[valid + invalid..];
                    }
                    None => break,
                }
            }
        }
    }
    out
}

const MAX_IMAGE_BYTES: usize = 6 * 1024 * 1024;

/// Strict v1 data URL parser.  Remote fetches, SVG, and unexpected image
/// types are deliberately excluded from the server attack surface.
fn decode_data_image(url: &str) -> Result<Vec<u8>, String> {
    let encoded = match url.strip_prefix("data:image/jpeg;base64,")
        .or_else(|| url.strip_prefix("data:image/png;base64,")) {
        Some(data) => data,
        None if url.starts_with("http://") || url.starts_with("https://") => return Err("remote image URLs are not supported; use a data:image/jpeg|png;base64 URL".into()),
        None => return Err("image_url must be a data:image/jpeg|png;base64 URL".into()),
    };
    if encoded.len() > MAX_IMAGE_BYTES.saturating_mul(4) / 3 + 8 {
        return Err(format!("decoded image exceeds {} byte limit", MAX_IMAGE_BYTES));
    }
    let bytes = decode_base64(encoded)?;
    if bytes.is_empty() { return Err("image input is empty".into()); }
    if bytes.len() > MAX_IMAGE_BYTES { return Err(format!("decoded image exceeds {} byte limit", MAX_IMAGE_BYTES)); }
    Ok(bytes)
}

/// Decode the standard RFC 4648 alphabet used in data URLs.  Keeping this
/// tiny parser local avoids adding a dependency to the server's deliberately
/// minimal request surface.
fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    fn value(byte: u8) -> Option<u8> {
        match byte {
            b'A'..=b'Z' => Some(byte - b'A'),
            b'a'..=b'z' => Some(byte - b'a' + 26),
            b'0'..=b'9' => Some(byte - b'0' + 52),
            b'+' => Some(62), b'/' => Some(63), _ => None,
        }
    }
    let bytes = input.as_bytes();
    if bytes.is_empty() || bytes.len() % 4 != 0 { return Err("image_url contains invalid base64".into()); }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for (group_index, group) in bytes.chunks_exact(4).enumerate() {
        let last = group_index + 1 == bytes.len() / 4;
        let a = value(group[0]).ok_or("image_url contains invalid base64")?;
        let b = value(group[1]).ok_or("image_url contains invalid base64")?;
        let c = if group[2] == b'=' { None } else {
            Some(value(group[2]).ok_or("image_url contains invalid base64")?)
        };
        let d = if group[3] == b'=' { None } else {
            Some(value(group[3]).ok_or("image_url contains invalid base64")?)
        };
        if c.is_none() && d.is_some() || (!last && (c.is_none() || d.is_none())) {
            return Err("image_url contains invalid base64 padding".into());
        }
        if c.is_none() && (b & 0x0f) != 0 { return Err("image_url contains invalid base64 padding".into()); }
        out.push((a << 2) | (b >> 4));
        if let Some(c) = c {
            if d.is_none() && (c & 0x03) != 0 { return Err("image_url contains invalid base64 padding".into()); }
            out.push((b << 4) | (c >> 2));
            if let Some(d) = d { out.push((c << 6) | d); }
        }
    }
    Ok(out)
}

// --- OpenAI response shaping -------------------------------------------

static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn completion_response(model: &str, text: &str, n_prompt: usize,
                       n_completion: usize, hit_eos: bool,
                       logprobs: &[TokenLogprob]) -> String {
    let id = format!("cmpl-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed));
    let choice = Json::Obj(vec![
        ("text".into(),          Json::Str(text.to_string())),
        ("index".into(),         Json::Num(0.0)),
        ("logprobs".into(),      render_text_logprobs(logprobs)),
        ("finish_reason".into(), Json::Str(
            if hit_eos { "stop" } else { "length" }.to_string())),
    ]);
    Json::Obj(vec![
        ("id".into(),      Json::Str(id)),
        ("object".into(),  Json::Str("text_completion".into())),
        ("created".into(), Json::Num(unix_now() as f64)),
        ("model".into(),   Json::Str(model.to_string())),
        ("choices".into(), Json::Arr(vec![choice])),
        ("usage".into(),   Json::Obj(vec![
            ("prompt_tokens".into(),     Json::Num(n_prompt as f64)),
            ("completion_tokens".into(), Json::Num(n_completion as f64)),
            ("total_tokens".into(),      Json::Num((n_prompt + n_completion) as f64)),
        ])),
    ]).to_string()
}

/// OpenAI-shaped `chat.completion` response. Same usage stats as the
/// Closing markers for instruction-tuned models' chain-of-thought
/// preambles. The user-facing response is everything after the LAST
/// occurrence of any of these in the full text:
///   * `</think>`     — Qwen 3.5/3.6 IT
///   * `<channel|>`   — Gemma 4 IT (channel/thought variant)
///   * `<|thought|>`  — Gemma 4 IT (alternate thought variant — also
///                      acts as its own "close" in the model's emit
///                      style: multiple bare `<|thought|>` then prose)
const THINK_CLOSERS: &[&str] = &["</think>", "<channel|>", "<|thought|>"];

/// Strip any thinking-mode preamble from a fully decoded response.
/// Returns the slice after the LAST closing marker, or the full text
/// unchanged if none are present.
fn strip_thinking_channels(text: &str) -> &str {
    let mut best_end: Option<usize> = None;
    for m in THINK_CLOSERS {
        if let Some(idx) = text.rfind(m) {
            let end = idx + m.len();
            if best_end.map_or(true, |b| end > b) {
                best_end = Some(end);
            }
        }
    }
    match best_end {
        Some(end) => text[end..].trim_start_matches(|c: char| c == '\n' || c == ' '),
        None      => text,
    }
}

/// Streaming-aware version of the same stripper. Buffers incoming text
/// until either a closing marker is seen (then emits everything after
/// it and switches to passthrough), or until enough text has been
/// buffered with no marker that we conclude this response simply isn't
/// using thinking mode (then flush the buffer verbatim and passthrough).
pub struct ThinkingStripStream {
    /// `true` once we've passed the closing marker (or decided there
    /// won't be one). All subsequent input is forwarded verbatim.
    passthrough: bool,
    /// Cumulative pre-passthrough text. Bounded by `MAX_BUFFER`.
    buf: String,
}

impl ThinkingStripStream {
    const MAX_BUFFER: usize = 256;

    pub fn new() -> Self {
        Self { passthrough: false, buf: String::new() }
    }

    /// Push a streaming delta. Returns the (possibly empty) clean text
    /// to forward downstream this call.
    pub fn push(&mut self, chunk: &str) -> String {
        if self.passthrough {
            return chunk.to_string();
        }
        self.buf.push_str(chunk);
        // Look for any closing marker in the cumulative buffer.
        let mut best_end: Option<usize> = None;
        for m in THINK_CLOSERS {
            if let Some(idx) = self.buf.rfind(m) {
                let end = idx + m.len();
                if best_end.map_or(true, |b| end > b) { best_end = Some(end); }
            }
        }
        if let Some(end) = best_end {
            let tail = self.buf[end..]
                .trim_start_matches(|c: char| c == '\n' || c == ' ')
                .to_string();
            self.passthrough = true;
            self.buf.clear();
            return tail;
        }
        // No closer yet. If the buffer has grown past our cap AND we
        // never saw any opening marker, this response isn't using
        // thinking mode — flush verbatim and start passing through.
        if self.buf.len() > Self::MAX_BUFFER {
            let opened = ["<think>", "<|channel>", "<|thought|>"]
                .iter().any(|o| self.buf.contains(o));
            if !opened {
                self.passthrough = true;
                let out = std::mem::take(&mut self.buf);
                return out;
            }
        }
        String::new()
    }

    /// Flush whatever is in the buffer at end-of-stream. The model may
    /// have stopped mid-thinking (rare) or emitted an opener with no
    /// matching closer — emit what we have so the user sees the response.
    pub fn flush(&mut self) -> String {
        if self.passthrough { return String::new(); }
        let out = std::mem::take(&mut self.buf);
        self.passthrough = true;
        out
    }
}

/// raw-completion shape, but the choice carries a `message` object
/// instead of a flat `text` field — what every chat SDK expects.
fn chat_completion_response(model: &str, text: &str, n_prompt: usize,
                            n_completion: usize, hit_eos: bool,
                            logprobs: &[TokenLogprob]) -> String {
    let text = strip_thinking_channels(text);
    let id = format!("chatcmpl-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed));
    let message = Json::Obj(vec![
        ("role".into(),    Json::Str("assistant".into())),
        ("content".into(), Json::Str(text.to_string())),
    ]);
    let choice = Json::Obj(vec![
        ("index".into(),         Json::Num(0.0)),
        ("message".into(),       message),
        ("logprobs".into(),      render_chat_logprobs(logprobs)),
        ("finish_reason".into(), Json::Str(
            if hit_eos { "stop" } else { "length" }.to_string())),
    ]);
    Json::Obj(vec![
        ("id".into(),      Json::Str(id)),
        ("object".into(),  Json::Str("chat.completion".into())),
        ("created".into(), Json::Num(unix_now() as f64)),
        ("model".into(),   Json::Str(model.to_string())),
        ("choices".into(), Json::Arr(vec![choice])),
        ("usage".into(),   Json::Obj(vec![
            ("prompt_tokens".into(),     Json::Num(n_prompt as f64)),
            ("completion_tokens".into(), Json::Num(n_completion as f64)),
            ("total_tokens".into(),      Json::Num((n_prompt + n_completion) as f64)),
        ])),
    ]).to_string()
}

fn error_body(message: &str, kind: &str) -> String {
    error_body_with_details(message, kind, None, None)
}

fn error_body_with_details(message: &str, kind: &str,
                           param: Option<&str>, code: Option<&str>) -> String {
    Json::Obj(vec![
        ("error".into(), Json::Obj(vec![
            ("message".into(), Json::Str(message.to_string())),
            ("type".into(),    Json::Str(kind.to_string())),
            ("param".into(),   param.map(|v| Json::Str(v.to_string())).unwrap_or(Json::Null)),
            ("code".into(),    code.map(|v| Json::Str(v.to_string())).unwrap_or(Json::Null)),
        ])),
    ]).to_string()
}

// --- the resident model ------------------------------------------------

/// A loaded GPU model — either runtime family — plus its tokenizer and a
/// reusable decode state. `generate` runs one prompt→completion.
enum ServerModel {
    Qwen {
        gpu: crate::runtime::qwen35::GpuQwen35,
        state: crate::runtime::qwen35::Qwen35GpuState,
        tok: crate::tokenizer::Tokenizer,
        eos: u32,
        max_seq: usize,
        name: String,
        vision: Option<crate::multimodal::MtmdProcessor>,
    },
    Gemma {
        gpu: crate::runtime::gemma4::GpuGemma4,
        state: crate::runtime::gemma4::Gemma4GpuState,
        tok: crate::tokenizer::GemmaTokenizer,
        eos: u32,
        bos: u32,
        max_seq: usize,
        name: String,
        /// MTP drafter for spec-decode. `None` ⇒ server was started
        /// without `--big-drafter`; every request runs plain decode.
        drafter: Option<GemmaDrafter>,
        /// KV prefix cache (LRU). On each request, scan all slots for
        /// the longest common prefix with the new prompt; restore the
        /// best match's snapshot and prefill only the suffix. Slashes
        /// TTFT for chat sessions where successive turns reuse a long
        /// system + history prefix. Multi-slot lets independent chat
        /// sessions (different users / conversations) each get cache
        /// hits even when interleaved on the same model.
        prefix_cache: PrefixCache,
    },
}

#[derive(Clone)]
struct VisionConfig {
    mmproj: PathBuf,
    bridge: PathBuf,
    threads: i32,
    image_min_tokens: i32,
    image_max_tokens: i32,
    use_gpu: bool,
}

struct PrefixCacheEntry {
    tokens: Vec<u32>,
    snapshot: crate::runtime::gemma4::Gemma4StateSnapshot,
}

/// Small LRU cache of `PrefixCacheEntry` per `ServerModel::Gemma`. On
/// each request we scan all slots for the longest common prefix with
/// the new prompt; on insert we evict the oldest. Multi-slot is the
/// difference between "two back-to-back turns share state" (1-slot)
/// and "multiple concurrent chat sessions on the same model can each
/// reuse state" (N-slot). Cap chosen by VRAM budget — each snapshot
/// is ~max_seq × per-layer-KV bytes (e.g. ~50 MB for a 500-token
/// gemma-26B-MoE prompt), so 4 slots ≈ ~200 MB.
struct PrefixCache {
    slots: std::collections::VecDeque<PrefixCacheEntry>,
    cap: usize,
}

impl PrefixCache {
    fn new(cap: usize) -> Self {
        Self { slots: std::collections::VecDeque::with_capacity(cap), cap }
    }

    /// Find the slot with the longest common prefix vs `prompt`. Returns
    /// `(slot_index, overlap_len)` when a slot has ≥ MIN_OVERLAP common
    /// tokens AND less than `prompt.len()` (need a non-empty suffix to
    /// prefill — full match means no work to do but also no need to
    /// snapshot again). None otherwise.
    fn best_match(&self, prompt: &[u32]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.slots.iter().enumerate() {
            let c = common_prefix_len(&e.tokens, prompt);
            if c >= PREFIX_CACHE_MIN_OVERLAP && c < prompt.len() {
                if best.map(|(_, bc)| c > bc).unwrap_or(true) {
                    best = Some((i, c));
                }
            }
        }
        best
    }

    /// Mark slot `idx` as most-recently-used and return a reference
    /// to its snapshot for restore. Moves entry to the back of the
    /// deque (most-recent end).
    fn touch(&mut self, idx: usize)
        -> &crate::runtime::gemma4::Gemma4StateSnapshot
    {
        if idx + 1 < self.slots.len() {
            let entry = self.slots.remove(idx).expect("idx valid");
            self.slots.push_back(entry);
            &self.slots.back().expect("just pushed").snapshot
        } else {
            &self.slots[idx].snapshot
        }
    }

    /// Add a fresh (tokens, snapshot) pair. Evicts the oldest if at
    /// cap. If a slot already holds an exact prefix duplicate, just
    /// updates it (no point keeping two identical entries).
    fn insert(&mut self, tokens: Vec<u32>,
              snapshot: crate::runtime::gemma4::Gemma4StateSnapshot)
    {
        // Dedup: if some slot's tokens are identical to ours, swap its
        // snapshot in place and move it to the back.
        if let Some(pos) = self.slots.iter().position(|e| e.tokens == tokens) {
            let mut e = self.slots.remove(pos).expect("pos valid");
            e.snapshot = snapshot;
            self.slots.push_back(e);
            return;
        }
        if self.slots.len() >= self.cap {
            self.slots.pop_front();
        }
        self.slots.push_back(PrefixCacheEntry { tokens, snapshot });
    }
}

/// Minimum prefix overlap to make snapshot restore worthwhile.
/// Restore + suffix-prefill needs to beat a plain prefill of N tokens —
/// at our gemma 26B-MoE prefill speeds (~770 tok/s) that means at least
/// ~32 tokens of overlap before the snapshot D2D copy starts paying off.
const PREFIX_CACHE_MIN_OVERLAP: usize = 32;

/// Default LRU slots per Gemma model. 4 = comfortable for a few
/// concurrent chat sessions without VRAM bloat. Override-able via
/// env in serve startup if a deployment wants more.
const PREFIX_CACHE_SLOTS: usize = 4;

fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// MTP drafter resources tied to a Gemma target. Captured graphs
/// (one per K seen) are lazily cached across requests since they're
/// K-shape-specific and reusable from any base_pos.
struct GemmaDrafter {
    runtime: crate::runtime::gemma4_assistant::GpuGemma4Assistant,
    /// `verify_graphs[k]` is a HIP-graph capture of `enqueue_verify_kernels`
    /// for K=k; reused across requests. Index 0 is unused (K must be ≥ 1).
    /// MoE targets keep this empty — they use the decode-loop verify path
    /// inside `verify_forward` (no graph to capture).
    verify_graphs: Vec<Option<crate::hip::GraphExec>>,
}

impl ServerModel {
    /// Load a GGUF, detecting the architecture, into a resident GPU model.
    /// `drafter_path` is honoured only on Gemma 4 targets — qwen35 has no
    /// supported drafter (Qwen 3.6 MTP loads but its forward path is
    /// unwritten; see the gemma4-mtp memory file for the round arithmetic).
    fn load(path: &PathBuf, drafter_path: Option<&PathBuf>, cache: &KernelCache,
            max_seq: usize, vision_config: Option<&VisionConfig>) -> Result<ServerModel, String>
    {
        let g = GgufFile::open(path).map_err(|e| e.to_string())?;
        let arch = g.metadata_get("general.architecture")
            .and_then(|v| v.as_str()).unwrap_or("<unknown>").to_string();
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string();

        // Detect the chat template family from the GGUF's jinja blob;
        // log the family + warn if serve can't apply it natively.
        if let Some(t) = g.metadata_get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
        {
            let fam = crate::chat::detect_chat_template(t);
            if fam.supported_by_serve() {
                info!("  chat template: {} (supported natively)", fam.label());
            } else {
                warn!("  chat template: {} — NOT applied by serve. \
                       /v1/chat/completions will fail at format time. \
                       Use /v1/completions and pre-template client-side.",
                      fam.label());
            }
        }

        if arch == "gemma4" {
            if vision_config.is_some() {
                return Err("server vision is supported only for Qwen 3.6 targets".into());
            }
            use crate::model::gemma4::Gemma4Model;
            use crate::model::gemma4_assistant::Gemma4AssistantModel;
            use crate::runtime::gemma4::{GpuGemma4, Gemma4GpuState};
            use crate::runtime::gemma4_assistant::GpuGemma4Assistant;
            use crate::tokenizer::GemmaTokenizer;
            let model = Gemma4Model::load(&g).map_err(|e| e.to_string())?;
            let eos = model.config.eos_token_id;
            let gpu = GpuGemma4::new(&model, &g, cache, max_seq)?;
            let state = Gemma4GpuState::new(&model, max_seq)?;
            let tok = GemmaTokenizer::from_gguf(&g).map_err(|e| e.to_string())?;
            let bos = tok.bos_id;
            let drafter = if let Some(dp) = drafter_path {
                info!("loading big-drafter   {} ...", dp.display());
                let t = std::time::Instant::now();
                let dg = GgufFile::open(dp).map_err(|e| e.to_string())?;
                let dm = Gemma4AssistantModel::load(&dg).map_err(|e| e.to_string())?;
                let dr = GpuGemma4Assistant::new(&dm, &dg, &gpu, cache)?;
                info!("  loaded drafter in {:.1}s", t.elapsed().as_secs_f32());
                let mut verify_graphs = Vec::with_capacity(5);
                for _ in 0..5 { verify_graphs.push(None); }
                Some(GemmaDrafter { runtime: dr, verify_graphs })
            } else { None };
            Ok(ServerModel::Gemma { gpu, state, tok, eos, bos, max_seq, name, drafter,
                                    prefix_cache: PrefixCache::new(PREFIX_CACHE_SLOTS) })
        } else {
            // qwen35 / qwen35moe — the dense + MoE Qwen runtime.
            use crate::model::qwen3_5::Qwen35Model;
            use crate::runtime::qwen35::{GpuQwen35, Qwen35GpuState};
            use crate::tokenizer::Tokenizer;
            let model = Qwen35Model::load(&g).map_err(|e| e.to_string())?;
            let eos = model.config.eos_token_id;
            let gpu = GpuQwen35::new(&model, &g, cache, max_seq)?;
            let state = Qwen35GpuState::new(&model, max_seq)?;
            let tok = Tokenizer::from_gguf(&g)?;
            let vision = match vision_config {
                Some(v) => {
                    info!("loading mtmd bridge {} ...", v.bridge.display());
                    Some(crate::multimodal::MtmdProcessor::load(
                        &v.bridge, path, &v.mmproj, model.config.hidden_size as usize,
                        v.use_gpu, v.threads, v.image_min_tokens, v.image_max_tokens,
                    )?)
                }
                None => None,
            };
            if drafter_path.is_some() {
                warn!("--big-drafter ignored on qwen35 target \
                       (no supported drafter; see gemma4-mtp memory file)");
            }
            Ok(ServerModel::Qwen { gpu, state, tok, eos, max_seq, name, vision })
        }
    }

    fn name(&self) -> &str {
        match self { ServerModel::Qwen { name, .. } | ServerModel::Gemma { name, .. } => name }
    }

    /// Run one completion. Returns generated text, token counts, finish state,
    /// per-token logprobs, and stage timings. `on_token` receives the
    /// decoded text DELTA for each emitted token plus an optional
    /// `TokenLogprob` when the request asked for logprobs. Returning
    /// `false` from it (e.g. because the streaming channel closed —
    /// the client disconnected) aborts generation early; the partial
    /// text accumulated so far is still returned.
    ///
    /// `per_token_logprobs` is the accumulated history for non-streaming
    /// responses to embed in the final `logprobs` field. Empty when the
    /// request didn't ask for logprobs OR when the model is on the spec-
    /// decode path (which doesn't surface per-token softmax probs today).
    fn generate(&mut self, req: &GenReq,
                mut on_token: impl FnMut(&str, Option<&TokenLogprob>) -> bool)
        -> Result<GenerationOutput, String>
    {
        use crate::sampling::{Rng, sample_chain_lp};
        let mut sp = req.sampler.clone();
        let want_lp = req.top_logprobs_n;
        let mut rng = Rng::new(sp.seed);
        // `history` is the decoded-so-far token sequence (for repetition
        // penalty); `counts` is the same data laid out per-vocab for
        // OpenAI-style frequency/presence penalties. Both are empty when
        // the per-request knobs leave their defaults.
        let deadline = req.request_timeout
            .map(|d| std::time::Instant::now() + d);

        match self {
            ServerModel::Qwen { gpu, state, tok, eos, max_seq, vision, .. } => {
                let prompt = match &req.prompt {
                    PromptInput::Raw(text) => tok.encode(text),
                    PromptInput::Chat(msgs) => {
                        // Qwen 3.5/3.6: render via the qwen template
                        // (no BOS — qwen expects to start at <|im_start|>),
                        // with assistant turn primed.
                        crate::chat::format_qwen3(tok, msgs, true)?
                    }
                    PromptInput::ChatVision { messages, .. } => {
                        // The bridge tokenizes the formatted text itself so it can
                        // replace the marker with image embeddings.  We count its
                        // physical rows after processing rather than estimating
                        // from the text tokenizer here.
                        let formatted = crate::chat::format_qwen3_with_media_marker(messages)?;
                        let marker_count = formatted.matches("<__media__>").count();
                        if marker_count != 1 { return Err(format!("multimodal chat needs exactly one image marker, found {marker_count}")); }
                        // A placeholder keeps the normal text-only prompt path
                        // below unused; the actual prefill happens after reset.
                        tok.encode(&formatted)
                    }
                };
                if prompt.is_empty() {
                    return Err("prompt encoded to zero tokens".into());
                }
                state.reset()?;
                let mut vision_profile = None;
                let prefill_started = std::time::Instant::now();
                let (mut logits, prompt_rows) = match &req.prompt {
                    PromptInput::ChatVision { messages, image } => {
                        let formatted = crate::chat::format_qwen3_with_media_marker(messages)?;
                        let processor = vision.as_mut().ok_or(
                            "image input is disabled; start serve with --mmproj PATH --mtmd-bridge PATH")?;
                        let mtmd_started = std::time::Instant::now();
                        let processed = processor.process(&formatted, image)?;
                        let mtmd_ms = mtmd_started.elapsed().as_secs_f64() * 1e3;
                        let rows: usize = processed.chunks.iter().map(|c| match c {
                            crate::multimodal::Chunk::Text { tokens, .. } => tokens.len(),
                            crate::multimodal::Chunk::Image { embeddings, embedding_dim, .. } => embeddings.len() / embedding_dim,
                        }).sum();
                        if rows + req.max_tokens + 4 > *max_seq {
                            return Err(format!("multimodal prompt ({rows} physical rows) + max_tokens ({}) exceeds context window ({})", req.max_tokens, max_seq));
                        }
                        let logits = gpu.forward_multimodal_chunks(&processed.chunks, state)?;
                        vision_profile = Some((mtmd_ms, processed.timings));
                        (logits, rows)
                    }
                    _ => {
                        if prompt.len() + req.max_tokens + 4 > *max_seq {
                            return Err(format!(
                                "prompt ({}) + max_tokens ({}) exceeds context window ({})",
                                prompt.len(), req.max_tokens, max_seq));
                        }
                        let logits = if prompt.len() > 1 {
                            gpu.forward_tokens_batched(&prompt, state)?
                        } else { gpu.forward_tokens(&prompt, state)? };
                        (logits, prompt.len())
                    }
                };
                let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1e3;
                let prefill_logical_pos = state.rope_pos;
                let vocab = logits.len();
                let mut counts: Vec<u16> = if sp.frequency_penalty != 0.0
                    || sp.presence_penalty != 0.0 { vec![0u16; vocab] } else { Vec::new() };
                let mut out: Vec<u32> = Vec::new();
                let mut hit_eos = false;
                let mut prev_text_len: usize = 0;
                let mut full_bytes = Vec::new();
                let mut full_text = String::new();
                let mut matched_text_stop = false;
                let mut all_lp: Vec<TokenLogprob> = Vec::new();
                let mut client_open = true;
                let decode_started = std::time::Instant::now();
                let mut first_token_ms = None;
                for _ in 0..req.max_tokens {
                    if let Some(d) = deadline {
                        if std::time::Instant::now() >= d { break; }
                    }
                    let res = sample_chain_lp(&mut logits, &mut sp, &out, &counts, &mut rng, want_lp);
                    let t = res.token;
                    if t == *eos { hit_eos = true; break; }
                    out.push(t);
                    if first_token_ms.is_none() {
                        first_token_ms = Some(prefill_started.elapsed().as_secs_f64() * 1e3);
                    }
                    if !counts.is_empty() { counts[t as usize] = counts[t as usize].saturating_add(1); }
                    // Re-decode the raw byte stream and expose only complete
                    // UTF-8. A byte-level token can end inside a multibyte
                    // glyph; lossy-decoding each intermediate token sequence
                    // would insert a replacement character that disappears
                    // on the next token and invalidate `prev_text_len`.
                    full_bytes = tok.decode_bytes(&out);
                    full_text = stable_utf8_prefix(&full_bytes);
                    let tlp = if want_lp > 0 {
                        Some(decode_token_logprob(|ids| tok.decode(ids), t, &res))
                    } else { None };
                    let (visible_end, matched_stop) = stop_visible_end(&full_text, &req.stop);
                    if visible_end > prev_text_len {
                        let delta = &full_text[prev_text_len..visible_end];
                        let ok = on_token(delta, tlp.as_ref());
                        if let Some(t) = tlp { all_lp.push(t); }
                        if !ok {
                            // Channel closed (client disconnected). Stop
                            // generating; return what we have.
                            client_open = false;
                            break;
                        }
                        prev_text_len = visible_end;
                    } else if let Some(t) = tlp {
                        all_lp.push(t);
                    }
                    if matched_stop {
                        full_text.truncate(visible_end);
                        matched_text_stop = true;
                        hit_eos = true;
                        break;
                    }
                    logits = gpu.forward_token(t, state)?;
                }
                if !matched_text_stop {
                    full_text = String::from_utf8_lossy(&full_bytes).into_owned();
                }
                if client_open && prev_text_len < full_text.len() {
                    let _ = on_token(&full_text[prev_text_len..], None);
                }
                let generation_ms = decode_started.elapsed().as_secs_f64() * 1e3;
                let ttft_ms = first_token_ms.unwrap_or(prefill_ms);
                if let Some((mtmd_ms, mtmd)) = vision_profile {
                    info!("vision profile rows={} logical_pos={} decode_image_ms={:.1} tokenize_ms={:.1} projector_ms={:.1} copy_ms={:.1} mtmd_ms={:.1} prefill_ms={:.1} ttft_ms={:.1} decode_ms={:.1} generated={}",
                        prompt_rows, prefill_logical_pos, mtmd.decode_ms, mtmd.tokenize_ms,
                        mtmd.encode_ms, mtmd.copy_ms, mtmd_ms, prefill_ms,
                        ttft_ms, generation_ms, out.len());
                }
                Ok(GenerationOutput { text: full_text, prompt_tokens: prompt_rows,
                    completion_tokens: out.len(), hit_stop: hit_eos, logprobs: all_lp,
                    prefill_ms, ttft_ms, generation_ms })
            }
            ServerModel::Gemma { gpu, state, tok, eos, bos, max_seq, drafter, prefix_cache, .. } => {
                let prompt = match &req.prompt {
                    PromptInput::Raw(text) => {
                        let mut p = vec![*bos];
                        p.extend(tok.encode(text));
                        p
                    }
                    PromptInput::Chat(msgs) => {
                        // Gemma 4: BOS + per-turn <|turn>role\n…<turn|>\n
                        // with assistant turn primed. The drafter was
                        // trained on this format — chat-templated input
                        // typically gets +30-40 percentage points of
                        // accept rate over raw user text.
                        crate::chat::format_gemma4(tok, msgs, true)?
                    }
                    PromptInput::ChatVision { .. } => return Err(
                        "image input is supported only by a Qwen server started with --mmproj and --mtmd-bridge".into()),
                };
                if prompt.is_empty() {
                    return Err("prompt encoded to zero tokens".into());
                }
                if prompt.len() + req.max_tokens + 8 > *max_seq {
                    return Err(format!(
                        "prompt ({}) + max_tokens ({}) exceeds context window ({})",
                        prompt.len(), req.max_tokens, *max_seq));
                }
                // Dispatch: spec-decode when a drafter is loaded AND the
                // request hasn't opted out. Default-on if drafter present.
                let want_spec = match req.use_speculative {
                    Some(b) => b,
                    None    => drafter.is_some(),
                };
                let do_spec = want_spec && drafter.is_some();
                if want_spec && drafter.is_none() {
                    return Err("use_speculative=true but server has no drafter loaded \
                                (start with --big-drafter PATH)".into());
                }

                // KV prefix cache: scan LRU for the slot with the longest
                // common prefix with this request. On hit, restore that
                // slot's snapshot + truncate to the common prefix, then
                // prefill only the suffix. Skipped entirely when the
                // state is SuperQuant — snapshot/restore/truncate are
                // all per-tier operations that SuperQuant does not
                // implement (see Gemma4GpuState::truncate).
                let mut overlap = 0usize;
                let restored = if state.is_superquant() {
                    state.reset();
                    false
                } else if let Some((idx, c)) = prefix_cache.best_match(&prompt) {
                    let snap = prefix_cache.touch(idx);
                    state.restore(snap)?;
                    state.truncate(c);
                    overlap = c;
                    true
                } else {
                    state.reset();
                    false
                };

                if !do_spec {
                    // Plain prefill + decode. If we hit the prefix cache,
                    // prefill only the suffix; otherwise full prompt.
                    let prefill_started = std::time::Instant::now();
                    let mut logits = if restored {
                        let suffix = &prompt[overlap..];
                        info!("req kv-cache hit: \
                               reused {overlap}/{} tokens; prefilling {} suffix",
                              prompt.len(), suffix.len());
                        gpu.prefill_forward(suffix, state)?
                    } else {
                        gpu.prefill_forward(&prompt, state)?
                    };
                    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1e3;
                    // Snapshot the post-prompt state for future requests
                    // that share a prefix. Best-effort — a snapshot
                    // allocation failure shouldn't abort the request,
                    // just skip caching this turn. SuperQuant states
                    // never support snapshot, so skip silently there.
                    if !state.is_superquant() {
                        match state.snapshot() {
                            Ok(snap) => prefix_cache.insert(prompt.clone(), snap),
                            Err(e) => warn!("prefix-cache snapshot failed: {e}"),
                        }
                    }
                    let vocab = logits.len();
                    let mut counts: Vec<u16> = if sp.frequency_penalty != 0.0
                        || sp.presence_penalty != 0.0 { vec![0u16; vocab] } else { Vec::new() };
                    let mut out: Vec<u32> = Vec::new();
                    let mut hit_eos = false;
                    let mut prev_text_len: usize = 0;
                    let mut full_bytes = Vec::new();
                    let mut full_text = String::new();
                    let mut matched_text_stop = false;
                    let mut all_lp: Vec<TokenLogprob> = Vec::new();
                    let mut client_open = true;
                    let decode_started = std::time::Instant::now();
                    let mut ttft_ms = None;
                    for _ in 0..req.max_tokens {
                        if let Some(d) = deadline {
                            if std::time::Instant::now() >= d { break; }
                        }
                        let res = sample_chain_lp(&mut logits, &mut sp, &out, &counts, &mut rng, want_lp);
                        let t = res.token;
                        if t == *eos { hit_eos = true; break; }
                        out.push(t);
                        if ttft_ms.is_none() {
                            ttft_ms = Some(prefill_ms + decode_started.elapsed().as_secs_f64() * 1e3);
                        }
                        if !counts.is_empty() {
                            counts[t as usize] = counts[t as usize].saturating_add(1);
                        }
                        full_bytes = tok.decode_bytes(&out);
                        full_text = stable_utf8_prefix(&full_bytes);
                        let tlp = if want_lp > 0 {
                            Some(decode_token_logprob(|ids| tok.decode(ids), t, &res))
                        } else { None };
                        let (visible_end, matched_stop) = stop_visible_end(&full_text, &req.stop);
                        if visible_end > prev_text_len {
                            let delta = &full_text[prev_text_len..visible_end];
                            let ok = on_token(delta, tlp.as_ref());
                            if let Some(t) = tlp { all_lp.push(t); }
                            if !ok {
                                client_open = false;
                                break;
                            }
                            prev_text_len = visible_end;
                        } else if let Some(t) = tlp {
                            all_lp.push(t);
                        }
                        if matched_stop {
                            full_text.truncate(visible_end);
                            matched_text_stop = true;
                            hit_eos = true;
                            break;
                        }
                        logits = gpu.forward_token(t, state)?;
                    }
                    if !matched_text_stop {
                        full_text = String::from_utf8_lossy(&full_bytes).into_owned();
                    }
                    if client_open && prev_text_len < full_text.len() {
                        let _ = on_token(&full_text[prev_text_len..], None);
                    }
                    let generation_ms = decode_started.elapsed().as_secs_f64() * 1e3;
                    return Ok(GenerationOutput { text: full_text, prompt_tokens: prompt.len(),
                        completion_tokens: out.len(), hit_stop: hit_eos, logprobs: all_lp,
                        prefill_ms, ttft_ms: ttft_ms.unwrap_or(prefill_ms), generation_ms });
                }

                // Spec-decode path: prefill, then K=req.speculative_k
                // (default 3) rounds via the shared loop. Verify graphs
                // are captured lazily per K and cached across requests.
                let d = drafter.as_mut().unwrap();
                let k = req.speculative_k.unwrap_or(3).clamp(1, 4);
                // Prefill all but the last token — its logits aren't
                // useful; the verify path immediately re-forwards it
                // through `forward_token` to seed the chain.
                let prefill_started = std::time::Instant::now();
                let _ = gpu.prefill_forward(&prompt[..prompt.len() - 1], state)?;
                let verify_logits = gpu.forward_token(*prompt.last().unwrap(), state)?;
                let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1e3;
                if d.verify_graphs[k].is_none() && !gpu.is_moe() {
                    d.verify_graphs[k] = Some(gpu.capture_verify_graph(state, k)?);
                }
                // Adaptive-K is ON by default in serve (chat is mixed
                // workload — α study showed plain MTP is net -2.3% across
                // 24-prompt benchmark; 0.55 threshold salvages structured-
                // output wins without paying the creative/longform losses).
                // Override via REINSTINCT_MTP_MIN_ALPHA env (set to 0 to
                // disable adaptive and run plain MTP regardless of α).
                let adaptive_alpha = std::env::var("REINSTINCT_MTP_MIN_ALPHA").ok()
                    .and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.55);
                let adaptive_window = std::env::var("REINSTINCT_MTP_WINDOW").ok()
                    .and_then(|s| s.parse::<usize>().ok()).unwrap_or(8);
                let decode_started = std::time::Instant::now();
                let (gen_toks, stats) = crate::runtime::spec_decode::spec_decode_generate(
                    gpu, &d.runtime, state,
                    d.verify_graphs[k].as_ref(), k,
                    verify_logits,
                    *prompt.last().unwrap(),
                    *eos,
                    req.max_tokens, k, req.sampler.temperature, req.sampler.seed,
                    req.speculative_p_min,
                    adaptive_alpha, adaptive_window,
                )?;
                let generation_ms = decode_started.elapsed().as_secs_f64() * 1e3;
                info!("spec-decode K={k}: {}/{} accept ({:.0}%){}",
                    stats.n_accepted, stats.n_drafted, 100.0 * stats.accept_rate(),
                    if stats.adaptive_disabled { " [adaptive: MTP off]" } else { "" });
                if want_lp > 0 {
                    warn!("logprobs requested but ignored on \
                           spec-decode path; pass use_speculative=false to enable");
                }
                // No per-token logprobs from spec-decode today; the response
                // shaper renders `logprobs: null` when the vec is empty.
                Ok(GenerationOutput { text: tok.decode(&gen_toks), prompt_tokens: prompt.len(),
                    completion_tokens: gen_toks.len(), hit_stop: stats.hit_eos,
                    logprobs: Vec::new(), prefill_ms, ttft_ms: prefill_ms, generation_ms })
            }
        }
    }
}

// --- the GPU worker ----------------------------------------------------

fn worker(rx: mpsc::Receiver<WorkerCommand>, mut current_config: config::ServeConfig, metrics: Arc<Metrics>,
          statuses: Vec<Arc<api::ApiStatus>>)
{
    let mut big = current_config.big.clone();
    let mut big_drafter = current_config.big_drafter.clone();
    let mut small = current_config.small.clone();
    let mut max_seq = current_config.max_seq;
    let mut vision = match (current_config.mmproj.clone(), current_config.mtmd_bridge.clone()) {
        (Some(mmproj), Some(bridge)) => Some(VisionConfig { mmproj, bridge, threads: current_config.vision_threads,
            image_min_tokens: current_config.vision_min_tokens, image_max_tokens: current_config.vision_max_tokens,
            use_gpu: !current_config.cpu_vision }),
        _ => None,
    };
    for status in &statuses {
        if status.target != "embed" { status.begin_loading(if status.target == "big" { &big } else { small.as_ref().unwrap_or(&big) }); }
    }
    let setup = (|| -> Result<(KernelCache, ServerModel, Option<ServerModel>), String> {
        crate::hip::Device::set(0)?;
        let cache = KernelCache::new()?;
        let load = |label: &str, path: &PathBuf, drafter: Option<&PathBuf>, vision_config: Option<&VisionConfig>|
            -> Result<ServerModel, String>
        {
            info!("loading {label:5} model {} ...", path.display());
            let t = std::time::Instant::now();
            let m = ServerModel::load(path, drafter, &cache, max_seq, vision_config)
                .map_err(|e| {
                    // VRAM-exhaustion → add a hint about model size vs VRAM.
                    if e.to_lowercase().contains("memory") {
                        let sz = std::fs::metadata(path).ok().map(|m| m.len()).unwrap_or(0);
                        format!("{label} model: {e}\n\
                                 [hint] model file is {:.1} GB on disk; \
                                 with KV cache for max_seq={max_seq} the GPU needs roughly \
                                 1.2-1.5× that. Check `rocm-smi --showmeminfo vram` \
                                 against your model's expected resident size, or lower \
                                 max_seq with --max-seq.",
                                 sz as f64 / (1024.0 * 1024.0 * 1024.0))
                    } else {
                        format!("{label} model: {e}")
                    }
                })?;
            info!("  loaded {} in {:.1}s", m.name(), t.elapsed().as_secs_f32());
            Ok(m)
        };
        let big_m   = load("big",   &big,   big_drafter.as_ref(), vision.as_ref())?;
        let small_m = match small.clone() {
            Some(sp) => Some(load("small", &sp, None, None)?),
            None => None,
        };
        Ok((cache, big_m, small_m))
    })();

    let (cache, big_loaded, mut small_m) = match setup {
        Ok(v) => v,
        Err(e) => {
            error!("FATAL: model load failed: {e}");
            for status in &statuses { status.set_phase("error", Some(e.clone())); }
            // Drain the queue with 503s so clients don't hang forever.
            for command in rx {
                match command {
                    WorkerCommand::Generate(job) => { let _ = job.reply.send(StreamMsg::Done(HttpReply {
                        status: 503, status_text: "Service Unavailable",
                        body: error_body(&format!("model load failed: {e}"), "server_error"),
                    })); }
                    WorkerCommand::Switch(switch) => {
                        for status in &statuses {
                            if status.target == "big" { status.fail_model_switch(switch.id, format!("model load failed: {e}"), false); }
                        }
                    }
                    WorkerCommand::ReloadConfig(reload) => {
                        for status in &statuses { if status.target == "big" { status.fail_config_reload(reload.id, format!("model load failed: {e}")); } }
                    }
                }
            }
            return;
        }
    };
    let mut big_m = Some(big_loaded);
    for status in &statuses {
        if status.target == "embed" {
            status.set_phase("unavailable", Some("Embedding runtime is not implemented".into()));
        } else {
            if let Ok(mut loading) = status.loading.lock() { *loading = None; }
            status.set_phase("ready", None);
        }
    }
    info!("ready — serving requests.");

    for command in rx {
        let job = match command {
            WorkerCommand::Generate(job) => job,
            WorkerCommand::Switch(switch) => {
                let Some(status) = statuses.iter().find(|s| s.target == "big").cloned() else {
                    continue;
                };
                let previous = status.model.lock().map(|m| m.clone()).unwrap_or(api::ModelStatus {
                    id: "unknown".into(), path: big.clone(), drafter: big_drafter.clone(), vision: None });
                let make_vision = |projector: Option<PathBuf>| projector.and_then(|mmproj| vision.as_ref().map(|base| VisionConfig {
                    mmproj, bridge: base.bridge.clone(), threads: base.threads,
                    image_min_tokens: base.image_min_tokens, image_max_tokens: base.image_max_tokens,
                    use_gpu: base.use_gpu,
                }));
                let selected_vision = make_vision(switch.projector.clone());
                status.start_model_switch(switch.id, &switch.path);
                info!("dashboard model switch: unloading {} and loading {}", previous.path.display(), switch.path.display());
                drop(big_m.take());
                match ServerModel::load(&switch.path, None, &cache, max_seq, selected_vision.as_ref()) {
                    Ok(loaded) => {
                        let id = loaded.name().to_string();
                        big_m = Some(loaded);
                        status.complete_model_switch(switch.id, api::ModelStatus { id: id.clone(), path: switch.path,
                            drafter: None, vision: selected_vision.as_ref().map(|v| api::VisionStatus {
                                projector: v.mmproj.clone(), threads: v.threads,
                                min_tokens: v.image_min_tokens, max_tokens: v.image_max_tokens,
                                device: if v.use_gpu { "gpu" } else { "cpu" },
                            }) });
                        info!("dashboard model switch complete: {id}");
                    }
                    Err(load_error) => {
                        error!("dashboard model switch failed: {load_error}; attempting rollback");
                        let rollback_vision = make_vision(previous.vision.as_ref().map(|v| v.projector.clone()));
                        match ServerModel::load(&previous.path, previous.drafter.as_ref(), &cache, max_seq, rollback_vision.as_ref()) {
                            Ok(loaded) => {
                                big_m = Some(loaded);
                                if let Ok(mut current) = status.model.lock() { *current = previous; }
                                status.fail_model_switch(switch.id,
                                    format!("selected model failed to load; previous model restored: {load_error}"), true);
                            }
                            Err(rollback_error) => {
                                status.fail_model_switch(switch.id,
                                    format!("model switch failed and rollback failed: {load_error}; {rollback_error}"), false);
                                return;
                            }
                        }
                    }
                }
                continue;
            }
            WorkerCommand::ReloadConfig(reload) => {
                let Some(status) = statuses.iter().find(|s| s.target == "big").cloned() else { continue; };
                status.start_config_reload(reload.id);
                let old_config = current_config.clone();
                let old_big = big.clone();
                let old_drafter = big_drafter.clone();
                let old_small = small.clone();
                let old_max_seq = max_seq;
                let old_vision = vision.clone();
                let new_vision = match (reload.config.mmproj.clone(), reload.config.mtmd_bridge.clone()) {
                    (Some(mmproj), Some(bridge)) => Some(VisionConfig { mmproj, bridge, threads: reload.config.vision_threads,
                        image_min_tokens: reload.config.vision_min_tokens, image_max_tokens: reload.config.vision_max_tokens,
                        use_gpu: !reload.config.cpu_vision }),
                    _ => None,
                };
                let load_one = |label: &str, path: &PathBuf, drafter: Option<&PathBuf>, vc: Option<&VisionConfig>, seq: usize| -> Result<ServerModel, String> {
                    info!("loading {label:5} model {} ...", path.display());
                    ServerModel::load(path, drafter, &cache, seq, vc)
                };
                drop(big_m.take());
                drop(small_m.take());
                let loaded = (|| {
                    let b = load_one("big", &reload.config.big, reload.config.big_drafter.as_ref(), new_vision.as_ref(), reload.config.max_seq)?;
                    let s = match reload.config.small.as_ref() { Some(p) => Some(load_one("small", p, None, None, reload.config.max_seq)?), None => None };
                    Ok::<_, String>((b, s))
                })();
                match loaded {
                    Ok((b, s)) => {
                        let big_id = b.name().to_string();
                        let small_id = s.as_ref().map(|m| m.name().to_string());
                        big_m = Some(b); small_m = s;
                        big = reload.config.big.clone(); big_drafter = reload.config.big_drafter.clone(); small = reload.config.small.clone();
                        max_seq = reload.config.max_seq; vision = new_vision; current_config = reload.config.clone();
                        if let Err(save_error) = current_config.persist() {
                            status.fail_config_reload(reload.id, format!("engine reloaded but configuration could not be saved: {save_error}"));
                        } else {
                            status.finish_loading(api::ModelStatus { id: big_id, path: current_config.big.clone(), drafter: current_config.big_drafter.clone(), vision: current_config.mmproj.as_ref().zip(current_config.mtmd_bridge.as_ref()).map(|(projector, _)| api::VisionStatus { projector: projector.clone(), threads: current_config.vision_threads, min_tokens: current_config.vision_min_tokens, max_tokens: current_config.vision_max_tokens, device: if current_config.cpu_vision { "cpu" } else { "gpu" } }) });
                            if let Some(id) = small_id { if let Some(small_status) = statuses.iter().find(|s| s.target == "small") { small_status.finish_loading(api::ModelStatus { id, path: current_config.small.clone().unwrap_or_default(), drafter: None, vision: None }); } }
                            status.complete_config_reload(reload.id, current_config.clone());
                            info!("dashboard configuration reload complete");
                        }
                    }
                    Err(load_error) => {
                        error!("configuration reload failed: {load_error}; attempting rollback");
                        match (load_one("big", &old_big, old_drafter.as_ref(), old_vision.as_ref(), old_max_seq), old_small.as_ref().map(|p| load_one("small", p, None, None, old_max_seq)).transpose()) {
                            (Ok(b), Ok(s)) => {
                                big_m = Some(b); small_m = s; big = old_big; big_drafter = old_drafter; small = old_small; max_seq = old_max_seq; vision = old_vision; current_config = old_config;
                                status.fail_config_reload(reload.id, format!("reload failed; previous engine restored: {load_error}"));
                            }
                            (Err(rb), _) | (_, Err(rb)) => { status.fail_config_reload(reload.id, format!("reload failed and rollback failed: {load_error}; {rb}")); return; }
                        }
                    }
                }
                continue;
            }
        };
        let job_status = statuses.iter().find(|s| s.target == job.target.label()).cloned();
        if let Some(status) = job_status.as_ref() {
            status.queued.fetch_sub(1, Ordering::Relaxed);
            status.active_request.store(job.request_id, Ordering::Relaxed);
            status.activate_run(job.request_id);
        }
        let mut captured_response: Option<String> = None;
        let mut captured_stats: Option<api::RunStats> = None;
        let reply = match job.req {
            Err((status, status_text, msg)) => {
                warn!("req={} target={} status={} reason={:?} msg={}",
                      job.request_id, job.target.label(), status, status_text, msg);
                metrics.requests_4xx.fetch_add(1, Ordering::Relaxed);
                HttpReply {
                    status, status_text, body: error_body(&msg, "invalid_request_error"),
                }
            }
            Ok(req) => match job.target {
                Target::Embed => {
                    warn!("req={} target=embed status=503 reason=not-yet-available",
                          job.request_id);
                    metrics.requests_5xx.fetch_add(1, Ordering::Relaxed);
                    HttpReply {
                        status: 503, status_text: "Service Unavailable",
                        body: error_body(
                            "embedder not yet available — nomic-bert encoder is a follow-up",
                            "server_error"),
                    }
                }
                Target::Big | Target::Small => {
                    let model: &mut ServerModel = if job.target == Target::Big {
                        big_m.as_mut().expect("worker invariant: big model loaded")
                    } else if let Some(ref mut sm) = small_m {
                        sm
                    } else {
                        warn!("req={} target=small status=503 reason=--small not provided",
                              job.request_id);
                        metrics.requests_5xx.fetch_add(1, Ordering::Relaxed);
                        let _ = job.reply.send(StreamMsg::Done(HttpReply {
                            status: 503, status_text: "Service Unavailable",
                            body: error_body(
                                "small model not loaded — start with --small PATH",
                                "server_error"),
                        }));
                        continue;
                    };
                    let t = std::time::Instant::now();
                    let is_chat = req.is_chat();
                    let is_stream = req.stream;
                    // For streaming requests, build a one-shot SSE id +
                    // first-chunk role frame (chat) up front so the
                    // per-token callback can emit just text deltas.
                    let stream_id = if is_chat {
                        format!("chatcmpl-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed))
                    } else {
                        format!("cmpl-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed))
                    };
                    let model_name = model.name().to_string();
                    let reply_tx = job.reply.clone();
                    // For chat streams, emit a role-only opener frame
                    // before any text content (matches OpenAI SDK
                    // expectations).
                    if is_stream && is_chat {
                        let frame = chat_stream_chunk(&stream_id, &model_name,
                            ChatDelta { role: Some("assistant"), content: None }, None, None);
                        let _ = reply_tx.send(StreamMsg::Chunk(frame));
                    }
                    // `catch_unwind` around generate(): a panic in a kernel
                    // launch, a slipped unwrap, or a numerical blowup
                    // shouldn't kill the worker thread (which would 503
                    // every subsequent request). On panic we 500 the
                    // current request, log, and continue.
                    let stream_id_for_cb = stream_id.clone();
                    let model_name_for_cb = model_name.clone();
                    let reply_for_cb = reply_tx.clone();
                    // Streaming thinking-marker stripper. Lives on the
                    // worker thread; the closure mutates it through a
                    // RefCell so we can also flush at end-of-stream from
                    // outside the closure.
                    let stripper = std::rc::Rc::new(std::cell::RefCell::new(
                        ThinkingStripStream::new()));
                    let stripper_cb = std::rc::Rc::clone(&stripper);
                    let progress_status = job_status.clone();
                    let progress_id = job.request_id;
                    let mut progress_tokens = 0usize;
                    let mut progress_started: Option<std::time::Instant> = None;
                    let on_token = move |delta: &str, lp: Option<&TokenLogprob>| -> bool {
                        progress_tokens += 1;
                        let started = progress_started.get_or_insert_with(std::time::Instant::now);
                        if let Some(status) = progress_status.as_ref() {
                            status.update_generation_progress(progress_id, progress_tokens,
                                started.elapsed().as_secs_f64() * 1e3);
                        }
                        if !is_stream { return true; }
                        let clean = stripper_cb.borrow_mut().push(delta);
                        if clean.is_empty() {
                            // Still buffering inside a thinking block;
                            // don't emit a frame this round.
                            return true;
                        }
                        let frame = if is_chat {
                            chat_stream_chunk(&stream_id_for_cb, &model_name_for_cb,
                                ChatDelta { role: None, content: Some(&clean) }, None, lp)
                        } else {
                            completion_stream_chunk(&stream_id_for_cb, &model_name_for_cb,
                                &clean, None, lp)
                        };
                        reply_for_cb.send(StreamMsg::Chunk(frame)).is_ok()
                    };
                    let result = std::panic::catch_unwind(
                        std::panic::AssertUnwindSafe(|| model.generate(&req, on_token)));
                    match result {
                        Ok(Ok(output)) => {
                            let GenerationOutput { text, prompt_tokens: n_p,
                                completion_tokens: n_c, hit_stop: eos, logprobs: lp,
                                prefill_ms, ttft_ms, generation_ms } = output;
                            let wall_us = t.elapsed().as_micros() as u64;
                            metrics.requests_ok.fetch_add(1, Ordering::Relaxed);
                            metrics.prompt_tokens.fetch_add(n_p as u64, Ordering::Relaxed);
                            metrics.completion_tokens.fetch_add(n_c as u64, Ordering::Relaxed);
                            metrics.prefill_us_total.fetch_add((prefill_ms * 1000.0) as u64, Ordering::Relaxed);
                            metrics.decode_us_total.fetch_add((generation_ms * 1000.0) as u64, Ordering::Relaxed);
                            metrics.ttft_us_total.fetch_add((ttft_ms * 1000.0) as u64, Ordering::Relaxed);
                            if eos { metrics.requests_eos.fetch_add(1, Ordering::Relaxed); }
                            else   { metrics.requests_length.fetch_add(1, Ordering::Relaxed); }
                            let tok_per_s = if n_c > 0 && generation_ms > 0.0 {
                                n_c as f64 * 1000.0 / generation_ms
                            } else { 0.0 };
                            let prompt_tok_per_s = if n_p > 0 && prefill_ms > 0.0 {
                                n_p as f64 * 1000.0 / prefill_ms
                            } else { 0.0 };
                            captured_stats = Some(api::RunStats {
                                prompt_tokens: n_p, completion_tokens: n_c,
                                queue_ms: 0.0, prefill_ms, ttft_ms, generation_ms,
                                total_ms: wall_us as f64 / 1000.0,
                                prompt_tokens_per_second: prompt_tok_per_s,
                                generation_tokens_per_second: tok_per_s,
                            });
                            info!("req={} target={} type={} status=200 \
                                   n_p={} n_c={} wall_ms={:.1} prefill_ms={:.1} ttft_ms={:.1} \
                                   prompt_tok_s={:.1} gen_tok_s={:.1} finish={} stream={}",
                                job.request_id, job.target.label(),
                                if is_chat { "chat" } else { "completion" },
                                n_p, n_c, wall_us as f64 / 1000.0, prefill_ms, ttft_ms,
                                prompt_tok_per_s, tok_per_s,
                                if eos { "stop" } else { "length" }, is_stream);
                            if is_stream {
                                // Flush any text still buffered by the
                                // thinking-marker stripper (e.g. model
                                // emitted an opener with no matching
                                // closer — show the user what we have).
                                let tail = stripper.borrow_mut().flush();
                                if !tail.is_empty() {
                                    let frame = if is_chat {
                                        chat_stream_chunk(&stream_id, &model_name,
                                            ChatDelta { role: None, content: Some(&tail) }, None, None)
                                    } else {
                                        completion_stream_chunk(&stream_id, &model_name,
                                            &tail, None, None)
                                    };
                                    let _ = reply_tx.send(StreamMsg::Chunk(frame));
                                }
                                // Final SSE frame: empty delta + finish_reason.
                                let fin = if eos { "stop" } else { "length" };
                                let frame = if is_chat {
                                    chat_stream_chunk(&stream_id, &model_name,
                                        ChatDelta { role: None, content: None }, Some(fin), None)
                                } else {
                                    completion_stream_chunk(&stream_id, &model_name,
                                        "", Some(fin), None)
                                };
                                let _ = reply_tx.send(StreamMsg::Chunk(frame));
                                // Optional usage chunk per OpenAI spec
                                // when stream_options.include_usage=true.
                                // Clients like Open WebUI use this to
                                // compute decode tok/s.
                                if req.stream_include_usage {
                                    // Three shapes coexist for max client
                                    // compatibility:
                                    //   * OpenAI usage (prompt_tokens / completion_tokens / total_tokens)
                                    //   * Ollama-style fields inside usage (eval_count / eval_duration ns)
                                    //   * llama.cpp-style `timings` at the
                                    //     top level (predicted_per_second
                                    //     pre-computed) — OWUI specifically
                                    //     merges this into the usage object
                                    //     in its stream handler.
                                    let wall_ns = wall_us.saturating_mul(1_000);
                                    let prefill_ns = (prefill_ms * 1_000_000.0) as u64;
                                    let generation_ns = (generation_ms * 1_000_000.0) as u64;
                                    let usage = Json::Obj(vec![
                                        ("id".into(),      Json::Str(stream_id.clone())),
                                        ("object".into(),  Json::Str(
                                            if is_chat { "chat.completion.chunk" }
                                            else       { "text_completion" }.into())),
                                        ("created".into(), Json::Num(unix_now() as f64)),
                                        ("model".into(),   Json::Str(model_name.clone())),
                                        ("choices".into(), Json::Arr(vec![])),
                                        ("usage".into(),   Json::Obj(vec![
                                            // OpenAI-spec
                                            ("prompt_tokens".into(),     Json::Num(n_p as f64)),
                                            ("completion_tokens".into(), Json::Num(n_c as f64)),
                                            ("total_tokens".into(),      Json::Num((n_p + n_c) as f64)),
                                            // Ollama-compat
                                            ("prompt_eval_count".into(),    Json::Num(n_p as f64)),
                                            ("prompt_eval_duration".into(), Json::Num(prefill_ns as f64)),
                                            ("eval_count".into(),           Json::Num(n_c as f64)),
                                            ("eval_duration".into(),        Json::Num(generation_ns as f64)),
                                            ("total_duration".into(),       Json::Num(wall_ns as f64)),
                                        ])),
                                        // llama.cpp `timings` — top-level
                                        // sibling of `usage`. OWUI's
                                        // middleware does:
                                        //   raw_usage.update(data.get('timings', {}))
                                        // so the per-second fields end up
                                        // in the message's usage object and
                                        // drive the tok/s display.
                                        ("timings".into(),  Json::Obj(vec![
                                            ("prompt_n".into(),                Json::Num(n_p as f64)),
                                            ("prompt_ms".into(),               Json::Num(prefill_ms)),
                                            ("prompt_per_token_ms".into(),     Json::Num(
                                                if n_p > 0 { prefill_ms / n_p as f64 } else { 0.0 })),
                                            ("prompt_per_second".into(),       Json::Num(prompt_tok_per_s)),
                                            ("predicted_n".into(),             Json::Num(n_c as f64)),
                                            ("predicted_ms".into(),            Json::Num(generation_ms)),
                                            ("predicted_per_token_ms".into(),  Json::Num(
                                                if n_c > 0 { generation_ms / n_c as f64 } else { 0.0 })),
                                            ("predicted_per_second".into(),    Json::Num(tok_per_s)),
                                        ])),
                                    ]).to_string();
                                    let _ = reply_tx.send(StreamMsg::Chunk(usage));
                                }
                                // Done signals the connection handler to
                                // write "data: [DONE]\n\n" and close.
                                captured_response = Some(Json::Obj(vec![
                                    ("stream".into(), Json::Bool(true)),
                                    ("assembled_text".into(), Json::Str(text.clone())),
                                    ("finish_reason".into(), Json::Str(
                                        if eos { "stop" } else { "length" }.into())),
                                ]).to_string());
                                HttpReply { status: 200, status_text: "OK",
                                            body: String::new() }
                            } else {
                                let body = if is_chat {
                                    chat_completion_response(&model_name, &text, n_p, n_c, eos, &lp)
                                } else {
                                    completion_response(&model_name, &text, n_p, n_c, eos, &lp)
                                };
                                captured_response = Some(body.clone());
                                HttpReply { status: 200, status_text: "OK", body }
                            }
                        }
                        Ok(Err(e)) => {
                            warn!("req={} target={} status=400 reason={:?}",
                                  job.request_id, job.target.label(), e);
                            metrics.requests_4xx.fetch_add(1, Ordering::Relaxed);
                            HttpReply {
                                status: 400, status_text: "Bad Request",
                                body: error_body(&e, "invalid_request_error"),
                            }
                        }
                        Err(payload) => {
                            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                                (*s).to_string()
                            } else if let Some(s) = payload.downcast_ref::<String>() {
                                s.clone()
                            } else {
                                "unknown panic in generate()".to_string()
                            };
                            error!("req={} target={} status=500 PANIC={}",
                                   job.request_id, job.target.label(), msg);
                            metrics.requests_5xx.fetch_add(1, Ordering::Relaxed);
                            metrics.panics_recovered.fetch_add(1, Ordering::Relaxed);
                            HttpReply {
                                status: 500, status_text: "Internal Server Error",
                                body: error_body(
                                    &format!("internal panic: {msg}"),
                                    "server_error"),
                            }
                        }
                    }
                }
            },
        };
        if let Some(status) = job_status.as_ref() {
            let response = captured_response.as_deref().unwrap_or(&reply.body);
            status.finish_run(job.request_id, reply.status, response, captured_stats);
        }
        let _ = job.reply.send(StreamMsg::Done(reply));
        if let Some(status) = job_status.as_ref() {
            status.active_request.store(0, Ordering::Relaxed);
        }
    }
}

// --- connection handling ----------------------------------------------

fn handle_conn(mut stream: std::net::TcpStream, target: Target,
               tx: mpsc::Sender<WorkerCommand>, metrics: Arc<Metrics>,
               status: Arc<api::ApiStatus>)
{
    let http_id = metrics.requests_total.fetch_add(1, Ordering::Relaxed) + 1;
    let mut request_id = http_id;
    let client_ip = stream.peer_addr().map(|addr| addr.ip().to_string())
        .unwrap_or_else(|_| "unknown".into());
    let request = match http::read_request(&stream) {
        Ok(r) => r,
        Err(e) => {
            metrics.requests_4xx.fetch_add(1, Ordering::Relaxed);
            warn!("http={http_id} target={} status=400 reason=malformed-http err={e}",
                  target.label());
            let _ = http::write_response(&mut stream, 400, "Bad Request",
                &error_body(&format!("malformed HTTP request: {e}"), "invalid_request_error"));
            return;
        }
    };

    // Plain GET /metrics on any port — serves Prometheus text. Cheap;
    // no GPU work. Operator point-of-entry for serving observability.
    let (path, query) = request.path.split_once('?').unwrap_or((&request.path, ""));
    let path = path.trim_end_matches('/');
    let is_get = request.method.eq_ignore_ascii_case("GET");
    let model_name = status.model.lock().map(|model| model.id.clone()).unwrap_or_default();
    if is_get && path.is_empty() {
        let _ = http::write_typed_response(&mut stream, 200, "OK",
            "text/html; charset=utf-8", api::INDEX_HTML_V3);
        return;
    }
    if is_get && path == "/docs" {
        let _ = http::write_typed_response(&mut stream, 200, "OK",
            "text/html; charset=utf-8", api::DOCS_HTML);
        return;
    }
    if is_get && path == "/openapi.json" {
        let body = api::openapi_json();
        let _ = http::write_typed_response(&mut stream, 200, "OK",
            "application/vnd.oai.openapi+json;version=3.1", &body);
        return;
    }
    if is_get && path == "/api/status" {
        let body = status.json(&metrics);
        let _ = http::write_response(&mut stream, 200, "OK", &body);
        return;
    }
    if is_get && path == "/api/runs" {
        let body = status.runs_json();
        let _ = http::write_response(&mut stream, 200, "OK", &body);
        return;
    }
    if is_get && path == "/api/logs" {
        let body = api::logs_json();
        let _ = http::write_response(&mut stream, 200, "OK", &body);
        return;
    }
    if is_get && path == "/api/models/switch" {
        let body = status.switch_json();
        let _ = http::write_response(&mut stream, 200, "OK", &body);
        return;
    }
    if is_get && path == "/api/models" {
        let body = status.models_json(query.split('&').any(|part| part == "refresh=1"));
        let _ = http::write_response(&mut stream, 200, "OK", &body);
        return;
    }
    if is_get && (path == "/api/config" || path == "/api/config/reload") {
        let body = if path.ends_with("/reload") {
            status.config_reload.lock().map(|r| serde_json::json!({"id":r.id,"state":r.state,"accepted_at_ms":r.accepted_at_ms,"completed_at_ms":r.completed_at_ms,"error":r.error}).to_string()).unwrap_or_else(|_| "null".into())
        } else { status.config_json() };
        let _ = http::write_response(&mut stream, 200, "OK", &body);
        return;
    }
    if request.method.eq_ignore_ascii_case("PUT") && path == "/api/config" {
        if request.headers.get("x-reinstinct-action").map(String::as_str) != Some("update-config") {
            let body = error_body("configuration changes require X-ReInstinct-Action: update-config", "invalid_request_error");
            let _ = http::write_response(&mut stream, 400, "Bad Request", &body);
            return;
        }
        if target != Target::Big {
            let body = error_body("configuration changes are available only on the big-model dashboard", "invalid_request_error");
            let _ = http::write_response(&mut stream, 400, "Bad Request", &body);
            return;
        }
        let update = match serde_json::from_str::<serde_json::Value>(&request.body) {
            Ok(v) => v,
            Err(e) => { let _ = http::write_response(&mut stream, 400, "Bad Request", &error_body(&format!("invalid configuration JSON: {e}"), "invalid_request_error")); return; }
        };
        let current = match status.config.lock() { Ok(c) => c.clone(), Err(_) => { let _ = http::write_response(&mut stream, 503, "Service Unavailable", &error_body("configuration state is unavailable", "server_error")); return; } };
        let next = match current.apply_update(&update) {
            Ok(v) => v,
            Err(e) => { let _ = http::write_response(&mut stream, 400, "Bad Request", &error_body(&e, "invalid_request_error")); return; }
        };
        if let Err(e) = next.persist() {
            let _ = http::write_response(&mut stream, 500, "Internal Server Error", &error_body(&e, "server_error"));
            return;
        }
        let reload = current.restart_required_fields(&next);
        let body = serde_json::json!({"state":"saved","restart_required_fields":reload}).to_string();
        let _ = http::write_response(&mut stream, 200, "OK", &body);
        return;
    }
    if request.method.eq_ignore_ascii_case("POST") && path == "/api/config/reload" {
        if request.headers.get("x-reinstinct-action").map(String::as_str) != Some("reload-config") {
            let body = error_body("configuration reload requires X-ReInstinct-Action: reload-config", "invalid_request_error");
            let _ = http::write_response(&mut stream, 400, "Bad Request", &body);
            return;
        }
        if target != Target::Big {
            let body = error_body("configuration reload is available only on the big-model dashboard", "invalid_request_error");
            let _ = http::write_response(&mut stream, 400, "Bad Request", &body);
            return;
        }
        let current = match status.config.lock() { Ok(c) => c.clone(), Err(_) => { let _ = http::write_response(&mut stream, 503, "Service Unavailable", &error_body("configuration state is unavailable", "server_error")); return; } };
        let next = match current.load_persisted() {
            Ok(v) => v,
            Err(e) => { let _ = http::write_response(&mut stream, 400, "Bad Request", &error_body(&e, "invalid_request_error")); return; }
        };
        let restart = current.restart_required_fields(&next);
        if !restart.is_empty() {
            let _ = http::write_response(&mut stream, 409, "Conflict", &error_body(&format!("service restart required for: {}", restart.join(", ")), "restart_required"));
            return;
        }
        let reload_id = match status.queue_config_reload(&next) {
            Ok(id) => id,
            Err(e) => { let _ = http::write_response(&mut stream, 409, "Conflict", &error_body(&e, "config_reload_error")); return; }
        };
        if tx.send(WorkerCommand::ReloadConfig(ConfigReload { id: reload_id, config: next })).is_err() {
            status.fail_config_reload(reload_id, "server worker is gone".into());
            let _ = http::write_response(&mut stream, 503, "Service Unavailable", &error_body("server worker is gone", "server_error"));
            return;
        }
        let body = serde_json::json!({"id":reload_id,"state":"queued","status_url":"/api/config/reload"}).to_string();
        let _ = http::write_response(&mut stream, 202, "Accepted", &body);
        return;
    }
    if request.method.eq_ignore_ascii_case("POST") && path == "/api/models/switch" {
        if request.headers.get("x-reinstinct-action").map(String::as_str) != Some("switch-model") {
            let body = error_body("model switching requires X-ReInstinct-Action: switch-model", "invalid_request_error");
            let _ = http::write_response(&mut stream, 400, "Bad Request", &body);
            return;
        }
        if target != Target::Big {
            let body = error_body("model switching is available only on the big-model dashboard", "invalid_request_error");
            let _ = http::write_response(&mut stream, 400, "Bad Request", &body);
            return;
        }
        let requested = serde_json::from_str::<serde_json::Value>(&request.body).ok()
            .and_then(|value| value.get("path")?.as_str().map(str::to_string));
        let Some((model_path, projector)) = requested.as_deref()
            .and_then(|path| api::resolve_catalog_model(&status.model_dir, path)) else {
            let body = error_body("path must select a model GGUF inside the configured catalog root", "invalid_request_error");
            let _ = http::write_response(&mut stream, 400, "Bad Request", &body);
            return;
        };
        let switch_id = match status.queue_model_switch(&model_path) {
            Ok(id) => id,
            Err(message) => {
                let _ = http::write_response(&mut stream, 409, "Conflict", &error_body(&message, "model_switch_error"));
                return;
            }
        };
        if tx.send(WorkerCommand::Switch(ModelSwitch { id: switch_id, path: model_path, projector })).is_err() {
            status.fail_model_switch(switch_id, "server worker is gone".into(), false);
            let body = error_body("server worker is gone", "server_error");
            let _ = http::write_response(&mut stream, 503, "Service Unavailable", &body);
            return;
        }
        let body = serde_json::json!({"id":switch_id,"state":"queued","status_url":"/api/models/switch"}).to_string();
        let _ = http::write_response(&mut stream, 202, "Accepted", &body);
        return;
    }
    if is_get && path.starts_with("/api/runs/") {
        let id = path["/api/runs/".len()..].parse::<u64>().ok();
        if let Some(body) = id.and_then(|id| status.run_json(id)) {
            let _ = http::write_response(&mut stream, 200, "OK", &body);
        } else {
            let body = error_body_with_details("run history entry was not found",
                "invalid_request_error", Some("id"), Some("run_not_found"));
            let _ = http::write_response(&mut stream, 404, "Not Found", &body);
        }
        return;
    }
    if is_get && path.ends_with("/metrics") {
        let body = metrics.render_prometheus();
        // Direct write — bypass JSON error_body shape.
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(), body);
        let _ = std::io::Write::write_all(&mut stream, resp.as_bytes());
        return;
    }
    // Plain GET /healthz — tiny liveness check.
    if is_get && path.ends_with("/healthz") {
        let body = "ok\n";
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(), body);
        let _ = std::io::Write::write_all(&mut stream, resp.as_bytes());
        return;
    }

    // GET /v1/models — OpenAI model-list endpoint. Each port advertises
    // the single model loaded on it (empty list for the embed port until
    // the encoder runtime lands). Open WebUI and other OpenAI-shaped
    // clients use this to populate their model dropdown.
    if is_get && path.ends_with("/v1/models") {
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let body = if model_name.is_empty() {
            r#"{"object":"list","data":[]}"#.to_string()
        } else {
            format!(
                r#"{{"object":"list","data":[{{"id":"{}","object":"model","created":{},"owned_by":"reinstinct"}}]}}"#,
                model_name, created)
        };
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(), body);
        let _ = std::io::Write::write_all(&mut stream, resp.as_bytes());
        return;
    }
    if is_get && path.ends_with("/readyz") {
        let phase = status.phase.lock().map(|p| p.name).unwrap_or("unknown");
        let (code, text) = if phase == "ready" { (200, "ready\n") } else { (503, "not ready\n") };
        let resp = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: text/plain\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            code, if code == 200 { "OK" } else { "Service Unavailable" }, text.len(), text);
        let _ = std::io::Write::write_all(&mut stream, resp.as_bytes());
        return;
    }
    // GET /v1/models/{model} — SDKs commonly probe the selected model after
    // listing it. Return the same exact ID advertised by /v1/models and a
    // stable OpenAI-shaped model_not_found error for anything else.
    if is_get && path.starts_with("/v1/models/") {
        let requested = &path["/v1/models/".len()..];
        if requested.is_empty() || model_name.is_empty() || requested != model_name.as_str() {
            metrics.requests_4xx.fetch_add(1, Ordering::Relaxed);
            let body = error_body_with_details(
                &format!("model '{requested}' was not found on this endpoint"),
                "invalid_request_error", Some("model"), Some("model_not_found"));
            let _ = http::write_response(&mut stream, 404, "Not Found", &body);
            return;
        }
        let body = format!(
            r#"{{"id":"{}","object":"model","created":{},"owned_by":"reinstinct"}}"#,
            model_name, unix_now());
        let _ = http::write_response(&mut stream, 200, "OK", &body);
        return;
    }

    // Route. LLM ports take /v1/completions (raw) or /v1/chat/completions
    // (messages, chat template applied server-side). Embed port takes
    // /v1/embeddings (answers 503 until the encoder lands).
    let is_post = request.method.eq_ignore_ascii_case("POST");
    let route = if target == Target::Embed {
        if is_post && path.ends_with("/v1/embeddings") { Some("embed") } else { None }
    } else if is_post && path.ends_with("/v1/chat/completions") {
        Some("chat")
    } else if is_post && path.ends_with("/v1/completions") {
        Some("completions")
    } else {
        None
    };

    if is_post && path.starts_with("/v1/") {
        let phase = status.phase.lock().map(|p| p.name).unwrap_or("unknown");
        if phase == "loading" || phase == "switching" {
            metrics.requests_5xx.fetch_add(1, Ordering::Relaxed);
            let body = error_body("model is loading; retry when /readyz returns 200", "server_error");
            let _ = http::write_response(&mut stream, 503, "Service Unavailable", &body);
            return;
        }
    }

    if is_post && path.starts_with("/v1/") {
        request_id = metrics.next_run_id.fetch_add(1, Ordering::Relaxed) + 1;
        metrics.inference_runs_total.fetch_add(1, Ordering::Relaxed);
        status.queue_run(request_id, client_ip, &request.method, path,
            route.unwrap_or("unknown"), &request.body);
    }

    let req = match route {
        None => Err((404u16, "Not Found",
            format!("no route for {} {} (expected POST /v1/completions or \
                     /v1/chat/completions on this port, or GET /metrics / /healthz)",
                    request.method, request.path))),
        Some("embed") => {
            // Worker answers 503; keep the shape.
            Ok(GenReq {
                prompt: PromptInput::Raw(String::new()),
                model: None,
                max_tokens: 0,
                stop: Vec::new(),
                sampler: crate::sampling::SamplerParams::default(),
                use_speculative: None,
                speculative_k: None,
                speculative_p_min: 0.0,
                request_timeout: None,
                stream: false,
                stream_include_usage: false,
                top_logprobs_n: 0,
            })
        }
        Some("chat") => parse_chat_completions(&request.body),
        Some("completions") => parse_completions(&request.body),
        Some(other) => unreachable!("unknown route tag {other}"),
    };

    // A request may omit model for compatibility with older clients, but a
    // supplied ID must select the model actually loaded on this port.
    if let Ok(parsed) = &req {
        if let Some(requested) = parsed.model.as_deref() {
            if model_name.is_empty() || requested != model_name.as_str() {
                metrics.requests_4xx.fetch_add(1, Ordering::Relaxed);
                let body = error_body_with_details(
                    &format!("model '{requested}' was not found on this endpoint"),
                    "invalid_request_error", Some("model"), Some("model_not_found"));
                status.finish_run(request_id, 404, &body, None);
                let _ = http::write_response(&mut stream, 404, "Not Found", &body);
                return;
            }
        }
    }

    let (rtx, rrx) = mpsc::channel();
    status.queued.fetch_add(1, Ordering::Relaxed);
    if tx.send(WorkerCommand::Generate(Job { request_id, target, req, reply: rtx })).is_err() {
        status.queued.fetch_sub(1, Ordering::Relaxed);
        metrics.requests_5xx.fetch_add(1, Ordering::Relaxed);
        let body = error_body("server worker is gone", "server_error");
        status.finish_run(request_id, 503, &body, None);
        let _ = http::write_response(&mut stream, 503, "Service Unavailable", &body);
        return;
    }
    // Receive the first message: if it's Done, plain response. If it's
    // Chunk, switch to SSE streaming mode (writing each chunk as it arrives
    // until a Done arrives or the channel closes).
    use std::io::Write;
    let first = rrx.recv();
    match first {
        Ok(StreamMsg::Done(reply)) => {
            let _ = http::write_response(&mut stream, reply.status, reply.status_text, &reply.body);
        }
        Ok(StreamMsg::Chunk(payload)) => {
            // SSE: open with the appropriate headers, then write each
            // chunk as `data: {payload}\n\n`. Final `data: [DONE]\n\n`
            // matches OpenAI's terminator. If a socket write fails
            // mid-stream, drop the connection — the worker will see the
            // channel close on its next send and stop generating.
            let header = "HTTP/1.1 200 OK\r\n\
                          Content-Type: text/event-stream\r\n\
                          Cache-Control: no-cache\r\n\
                          Connection: close\r\n\r\n";
            if stream.write_all(header.as_bytes()).is_err() { return; }
            if stream.write_all(format!("data: {payload}\n\n").as_bytes()).is_err() { return; }
            let _ = stream.flush();
            loop {
                match rrx.recv() {
                    Ok(StreamMsg::Chunk(p)) => {
                        if stream.write_all(format!("data: {p}\n\n").as_bytes()).is_err() { return; }
                        let _ = stream.flush();
                    }
                    Ok(StreamMsg::Done(_reply)) => {
                        // Standard OpenAI SSE terminator. The summary
                        // reply body isn't sent in streaming mode —
                        // clients accumulate the chunks.
                        let _ = stream.write_all(b"data: [DONE]\n\n");
                        let _ = stream.flush();
                        return;
                    }
                    Err(_) => {
                        // Worker dropped the channel. Close stream.
                        return;
                    }
                }
            }
        }
        Err(_) => {
            metrics.requests_5xx.fetch_add(1, Ordering::Relaxed);
            let _ = http::write_response(&mut stream, 500, "Internal Server Error",
                &error_body("worker dropped the request", "server_error"));
        }
    }
}

fn acceptor(port: u16, target: Target, tx: mpsc::Sender<WorkerCommand>,
            metrics: Arc<Metrics>, status: Arc<api::ApiStatus>) {
    let listener = match std::net::TcpListener::bind(("0.0.0.0", port)) {
        Ok(l) => l,
        Err(e) => { error!("FATAL: cannot bind port {port}: {e}"); return; }
    };
    info!("{} listening on :{port}", target.label());
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let tx = tx.clone();
                let metrics = Arc::clone(&metrics);
                let status = Arc::clone(&status);
                thread::spawn(move || handle_conn(stream, target, tx, metrics, status));
            }
            Err(e) => warn!("accept error on :{port}: {e}"),
        }
    }
}

/// Start the three-port multi-model server. Blocks forever.
fn panic_detail(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() { (*message).to_string() }
    else if let Some(message) = payload.downcast_ref::<String>() { message.clone() }
    else { "unknown panic payload".into() }
}

pub fn run(overrides: config::ServeOverrides)
    -> Result<(), String>
{
    let effective = config::ServeConfig::from_overrides(overrides)?;
    let big = effective.big.clone();
    let model_dir = effective.model_dir.clone();
    let big_drafter = effective.big_drafter.clone();
    let small = effective.small.clone();
    let embed = effective.embed.clone();
    let big_port = effective.big_port;
    let small_port = effective.small_port;
    let embed_port = effective.embed_port;
    let max_seq = effective.max_seq;
    let vision_use_gpu = !effective.cpu_vision;
    let mmproj = effective.mmproj.clone();
    let mtmd_bridge = effective.mtmd_bridge.clone();
    let vision_threads = effective.vision_threads;
    let vision_min_tokens = effective.vision_min_tokens;
    let vision_max_tokens = effective.vision_max_tokens;
    let vision = match (mmproj, mtmd_bridge) {
        (None, None) => None,
        (Some(mmproj), Some(bridge)) if vision_threads > 0 &&
            vision_min_tokens != 0 && vision_max_tokens != 0 &&
            !(vision_min_tokens > 0 && vision_max_tokens > 0 && vision_min_tokens > vision_max_tokens) => Some(VisionConfig {
            mmproj, bridge, threads: vision_threads, image_min_tokens: vision_min_tokens,
            image_max_tokens: vision_max_tokens, use_gpu: vision_use_gpu,
        }),
        (Some(_), None) | (None, Some(_)) => return Err(
            "server vision requires both --mmproj PATH and --mtmd-bridge PATH".into()),
        (_, _) => return Err("vision threads must be positive and token limits must be -1 or positive with minimum <= maximum".into()),
    };
    // Surface any REINSTINCT_* env vars at startup. Several of them are
    // perf-killers if set unintentionally on a serve box (graph capture
    // off, dp4a path off, etc) — better to log them than have an
    // operator chasing a silent regression weeks later.
    let env_warnings: Vec<(&str, bool)> = vec![
        ("REINSTINCT_NO_GRAPH",        true),
        ("REINSTINCT_MOE_PROFILE",     true),
        ("REINSTINCT_NO_DP4A_Q4",      true),
        ("REINSTINCT_NO_DP4A_Q5",      true),
        ("REINSTINCT_NO_DP4A_Q6",      true),
        ("REINSTINCT_NO_DP4A_Q8",      true),
        ("REINSTINCT_GEMMA_NO_DP4A",   true),
        ("REINSTINCT_GDN_NO_LDS128",   true),
        ("REINSTINCT_OLD_ATTN",        true),
        ("REINSTINCT_PREFILL_NO_GRAPH", true),
        ("REINSTINCT_PREFILL_DEBUG",   false),
        ("REINSTINCT_DECODE_DEBUG",    false),
        ("REINSTINCT_PREFILL_TRACE",   false),
    ];
    for (var, is_perf_killer) in &env_warnings {
        if std::env::var_os(var).is_some() {
            if *is_perf_killer {
                warn!("{var} is set — perf will regress significantly; \
                       unset for production");
            } else {
                info!("{var} is set — tracing on; expect verbose logs");
            }
        }
    }

    if let Some(e) = &embed {
        info!("--embed {} accepted but deferred — \
               the :{embed_port} port will answer 503 until the \
               nomic-bert encoder lands.", e.display());
    }

    let (tx, rx) = mpsc::channel::<WorkerCommand>();
    let metrics = Arc::new(Metrics::new());

    // Derive each port's advertised model name from the GGUF filename
    // stem (same rule the worker uses for ServerModel.name). The embed
    // port has no model loaded today, so its /v1/models returns an
    // empty list.
    let stem = |p: &PathBuf| -> String {
        p.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string()
    };
    let mk_status = |port, target: Target, model: &PathBuf, drafter: Option<PathBuf>,
                     vision: Option<&VisionConfig>| Arc::new(api::ApiStatus {
        started_at: unix_now(), max_seq: std::sync::Mutex::new(max_seq), target: target.label(), port,
        model: std::sync::Mutex::new(api::ModelStatus { id: stem(model), path: model.clone(), drafter,
        vision: vision.map(|v| api::VisionStatus {
            projector: v.mmproj.clone(), threads: v.threads,
            min_tokens: v.image_min_tokens, max_tokens: v.image_max_tokens,
            device: if v.use_gpu { "gpu" } else { "cpu" },
        }) }), model_dir: model_dir.clone(),
        phase: std::sync::Mutex::new(api::Phase { name: "starting", detail: None }),
        loading: std::sync::Mutex::new(None),
        network: std::sync::Mutex::new(api::NetworkSample::default()),
        catalog: std::sync::Mutex::new(None),
        model_switch: std::sync::Mutex::new(api::ModelSwitchStatus::default()),
        next_switch_id: AtomicU64::new(0),
        config: std::sync::Mutex::new(effective.clone()),
        config_reload: std::sync::Mutex::new(api::ConfigReloadStatus::default()),
        next_config_reload_id: AtomicU64::new(0),
        queued: AtomicU64::new(0), active_request: AtomicU64::new(0),
        history: std::sync::Mutex::new(api::RunHistory::new()),
    });
    let big_status = mk_status(big_port, Target::Big, &big, big_drafter.clone(), vision.as_ref());
    let embed_path = embed.clone().unwrap_or_else(|| PathBuf::from("embedder-not-configured"));
    let embed_status = mk_status(embed_port, Target::Embed, &embed_path, None, None);

    let acceptors: Vec<(u16, Target, Arc<api::ApiStatus>)> = if let Some(ref sp) = small {
        let small_status = mk_status(small_port, Target::Small, sp, None, None);
        vec![
            (big_port, Target::Big, Arc::clone(&big_status)),
            (small_port, Target::Small, small_status),
            (embed_port, Target::Embed, Arc::clone(&embed_status)),
        ]
    } else {
        info!("--small not given — small-model port disabled (VRAM saved)");
        vec![
            (big_port, Target::Big, Arc::clone(&big_status)),
            (embed_port, Target::Embed, Arc::clone(&embed_status)),
        ]
    };

    let statuses: Vec<Arc<api::ApiStatus>> = acceptors.iter().map(|(_, _, s)| Arc::clone(s)).collect();
    let worker_handle = {
        let metrics = Arc::clone(&metrics);
        thread::Builder::new().name("gpu-worker".into())
            .spawn(move || {
                let statuses_for_panic = statuses.clone();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    worker(rx, effective, metrics, statuses_for_panic.clone())
                }));
                if let Err(payload) = result {
                    let detail = panic_detail(payload);
                    error!("GPU worker panicked during startup or service: {detail}");
                    for status in &statuses_for_panic {
                        status.set_phase("error", Some(format!("GPU worker panic: {detail}")));
                    }
                    // Keep the process and dashboard alive so the operator
                    // can read the failure and restart through the service
                    // manager after correcting the cause.
                    loop { thread::park(); }
                }
            })
            .map_err(|e| e.to_string())?
    };

    for (port, target, status) in acceptors {
        let tx = tx.clone();
        let metrics = Arc::clone(&metrics);
        thread::Builder::new().name(format!("accept-{}", target.label()))
            .spawn(move || acceptor(port, target, tx, metrics, status))
            .map_err(|e| e.to_string())?;
    }
    drop(tx);   // only the acceptors hold senders now

    worker_handle.join().map_err(|_| "gpu worker panicked".to_string())?;
    Ok(())
}

#[cfg(test)]
mod logprobs_tests {
    use super::*;

    fn parse_lp(body: &str) -> usize {
        // tuple positions 0..8: max_tokens, sampler, use_speculative,
        // speculative_k, speculative_p_min, request_timeout, stream,
        // stream_include_usage, top_logprobs_n  (← .8)
        parse_common_fields(&Json::parse(body).unwrap(), COMPLETION_DEFAULTS).unwrap().9
    }

    #[test]
    fn logprobs_bool_true_defaults_to_5() {
        assert_eq!(parse_lp(r#"{"logprobs": true}"#), 5);
    }

    #[test]
    fn logprobs_int_chooses_count() {
        assert_eq!(parse_lp(r#"{"logprobs": 10}"#), 10);
    }

    #[test]
    fn logprobs_capped_at_20() {
        assert_eq!(parse_lp(r#"{"logprobs": 999}"#), 20);
    }

    #[test]
    fn logprobs_omitted_is_zero() {
        assert_eq!(parse_lp(r#"{}"#), 0);
    }

    #[test]
    fn logprobs_false_is_zero() {
        assert_eq!(parse_lp(r#"{"logprobs": false}"#), 0);
    }

    #[test]
    fn render_text_logprobs_has_parallel_arrays() {
        let lp = vec![
            TokenLogprob { token: "hi".into(), logprob: -0.5,
                top_alts: vec![("hi".into(), -0.5), ("hello".into(), -1.2)] },
            TokenLogprob { token: " world".into(), logprob: -0.7,
                top_alts: vec![(" world".into(), -0.7)] },
        ];
        let s = render_text_logprobs(&lp).to_string();
        assert!(s.contains("\"tokens\""));
        assert!(s.contains("\"token_logprobs\""));
        assert!(s.contains("\"top_logprobs\""));
        assert!(s.contains("\"hi\""));
        assert!(s.contains("\" world\""));
    }

    #[test]
    fn render_chat_logprobs_has_content_array() {
        let lp = vec![TokenLogprob {
            token: "hi".into(), logprob: -0.5,
            top_alts: vec![("hi".into(), -0.5), ("hello".into(), -1.2)] }];
        let s = render_chat_logprobs(&lp).to_string();
        assert!(s.contains("\"content\""));
        assert!(s.contains("\"token\""));
        assert!(s.contains("\"logprob\""));
        assert!(s.contains("\"top_logprobs\""));
    }

    #[test]
    fn empty_logprobs_render_as_null() {
        assert_eq!(render_text_logprobs(&[]).to_string(), "null");
        assert_eq!(render_chat_logprobs(&[]).to_string(), "null");
    }

    #[test]
    fn chat_structured_content_accepts_one_data_png() {
        let req = parse_chat_completions(r#"{
            "messages":[{"role":"user","content":[
              {"type":"text","text":"What is this? "},
              {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}
            ]}]
        }"#).unwrap();
        match req.prompt {
            PromptInput::ChatVision { messages, image } => {
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].content, "What is this? <__media__>");
                assert_eq!(image, vec![137, 80, 78, 71, 13, 10, 26, 10]);
            }
            _ => panic!("expected vision request"),
        }
    }

    #[test]
    fn chat_structured_content_rejects_remote_or_multiple_images() {
        let remote = r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/x.png"}}]}]}"#;
        let remote_error = parse_chat_completions(remote).err().expect("remote URL must fail");
        assert!(remote_error.2.contains("remote image URLs"));
        let two = r#"{"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,/9j/"}},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}
        ]}]}"#;
        assert!(parse_chat_completions(two).is_err());
    }

    #[test]
    fn common_fields_accept_max_completion_tokens_and_stop() {
        let req = parse_completions(r#"{
            "prompt":"hello", "max_completion_tokens":17,
            "stop":["END", "DONE"], "n":1, "user":"test"
        }"#).unwrap();
        assert_eq!(req.max_tokens, 17);
        assert_eq!(req.stop, ["END", "DONE"]);
    }

    #[test]
    fn common_fields_reject_conflicting_or_unsupported_options() {
        assert!(parse_completions(r#"{"prompt":"x","max_tokens":1,"max_completion_tokens":2}"#).is_err());
        assert!(parse_completions(r#"{"prompt":"x","n":2}"#).is_err());
        assert!(parse_completions(r#"{"prompt":"x","tools":[]}"#).is_err());
    }

    #[test]
    fn stop_matching_handles_token_boundary_and_holds_prefixes() {
        let stops = vec!["END".to_string()];
        assert_eq!(stop_visible_end("answer EN", &stops), (7, false));
        assert_eq!(stop_visible_end("answer END trailing", &stops), (7, true));
        assert_eq!(stop_visible_end("answer", &stops), (6, false));

        let unicode_stops = vec!["😊END".to_string()];
        assert_eq!(stop_visible_end("answer 😊", &unicode_stops), (7, false));
    }

    #[test]
    fn stable_utf8_decode_holds_incomplete_multibyte_suffix() {
        let complete = "prefix 😊".as_bytes();
        for end in 8..complete.len() {
            assert_eq!(stable_utf8_prefix(&complete[..end]), "prefix ");
        }
        assert_eq!(stable_utf8_prefix(complete), "prefix 😊");
        assert_eq!(stable_utf8_prefix(b"ok\xff!"), "ok\u{fffd}!");
    }
}
