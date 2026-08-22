#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::io::{self, Write};
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use super::Metrics;
use super::config::ServeConfig;
use super::gpu::GpuInventory;
use super::thermal::ThermalGuard;
use serde_json::{Value, json};

pub struct ApiStatus {
    pub started_at: u64,
    pub max_seq: Mutex<usize>,
    pub target: &'static str,
    pub port: u16,
    pub model: Mutex<ModelStatus>,
    pub model_dir: PathBuf,
    pub phase: Mutex<Phase>,
    pub loading: Mutex<Option<LoadingStatus>>,
    pub network: Mutex<NetworkSample>,
    pub queued: AtomicU64,
    pub active_request: AtomicU64,
    /// Physical context positions currently occupied by the active model
    /// state. This is deliberately separate from reserved context bytes.
    pub context_used_tokens: AtomicU64,
    pub history: Mutex<RunHistory>,
    pub catalog: Mutex<Option<CatalogCache>>,
    pub model_switch: Mutex<ModelSwitchStatus>,
    pub next_switch_id: AtomicU64,
    pub config: Mutex<ServeConfig>,
    pub config_reload: Mutex<ConfigReloadStatus>,
    pub next_config_reload_id: AtomicU64,
    pub inventory: std::sync::Arc<GpuInventory>,
    pub thermal: std::sync::Arc<ThermalGuard>,
}

#[derive(Clone)]
pub struct ConfigReloadStatus {
    pub id: u64,
    pub state: &'static str,
    pub accepted_at_ms: Option<u64>,
    pub completed_at_ms: Option<u64>,
    pub error: Option<String>,
}

impl Default for ConfigReloadStatus {
    fn default() -> Self { Self { id: 0, state: "idle", accepted_at_ms: None, completed_at_ms: None, error: None } }
}

pub struct CatalogCache {
    value: Value,
    scanned_at_ms: u64,
    duration_ms: u64,
}

#[derive(Clone)]
pub struct ModelSwitchStatus {
    pub id: u64,
    pub state: &'static str,
    pub requested_path: Option<PathBuf>,
    pub previous_model: Option<String>,
    pub accepted_at_ms: Option<u64>,
    pub completed_at_ms: Option<u64>,
    pub error: Option<String>,
}

impl Default for ModelSwitchStatus {
    fn default() -> Self {
        Self { id: 0, state: "idle", requested_path: None, previous_model: None,
            accepted_at_ms: None, completed_at_ms: None, error: None }
    }
}

#[derive(Clone)]
pub struct ModelStatus {
    pub id: String,
    pub path: PathBuf,
    pub drafter: Option<PathBuf>,
    pub vision: Option<VisionStatus>,
}

pub struct LoadingStatus {
    pub started_at_ms: u64,
    pub expected_bytes: u64,
    pub network_rx_start: u64,
}

#[derive(Default)]
pub struct NetworkSample {
    at_ms: u64,
    rx_bytes: u64,
    tx_bytes: u64,
    rx_mbps: f64,
    tx_mbps: f64,
}

pub const RUN_HISTORY_CAPACITY: usize = 128;
const CAPTURE_TEXT_BYTES: usize = 64 * 1024;
const LOG_CAPACITY: usize = 1000;

static ENGINE_LOGS: OnceLock<Mutex<VecDeque<String>>> = OnceLock::new();

pub struct DashboardLogWriter;

impl Write for DashboardLogWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let text = String::from_utf8_lossy(bytes);
        let logs = ENGINE_LOGS.get_or_init(|| Mutex::new(VecDeque::with_capacity(LOG_CAPACITY)));
        if let Ok(mut logs) = logs.lock() {
            for line in text.lines().filter(|line| !line.trim().is_empty()) {
                while logs.len() >= LOG_CAPACITY { logs.pop_front(); }
                logs.push_back(line.to_string());
            }
        }
        io::stderr().write_all(bytes)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> { io::stderr().flush() }
}

pub fn logs_json() -> String {
    let logs = ENGINE_LOGS.get_or_init(|| Mutex::new(VecDeque::with_capacity(LOG_CAPACITY)));
    let lines = logs.lock().map(|logs| logs.iter().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    json!({"capacity":LOG_CAPACITY,"retained":lines.len(),"lines":lines}).to_string()
}

fn logs_summary_json() -> Value {
    let retained = ENGINE_LOGS.get().and_then(|logs| logs.lock().ok().map(|logs| logs.len())).unwrap_or(0);
    json!({"capacity":LOG_CAPACITY,"retained":retained})
}

#[derive(Clone, Default)]
pub struct RunStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub queue_ms: f64,
    pub prefill_ms: f64,
    pub ttft_ms: f64,
    pub generation_ms: f64,
    pub thermal_wait_ms: f64,
    pub total_ms: f64,
    pub prompt_tokens_per_second: f64,
    pub generation_tokens_per_second: f64,
}

pub struct RunRecord {
    id: u64,
    client_ip: String,
    method: String,
    path: String,
    request_type: String,
    state: &'static str,
    queued_at_ms: u64,
    started_at_ms: Option<u64>,
    completed_at_ms: Option<u64>,
    status_code: Option<u16>,
    request: Value,
    response: Option<Value>,
    error: Option<String>,
    stats: RunStats,
}

pub struct RunHistory {
    entries: VecDeque<RunRecord>,
    capacity: usize,
}

impl RunHistory {
    pub fn new() -> Self {
        Self { entries: VecDeque::with_capacity(RUN_HISTORY_CAPACITY),
               capacity: RUN_HISTORY_CAPACITY }
    }

    fn push(&mut self, run: RunRecord) {
        while self.entries.len() >= self.capacity { self.entries.pop_front(); }
        self.entries.push_back(run);
    }

    fn find_mut(&mut self, id: u64) -> Option<&mut RunRecord> {
        self.entries.iter_mut().find(|run| run.id == id)
    }

    fn find(&self, id: u64) -> Option<&RunRecord> {
        self.entries.iter().find(|run| run.id == id)
    }
}

#[derive(Clone)]
pub struct VisionStatus {
    pub projector: PathBuf,
    pub threads: i32,
    pub min_tokens: i32,
    pub max_tokens: i32,
    pub device: &'static str,
}

pub struct Phase {
    pub name: &'static str,
    pub detail: Option<String>,
}

impl ApiStatus {
    pub fn set_context_used(&self, tokens: usize) {
        self.context_used_tokens.store(tokens as u64, Ordering::Relaxed);
    }

    pub fn set_phase(&self, name: &'static str, detail: Option<String>) {
        if let Ok(mut phase) = self.phase.lock() {
            *phase = Phase { name, detail };
        }
    }

    pub fn begin_loading(&self, path: &std::path::Path) {
        let expected_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let (network_rx_start, _) = network_totals();
        if let Ok(mut loading) = self.loading.lock() {
            *loading = Some(LoadingStatus { started_at_ms: unix_now_ms(), expected_bytes,
                network_rx_start });
        }
        self.set_phase("loading", Some(format!("Loading {}", path.display())));
    }

    pub fn finish_loading(&self, model: ModelStatus) {
        if let Ok(mut current) = self.model.lock() { *current = model; }
        if let Ok(mut loading) = self.loading.lock() { *loading = None; }
        self.set_phase("ready", None);
    }

    pub fn queue_run(&self, id: u64, client_ip: String, method: &str,
                     path: &str, request_type: &str, body: &str) {
        if let Ok(mut history) = self.history.lock() {
            history.push(RunRecord {
                id, client_ip, method: method.into(), path: path.into(),
                request_type: request_type.into(), state: "queued",
                queued_at_ms: unix_now_ms(), started_at_ms: None,
                completed_at_ms: None, status_code: None,
                request: capture_json(body), response: None, error: None,
                stats: RunStats::default(),
            });
        }
    }

    pub fn activate_run(&self, id: u64) {
        if let Ok(mut history) = self.history.lock() {
            if let Some(run) = history.find_mut(id) {
                let now = unix_now_ms();
                run.state = "active";
                run.started_at_ms = Some(now);
                run.stats.queue_ms = now.saturating_sub(run.queued_at_ms) as f64;
            }
        }
    }

    pub fn update_generation_progress(&self, id: u64, completion_tokens: usize,
                                      generation_ms: f64) {
        if let Ok(mut history) = self.history.lock() {
            if let Some(run) = history.find_mut(id) {
                run.stats.completion_tokens = completion_tokens;
                run.stats.generation_ms = generation_ms;
                run.stats.generation_tokens_per_second = per_second(completion_tokens, generation_ms);
            }
        }
    }

    pub fn finish_run(&self, id: u64, status_code: u16, response: &str,
                      stats: Option<RunStats>) {
        if let Ok(mut history) = self.history.lock() {
            if let Some(run) = history.find_mut(id) {
                run.state = if status_code < 400 { "complete" } else { "errored" };
                run.completed_at_ms = Some(unix_now_ms());
                run.status_code = Some(status_code);
                run.response = Some(capture_json(response));
                run.error = if status_code >= 400 { error_message(response) } else { None };
                if let Some(mut stats) = stats {
                    stats.queue_ms = run.stats.queue_ms;
                    run.stats = stats;
                }
            }
        }
    }

    pub fn runs_json(&self) -> String {
        let history = match self.history.lock() {
            Ok(history) => history,
            Err(_) => return json!({"capacity":RUN_HISTORY_CAPACITY,"retained":0,
                "active_count":0,"completed":0,"errored":0,"runs":[]}).to_string(),
        };
        runs_value(&history).to_string()
    }

    pub fn run_json(&self, id: u64) -> Option<String> {
        let history = self.history.lock().ok()?;
        history.find(id).map(|run| run_detail_json(run).to_string())
    }

    pub fn models_json(&self, refresh: bool) -> String {
        if !refresh {
            if let Ok(cache) = self.catalog.lock() {
                if let Some(cache) = cache.as_ref() { return catalog_response(cache, true).to_string(); }
            }
        }
        let started = std::time::Instant::now();
        let value = catalog_value(&self.model_dir);
        let cache = CatalogCache { value, scanned_at_ms: unix_now_ms(),
            duration_ms: started.elapsed().as_millis() as u64 };
        let response = catalog_response(&cache, false).to_string();
        if let Ok(mut slot) = self.catalog.lock() { *slot = Some(cache); }
        response
    }

    pub fn queue_model_switch(&self, path: &std::path::Path) -> Result<u64, String> {
        let phase = self.phase.lock().map_err(|_| "service state is unavailable")?;
        if phase.name != "ready" { return Err(format!("service is {}", phase.name)); }
        drop(phase);
        if self.queued.load(Ordering::Relaxed) > 0 || self.active_request.load(Ordering::Relaxed) > 0 {
            return Err("model switch requires an idle worker and empty queue".into());
        }
        let id = self.next_switch_id.fetch_add(1, Ordering::Relaxed) + 1;
        let previous_model = self.model.lock().ok().map(|m| m.id.clone());
        if let Ok(mut switch) = self.model_switch.lock() {
            *switch = ModelSwitchStatus { id, state: "queued", requested_path: Some(path.to_path_buf()),
                previous_model, accepted_at_ms: Some(unix_now_ms()), completed_at_ms: None, error: None };
        }
        self.set_phase("switching", Some(format!("Queued model switch to {}", path.display())));
        Ok(id)
    }

    pub fn start_model_switch(&self, id: u64, path: &std::path::Path) {
        if let Ok(mut switch) = self.model_switch.lock() {
            if switch.id == id { switch.state = "loading"; }
        }
        self.begin_loading(path);
    }

    pub fn complete_model_switch(&self, id: u64, model: ModelStatus) {
        self.finish_loading(model);
        if let Ok(mut switch) = self.model_switch.lock() {
            if switch.id == id { switch.state = "complete"; switch.completed_at_ms = Some(unix_now_ms()); }
        }
    }

    pub fn fail_model_switch(&self, id: u64, message: String, restored: bool) {
        if let Ok(mut loading) = self.loading.lock() { *loading = None; }
        if restored { self.set_phase("ready", Some("Selected model failed; previous model restored".into())); }
        else { self.set_phase("error", Some(message.clone())); }
        if let Ok(mut switch) = self.model_switch.lock() {
            if switch.id == id { switch.state = "failed"; switch.completed_at_ms = Some(unix_now_ms()); switch.error = Some(message); }
        }
    }

    pub fn switch_json(&self) -> String { self.switch_value().to_string() }

    pub fn config_json(&self) -> String {
        let cfg = self.config.lock().ok();
        let reload = self.config_reload.lock().ok();
        let saved = cfg.as_ref().and_then(|c| c.load_persisted().ok());
        let reload_required = saved.as_ref().zip(cfg.as_ref()).map(|(saved, active)| {
            saved.json_value().as_object().into_iter().flat_map(|values| values.iter())
                .filter(|(key, value)| !matches!(key.as_str(), "big_port" | "small_port" | "embed_port")
                    && active.json_value().get(*key) != Some(*value))
                .map(|(key, _)| key.clone()).collect::<Vec<_>>()
        }).unwrap_or_default();
        let restart_required = saved.as_ref().zip(cfg.as_ref())
            .map(|(saved, active)| active.restart_required_fields(saved)).unwrap_or_default();
        json!({
            "path": cfg.as_ref().and_then(|c| c.config_path.clone()),
            "persisted": cfg.as_ref().map(|c| c.config_path.is_some()).unwrap_or(false),
            "effective": cfg.as_ref().map(|c| c.json_value()).unwrap_or(Value::Null),
            "saved": saved.as_ref().map(|c| c.json_value()).unwrap_or(Value::Null),
            "reload_required_fields": reload_required,
            "restart_required_fields": restart_required,
            "cli_locked": cfg.as_ref().map(|c| c.cli_locked.iter().cloned().collect::<Vec<_>>()).unwrap_or_default(),
            "reload": reload.map(|r| json!({"id":r.id,"state":r.state,"accepted_at_ms":r.accepted_at_ms,"completed_at_ms":r.completed_at_ms,"error":r.error})).unwrap_or(Value::Null)
        }).to_string()
    }

    pub fn gpus_json(&self) -> String { self.inventory.json().to_string() }

    pub fn apply_thermal_config(&self, next: &ServeConfig) -> Result<(), String> {
        let changed = self.config.lock().map(|current| {
            current.gpu_thermal_guard_enabled != next.gpu_thermal_guard_enabled
                || current.gpu_max_temp_c != next.gpu_max_temp_c
                || current.gpu_max_temp_seconds != next.gpu_max_temp_seconds
                || current.gpu_resume_temp_c != next.gpu_resume_temp_c
        }).map_err(|_| "configuration state is unavailable")?;
        if !changed { return Ok(()); }
        self.thermal.update_config(next.thermal_config())?;
        if let Ok(mut current) = self.config.lock() {
            current.gpu_thermal_guard_enabled = next.gpu_thermal_guard_enabled;
            current.gpu_max_temp_c = next.gpu_max_temp_c;
            current.gpu_max_temp_seconds = next.gpu_max_temp_seconds;
            current.gpu_resume_temp_c = next.gpu_resume_temp_c;
        }
        Ok(())
    }

    pub fn queue_config_reload(&self, next: &ServeConfig) -> Result<u64, String> {
        let phase = self.phase.lock().map_err(|_| "service state is unavailable")?;
        if phase.name != "ready" { return Err(format!("service is {}", phase.name)); }
        drop(phase);
        if self.queued.load(Ordering::Relaxed) > 0 || self.active_request.load(Ordering::Relaxed) > 0 {
            return Err("configuration reload requires an idle worker and empty queue".into());
        }
        if next.config_path.is_none() { return Err("server was not started with --config".into()); }
        let id = self.next_config_reload_id.fetch_add(1, Ordering::Relaxed) + 1;
        if let Ok(mut state) = self.config_reload.lock() { *state = ConfigReloadStatus { id, state: "queued", accepted_at_ms: Some(unix_now_ms()), completed_at_ms: None, error: None }; }
        self.set_phase("reloading", Some("Queued engine configuration reload".into()));
        Ok(id)
    }

    pub fn start_config_reload(&self, id: u64) {
        if let Ok(mut state) = self.config_reload.lock() { if state.id == id { state.state = "loading"; } }
        self.set_phase("reloading", Some("Loading engine configuration".into()));
    }

    pub fn complete_config_reload(&self, id: u64, cfg: ServeConfig) {
        if let Ok(mut max_seq) = self.max_seq.lock() { *max_seq = cfg.max_seq; }
        if let Ok(mut current) = self.config.lock() { *current = cfg; }
        if let Ok(mut state) = self.config_reload.lock() { if state.id == id { state.state = "complete"; state.completed_at_ms = Some(unix_now_ms()); } }
        self.set_phase("ready", None);
    }

    pub fn fail_config_reload(&self, id: u64, error: String) {
        if let Ok(mut state) = self.config_reload.lock() { if state.id == id { state.state = "failed"; state.completed_at_ms = Some(unix_now_ms()); state.error = Some(error.clone()); } }
        self.set_phase("ready", Some("Configuration reload failed; previous engine restored".into()));
    }

    fn switch_value(&self) -> Value {
        self.model_switch.lock().map(|s| json!({"id":s.id,"state":s.state,
            "requested_path":s.requested_path,"previous_model":s.previous_model,
            "accepted_at_ms":s.accepted_at_ms,"completed_at_ms":s.completed_at_ms,"error":s.error}))
            .unwrap_or(Value::Null)
    }

    pub fn json(&self, metrics: &Metrics) -> String {
        let (phase, detail) = self
            .phase
            .lock()
            .map(|p| (p.name, p.detail.clone()))
            .unwrap_or(("unknown", None));
        let model = self.model.lock().map(|m| m.clone()).unwrap_or(ModelStatus {
            id: "unknown".into(), path: PathBuf::new(), drafter: None, vision: None });
        let vision = model.vision.as_ref().map(|v| json!({"enabled":true,"projector":v.projector,
            "threads":v.threads,"min_tokens":v.min_tokens,"max_tokens":v.max_tokens,"device":v.device}))
            .unwrap_or_else(|| json!({"enabled":false}));
        let network = self.network_json();
        let loading = self.loading_json();
        let (history_summary, performance) = self.history.lock().map(|history| {
            let active = history.entries.iter().rev().find(|run| run.state == "active")
                .map(run_summary_json).unwrap_or(Value::Null);
            let last = history.entries.iter().rev().find(|run| run.state == "complete")
                .map(run_summary_json).unwrap_or(Value::Null);
            let errored = history.entries.iter().filter(|run| run.state == "errored").count();
            (json!({"capacity":history.capacity,"retained":history.entries.len(),
                    "active_count":history.entries.iter().filter(|run| run.state == "active").count(),
                    "completed":history.entries.iter().filter(|run| run.state == "complete").count(),
                    "errored":errored,
                    "active":active,"last_completed":last,
                    "capture":"requests and responses retained in process memory; image data redacted"}),
             performance_json(metrics, &history))
        }).unwrap_or((json!({"capacity":RUN_HISTORY_CAPACITY,"retained":0}), Value::Null));
        json!({
            "service":{"name":"reinstinct","version":env!("CARGO_PKG_VERSION")},
            "status":phase,"status_detail":detail,"started_at":self.started_at,
            "uptime_seconds":super::unix_now().saturating_sub(self.started_at),
            "endpoint":{"target":self.target,"port":self.port},
            "model":{"id":model.id,"path":model.path,"max_context_tokens":self.max_seq.lock().map(|v| *v).unwrap_or(0),
                "drafter":model.drafter,"vision":vision,"catalog_root":self.model_dir},
            "worker":{"queued_requests":self.queued.load(Ordering::Relaxed),
                "active_request_id":match self.active_request.load(Ordering::Relaxed){0=>Value::Null,n=>json!(n)}},
            "performance":performance,
            "loading":loading,
            "network":network,
            "run_history":history_summary,
            "logs":logs_summary_json(),
            "management":{"model_switch":self.switch_value(),"config_reload":self.config_reload.lock().map(|r| json!({"id":r.id,"state":r.state,"error":r.error})).unwrap_or(Value::Null),"switch_requires_action_header":true},
            "gpu":gpu_json(&self.inventory, &self.thermal,
                self.max_seq.lock().map(|v| *v).unwrap_or(0),
                self.context_used_tokens.load(Ordering::Relaxed)),
            "metrics":{"requests_total":metrics.requests_total.load(Ordering::Relaxed),
                "http_requests_total":metrics.requests_total.load(Ordering::Relaxed),
                "inference_runs_total":metrics.inference_runs_total.load(Ordering::Relaxed),
                "requests_ok":metrics.requests_ok.load(Ordering::Relaxed),
                "requests_4xx":metrics.requests_4xx.load(Ordering::Relaxed),
                "requests_5xx":metrics.requests_5xx.load(Ordering::Relaxed),
                "prompt_tokens":metrics.prompt_tokens.load(Ordering::Relaxed),
                "completion_tokens":metrics.completion_tokens.load(Ordering::Relaxed),
                "prefill_us_total":metrics.prefill_us_total.load(Ordering::Relaxed),
                "generation_us_total":metrics.decode_us_total.load(Ordering::Relaxed),
                "ttft_us_total":metrics.ttft_us_total.load(Ordering::Relaxed),
                "panics_recovered":metrics.panics_recovered.load(Ordering::Relaxed)},
            "api":{"openapi":"/openapi.json","docs":"/docs","health":"/healthz","ready":"/readyz",
                "models":"/v1/models","runs":"/api/runs","logs":"/api/logs","catalog":"/api/models"}
        }).to_string()
    }

    fn network_json(&self) -> Value {
        let now = unix_now_ms();
        let (rx, tx) = network_totals();
        let mut sample = match self.network.lock() { Ok(sample) => sample, Err(_) => return Value::Null };
        if sample.at_ms > 0 && now > sample.at_ms {
            let seconds = (now - sample.at_ms) as f64 / 1000.0;
            sample.rx_mbps = rx.saturating_sub(sample.rx_bytes) as f64 * 8.0 / seconds / 1_000_000.0;
            sample.tx_mbps = tx.saturating_sub(sample.tx_bytes) as f64 * 8.0 / seconds / 1_000_000.0;
        }
        sample.at_ms = now; sample.rx_bytes = rx; sample.tx_bytes = tx;
        json!({"receive_mbps":sample.rx_mbps,"transmit_mbps":sample.tx_mbps,
            "receive_bytes":rx,"transmit_bytes":tx,
            "scope":"sum of non-loopback Linux interfaces"})
    }

    fn loading_json(&self) -> Value {
        let loading = match self.loading.lock() { Ok(loading) => loading, Err(_) => return Value::Null };
        let Some(load) = loading.as_ref() else { return Value::Null };
        let elapsed_ms = unix_now_ms().saturating_sub(load.started_at_ms);
        let (rx, _) = network_totals();
        let observed = rx.saturating_sub(load.network_rx_start).min(load.expected_bytes);
        let transfer_fraction = if load.expected_bytes > 0 { observed as f64 / load.expected_bytes as f64 } else { 0.0 };
        // Model parsing, GPU allocation and graph setup continue after NFS
        // transfer. Never claim completion until the worker is actually ready.
        let progress = transfer_fraction.min(0.98);
        let bytes_per_second = if elapsed_ms > 0 { observed as f64 * 1000.0 / elapsed_ms as f64 } else { 0.0 };
        let eta_seconds = if bytes_per_second > 0.0 && observed < load.expected_bytes {
            Some((load.expected_bytes - observed) as f64 / bytes_per_second) } else { None };
        let stage = if transfer_fraction >= 1.0 { "finalizing_gpu" } else { "receiving_model" };
        json!({"elapsed_seconds":elapsed_ms as f64 / 1000.0,"expected_bytes":load.expected_bytes,
            "observed_network_bytes":observed,"estimated_fraction":progress,
            "transfer_fraction":transfer_fraction,"stage":stage,
            "estimated_eta_seconds":eta_seconds,"estimate_basis":"network receive bytes during load"})
    }
}

pub fn resolve_catalog_model(root: &std::path::Path, requested: &str) -> Option<(PathBuf, Option<PathBuf>)> {
    let wanted = std::fs::canonicalize(requested).ok()?;
    let canonical_root = std::fs::canonicalize(root).ok()?;
    if !wanted.starts_with(&canonical_root) { return None; }
    let metadata = std::fs::metadata(&wanted).ok()?;
    if !metadata.is_file() || !is_model_gguf(&wanted) { return None; }
    Some((wanted.clone(), projector_for(&wanted)))
}

fn catalog_value(root: &std::path::Path) -> Value {
    let canonical_root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let mut files = Vec::new();
    scan_gguf(&canonical_root, &mut files);
    let mut projectors: HashMap<PathBuf, PathBuf> = HashMap::new();
    for path in &files {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
        if name.starts_with("mmproj") && name.ends_with(".gguf") {
            if let Some(parent) = path.parent() {
                projectors.entry(parent.to_path_buf()).and_modify(|current| {
                    if path < current { *current = path.clone(); }
                }).or_insert_with(|| path.clone());
            }
        }
    }
    let mut models: Vec<Value> = files.into_iter().filter(|path| is_model_gguf(path)).map(|path| {
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let projector = path.parent().and_then(|parent| projectors.get(parent)).cloned();
        json!({"id":path.file_stem().and_then(|s| s.to_str()).unwrap_or("model"),
            "path":path,"size_bytes":size,"size_gib":size as f64 / 1_073_741_824.0,
            "image_projector":projector,"vision_capable":projector.is_some()})
    }).collect();
    models.sort_by(|a,b| a["path"].as_str().cmp(&b["path"].as_str()));
    json!({"root":canonical_root,"count":models.len(),"models":models})
}

fn catalog_response(cache: &CatalogCache, cached: bool) -> Value {
    let mut value = cache.value.clone();
    if let Value::Object(ref mut map) = value {
        map.insert("cached".into(), json!(cached));
        map.insert("scanned_at_ms".into(), json!(cache.scanned_at_ms));
        map.insert("scan_duration_ms".into(), json!(cache.duration_ms));
    }
    value
}

fn scan_gguf(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else { continue };
        if kind.is_dir() { scan_gguf(&path, out); }
        else if kind.is_file() && path.extension().and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false) {
            out.push(path);
        }
    }
}

fn is_model_gguf(path: &std::path::Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
    path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false)
        && !name.starts_with("mmproj") && !name.contains("projector")
}

fn projector_for(model: &std::path::Path) -> Option<PathBuf> {
    let parent = model.parent()?;
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(parent).ok()?.flatten()
        .map(|e| e.path()).filter(|path| {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
            path.is_file() && name.starts_with("mmproj") && name.ends_with(".gguf")
        }).collect();
    candidates.sort();
    candidates.into_iter().next().map(|p| std::fs::canonicalize(&p).unwrap_or(p))
}

fn network_totals() -> (u64, u64) {
    let text = std::fs::read_to_string("/proc/net/dev").unwrap_or_default();
    text.lines().filter_map(|line| {
        let (name, values) = line.split_once(':')?;
        if name.trim() == "lo" { return None; }
        let fields: Vec<&str> = values.split_whitespace().collect();
        Some((fields.first()?.parse::<u64>().ok()?, fields.get(8)?.parse::<u64>().ok()?))
    }).fold((0, 0), |(ar, at), (rx, tx)| (ar.saturating_add(rx), at.saturating_add(tx)))
}

fn unix_now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn per_second(tokens: usize, milliseconds: f64) -> f64 {
    if tokens > 0 && milliseconds > 0.0 { tokens as f64 * 1000.0 / milliseconds } else { 0.0 }
}

fn nullable_number(value: f64) -> Value {
    if value > 0.0 && value.is_finite() { json!(value) } else { Value::Null }
}

fn stats_json(stats: &RunStats) -> Value {
    json!({
        "prompt_tokens":stats.prompt_tokens,
        "completion_tokens":stats.completion_tokens,
        "queue_ms":stats.queue_ms,
        "prefill_ms":nullable_number(stats.prefill_ms),
        "ttft_ms":nullable_number(stats.ttft_ms),
        "generation_ms":nullable_number(stats.generation_ms),
        "thermal_wait_ms":stats.thermal_wait_ms,
        "total_ms":nullable_number(stats.total_ms),
        "prompt_tokens_per_second":nullable_number(stats.prompt_tokens_per_second),
        "generation_tokens_per_second":nullable_number(stats.generation_tokens_per_second)
    })
}

fn run_summary_json(run: &RunRecord) -> Value {
    json!({
        "id":run.id,"client_ip":run.client_ip,"method":run.method,"path":run.path,
        "request_type":run.request_type,"state":run.state,
        "queued_at_ms":run.queued_at_ms,"started_at_ms":run.started_at_ms,
        "completed_at_ms":run.completed_at_ms,"status_code":run.status_code,
        "error":run.error,"stats":stats_json(&run.stats)
    })
}

fn runs_value(history: &RunHistory) -> Value {
    let runs: Vec<Value> = history.entries.iter().rev().map(run_summary_json).collect();
    json!({"capacity":history.capacity,"retained":runs.len(),
        "active_count":history.entries.iter().filter(|run| run.state == "active").count(),
        "completed":history.entries.iter().filter(|run| run.state == "complete").count(),
        "errored":history.entries.iter().filter(|run| run.state == "errored").count(),
        "runs":runs})
}

fn run_detail_json(run: &RunRecord) -> Value {
    let mut value = run_summary_json(run);
    if let Value::Object(ref mut map) = value {
        map.insert("request".into(), run.request.clone());
        map.insert("response".into(), run.response.clone().unwrap_or(Value::Null));
    }
    value
}

fn performance_json(metrics: &Metrics, history: &RunHistory) -> Value {
    let prompt_tokens = metrics.prompt_tokens.load(Ordering::Relaxed);
    let completion_tokens = metrics.completion_tokens.load(Ordering::Relaxed);
    let prefill_us = metrics.prefill_us_total.load(Ordering::Relaxed);
    let generation_us = metrics.decode_us_total.load(Ordering::Relaxed);
    let prompt_tps = if prefill_us > 0 { prompt_tokens as f64 * 1_000_000.0 / prefill_us as f64 } else { 0.0 };
    let generation_tps = if generation_us > 0 { completion_tokens as f64 * 1_000_000.0 / generation_us as f64 } else { 0.0 };
    let last = history.entries.iter().rev().find(|run| run.state == "complete");
    let completed = history.entries.iter().filter(|run| run.state == "complete");
    let mut prompt_samples = 0usize;
    let mut generation_samples = 0usize;
    let mut prompt_sum = 0.0;
    let mut generation_sum = 0.0;
    let mut prompt_max: f64 = 0.0;
    let mut generation_max: f64 = 0.0;
    for run in completed {
        let prompt = run.stats.prompt_tokens_per_second;
        if prompt > 0.0 && prompt.is_finite() {
            prompt_samples += 1;
            prompt_sum += prompt;
            prompt_max = prompt_max.max(prompt);
        }
        let generation = run.stats.generation_tokens_per_second;
        if generation > 0.0 && generation.is_finite() {
            generation_samples += 1;
            generation_sum += generation;
            generation_max = generation_max.max(generation);
        }
    }
    let prompt_average = if prompt_samples > 0 { prompt_sum / prompt_samples as f64 } else { 0.0 };
    let generation_average = if generation_samples > 0 { generation_sum / generation_samples as f64 } else { 0.0 };
    json!({
        "aggregate":{"prompt_tokens_per_second":nullable_number(prompt_tps),
                     "generation_tokens_per_second":nullable_number(generation_tps)},
        "average":{"prompt_tokens_per_second":nullable_number(prompt_average),
                    "generation_tokens_per_second":nullable_number(generation_average)},
        "max":{"prompt_tokens_per_second":nullable_number(prompt_max),
                "generation_tokens_per_second":nullable_number(generation_max)},
        "sample_count":{"prompt":prompt_samples,"generation":generation_samples},
        "last":{"prompt_tokens_per_second":last.map(|r| nullable_number(r.stats.prompt_tokens_per_second)).unwrap_or(Value::Null),
                "generation_tokens_per_second":last.map(|r| nullable_number(r.stats.generation_tokens_per_second)).unwrap_or(Value::Null)}
    })
}

fn capture_json(body: &str) -> Value {
    if body.is_empty() { return Value::Null; }
    match serde_json::from_str::<Value>(body) {
        Ok(mut value) => {
            redact_images(&mut value);
            let compact = value.to_string();
            if compact.len() <= CAPTURE_TEXT_BYTES { value }
            else { json!({"truncated":true,"preview":truncate_utf8(&compact, CAPTURE_TEXT_BYTES)}) }
        }
        Err(_) => json!({"raw":truncate_utf8(body, CAPTURE_TEXT_BYTES),
                         "truncated":body.len() > CAPTURE_TEXT_BYTES}),
    }
}

fn redact_images(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                if key == "url" {
                    if let Value::String(url) = child {
                        if url.starts_with("data:image/") {
                            *url = format!("[image data URL redacted: {} characters]", url.len());
                            continue;
                        }
                    }
                }
                redact_images(child);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(redact_images),
        _ => {}
    }
}

fn truncate_utf8(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes { return text.to_string(); }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) { end -= 1; }
    format!("{}…", &text[..end])
}

fn error_message(body: &str) -> Option<String> {
    serde_json::from_str::<Value>(body).ok()
        .and_then(|v| v.get("error")?.get("message")?.as_str().map(str::to_string))
}

fn gpu_json(inventory: &GpuInventory, thermal: &ThermalGuard,
            context_capacity_tokens: usize, context_used_tokens: u64) -> Value {
    let inventory_value = inventory.json();
    match crate::hip::device_count() {
        Ok(n) => {
            let name = (n > 0).then(|| crate::hip::device_name(0).ok()).flatten();
            let memory=(n > 0).then(|| crate::hip::mem_info().ok()).flatten()
            .map(|(free,total)| {
                let used = total.saturating_sub(free);
                let tracked = crate::hip::memory_snapshot();
                let tracked_bytes = tracked.tracked_bytes();
                let residual = (used as u64).saturating_sub(tracked_bytes);
                let capacity = context_capacity_tokens as u64;
                let used_tokens = context_used_tokens.min(capacity);
                let reserved = tracked.context_bytes;
                json!({
                    "free_bytes":free,"used_bytes":used,"total_bytes":total,
                    "tracked_bytes":tracked_bytes,
                    "unattributed_bytes":residual,
                    "sections": {
                        "model_weights_bytes":tracked.model_weights_bytes,
                        "context_reserved_bytes":tracked.context_bytes,
                        "runtime_bytes":tracked.runtime_bytes,
                        "scratch_bytes":tracked.scratch_bytes,
                        "vision_bytes":tracked.vision_bytes,
                        "unattributed_bytes":tracked.unattributed_bytes.saturating_add(residual)
                    },
                    "context": {
                        "available": context_capacity_tokens > 0,
                        "capacity_tokens": context_capacity_tokens,
                        "used_tokens": used_tokens,
                        "utilization_fraction": if capacity > 0 { used_tokens as f64 / capacity as f64 } else { 0.0 },
                        "reserved_bytes": reserved,
                        "used_equivalent_bytes": if capacity > 0 { reserved.saturating_mul(used_tokens) / capacity } else { 0 }
                    }
                })
            });
            let architecture = std::env::var("REINSTINCT_OFFLOAD_ARCH")
                .unwrap_or_else(|_| crate::runtime::DEFAULT_ARCH.into());
            json!({"available":n>0,"device_count":n,"name":name,"architecture":architecture,"memory":memory,
                "thermal_guard":thermal.json(),"inventory":inventory_value})
        }
        Err(e) => json!({"available":false,"error":e,"thermal_guard":thermal.json(),"inventory":inventory_value}),
    }
}

pub fn openapi_json() -> String {
    json!({"openapi":"3.1.0","info":{"title":"ReInstinct API","version":env!("CARGO_PKG_VERSION"),
      "description":"OpenAI Chat Completions-compatible inference API. Unsupported API families are intentionally omitted."},
      "servers":[{"url":"/"}],"paths":{
        "/":{"get":{"summary":"Server dashboard","responses":{"200":{"description":"HTML status page","content":{"text/html":{}}}}}},
        "/healthz":{"get":{"summary":"Liveness","responses":{"200":{"description":"Alive","content":{"text/plain":{"schema":{"type":"string"}}}}}}},
        "/readyz":{"get":{"summary":"Readiness","responses":{"200":{"description":"Model ready","content":{"text/plain":{"schema":{"type":"string"}}}}},"503":{"description":"Loading or unavailable"}}},
        "/api/status":{"get":{"summary":"Live server status","responses":{"200":{"description":"Runtime, model, queue, GPU and counters","content":{"application/json":{"schema":{"$ref":"#/components/schemas/ServerStatus"}}}}}}},
        "/api/gpus":{"get":{"summary":"Inventory and telemetry for every installed GPU","responses":{"200":{"description":"PCI-keyed GPU inventory with virtualization evidence","content":{"application/json":{"schema":{"$ref":"#/components/schemas/GpuInventory"}}}}}}},
        "/api/gpus/{pci_address}/power-limit":{"put":{"summary":"Set an AMD runtime power cap through the privileged helper","parameters":[{"name":"pci_address","in":"path","required":true,"schema":{"type":"string","pattern":"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\\.[0-7]$"}},{"name":"X-ReInstinct-Action","in":"header","required":true,"schema":{"const":"set-power-limit"}}],"requestBody":{"required":true,"content":{"application/json":{"schema":{"type":"object","required":["watts"],"properties":{"watts":{"type":"number"}}}}}},"responses":{"200":{"description":"Helper readback"},"400":{"description":"Invalid identity, range, JSON, or action"},"409":{"description":"Read-only or unsupported device"},"503":{"description":"Helper unavailable"}}}},
        "/api/gpus/{pci_address}/tuning":{"put":{"summary":"Set bounded AMD runtime tuning through the privileged helper","parameters":[{"name":"pci_address","in":"path","required":true,"schema":{"type":"string"}},{"name":"X-ReInstinct-Action","in":"header","required":true,"schema":{"const":"set-tuning"}}],"requestBody":{"required":true,"content":{"application/json":{"schema":{"type":"object"}}}},"responses":{"200":{"description":"Helper readback"},"400":{"description":"Invalid request"},"409":{"description":"Read-only or unsupported device"},"503":{"description":"Helper unavailable"}}}},
        "/api/gpus/{pci_address}/reset":{"post":{"summary":"Reset an AMD GPU through the privileged helper","parameters":[{"name":"pci_address","in":"path","required":true,"schema":{"type":"string"}},{"name":"X-ReInstinct-Action","in":"header","required":true,"schema":{"const":"reset-gpu"}}],"requestBody":{"required":true,"content":{"application/json":{"schema":{"type":"object"}}}},"responses":{"200":{"description":"Reset result"},"400":{"description":"Invalid request"},"409":{"description":"Unsupported device"},"503":{"description":"Helper unavailable"}}}},
        "/api/config":{"get":{"summary":"Get active and saved server configuration"},"put":{"summary":"Validate and persist server configuration","parameters":[{"name":"X-ReInstinct-Action","in":"header","required":true,"schema":{"const":"update-config"}}],"responses":{"200":{"description":"Configuration saved"},"400":{"description":"Invalid configuration"}}}},
        "/api/config/reload":{"get":{"summary":"Get configuration reload state"},"post":{"summary":"Apply the saved engine configuration","parameters":[{"name":"X-ReInstinct-Action","in":"header","required":true,"schema":{"const":"reload-config"}}],"responses":{"202":{"description":"Reload queued"},"409":{"description":"Busy or service restart required"}}}},
        "/api/runs":{"get":{"summary":"List bounded in-memory inference run history","responses":{"200":{"description":"Newest retained runs first","content":{"application/json":{"schema":{"$ref":"#/components/schemas/RunList"}}}}}}},
        "/api/runs/{id}":{"get":{"summary":"Get one retained run with captured request and response","parameters":[{"name":"id","in":"path","required":true,"schema":{"type":"integer"}}],"responses":{"200":{"description":"Run detail","content":{"application/json":{"schema":{"$ref":"#/components/schemas/RunDetail"}}}},"404":{"description":"Run was evicted or does not exist","content":{"application/json":{"schema":{"$ref":"#/components/schemas/Error"}}}}}}},
        "/api/logs":{"get":{"summary":"List bounded in-memory engine logs","responses":{"200":{"description":"Oldest-to-newest log lines"}}}},
        "/api/models":{"get":{"summary":"Scan the configured model catalog","responses":{"200":{"description":"Canonical model paths, sizes, and associated image projectors"}}}},
        "/api/models/switch":{"get":{"summary":"Get current model-switch state"},"post":{"summary":"Queue replacement of the resident big model","parameters":[{"name":"X-ReInstinct-Action","in":"header","required":true,"schema":{"const":"switch-model"}}],"responses":{"202":{"description":"Switch queued"},"400":{"description":"Missing action header or invalid model"},"409":{"description":"Busy or unavailable"}}}},
        "/v1/models":{"get":{"summary":"List models","responses":{"200":{"description":"Models available on this port","content":{"application/json":{}}}}}},
        "/v1/models/{model}":{"get":{"summary":"Retrieve a model","parameters":[{"name":"model","in":"path","required":true,"schema":{"type":"string"}}],"responses":{"200":{"description":"Selected model","content":{"application/json":{}}},"404":{"description":"Model not found","content":{"application/json":{"schema":{"$ref":"#/components/schemas/Error"}}}}}}},
        "/v1/completions":{"post":{"summary":"Create a text completion","requestBody":{"required":true,"content":{"application/json":{"schema":{"$ref":"#/components/schemas/CompletionRequest"}}}},"responses":response_set()}},
        "/v1/chat/completions":{"post":{"summary":"Create a chat completion","description":"Text plus one JPEG/PNG data-URL image in user content.","requestBody":{"required":true,"content":{"application/json":{"schema":{"$ref":"#/components/schemas/ChatCompletionRequest"}}}},"responses":response_set()}}
      },"components":{"schemas":{
        "CompletionRequest":{"type":"object","required":["prompt"],"properties":common_request(json!({"prompt":{"type":"string"}}))},
        "ChatCompletionRequest":{"type":"object","required":["messages"],"properties":common_request(json!({"messages":{"type":"array","minItems":1,"items":{"$ref":"#/components/schemas/ChatMessage"}}}))},
        "ChatMessage":{"type":"object","required":["role","content"],"properties":{"role":{"type":"string","enum":["system","user","assistant"]},"content":{"oneOf":[{"type":"string"},{"type":"array","items":{"oneOf":[{"$ref":"#/components/schemas/TextPart"},{"$ref":"#/components/schemas/ImagePart"}]}}]}}},
        "TextPart":{"type":"object","required":["type","text"],"properties":{"type":{"const":"text"},"text":{"type":"string"}}},
        "ImagePart":{"type":"object","required":["type","image_url"],"properties":{"type":{"const":"image_url"},"image_url":{"type":"object","required":["url"],"properties":{"url":{"type":"string","pattern":"^data:image/(jpeg|png);base64,"}}}}},
        "ServerStatus":{"type":"object","additionalProperties":true},
        "GpuInventory":{"type":"object","required":["virtualization","gpus"],"properties":{"virtualization":{"type":"object"},"gpus":{"type":"object","additionalProperties":{"type":"object"}}}},
        "RunList":{"type":"object","additionalProperties":true},
        "RunDetail":{"type":"object","additionalProperties":true},
        "Error":{"type":"object","required":["error"],"properties":{"error":{"type":"object","required":["message","type","param","code"],"properties":{"message":{"type":"string"},"type":{"type":"string"},"param":{"oneOf":[{"type":"string"},{"type":"null"}]},"code":{"oneOf":[{"type":"string"},{"type":"null"}]}}}}}
      }}}).to_string()
}

fn common_request(mut v: Value) -> Value {
    let Value::Object(ref mut m) = v else {
        return v;
    };
    m.extend(serde_json::Map::from_iter([
        ("model".into(), json!({"type":"string"})),
        (
            "max_tokens".into(),
            json!({"type":"integer","minimum":1,"maximum":4096,"default":256}),
        ),
        (
            "max_completion_tokens".into(),
            json!({"type":"integer","minimum":1,"maximum":4096,"description":"Alias for max_tokens; do not send both."}),
        ),
        (
            "stop".into(),
            json!({"oneOf":[{"type":"string","minLength":1},{"type":"array","maxItems":4,"items":{"type":"string","minLength":1}}]}),
        ),
        ("n".into(), json!({"type":"integer","const":1,"default":1})),
        ("user".into(), json!({"type":"string","description":"Accepted for attribution and ignored by local inference."})),
        ("metadata".into(), json!({"type":"object","description":"Accepted for attribution and ignored by local inference."})),
        ("temperature".into(), json!({"type":"number","minimum":0})),
        ("top_k".into(), json!({"type":"integer","minimum":0})),
        (
            "top_p".into(),
            json!({"type":"number","minimum":0,"maximum":1}),
        ),
        (
            "min_p".into(),
            json!({"type":"number","minimum":0,"maximum":1}),
        ),
        ("frequency_penalty".into(), json!({"type":"number"})),
        ("presence_penalty".into(), json!({"type":"number"})),
        ("seed".into(), json!({"type":"integer"})),
        ("stream".into(), json!({"type":"boolean","default":false})),
        (
            "stream_options".into(),
            json!({"type":"object","properties":{"include_usage":{"type":"boolean"}}}),
        ),
        (
            "logprobs".into(),
            json!({"oneOf":[{"type":"boolean"},{"type":"integer","minimum":0,"maximum":20}]}),
        ),
        (
            "top_logprobs".into(),
            json!({"type":"integer","minimum":0,"maximum":20}),
        ),
        ("use_speculative".into(), json!({"type":"boolean"})),
        (
            "speculative_k".into(),
            json!({"type":"integer","minimum":1,"maximum":4}),
        ),
        (
            "request_timeout_seconds".into(),
            json!({"type":"number","minimum":0.1,"maximum":600}),
        ),
    ]));
    v
}
fn response_set() -> Value {
    json!({"200":{"description":"Completion or SSE stream","content":{"application/json":{},"text/event-stream":{}}},"400":{"description":"Invalid request","content":{"application/json":{"schema":{"$ref":"#/components/schemas/Error"}}}},"503":{"description":"Unavailable","content":{"application/json":{"schema":{"$ref":"#/components/schemas/Error"}}}}})
}

pub const DOCS_HTML: &str = r#"<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>ReInstinct API docs</title><link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"></head><body><div id="swagger-ui"><a href="/openapi.json">OpenAPI document</a></div><script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script><script>if(window.SwaggerUIBundle)SwaggerUIBundle({url:'/openapi.json',dom_id:'#swagger-ui',deepLinking:true});</script></body></html>"#;
pub const INDEX_HTML: &str = r#"<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>ReInstinct status</title><style>:root{color-scheme:dark;font:15px system-ui;background:#0b1017;color:#dbe7f3}body{max-width:1050px;margin:auto;padding:38px 22px}header{display:flex;justify-content:space-between;align-items:center}h1{margin:0}.badge{padding:6px 11px;border-radius:99px;background:#243243}.ready{background:#123d2c;color:#83f0b4}.error{background:#53242a;color:#ffabb3}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px;margin-top:24px}.card{background:#121a24;border:1px solid #263343;border-radius:12px;padding:17px}.card h2{font-size:13px;text-transform:uppercase;color:#8fa5bb;margin:0 0 14px}dl{display:grid;grid-template-columns:1fr 1.5fr;gap:8px;margin:0}dt{color:#8fa5bb}dd{margin:0;overflow-wrap:anywhere}nav{display:flex;gap:16px;margin-top:28px}a{color:#68d8ff}small{color:#7890a8}</style></head><body><header><div><h1>ReInstinct</h1><small id="version">Inference server</small></div><span id="phase" class="badge">connecting</span></header><div id="cards" class="grid"></div><nav><a href="/docs">API docs</a><a href="/openapi.json">OpenAPI 3.1</a><a href="/metrics">Metrics</a><a href="/v1/models">Models</a></nav><script>const e=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),z=n=>n==null?'—':(n/1073741824).toFixed(2)+' GiB',c=(t,r)=>`<section class="card"><h2>${e(t)}</h2><dl>${r.map(x=>`<dt>${e(x[0])}</dt><dd>${e(x[1])}</dd>`).join('')}</dl></section>`;async function l(){try{let s=await(await fetch('/api/status',{cache:'no-store'})).json(),g=s.gpu||{},m=g.memory||{},v=s.model.vision||{};version.textContent=`v${s.service.version} · uptime ${s.uptime_seconds}s`;phase.textContent=s.status;phase.className='badge '+s.status;cards.innerHTML=c('Model',[['ID',s.model.id],['State',s.status_detail||s.status],['Context',s.model.max_context_tokens+' tokens'],['Drafter',s.model.drafter],['Vision',v.enabled?`${v.device}, ${v.threads} threads`:'disabled']])+c('Worker',[['Target',s.endpoint.target+':'+s.endpoint.port],['Active',s.worker.active_request_id],['Queued',s.worker.queued_requests],['Completed',s.metrics.requests_ok],['Errors',s.metrics.requests_4xx+s.metrics.requests_5xx]])+c('GPU',[['Available',g.available],['Device',g.name],['Architecture',g.architecture],['VRAM used',z(m.used_bytes)],['VRAM free',z(m.free_bytes)],['VRAM total',z(m.total_bytes)]])+c('Tokens',[['Prompt',s.metrics.prompt_tokens],['Completion',s.metrics.completion_tokens],['Total',s.metrics.prompt_tokens+s.metrics.completion_tokens]])}catch(_){phase.textContent='unavailable';phase.className='badge error'}}l();setInterval(l,3000)</script></body></html>"#;

pub const INDEX_HTML_V2: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ReInstinct flight recorder</title>
<style>
:root{color-scheme:dark;--ink:#e8efe9;--muted:#89958f;--panel:#111816;--line:#2a3731;--accent:#b7ff5a;--cyan:#62d9d1;--warn:#ffc857;--bad:#ff6b6b;--bg:#080d0b;font-family:"Aptos Mono","Cascadia Code","IBM Plex Mono",monospace;background:var(--bg);color:var(--ink)}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:linear-gradient(90deg,rgba(183,255,90,.035) 1px,transparent 1px),linear-gradient(rgba(183,255,90,.025) 1px,transparent 1px),radial-gradient(circle at 82% 4%,#17352b 0,transparent 32%);background-size:32px 32px,32px 32px,auto}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.18;background:repeating-linear-gradient(0deg,transparent 0 3px,#000 4px)}
main{position:relative;width:min(1480px,calc(100% - 36px));margin:auto;padding:30px 0 48px}header{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:1px solid var(--line);padding:8px 0 20px;margin-bottom:18px}.eyebrow{color:var(--accent);letter-spacing:.18em;font-size:11px;text-transform:uppercase}.title{font-family:Bahnschrift,"DIN Condensed",sans-serif;font-size:clamp(34px,6vw,70px);line-height:.9;letter-spacing:-.035em;margin:6px 0 0;text-transform:uppercase}.sub{color:var(--muted);font-size:12px;margin-top:10px}.badge{border:1px solid var(--line);padding:9px 13px;text-transform:uppercase;font-size:11px;letter-spacing:.13em;background:#131c18}.badge.ready{border-color:#5c8d40;color:var(--accent);box-shadow:0 0 22px rgba(183,255,90,.12)}.badge.error,.badge.unavailable{border-color:#733;color:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.card{background:linear-gradient(145deg,rgba(20,29,25,.96),rgba(11,17,14,.96));border:1px solid var(--line);padding:15px;min-height:122px;position:relative;overflow:hidden}.card:after{content:attr(data-index);position:absolute;right:9px;top:7px;color:#34423b;font-size:10px}.card h2{font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:0 0 14px}.metric{font-family:Bahnschrift,"DIN Condensed",sans-serif;font-size:30px;letter-spacing:-.02em}.metric small{font:11px "Aptos Mono",monospace;color:var(--muted)}dl{display:grid;grid-template-columns:1fr auto;gap:7px;margin:0;font-size:11px}dt{color:var(--muted)}dd{margin:0;text-align:right;overflow-wrap:anywhere}.accent{color:var(--accent)}.cyan{color:var(--cyan)}.bad{color:var(--bad)}.memory-panel{margin-top:10px;background:linear-gradient(145deg,rgba(20,29,25,.96),rgba(11,17,14,.96));border:1px solid var(--line);padding:15px}.memory-head{display:flex;justify-content:space-between;gap:12px;align-items:baseline}.memory-title{font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}.memory-total{font:22px Bahnschrift,"DIN Condensed",sans-serif}.memory-bar,.context-bar{display:flex;height:18px;margin:12px 0 9px;background:#070b09;border:1px solid var(--line);overflow:hidden}.context-bar{height:9px;margin-top:8px}.memory-segment{height:100%;min-width:1px}.memory-legend{display:flex;flex-wrap:wrap;gap:8px 16px;font-size:10px;color:var(--muted)}.memory-legend span:before{content:"";display:inline-block;width:8px;height:8px;margin-right:5px;background:var(--swatch);border-radius:2px}.memory-detail{display:grid;grid-template-columns:1fr auto;gap:5px;margin-top:10px;font-size:11px}.memory-detail b{text-align:right;color:var(--ink)}
.runs{margin-top:20px;border:1px solid var(--line);background:rgba(9,14,12,.94)}.runs-head{display:flex;align-items:center;justify-content:space-between;padding:15px 16px;border-bottom:1px solid var(--line)}.runs-head h2{font:20px Bahnschrift,sans-serif;text-transform:uppercase;letter-spacing:.04em;margin:0}.runs-head p{font-size:10px;color:var(--muted);margin:3px 0 0}.live{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:10px;text-transform:uppercase}.live:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--accent);animation:pulse 1.8s infinite}@keyframes pulse{50%{opacity:.35}}
.table-wrap{overflow:auto;max-height:520px}table{width:100%;border-collapse:collapse;font-size:11px;white-space:nowrap}th{position:sticky;top:0;z-index:1;background:#121b17;text-align:left;color:var(--muted);font-weight:500;text-transform:uppercase;letter-spacing:.09em;padding:10px 12px;border-bottom:1px solid var(--line)}td{padding:11px 12px;border-bottom:1px solid #1c2822}tbody tr{cursor:pointer;transition:background .15s,color .15s}tbody tr:hover{background:#19251f;color:#fff}.state{display:inline-block;min-width:72px;padding:4px 7px;border:1px solid var(--line);text-align:center;text-transform:uppercase;font-size:9px;letter-spacing:.08em}.state.complete{color:var(--accent);border-color:#42622f}.state.active{color:var(--cyan);border-color:#2b6b67}.state.queued{color:var(--warn);border-color:#765f30}.state.errored{color:var(--bad);border-color:#713535}.empty{padding:34px;text-align:center;color:var(--muted)}
nav{display:flex;gap:18px;flex-wrap:wrap;margin-top:15px;font-size:11px}a{color:var(--cyan);text-decoration:none;border-bottom:1px solid transparent}a:hover{border-color:currentColor}.privacy{margin-left:auto;color:var(--muted)}
dialog{width:min(900px,calc(100% - 28px));max-height:88vh;border:1px solid #53675d;background:#0d1411;color:var(--ink);padding:0;box-shadow:0 32px 90px #000}dialog::backdrop{background:rgba(0,0,0,.78);backdrop-filter:blur(3px)}.detail-head{position:sticky;top:0;display:flex;justify-content:space-between;align-items:center;background:#121b17;border-bottom:1px solid var(--line);padding:15px 18px;z-index:2}.detail-head h2{margin:0;font:22px Bahnschrift,sans-serif;text-transform:uppercase}.close{background:transparent;border:1px solid var(--line);color:var(--ink);padding:7px 10px;cursor:pointer}.detail-body{padding:18px}.detail-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px}.detail-grid div{border:1px solid var(--line);padding:10px}.detail-grid span{display:block;color:var(--muted);font-size:9px;text-transform:uppercase;margin-bottom:5px}.payload{margin-top:13px}.payload h3{font-size:10px;color:var(--muted);letter-spacing:.14em;text-transform:uppercase}pre{margin:0;max-height:310px;overflow:auto;background:#080c0a;border:1px solid var(--line);padding:14px;font:11px/1.55 "Cascadia Code",monospace;white-space:pre-wrap;overflow-wrap:anywhere}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}.detail-grid{grid-template-columns:repeat(2,1fr)}header{align-items:flex-start}.badge{margin-top:4px}}@media(max-width:520px){main{width:min(100% - 20px,1480px);padding-top:18px}.grid{grid-template-columns:1fr}.title{font-size:42px}.privacy{margin-left:0}.runs-head{align-items:flex-start;gap:12px}}
</style></head>
<body><main><header><div><div class="eyebrow">gfx906 / inference telemetry</div><h1 class="title">ReInstinct</h1><div id="version" class="sub">Connecting to flight recorder…</div></div><span id="phase" class="badge">connecting</span></header>
<section id="cards" class="grid"></section>
<section id="memoryPanel" class="memory-panel" aria-live="polite"></section>
<section class="runs"><div class="runs-head"><div><h2>Recent runs</h2><p>Newest first · click any row for captured request, response, and stage timings</p></div><div class="live">live refresh</div></div><div class="table-wrap"><table><thead><tr><th>ID / time</th><th>Client</th><th>Type</th><th>State</th><th>Tokens p / g</th><th>Prompt tok/s</th><th>Gen tok/s</th><th>TTFT</th><th>Total</th></tr></thead><tbody id="runRows"><tr><td colspan="9" class="empty">No inference runs retained yet.</td></tr></tbody></table></div></section>
<nav><a href="/docs">API docs</a><a href="/openapi.json">OpenAPI 3.1</a><a href="/metrics">Prometheus</a><a href="/v1/models">Models</a><a href="/api/runs">Run JSON</a><span class="privacy">128-run memory buffer · image bytes redacted · clears on restart</span></nav>
</main>
<dialog id="detail"><div class="detail-head"><h2 id="detailTitle">Run detail</h2><button class="close" type="button" onclick="detail.close()">Close</button></div><div class="detail-body"><div id="detailGrid" class="detail-grid"></div><div class="payload"><h3>Request</h3><pre id="requestPayload"></pre></div><div class="payload"><h3>Response</h3><pre id="responsePayload"></pre></div></div></dialog>
<script>
const phase=document.getElementById('phase'),version=document.getElementById('version'),cards=document.getElementById('cards'),runRows=document.getElementById('runRows'),detail=document.getElementById('detail'),detailTitle=document.getElementById('detailTitle'),detailGrid=document.getElementById('detailGrid'),requestPayload=document.getElementById('requestPayload'),responsePayload=document.getElementById('responsePayload');
const e=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const rate=n=>n==null?'—':Number(n).toFixed(1);const ms=n=>n==null?'—':Number(n)<1000?Number(n).toFixed(0)+' ms':(Number(n)/1000).toFixed(2)+' s';const gib=n=>n==null?'—':(n/1073741824).toFixed(1)+' GiB';
const clock=n=>n?new Date(n).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'}):'—';
const card=(i,t,hero,rows)=>`<article class="card" data-index="0${i}"><h2>${e(t)}</h2><div class="metric">${hero}</div><dl>${rows.map(r=>`<dt>${e(r[0])}</dt><dd>${r[1]}</dd>`).join('')}</dl></article>`;
async function refresh(){try{const [s,h]=await Promise.all([fetch('/api/status',{cache:'no-store'}).then(r=>r.json()),fetch('/api/runs',{cache:'no-store'}).then(r=>r.json())]);renderStatus(s,h);renderRuns(h.runs||[])}catch(err){phase.textContent='unavailable';phase.className='badge unavailable'}}
function renderStatus(s,h){const g=s.gpu||{},m=g.memory||{},v=s.model.vision||{},p=s.performance||{},a=p.average||p.aggregate||{},aggregate=p.aggregate||{},max=p.max||{},l=p.last||{},runs=h.runs||[],runErrors=runs.filter(r=>r.state==='errored').length;version.textContent=`v${s.service.version} · ${s.model.id} · uptime ${s.uptime_seconds}s`;phase.textContent=s.status;phase.className='badge '+s.status;cards.innerHTML=card(1,'Prompt throughput',`<span class="accent">${rate(a.prompt_tokens_per_second)}</span> <small>tok/s average</small>`,[['Max',`${rate(max.prompt_tokens_per_second)} tok/s`],['Aggregate',`${rate(aggregate.prompt_tokens_per_second)} tok/s`],['Last run',`${rate(l.prompt_tokens_per_second)} tok/s`],['Prompt tokens',e(s.metrics.prompt_tokens)]])+card(2,'Generation throughput',`<span class="cyan">${rate(a.generation_tokens_per_second)}</span> <small>tok/s average</small>`,[['Max',`${rate(max.generation_tokens_per_second)} tok/s`],['Aggregate',`${rate(aggregate.generation_tokens_per_second)} tok/s`],['Last run',`${rate(l.generation_tokens_per_second)} tok/s`],['Generated tokens',e(s.metrics.completion_tokens)]])+card(3,'Worker',s.worker.active_request_id?`<span class="cyan">ACTIVE #${e(s.worker.active_request_id)}</span>`:`<span class="accent">IDLE</span>`,[['Queued',e(s.worker.queued_requests)],['Completed',e(s.metrics.requests_ok)],['Retained errors',`<span class="${runErrors?'bad':''}">${runErrors}</span>`],['HTTP 4xx / 5xx',`${e(s.metrics.requests_4xx)} / ${e(s.metrics.requests_5xx)}`]])+card(4,'GPU',`<span>${gib(m.used_bytes)}</span> <small>used</small>`,[['Device',e(g.name)],['Architecture',e(g.architecture)],['Free',gib(m.free_bytes)],['Vision',v.enabled?`${e(v.device)} / ${e(v.threads)} threads`:'disabled']]);}
function renderRuns(runs){runRows.innerHTML=runs.length?runs.map(r=>{const x=r.stats||{};return `<tr data-id="${r.id}"><td><b>#${r.id}</b><br><span style="color:var(--muted)">${clock(r.queued_at_ms)}</span></td><td>${e(r.client_ip)}</td><td>${e(r.request_type)}${r.path.includes('chat')?'<br><span style="color:var(--muted)">chat</span>':''}</td><td><span class="state ${e(r.state)}">${e(r.state)}</span>${r.status_code?`<br><span style="color:var(--muted)">HTTP ${r.status_code}</span>`:''}</td><td>${e(x.prompt_tokens)} / ${e(x.completion_tokens)}</td><td class="accent">${rate(x.prompt_tokens_per_second)}</td><td class="cyan">${rate(x.generation_tokens_per_second)}</td><td>${ms(x.ttft_ms)}</td><td>${ms(x.total_ms)}</td></tr>`}).join(''):'<tr><td colspan="9" class="empty">No inference runs retained yet.</td></tr>';}
runRows.addEventListener('click',async ev=>{const row=ev.target.closest('tr[data-id]');if(!row)return;const r=await fetch('/api/runs/'+row.dataset.id,{cache:'no-store'}).then(x=>x.json()),x=r.stats||{};detailTitle.textContent=`Run #${r.id} · ${r.state}`;detailGrid.innerHTML=[['Client',r.client_ip],['Route',r.method+' '+r.path],['State',r.state+(r.status_code?' / HTTP '+r.status_code:'')],['Queued',ms(x.queue_ms)],['Prompt',x.prompt_tokens+' tok / '+rate(x.prompt_tokens_per_second)+' tok/s'],['Generation',x.completion_tokens+' tok / '+rate(x.generation_tokens_per_second)+' tok/s'],['TTFT',ms(x.ttft_ms)],['Total',ms(x.total_ms)]].map(v=>`<div><span>${e(v[0])}</span>${e(v[1])}</div>`).join('');requestPayload.textContent=JSON.stringify(r.request,null,2);responsePayload.textContent=JSON.stringify(r.response,null,2);detail.showModal()});
</script></body></html>"#;

pub const INDEX_HTML_V3: &str = r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ReInstinct operations</title><style>
:root{color-scheme:dark;--bg:#080d0b;--panel:#101814;--line:#2a3931;--ink:#e8efe9;--muted:#8b9991;--lime:#b7ff5a;--cyan:#62d9d1;--red:#ff6b6b;--amber:#ffc857;font:13px "Cascadia Code",monospace;background:var(--bg);color:var(--ink)}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 0,#17352b,transparent 32%),linear-gradient(90deg,#b7ff5a08 1px,transparent 1px),linear-gradient(#b7ff5a06 1px,transparent 1px);background-size:auto,32px 32px,32px 32px}main{width:min(1500px,calc(100% - 32px));margin:auto;padding:28px 0 44px}header,.row,.section-head{display:flex;align-items:center;justify-content:space-between;gap:14px}header{align-items:flex-end;border-bottom:1px solid var(--line);padding-bottom:18px}.eyebrow,summary,.label{color:var(--muted);text-transform:uppercase;letter-spacing:.13em;font-size:10px}h1{font:700 clamp(38px,6vw,68px)/.9 Bahnschrift,sans-serif;text-transform:uppercase;margin:6px 0}.sub{color:var(--muted);font-size:11px}.badge,.btn,input{border:1px solid var(--line);background:#131d18;color:var(--ink);padding:8px 11px}.badge{text-transform:uppercase;letter-spacing:.12em;font-size:10px}.ready{color:var(--lime);border-color:#567a3c}.loading{color:var(--amber)}.error,.bad{color:var(--red)}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px;margin:18px 0}.card,details{background:linear-gradient(145deg,#141d19f5,#0b110ef5);border:1px solid var(--line)}.card{min-height:118px;padding:14px}.card h2{margin:0 0 13px;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em}.hero{font:28px Bahnschrift,sans-serif;margin-bottom:10px}.lime{color:var(--lime)}.cyan{color:var(--cyan)}dl{display:grid;grid-template-columns:1fr auto;gap:6px;margin:0;font-size:10px}dt{color:var(--muted)}dd{margin:0;text-align:right;overflow-wrap:anywhere}details{margin-top:12px}summary{cursor:pointer;padding:14px 16px;list-style:none;display:flex;justify-content:space-between}summary::-webkit-details-marker{display:none}summary:after{content:"+";color:var(--lime)}details[open]>summary{border-bottom:1px solid var(--line)}details[open]>summary:after{content:"−"}.summary-meta{margin-left:auto;margin-right:14px;color:var(--ink);letter-spacing:0;text-transform:none}.tools{padding:10px 14px;border-bottom:1px solid var(--line);display:flex;gap:9px;align-items:center;flex-wrap:wrap}.btn{cursor:pointer;font:11px inherit}.btn:hover{border-color:var(--cyan);color:var(--cyan)}input{min-width:260px;font:11px inherit}.progress{height:10px;border:1px solid var(--line);background:#070b09;overflow:hidden;margin-top:9px}.progress i{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--lime));transition:width .4s}.table-wrap{overflow:auto;max-height:500px}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:10px}th{position:sticky;top:0;background:#121b17;color:var(--muted);text-align:left;text-transform:uppercase;letter-spacing:.08em;padding:9px 11px}td{padding:10px 11px;border-top:1px solid #1c2822}tbody tr{cursor:pointer}tbody tr:hover{background:#19251f}.state{padding:3px 6px;border:1px solid var(--line);text-transform:uppercase}.state.complete{color:var(--lime)}.state.active{color:var(--cyan)}.state.errored{color:var(--red)}#logLines{height:360px;overflow:auto;margin:0;padding:14px;background:#060a08;white-space:pre-wrap;font:11px/1.55 "Cascadia Code",monospace;color:#b9c8bf}.model-list{padding:12px;display:grid;gap:8px}.model{border:1px solid var(--line);padding:11px;display:grid;grid-template-columns:1fr auto;gap:5px 15px}.model .path{color:var(--muted);font-size:10px;overflow-wrap:anywhere}.model.active-model{border-color:#567a3c}.model button{grid-row:1/4;grid-column:2;align-self:center}nav{display:flex;gap:16px;flex-wrap:wrap;margin-top:14px;font-size:10px}a{color:var(--cyan);text-decoration:none}.privacy{margin-left:auto;color:var(--muted)}dialog{width:min(900px,calc(100% - 24px));max-height:88vh;border:1px solid #53675d;background:#0d1411;color:var(--ink);padding:0}dialog::backdrop{background:#000c}.dialog-head{position:sticky;top:0;background:#121b17;border-bottom:1px solid var(--line);padding:13px 16px;display:flex;justify-content:space-between}.dialog-body{padding:16px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.stats div{border:1px solid var(--line);padding:9px}.stats span{display:block;color:var(--muted);font-size:9px}pre.payload{max-height:280px;overflow:auto;background:#060a08;border:1px solid var(--line);padding:12px;white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:1050px){.grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:580px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}header{align-items:flex-start}.privacy{margin-left:0}}
.config-tools{padding:14px 16px;background:#0b120f;display:grid;grid-template-columns:minmax(260px,1fr) auto;gap:12px 18px}.config-path{min-width:0;display:flex;align-items:center;gap:9px;overflow:hidden}.config-path:before{content:"JSON";flex:none;color:var(--cyan);border:1px solid #31534a;padding:3px 6px;font-size:9px;letter-spacing:.12em}.config-path span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.config-actions{display:flex;gap:8px}.btn.primary{border-color:#628a43;background:#172319;color:var(--lime)}.btn:disabled{cursor:not-allowed;opacity:.42}.config-key{grid-column:1/-1;display:flex;gap:17px;padding-top:11px;border-top:1px solid #1d2923;color:var(--muted);font-size:10px}.config-key b{color:var(--cyan);font-size:14px}.config-notice{display:none;margin:12px 16px 0;padding:10px 12px;border-left:2px solid var(--cyan);background:#10201b;color:#b8cbc1;font-size:11px}.config-notice.show{display:block}.config-notice.warn{border-color:var(--amber);background:#211c10;color:#f0d794}.config-notice.error{border-color:var(--red);background:#241315;color:#ffb0b0}.config-form{padding:16px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:#26352e}.config-field{min-width:0;background:#0e1612;padding:13px 14px;display:grid;grid-template-columns:minmax(130px,.65fr) minmax(160px,1.35fr) 18px;align-items:center;gap:12px}.config-field:hover{background:#111c17}.config-field label{color:#b7c4bd;font-size:11px}.config-field input:not([type=checkbox]){width:100%;min-width:0;background:#0a100d;border-color:#34463d;padding:9px 10px}.config-field input:focus{outline:1px solid var(--cyan);outline-offset:1px}.config-field .apply-mark{color:var(--cyan);font-size:15px;text-align:center}.config-field .restart-mark{color:var(--amber);font-size:12px;text-align:center}.config-field.locked{opacity:.58}.config-field.checkbox-field{grid-template-columns:1fr 18px}.config-field.checkbox-field label{display:flex;align-items:center;gap:9px}.config-field.checkbox-field input{min-width:0;accent-color:var(--lime)}@media(max-width:1050px){.config-form{grid-template-columns:1fr}}@media(max-width:680px){.config-tools{grid-template-columns:1fr}.config-actions{grid-row:2}.config-key{grid-row:3;flex-direction:column;gap:8px}.config-field{grid-template-columns:1fr 18px}.config-field label{grid-column:1/-1}.config-field input:not([type=checkbox]){grid-column:1}}
</style></head><body><main><header><div><div class="eyebrow">gfx906 / operations console</div><h1>ReInstinct</h1><div id="version" class="sub">Connecting…</div></div><span id="phase" class="badge">connecting</span></header><section id="cards" class="grid"></section><section id="loadPanel" class="card" style="display:none"><div class="row"><div><div class="label">Model loading estimate</div><strong id="loadText"></strong></div><strong id="loadPct" class="cyan"></strong></div><div class="progress"><i id="loadBar"></i></div><div id="loadHint" class="sub" style="margin-top:8px"></div></section>
<section id="errorPanel" class="card" style="display:none;border-color:var(--red)"><div class="label">Engine startup failure</div><pre id="errorText" style="margin-top:10px;color:var(--red);max-height:260px;white-space:pre-wrap;overflow:auto"></pre><div class="sub" style="margin-top:10px">The HTTP dashboard is still available. Fix the startup configuration or model, then restart the service.</div></section>
<details id="gpusPanel"><summary>GPUs <span id="gpuSummary" class="summary-meta">loading</span></summary><div class="tools"><span class="sub">All PCI devices are read-only unless AMD driver capabilities and the privileged helper permit a bounded runtime operation.</span></div><div class="table-wrap"><table><thead><tr><th>PCI / identity</th><th>Driver / virtualization</th><th>PCIe</th><th>Power</th><th>Temperature</th><th>VRAM</th><th>Controls</th></tr></thead><tbody id="gpuRows"></tbody></table></div></details>
<details id="thermalPanel"><summary>Thermal guard <span id="thermalSummary" class="summary-meta">loading</span></summary><div class="tools"><span id="thermalDetail" class="sub">The inference GPU is monitored at one-second intervals.</span></div></details>
<details id="amdPanel"><summary>AMD GPU configuration <span id="amdSummary" class="summary-meta">select a writable GPU</span></summary><div class="tools"><select id="amdPci" class="btn"><option value="">No writable AMD GPU</option></select><input id="amdWatts" type="number" min="1" step="0.1" placeholder="Power limit (W)"><select id="amdLevel" class="btn"><option value="auto">auto</option><option value="low">low</option><option value="high">high</option><option value="manual">manual</option></select><button id="amdPower" class="btn primary">Apply power</button><button id="amdTune" class="btn">Apply tuning</button><button id="amdReset" class="btn">Reset</button><span class="sub">Runtime-only, driver-bounded, and subject to the thermal interlock. Confirm each mutation.</span></div></details>
<details id="modelsPanel"><summary>Model catalog <span id="modelSummary" class="summary-meta">not scanned</span></summary><div class="tools"><button id="scanModels" class="btn">Refresh models</button><input id="modelFilter" type="search" placeholder="Filter models or paths…"><span class="sub">NFS scan is cached until refreshed. Switching waits for an idle worker.</span></div><div id="modelList" class="model-list"></div></details>
<details id="configPanel"><summary>Engine configuration <span id="configSummary" class="summary-meta">loading</span></summary><div class="config-tools"><div class="config-path"><span id="configPath"></span></div><div class="config-actions"><button id="configSave" class="btn primary">Save</button><button id="configReload" class="btn">Reload</button><button id="configReset" class="btn">Reset</button></div><div class="config-key"><span><b>↻</b> applied after Reload</span><span><b>◆</b> applied after service restart</span><span>CLI-locked settings are read-only</span></div></div><div id="configNotice" class="config-notice" role="status"></div><form id="configForm" class="config-form"></form></details>
<details id="runsPanel" open><summary>Recent runs <span id="runSummary" class="summary-meta">0 retained</span></summary><div class="tools"><input id="runFilter" type="search" placeholder="Filter runs by IP, route, state…"></div><div class="table-wrap"><table><thead><tr><th>ID / time</th><th>Client</th><th>Type</th><th>State</th><th>Tokens p/g</th><th>Prompt tok/s</th><th>Gen tok/s</th><th>TTFT</th><th>Total</th></tr></thead><tbody id="runRows"></tbody></table></div></details>
<details id="logsPanel"><summary>Engine logs <span id="logSummary" class="summary-meta">0 retained</span></summary><div class="tools"><input id="logFilter" type="search" placeholder="Filter logs (text or level)…"><button id="logTail" class="btn">Jump to latest</button></div><pre id="logLines">Open this panel to load captured logs.</pre></details>
<nav><a href="/docs">API docs</a><a href="/openapi.json">OpenAPI</a><a href="/metrics">Prometheus</a><a href="/api/runs">Run JSON</a><a href="/api/logs">Log JSON</a><span class="privacy">memory-only telemetry · clears on restart</span></nav></main>
<dialog id="detail"><div class="dialog-head"><strong id="detailTitle">Run detail</strong><button class="btn" onclick="detail.close()">Close</button></div><div class="dialog-body"><div id="detailStats" class="stats"></div><h3 class="label">Request</h3><pre id="requestPayload" class="payload"></pre><h3 class="label">Response</h3><pre id="responsePayload" class="payload"></pre></div></dialog>
<script>
const $=id=>document.getElementById(id),esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),rate=n=>n==null?'—':Number(n).toFixed(1),ms=n=>n==null?'—':n<1000?Number(n).toFixed(0)+' ms':(n/1000).toFixed(2)+' s',gib=n=>n==null?'—':(n/1073741824).toFixed(1)+' GiB',net=n=>n==null?'—':n>=1000?(n/1000).toFixed(2)+' Gbps':Number(n).toFixed(1)+' Mbps',clock=n=>n?new Date(n).toLocaleTimeString():'—';let cachedLogs=[],cachedRuns=[],lastHistory={},refreshInFlight=false;
const card=(t,h,rows)=>`<article class="card"><h2>${esc(t)}</h2><div class="hero">${h}</div><dl>${rows.map(r=>`<dt>${esc(r[0])}</dt><dd>${r[1]}</dd>`).join('')}</dl></article>`;
async function refresh(){if(refreshInFlight)return;refreshInFlight=true;try{const s=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json());renderStatus(s);if($('runsPanel').open){const h=await fetch('/api/runs',{cache:'no-store'}).then(r=>r.json());cachedRuns=h.runs||[];lastHistory=h;renderRuns(h)}else{lastHistory=s.run_history||{};renderRunSummary(lastHistory)}if($('logsPanel').open){const l=await fetch('/api/logs',{cache:'no-store'}).then(r=>r.json());cachedLogs=l.lines||[];renderLogs();$('logSummary').textContent=`${l.retained||0} / ${l.capacity||0} retained`}else{$('logSummary').textContent=`${(s.logs||{}).retained||0} / ${(s.logs||{}).capacity||0} retained`}}catch(e){$('phase').textContent='unavailable';$('phase').className='badge error'}finally{refreshInFlight=false}}
function renderStatus(s){window.currentModel=s.model.id;const p=s.performance||{},a=p.average||p.aggregate||{},aggregate=p.aggregate||{},max=p.max||{},last=p.last||{},g=s.gpu||{},m=g.memory||{},n=s.network||{},v=s.model.vision||{},rh=s.run_history||{},sw=s.management?.model_switch||{};$('version').textContent=`v${s.service.version} · ${s.model.id} · uptime ${s.uptime_seconds}s`;$('phase').textContent=s.status;$('phase').className='badge '+s.status;if(sw.state==='queued'||sw.state==='loading')$('version').textContent+=` · model switch ${sw.state} #${sw.id}`;if(sw.state==='failed')$('version').textContent+=` · switch failed: ${sw.error||'unknown error'}`;$('cards').innerHTML=card('Prompt throughput',`<span class="lime">${rate(a.prompt_tokens_per_second)}</span> <small>tok/s average</small>`,[['Max',`${rate(max.prompt_tokens_per_second)} tok/s`],['Aggregate',`${rate(aggregate.prompt_tokens_per_second)} tok/s`],['Last run',`${rate(last.prompt_tokens_per_second)} tok/s`],['Prompt tokens',s.metrics.prompt_tokens]])+card('Generation throughput',`<span class="cyan">${rate(a.generation_tokens_per_second)}</span> <small>tok/s average</small>`,[['Max',`${rate(max.generation_tokens_per_second)} tok/s`],['Aggregate',`${rate(aggregate.generation_tokens_per_second)} tok/s`],['Last run',`${rate(last.generation_tokens_per_second)} tok/s`],['Generated tokens',s.metrics.completion_tokens]])+card('Worker',s.worker.active_request_id?`<span class="cyan">ACTIVE #${s.worker.active_request_id}</span>`:'<span class="lime">IDLE</span>',[['Queued',s.worker.queued_requests],['Complete / errors',`${s.metrics.requests_ok} / ${rh.errored||0}`]])+card('GPU',gib(m.used_bytes),[['Free',gib(m.free_bytes)],['Vision',v.enabled?`yes · ${esc(v.device)}`:'no']])+card('Network',`<span class="cyan">${net(n.receive_mbps)}</span>`,[['Transmit',net(n.transmit_mbps)],['Scope',esc(n.scope)]]);const x=s.loading;if(x){$('loadPanel').style.display='block';const pct=Math.min(98,(x.estimated_fraction||0)*100);$('loadBar').style.width=pct+'%';$('loadPct').textContent=pct.toFixed(1)+'%';$('loadText').textContent=`${x.elapsed_seconds.toFixed(0)}s elapsed · ${x.stage==='finalizing_gpu'?'finalizing GPU setup':x.estimated_eta_seconds==null?'ETA calculating':Math.ceil(x.estimated_eta_seconds)+'s estimated remaining'}`;$('loadHint').textContent=`${gib(x.observed_network_bytes)} observed / ${gib(x.expected_bytes)} model · ${x.estimate_basis}`}else $('loadPanel').style.display='none';document.querySelectorAll('#modelList button[data-path]').forEach(b=>b.disabled=sw.state==='queued'||sw.state==='loading');renderRunSummary(rh);renderThermal(g.thermal_guard);decorateGpuCard(s);$('errorPanel').style.display=s.status==='error'?'block':'none';$('errorText').textContent=s.status==='error'?(s.status_detail||'Engine failed without a reported detail.'):''}
function renderRunSummary(h){const active=h.active_count??(h.active?1:0);$('runSummary').textContent=`${h.retained||0} retained · ${active} active · ${h.errored||0} errors`}
function renderThermal(t){t=t||{};const state=t.state||'unavailable',temp=t.current_temperature_c==null?'—':Number(t.current_temperature_c).toFixed(1)+'°C',resume=t.resume_temperature_c==null?'—':Number(t.resume_temperature_c).toFixed(0)+'°C';$('thermalSummary').textContent=`${state} · ${temp}`;$('thermalDetail').textContent=state==='threshold_pending'?`HOT · pausing in ${Math.ceil(t.threshold_remaining_seconds||0)}s · current ${temp}`:state==='paused'||state==='cooling'?`Request ${t.affected_request||'queued'} held ${Number(t.paused_seconds||0).toFixed(1)}s · current ${temp} · resumes at ${resume} · queue remains alive.`:`State ${state} · threshold ${t.maximum_temperature_threshold_c||'—'}°C for ${t.maximum_temperature_seconds||'—'}s · resume ${resume}.`}
function decorateGpuCard(s){const g=s.gpu||{},m=g.memory||{},all=g.inventory?.gpus||{},x=Object.values(all).find(x=>x.hip_device_index===0)||{},p=x.pcie||{},w=x.power||{},t=(x.temperatures||[]).slice().sort((a,b)=>(b.temperature_c||-1)-(a.temperature_c||-1))[0]||{},guard=g.thermal_guard||{};const c=$('cards')?.children[3];if(!c)return;c.querySelector('.hero').innerHTML=`${gib(m.used_bytes)} <small>VRAM · ${esc(guard.state||'unavailable')} · ${t.temperature_c==null?'—':Number(t.temperature_c).toFixed(1)+'°C'}</small>`;c.querySelector('dl').innerHTML=[['Power',`${w.watts==null?'—':Number(w.watts).toFixed(1)+' W'} / ${w.limit_watts==null?'—':Number(w.limit_watts).toFixed(1)+' W'}`],['Temperature',t.temperature_c==null?'—':Number(t.temperature_c).toFixed(1)+'°C'],['PCIe',`Gen ${p.current_generation||'—'} ×${p.current_lanes||'—'}`],['Free VRAM',gib(m.free_bytes)]].map(r=>`<dt>${esc(r[0])}</dt><dd>${r[1]}</dd>`).join('')}
async function refreshGpus(){try{const j=await fetch('/api/gpus',{cache:'no-store'}).then(r=>r.json()),g=Object.values(j.gpus||{});$('gpuSummary').textContent=`${g.length} device${g.length===1?'':'s'} · ${j.virtualization?.mode||'unknown'}`;$('gpuRows').innerHTML=g.length?g.map(x=>{const p=x.pcie||{},w=x.power||{},t=(x.temperatures||[]).slice().sort((a,b)=>(b.temperature_c||-1)-(a.temperature_c||-1))[0]||{};return `<tr><td><b>${esc(x.pci_address)}</b><br>${esc(x.vendor||'Unknown')} ${esc(x.name||'—')}<br><span class="sub">${esc(x.vendor_id||'—')}/${esc(x.device_id||'—')} · HIP ${esc(x.hip_device_index)}</span></td><td>${esc(x.driver||'—')}<br>${esc(j.virtualization?.mode||'unknown')}</td><td>Gen ${esc(p.current_generation)} ×${esc(p.current_lanes)}<br>${esc(p.theoretical_gbps_per_direction)} GB/s</td><td>${w.watts==null?'—':Number(w.watts).toFixed(1)+' W'} / ${w.limit_watts==null?'—':Number(w.limit_watts).toFixed(1)+' W'}</td><td>${t.temperature_c==null?'—':Number(t.temperature_c).toFixed(1)+'°C'}<br>${esc(t.label)}</td><td>${gib(x.vram?.used_bytes)} / ${gib(x.vram?.total_bytes)}</td><td>${x.controls?.helper_installed&&x.controls?.power_limit_writable?'AMD writable':'read-only'}</td></tr>`}).join(''):'<tr><td colspan="7">No GPU inventory available.</td></tr>'}catch(e){$('gpuSummary').textContent='unavailable';$('gpuRows').innerHTML='<tr><td colspan="7">GPU inventory unavailable.</td></tr>'}}
async function refreshAmd(){try{const j=await fetch('/api/gpus',{cache:'no-store'}).then(r=>r.json()),w=Object.values(j.gpus||{}).filter(x=>x.vendor==='AMD'&&x.controls?.helper_installed&&x.controls?.power_limit_writable);$('amdPci').innerHTML=w.length?w.map(x=>`<option value="${esc(x.pci_address)}">${esc(x.pci_address)} · ${esc(x.name||x.driver||'AMD')}</option>`).join(''):'<option value="">No writable AMD GPU</option>';$('amdSummary').textContent=w.length?`${w.length} writable AMD device${w.length===1?'':'s'}`:'No helper-enabled writable AMD GPU'}catch(_){$('amdSummary').textContent='unavailable'}}
async function amdMutation(operation){const pci=$('amdPci').value;if(!pci){alert('Select a writable AMD GPU first.');return}if(!confirm(`Apply runtime ${operation} to ${pci}? The thermal guard remains authoritative.`))return;const specs={power:['PUT','set-power-limit',{watts:Number($('amdWatts').value)}],tuning:['PUT','set-tuning',{performance_level:$('amdLevel').value}],reset:['POST','reset-gpu',{}]}[operation];const r=await fetch(`/api/gpus/${encodeURIComponent(pci)}/${operation==='power'?'power-limit':operation==='tuning'?'tuning':'reset'}`,{method:specs[0],headers:{'Content-Type':'application/json','X-ReInstinct-Action':specs[1]},body:JSON.stringify(specs[2])}),j=await r.json();$('amdSummary').textContent=r.ok?`Applied ${operation}; readback ${JSON.stringify(j.readback||j)}`:`${r.status}: ${j.error?.message||'operation failed'}`;refreshGpus();refreshAmd()}
function renderRuns(h){const q=$('runFilter').value.toLowerCase(),runs=(h.runs||[]).filter(r=>!q||[r.client_ip,r.path,r.request_type,r.state,String(r.status_code||'')].join(' ').toLowerCase().includes(q));renderRunSummary(h);$('runRows').innerHTML=runs.length?runs.map(r=>{const x=r.stats||{};return `<tr data-id="${r.id}"><td><b>#${r.id}</b><br>${clock(r.queued_at_ms)}</td><td>${esc(r.client_ip)}</td><td>${esc(r.request_type)}</td><td><span class="state ${esc(r.state)}">${esc(r.state)}</span></td><td>${x.prompt_tokens||0}/${x.completion_tokens||0}</td><td class="lime">${rate(x.prompt_tokens_per_second)}</td><td class="cyan">${rate(x.generation_tokens_per_second)}</td><td>${ms(x.ttft_ms)}</td><td>${ms(x.total_ms)}</td></tr>`}).join(''):'<tr><td colspan="9">No matching retained runs.</td></tr>'}
function renderLogs(){const q=$('logFilter').value.toLowerCase(),lines=cachedLogs.filter(x=>x.toLowerCase().includes(q));$('logLines').textContent=lines.join('\n')||'No matching logs.'}
 $('logFilter').addEventListener('input',renderLogs);$('runFilter').addEventListener('input',()=>renderRuns(Object.assign({runs:cachedRuns},lastHistory)));$('modelFilter').addEventListener('input',()=>filterModels());$('logTail').onclick=()=>{$('logLines').scrollTop=$('logLines').scrollHeight};$('logsPanel').addEventListener('toggle',()=>{if($('logsPanel').open)refresh()});$('runsPanel').addEventListener('toggle',()=>{if($('runsPanel').open)refresh()});$('runRows').onclick=async e=>{const row=e.target.closest('tr[data-id]');if(!row)return;const r=await fetch('/api/runs/'+row.dataset.id).then(x=>x.json()),x=r.stats||{};$('detailTitle').textContent=`Run #${r.id} · ${r.state}`;$('detailStats').innerHTML=[['Client',r.client_ip],['Route',r.method+' '+r.path],['Prompt',`${x.prompt_tokens} · ${rate(x.prompt_tokens_per_second)} tok/s`],['Generation',`${x.completion_tokens} · ${rate(x.generation_tokens_per_second)} tok/s`],['TTFT',ms(x.ttft_ms)],['Total',ms(x.total_ms)]].map(v=>`<div><span>${esc(v[0])}</span>${esc(v[1])}</div>`).join('');$('requestPayload').textContent=JSON.stringify(r.request,null,2);$('responsePayload').textContent=JSON.stringify(r.response,null,2);$('detail').showModal()};
function filterModels(){const q=$('modelFilter').value.toLowerCase();document.querySelectorAll('.model').forEach(row=>row.hidden=!!q&&!row.textContent.toLowerCase().includes(q))}
$('scanModels').onclick=async()=>{const b=$('scanModels');b.disabled=true;b.textContent='Refreshing…';try{const c=await fetch('/api/models?refresh=1',{cache:'no-store'}).then(r=>r.json());$('modelSummary').textContent=`${c.count} models · ${c.root} · ${c.scan_duration_ms} ms`;$('modelList').innerHTML=c.models.map(m=>`<div class="model ${m.id===window.currentModel?'active-model':''}"><strong>${esc(m.id)}</strong><button class="btn" data-path="${esc(m.path)}" ${m.id===window.currentModel?'disabled':''}>${m.id===window.currentModel?'Loaded':'Load model'}</button><span>${Number(m.size_gib).toFixed(2)} GiB · ${m.vision_capable?'image projector found':'text only'}</span><span class="path">${esc(m.path)}${m.image_projector?'<br>projector: '+esc(m.image_projector):''}</span></div>`).join('');filterModels()}catch(e){$('modelList').textContent='Catalog scan failed: '+e}finally{b.disabled=false;b.textContent='Refresh models'}};
$('modelList').onclick=async e=>{const b=e.target.closest('button[data-path]');if(!b)return;const path=b.dataset.path;if(!confirm(`Unload the current model and load:\n${path}\n\nInference will pause during loading.`))return;b.disabled=true;b.textContent='Queued…';try{const r=await fetch('/api/models/switch',{method:'POST',headers:{'Content-Type':'application/json','X-ReInstinct-Action':'switch-model'},body:JSON.stringify({path})}),j=await r.json();if(!r.ok)throw new Error(j.error?.message||'switch failed');$('modelSummary').textContent=`switch #${j.id} queued`;refresh()}catch(e){alert(e.message)}finally{b.disabled=false}};
let configSnapshot=null;const restartKeys=new Set(['big_port','small_port','embed_port']);const configFields=[['big','Big model','path'],['model_dir','Model catalog root','path'],['big_drafter','Big drafter (optional)','path'],['small','Small model (optional)','path'],['embed','Embedder (optional)','path'],['big_port','Big port','number'],['small_port','Small port','number'],['embed_port','Embed port','number'],['max_seq','Context size','number'],['mmproj','Vision projector (optional)','path'],['mtmd_bridge','Vision bridge (optional)','path'],['vision_threads','Vision threads','number'],['vision_min_tokens','Vision minimum tokens','number'],['vision_max_tokens','Vision maximum tokens','number'],['cpu_vision','CPU vision','bool'],['gpu_thermal_guard_enabled','Thermal guard enabled','bool'],['gpu_max_temp_c','Maximum temperature °C','number'],['gpu_max_temp_seconds','Sustained duration seconds','number'],['gpu_resume_temp_c','Resume temperature °C','number']];
function configNotice(message,tone=''){const n=$('configNotice');n.textContent=message;n.className='config-notice'+(message?' show':'')+(tone?' '+tone:'')}
function renderConfig(c){configSnapshot=c;const locked=new Set(c.cli_locked||[]),reloadFields=c.reload_required_fields||[],restartFields=c.restart_required_fields||[];$('configPath').textContent=c.path||'No configuration file — settings are read-only';$('configSummary').textContent=reloadFields.length?'reload needed':restartFields.length?'restart needed':(c.persisted?'saved':'read-only');$('configReload').disabled=!c.persisted||!reloadFields.length||restartFields.length>0;$('configSave').disabled=!c.persisted;if(c.reload?.error)configNotice(c.reload.error,'error');else if(restartFields.length)configNotice(`Saved. Restart the service to apply: ${restartFields.join(', ')}.`,'warn');else if(reloadFields.length)configNotice(`Saved changes are waiting. Reload the engine to apply ${reloadFields.length} setting${reloadFields.length===1?'':'s'}.`,'warn');else if(!c.persisted)configNotice('Start the service with --config PATH to enable dashboard persistence.','warn');else configNotice('');const v=c.saved||c.effective||{};$('configForm').innerHTML=configFields.map(([key,label,type])=>{const lock=locked.has(key),val=v[key],restart=restartKeys.has(key),mark=lock?'':`<span class="${restart?'restart-mark':'apply-mark'}" title="${restart?'Applied after service restart':'Applied after Reload'}">${restart?'◆':'↻'}</span>`;if(type==='bool')return `<div class="config-field checkbox-field ${lock?'locked':''}"><label><input data-config="${key}" type="checkbox" ${val?'checked':''} ${lock?'disabled':''}> ${esc(label)}</label>${mark}</div>`;return `<div class="config-field ${lock?'locked':''}"><label for="cfg-${key}">${esc(label)}</label><input id="cfg-${key}" data-config="${key}" type="${type==='number'?'number':'text'}" value="${val==null?'':esc(val)}" ${lock?'disabled':''}>${mark}</div>`}).join('')}
async function loadConfig(){try{const c=await fetch('/api/config',{cache:'no-store'}).then(r=>r.json());renderConfig(c)}catch(e){configNotice('Configuration unavailable: '+e,'error')}}
$('configPanel').addEventListener('toggle',()=>{if($('configPanel').open)loadConfig()});$('configForm').addEventListener('input',()=>{$('configSummary').textContent='unsaved';$('configReload').disabled=true;configNotice('Unsaved changes. Save to validate them and see whether Reload is required.','warn')});$('configReset').onclick=()=>{if(configSnapshot)renderConfig(configSnapshot)};$('configSave').onclick=async()=>{if(!configSnapshot?.persisted){configNotice('Start the service with --config PATH before saving.','warn');return}const body={};for(const el of document.querySelectorAll('[data-config]')){if(el.disabled)continue;body[el.dataset.config]=el.type==='checkbox'?el.checked:(el.type==='number'?(el.value===''?null:Number(el.value)):(el.value===''?null:el.value))}const b=$('configSave');b.disabled=true;b.textContent='Saving…';try{const r=await fetch('/api/config',{method:'PUT',headers:{'Content-Type':'application/json','X-ReInstinct-Action':'update-config'},body:JSON.stringify(body)}),j=await r.json();if(!r.ok)throw new Error(j.error?.message||'configuration update failed');await loadConfig()}catch(e){configNotice(e.message,'error')}finally{b.disabled=!configSnapshot?.persisted;b.textContent='Save'}};$('configReload').onclick=async()=>{if(!confirm('Reload the engine now? Inference will pause while models are reloaded.'))return;const b=$('configReload');b.disabled=true;b.textContent='Queuing…';try{const r=await fetch('/api/config/reload',{method:'POST',headers:{'X-ReInstinct-Action':'reload-config'}}),j=await r.json();if(!r.ok)throw new Error(j.error?.message||'reload failed');configNotice(`Reload #${j.id} queued.`);$('configSummary').textContent='reload queued';loadConfig()}catch(e){configNotice(e.message,'error')}finally{b.textContent='Reload'}};loadConfig();
 $('amdPower').onclick=()=>amdMutation('power');$('amdTune').onclick=()=>amdMutation('tuning');$('amdReset').onclick=()=>amdMutation('reset');refresh();setInterval(refresh,2000);refreshGpus();refreshAmd();setInterval(refreshGpus,2000);setInterval(refreshAmd,5000);
</script></body></html>"#;

/// Add the driver-aware power-limit control without duplicating the large
/// dashboard template in the API source. The base page remains a single
/// self-contained document for the no-build deployment model.
pub fn dashboard_html() -> String {
    let mut html = INDEX_HTML_V3.replace(
        "</style></head>",
        r#"<style>
  .power-control{flex:1 1 390px;min-width:280px;display:grid;grid-template-columns:1fr auto;gap:6px 12px;align-items:center}.power-control label{color:var(--muted);font-size:10px;letter-spacing:.1em;text-transform:uppercase}.power-control output{font:18px Bahnschrift,sans-serif;color:var(--lime);white-space:nowrap}.power-control input[type=range]{grid-column:1/-1;width:100%;min-width:220px;accent-color:var(--lime);cursor:pointer}.power-control input[type=range]:disabled{cursor:not-allowed;opacity:.4}.power-meta{grid-column:1/-1;display:flex;justify-content:space-between;gap:12px;color:var(--muted);font-size:10px}.power-meta strong{color:var(--ink);font-weight:400}
  .thermal-body{padding:14px 16px}.thermal-settings{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:#26352e}.thermal-field{background:#0e1612;padding:12px;display:grid;gap:7px}.thermal-field label{color:#b7c4bd;font-size:10px;text-transform:uppercase;letter-spacing:.08em}.thermal-field input{width:100%;min-width:0;background:#0a100d;border-color:#34463d;padding:9px 10px;font:11px inherit}.thermal-field input:focus{outline:1px solid var(--cyan);outline-offset:1px}.thermal-field.checkbox{display:flex;align-items:center}.thermal-field.checkbox label{display:flex;align-items:center;gap:9px;text-transform:none;letter-spacing:0}.thermal-field.checkbox input{width:auto;accent-color:var(--lime)}.thermal-field.locked{opacity:.58}.thermal-actions{display:flex;align-items:center;gap:9px;margin-top:12px}.thermal-notice{color:var(--muted);font-size:10px}.thermal-notice.warn{color:var(--amber)}.thermal-notice.error{color:var(--red)}.gpu-meters{display:grid;gap:8px;margin-top:14px}.gpu-meter{display:grid;gap:4px}.meter-label{display:flex;justify-content:space-between;gap:10px;color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.08em}.meter-label b{color:var(--ink);font-weight:400;letter-spacing:0;text-transform:none}.meter{height:7px;border:1px solid var(--line);background:#070b09;overflow:hidden}.meter i{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--lime));transition:width .4s}.meter i.warn{background:linear-gradient(90deg,var(--amber),#ff8f5a)}.meter i.bad{background:var(--red)}
</style></head>"#,
    );
    html = html.replace(
        "</style></head>",
        r#"<style>.memory-panel{margin:0 0 12px;padding:14px 16px;background:linear-gradient(145deg,#141d19f5,#0b110ef5);border:1px solid var(--line)}.memory-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px}.memory-title{color:var(--muted);font-size:10px;letter-spacing:.13em;text-transform:uppercase}.memory-total{font:22px Bahnschrift,sans-serif}.memory-total small{font:10px inherit;color:var(--muted)}.memory-bar,.context-bar{display:flex;height:18px;margin:11px 0 8px;background:#070b09;border:1px solid var(--line);overflow:hidden}.context-bar{height:8px;margin-top:8px}.memory-segment{height:100%;min-width:1px}.memory-legend{display:flex;flex-wrap:wrap;gap:7px 15px;color:var(--muted);font-size:10px}.memory-legend span:before{content:"";display:inline-block;width:8px;height:8px;margin-right:5px;background:var(--swatch);border-radius:2px}.memory-detail{display:grid;grid-template-columns:1fr auto;gap:5px;margin-top:9px;font-size:10px}.memory-detail b{text-align:right;color:var(--ink);font-weight:400}</style></head>"#,
    );
    html = html.replace(
        r#"<details id="thermalPanel"><summary>Thermal guard <span id="thermalSummary" class="summary-meta">loading</span></summary><div class="tools"><span id="thermalDetail" class="sub">The inference GPU is monitored at one-second intervals.</span></div></details>"#,
        r#"<details id="thermalPanel"><summary>Thermal guard <span id="thermalSummary" class="summary-meta">loading</span></summary><div class="thermal-body"><div class="tools"><span id="thermalDetail" class="sub">The inference GPU is monitored at one-second intervals.</span></div><div id="thermalForm" class="thermal-settings"></div><div class="thermal-actions"><button id="thermalSave" class="btn primary">Save thermal settings</button><button id="thermalReset" class="btn">Reset</button><span id="thermalNotice" class="thermal-notice" role="status"></span></div></div></details>"#,
    );
    html = html.replace(
        r#"<details id="runsPanel" open>"#,
        r#"<details id="runsPanel">"#,
    );
    html = html.replace(
        r#"<section id="cards" class="grid"></section>"#,
        r#"<section id="cards" class="grid"></section><section id="memoryPanel" class="memory-panel" aria-live="polite"></section>"#,
    );
    html = html.replace(
        r#"${esc(p.theoretical_gbps_per_direction)} GB/s"#,
        r#"${Number.isFinite(Number(p.theoretical_gbps_per_direction)) ? Number(p.theoretical_gbps_per_direction).toFixed(1) : '—'} GB/s"#,
    );
    html = html.replace(
        r#"['cpu_vision','CPU vision','bool'],['gpu_thermal_guard_enabled','Thermal guard enabled','bool'],['gpu_max_temp_c','Maximum temperature °C','number'],['gpu_max_temp_seconds','Sustained duration seconds','number'],['gpu_resume_temp_c','Resume temperature °C','number']"#,
        r#"['cpu_vision','CPU vision','bool']"#,
    );
    html = html.replace(
        r#"const v=c.saved||c.effective||{};$('configForm').innerHTML="#,
        r#"const v=c.saved||c.effective||{};renderThermalConfig(v,locked);$('configForm').innerHTML="#,
    );
    html = html.replace(
        r#"<input id="amdWatts" type="number" min="1" step="0.1" placeholder="Power limit (W)">"#,
        r#"<input id="amdWatts" type="range" min="0" max="1" step="0.1" value="0" disabled aria-label="Power limit in watts">"#,
    );
    html = html.replace(
        "</script></body></html>",
         r#"</script><script>
 function renderMemory(s){const m=s.gpu?.memory||{},sec=m.sections||{},total=Number(m.total_bytes)||0,used=Number(m.used_bytes)||0,free=Number(m.free_bytes)||0;const rows=[['Model weights',Number(sec.model_weights_bytes)||0,'#b7ff5a'],['Context reserved',Number(sec.context_reserved_bytes)||0,'#62d9d1'],['Runtime',Number(sec.runtime_bytes)||0,'#a78bfa'],['Scratch / pools',Number(sec.scratch_bytes)||0,'#ffc857'],['Vision',Number(sec.vision_bytes)||0,'#fb923c'],['Unattributed',Number(sec.unattributed_bytes)||0,'#718096'],['Free',free,'#26332c']];const pct=v=>total>0?Math.max(0,Math.min(100,v/total*100)):0;const fmt=v=>`${(v/1073741824).toFixed(2)} GiB`;const c=m.context||{},contextPct=Number(c.utilization_fraction)||0;const panel=document.getElementById('memoryPanel');if(!panel)return;panel.innerHTML=`<div class="memory-head"><div class="memory-title">VRAM allocation</div><div class="memory-total">${fmt(used)} <small>/ ${fmt(total)} used</small></div></div><div class="memory-bar" role="img" aria-label="VRAM allocation breakdown">${rows.map(r=>`<i class="memory-segment" style="width:${pct(r[1])}%;background:${r[2]}" title="${r[0]}: ${fmt(r[1])}"></i>`).join('')}</div><div class="memory-legend">${rows.map(r=>`<span style="--swatch:${r[2]}">${r[0]}</span>`).join('')}</div><div class="memory-title" style="margin-top:16px">Context capacity</div><div class="context-bar" role="progressbar" aria-valuenow="${c.used_tokens||0}" aria-valuemax="${c.capacity_tokens||0}"><i class="memory-segment" style="width:${Math.max(0,Math.min(100,contextPct*100))}%;background:#62d9d1"></i></div><div class="memory-detail"><span>Positions used</span><b>${c.used_tokens??'—'} / ${c.capacity_tokens??'—'} · ${(contextPct*100).toFixed(1)}%</b><span>Reserved context VRAM</span><b>${fmt(Number(c.reserved_bytes)||0)} · used equivalent ${fmt(Number(c.used_equivalent_bytes)||0)}</b></div><div class="sub" style="margin-top:10px">Categorized ${fmt(Number(m.tracked_bytes)||0)} · residual ${fmt(Number(m.unattributed_bytes)||0)}</div>`}
 const _renderStatus=renderStatus;renderStatus=s=>{_renderStatus(s);renderMemory(s)};
 (() => {
  const input = document.getElementById('amdWatts');
  const select = document.getElementById('amdPci');
  if (!input || !select) return;
  input.type = 'range'; input.disabled = true; input.removeAttribute('placeholder');
  const control = document.createElement('div'); control.className = 'power-control';
  const label = document.createElement('label'); label.htmlFor = 'amdWatts'; label.textContent = 'Power limit';
  const output = document.createElement('output'); output.id = 'amdWattsValue';
  const meta = document.createElement('div'); meta.className = 'power-meta';
  const current = document.createElement('span'); current.id = 'amdPowerCurrent';
  const range = document.createElement('span'); range.id = 'amdPowerRange';
  meta.append(current, range); input.replaceWith(control); control.append(label, output, input, meta);
  const devices = {};
  const finite = value => Number.isFinite(Number(value));
  const watts = value => finite(value) ? `${Number(value).toFixed(1)} W` : '—';
  const updateReadout = () => { output.textContent = watts(input.value); };
  input.addEventListener('input', updateReadout);
  select.addEventListener('change', () => { input.dataset.pci = ''; refreshPowerLimit(); });
  async function refreshPowerLimit() {
    try {
      const payload = await fetch('/api/gpus', {cache:'no-store'}).then(response => response.json());
      Object.keys(devices).forEach(key => delete devices[key]);
      Object.values(payload.gpus || {}).filter(gpu => gpu.vendor === 'AMD' && gpu.controls?.helper_installed && gpu.controls?.power_limit_writable).forEach(gpu => { devices[gpu.pci_address] = gpu; });
      const gpu = devices[select.value], power = gpu?.power || {};
      const min = Number(power.minimum_watts), max = Number(power.maximum_watts), limit = Number(power.limit_watts);
      const valid = [min, max, limit].every(Number.isFinite) && max > min;
      input.disabled = !valid;
      if (!valid) { output.textContent = '—'; current.innerHTML = 'Current limit <strong>—</strong>'; range.innerHTML = 'Driver range <strong>unavailable</strong>'; return; }
      input.min = min; input.max = max; input.step = Math.max(.1, Math.round((max - min) / 100 * 10) / 10);
      if (input.dataset.pci !== select.value || document.activeElement !== input) input.value = Math.min(max, Math.max(min, limit));
      input.dataset.pci = select.value; updateReadout();
      current.innerHTML = `Current limit <strong>${watts(limit)}</strong>`;
      range.innerHTML = `Driver range <strong>${watts(min)} – ${watts(max)}</strong>`;
    } catch (_) { input.disabled = true; output.textContent = '—'; current.innerHTML = 'Current limit <strong>unavailable</strong>'; range.innerHTML = 'Driver range <strong>unavailable</strong>'; }
  }
   refreshPowerLimit(); setInterval(refreshPowerLimit, 2000);
 })();
 const thermalFields=[['gpu_thermal_guard_enabled','Guard enabled','bool'],['gpu_max_temp_c','Pause threshold °C','number'],['gpu_max_temp_seconds','Sustained duration seconds','number'],['gpu_resume_temp_c','Resume below °C','number']];
 function renderThermalConfig(values,locked){const form=$('thermalForm');if(!form)return;form.innerHTML=thermalFields.map(([key,label,type])=>{const lock=locked.has(key),value=values[key],mark=lock?'':'<span class="apply-mark" title="Applied immediately">↻</span>';if(type==='bool')return `<div class="thermal-field checkbox ${lock?'locked':''}"><label><input data-thermal="${key}" type="checkbox" ${value?'checked':''} ${lock?'disabled':''}> ${esc(label)}</label>${mark}</div>`;return `<div class="thermal-field ${lock?'locked':''}"><label for="thermal-${key}">${esc(label)} ${mark}</label><input id="thermal-${key}" data-thermal="${key}" type="number" value="${value==null?'':esc(value)}" ${lock?'disabled':''}></div>`}).join('')}
 function thermalNotice(message,tone=''){const n=$('thermalNotice');if(!n)return;n.textContent=message;n.className='thermal-notice'+(tone?' '+tone:'')}
 $('thermalForm').addEventListener('input',()=>{thermalNotice('Unsaved thermal changes. Save to validate and apply them.','warn');$('thermalSave').disabled=false});
 $('thermalReset').onclick=()=>{if(configSnapshot){const values=configSnapshot.saved||configSnapshot.effective||{};renderThermalConfig(values,new Set(configSnapshot.cli_locked||[]));thermalNotice('')}};
 $('thermalSave').onclick=async()=>{if(!configSnapshot?.persisted){thermalNotice('Start the service with --config PATH before saving.','warn');return}const body={};for(const el of document.querySelectorAll('[data-thermal]')){if(el.disabled)continue;body[el.dataset.thermal]=el.type==='checkbox'?el.checked:(el.value===''?null:Number(el.value))}const b=$('thermalSave');b.disabled=true;b.textContent='Saving…';try{const r=await fetch('/api/config',{method:'PUT',headers:{'Content-Type':'application/json','X-ReInstinct-Action':'update-config'},body:JSON.stringify(body)}),j=await r.json();if(!r.ok)throw new Error(j.error?.message||'thermal configuration update failed');thermalNotice('Saved and applied immediately.');await loadConfig()}catch(e){thermalNotice(e.message,'error');b.disabled=false}finally{b.textContent='Save thermal settings';if(configSnapshot?.persisted)b.disabled=false}};
 const baseDecorateGpuCard=decorateGpuCard;
 window.decorateGpuCard=function(s){baseDecorateGpuCard(s);const g=s.gpu||{},m=g.memory||{},all=g.inventory?.gpus||{},x=Object.values(all).find(x=>x.hip_device_index===0)||{},w=x.power||{},t=(x.temperatures||[]).slice().sort((a,b)=>(b.temperature_c??-1)-(a.temperature_c??-1))[0]||{},guard=g.thermal_guard||{},threshold=Number(guard.maximum_temperature_threshold_c),powerMax=Number(w.limit_watts??w.maximum_watts),power=Number(w.watts),vramTotal=Number(m.total_bytes),vramUsed=Number(m.used_bytes),temp=Number(t.temperature_c),compute=Number(x.utilization?.percent),c=$('cards')?.children[3];if(!c)return;const meter=(label,percent,text,tone='')=>{const valid=Number.isFinite(percent);const width=valid?Math.min(100,Math.max(0,percent)):0;return `<div class="gpu-meter"><div class="meter-label"><span>${label}</span><b>${esc(valid?text:'unavailable')}</b></div><div class="meter"><i class="${tone}" style="width:${width}%"></i></div></div>`};let meters=c.querySelector('.gpu-meters');if(!meters){meters=document.createElement('div');meters.className='gpu-meters';c.append(meters)}const powerPercent=powerMax>0?power/powerMax*100:null,vramPercent=vramTotal>0?vramUsed/vramTotal*100:null,tempPercent=threshold>0?temp/threshold*100:null;meters.innerHTML=meter('Power',powerPercent,Number.isFinite(power)?`${power.toFixed(1)} W / ${Number.isFinite(powerMax)?powerMax.toFixed(1)+' W':'—'}`:'—',powerPercent>=90?'warn':'')+meter('VRAM',vramPercent,`${gib(vramUsed)} / ${gib(vramTotal)}`,vramPercent>=90?'warn':'')+meter('Temp',tempPercent,Number.isFinite(temp)?`${temp.toFixed(1)}°C / ${Number.isFinite(threshold)?threshold.toFixed(0)+'°C':'—'}`:'—',tempPercent>=100?'bad':tempPercent>=90?'warn':'')+meter('Compute',compute,Number.isFinite(compute)?`${compute.toFixed(0)}%`:'—');c.querySelector('dl').innerHTML=''};
 refresh();
 $('thermalSave').disabled=!configSnapshot?.persisted;
 </script></body></html>"#,
    );
    html
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_openapi() {
        let v: serde_json::Value = serde_json::from_str(&openapi_json()).unwrap();
        assert_eq!(v["openapi"], "3.1.0");
        assert!(v["paths"]["/v1/chat/completions"].is_object());
        assert!(v["paths"]["/api/runs"].is_object());
        assert!(v["paths"]["/api/runs/{id}"].is_object());
        assert!(v["paths"]["/api/config"]["put"].is_object());
        assert!(v["paths"]["/api/config/reload"]["post"].is_object());
    }

    #[test]
    fn capture_redacts_image_data_but_keeps_prompt() {
        let captured = capture_json(r#"{"messages":[{"content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}"#);
        let text = captured.to_string();
        assert!(text.contains("describe"));
        assert!(text.contains("image data URL redacted"));
        assert!(!text.contains("base64,AAAA"));
    }

    #[test]
    fn run_history_evicts_oldest_at_capacity() {
        let mut history = RunHistory::new();
        for id in 1..=(RUN_HISTORY_CAPACITY as u64 + 1) {
            history.push(RunRecord { id, client_ip: "127.0.0.1".into(),
                method: "POST".into(), path: "/v1/chat/completions".into(),
                request_type: "chat".into(), state: "complete", queued_at_ms: id,
                started_at_ms: Some(id), completed_at_ms: Some(id), status_code: Some(200),
                request: Value::Null, response: Some(Value::Null), error: None,
                stats: RunStats::default() });
        }
        assert_eq!(history.entries.len(), RUN_HISTORY_CAPACITY);
        assert!(history.find(1).is_none());
        assert!(history.find(RUN_HISTORY_CAPACITY as u64 + 1).is_some());
    }

    #[test]
    fn run_list_keeps_summary_counts_with_rows() {
        let mut history = RunHistory::new();
        for (id, state, status_code) in [(1, "complete", 200), (2, "errored", 500),
                                         (3, "active", 0)] {
            history.push(RunRecord { id, client_ip: "127.0.0.1".into(),
                method: "POST".into(), path: "/v1/chat/completions".into(),
                request_type: "chat".into(), state, queued_at_ms: id,
                started_at_ms: Some(id), completed_at_ms: (status_code > 0).then_some(id),
                status_code: (status_code > 0).then_some(status_code), request: Value::Null,
                response: Some(Value::Null), error: None, stats: RunStats::default() });
        }
        let value = runs_value(&history);
        assert_eq!(value["retained"], 3);
        assert_eq!(value["active_count"], 1);
        assert_eq!(value["completed"], 1);
        assert_eq!(value["errored"], 1);
        assert_eq!(value["runs"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn performance_reports_average_and_max_retained_rates() {
        let metrics = Metrics::new();
        let mut history = RunHistory::new();
        for (id, prompt, generation) in [(1, 100.0, 20.0), (2, 140.0, 30.0), (3, 0.0, 0.0)] {
            history.push(RunRecord {
                id, client_ip: "127.0.0.1".into(), method: "POST".into(),
                path: "/v1/chat/completions".into(), request_type: "chat".into(),
                state: "complete", queued_at_ms: id, started_at_ms: Some(id),
                completed_at_ms: Some(id), status_code: Some(200), request: Value::Null,
                response: Some(Value::Null), error: None,
                stats: RunStats { prompt_tokens_per_second: prompt,
                    generation_tokens_per_second: generation, ..RunStats::default() },
            });
        }
        let value = performance_json(&metrics, &history);
        assert_eq!(value["average"]["prompt_tokens_per_second"], 120.0);
        assert_eq!(value["average"]["generation_tokens_per_second"], 25.0);
        assert_eq!(value["max"]["prompt_tokens_per_second"], 140.0);
        assert_eq!(value["max"]["generation_tokens_per_second"], 30.0);
        assert_eq!(value["sample_count"]["prompt"], 2);
        assert_eq!(value["sample_count"]["generation"], 2);
    }

    #[test]
    fn operations_dashboard_uses_one_status_refresh_path() {
        assert_eq!(INDEX_HTML_V3.matches("fetch('/api/status'").count(), 1);
        assert!(!INDEX_HTML_V3.contains("syncThermal"));
        assert!(!INDEX_HTML_V3.contains("refreshError"));
        assert!(INDEX_HTML_V3.contains("renderThermal(g.thermal_guard);decorateGpuCard(s)"));
        assert!(INDEX_HTML_V3.contains("refreshInFlight"));
        let dashboard = dashboard_html();
        assert!(dashboard.contains("Current limit"));
        assert!(dashboard.contains("Driver range"));
        assert!(dashboard.contains("input.type = 'range'"));
        assert!(dashboard.contains(r#"<details id="runsPanel">"#));
        assert!(!dashboard.contains(r#"<details id="runsPanel" open>"#));
        assert!(dashboard.contains("thermalForm"));
        assert!(dashboard.contains("data-thermal"));
        assert!(dashboard.contains("Pause threshold °C"));
        assert!(dashboard.contains("memoryPanel"));
        assert!(dashboard.contains("Context capacity"));
        assert!(dashboard.contains("Model weights"));
        assert!(dashboard.contains("tok/s average"));
        assert!(dashboard.contains("['Max'"));
        assert!(!dashboard.contains("Thermal guard enabled','bool'"));
        assert!(dashboard.contains("meter('Power'"));
        assert!(dashboard.contains("meter('Compute'"));
        assert!(dashboard.contains("window.decorateGpuCard=function(s){baseDecorateGpuCard(s);"));
        assert!(dashboard.contains("c.querySelector('dl').innerHTML=''"));
        assert!(dashboard.contains("theoretical_gbps_per_direction)) ? Number(p.theoretical_gbps_per_direction).toFixed(1)"));
        assert!(dashboard.contains("</script><script>"));
    }
}
