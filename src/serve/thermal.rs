//! Cooperative, fail-closed thermal interlock.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use super::gpu::{GpuInventory, SensorReading};

#[derive(Clone, Debug)]
pub struct ThermalConfig {
    pub enabled: bool,
    pub max_temp_c: f64,
    pub max_temp_seconds: u64,
    pub resume_temp_c: f64,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_temp_c: 90.0,
            max_temp_seconds: 5,
            resume_temp_c: 82.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThermalState {
    Monitoring,
    ThresholdPending,
    Paused,
    Cooling,
    Unavailable,
    Disabled,
}

impl ThermalState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Monitoring => "monitoring",
            Self::ThresholdPending => "threshold_pending",
            Self::Paused => "paused",
            Self::Cooling => "cooling",
            Self::Unavailable => "unavailable",
            Self::Disabled => "disabled",
        }
    }
}

struct Inner {
    config: ThermalConfig,
    state: ThermalState,
    current_temp_c: Option<f64>,
    maximum_temp_c: Option<f64>,
    hot_since: Option<Instant>,
    pause_started: Option<Instant>,
    pause_start_unix_ms: Option<u64>,
    affected_request: Option<u64>,
    sensor: Option<String>,
    consecutive_failures: u32,
    ever_monitored: bool,
}

pub struct ThermalGuard {
    inner: Mutex<Inner>,
    wake: Condvar,
    thermal_pauses: AtomicU64,
    paused_ms: AtomicU64,
    sensor_failures: AtomicU64,
    maximum_observed_milli_c: AtomicU64,
    held_requests: AtomicU64,
}

impl ThermalGuard {
    pub fn new(config: ThermalConfig) -> Self {
        let state = if !config.enabled {
            ThermalState::Disabled
        } else {
            ThermalState::Unavailable
        };
        Self {
            inner: Mutex::new(Inner {
                config,
                state,
                current_temp_c: None,
                maximum_temp_c: None,
                hot_since: None,
                pause_started: None,
                pause_start_unix_ms: None,
                affected_request: None,
                sensor: None,
                consecutive_failures: 0,
                ever_monitored: false,
            }),
            wake: Condvar::new(),
            thermal_pauses: AtomicU64::new(0),
            paused_ms: AtomicU64::new(0),
            sensor_failures: AtomicU64::new(0),
            maximum_observed_milli_c: AtomicU64::new(0),
            held_requests: AtomicU64::new(0),
        }
    }

    pub fn start(self: &std::sync::Arc<Self>, inventory: std::sync::Arc<GpuInventory>) {
        let guard = std::sync::Arc::clone(self);
        std::thread::Builder::new()
            .name("thermal-monitor".into())
            .spawn(move || loop {
                let reading = inventory.inference_sensor();
                guard.sample(reading);
                std::thread::sleep(Duration::from_secs(1));
            })
            .expect("thermal monitor thread");
    }

    pub fn update_config(&self, config: ThermalConfig) -> Result<(), String> {
        if !config.max_temp_c.is_finite()
            || !config.resume_temp_c.is_finite()
            || config.resume_temp_c >= config.max_temp_c
        {
            return Err("gpu_resume_temp_c must be lower than gpu_max_temp_c".into());
        }
        if config.max_temp_seconds == 0 {
            return Err("gpu_max_temp_seconds must be positive".into());
        }
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "thermal state is unavailable")?;
        tracing::info!("thermal guard configuration changed: enabled={} max_temp_c={} max_seconds={} resume_temp_c={}", config.enabled, config.max_temp_c, config.max_temp_seconds, config.resume_temp_c);
        let was_holding = matches!(inner.state, ThermalState::Paused | ThermalState::Cooling);
        inner.config = config;
        if !inner.config.enabled {
            if let Some(start) = inner.pause_started.take() {
                self.paused_ms
                    .fetch_add(start.elapsed().as_millis() as u64, Ordering::Relaxed);
            }
            inner.state = ThermalState::Disabled;
            inner.hot_since = None;
            inner.pause_start_unix_ms = None;
            inner.affected_request = None;
        } else if was_holding
            && (inner.consecutive_failures >= 5
                || inner
                    .current_temp_c
                    .is_none_or(|temp| temp > inner.config.resume_temp_c))
        {
            // A settings edit must never release an existing safety hold until
            // a valid sample confirms the configured resume threshold.
            inner.state = ThermalState::Cooling;
        } else if !inner.ever_monitored {
            inner.state = ThermalState::Unavailable;
            inner.hot_since = None;
        } else if inner
            .current_temp_c
            .is_some_and(|temp| temp >= inner.config.max_temp_c)
        {
            inner.state = ThermalState::ThresholdPending;
            inner.hot_since = Some(Instant::now());
        } else {
            if let Some(start) = inner.pause_started.take() {
                self.paused_ms
                    .fetch_add(start.elapsed().as_millis() as u64, Ordering::Relaxed);
            }
            inner.state = ThermalState::Monitoring;
            inner.hot_since = None;
            inner.pause_start_unix_ms = None;
            inner.affected_request = None;
        }
        self.wake.notify_all();
        Ok(())
    }

    pub fn checkpoint(
        &self,
        request_id: u64,
        deadline: &mut Option<Instant>,
    ) -> Result<f64, String> {
        let wait_started = Instant::now();
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| "thermal state is unavailable")?;
        loop {
            if !guard.config.enabled
                || matches!(
                    guard.state,
                    ThermalState::Monitoring
                        | ThermalState::ThresholdPending
                        | ThermalState::Unavailable
                        | ThermalState::Disabled
                )
            {
                break;
            }
            guard.affected_request = Some(request_id);
            self.held_requests.store(1, Ordering::Relaxed);
            guard = self
                .wake
                .wait(guard)
                .map_err(|_| "thermal state is unavailable")?;
        }
        self.held_requests.store(0, Ordering::Relaxed);
        let waited_ms = wait_started.elapsed().as_secs_f64() * 1000.0;
        if waited_ms > 0.0 {
            if let Some(d) = deadline.as_mut() {
                *d += wait_started.elapsed();
            }
        }
        Ok(waited_ms)
    }

    pub fn sample(&self, reading: SensorReading) {
        self.sample_at(reading, Instant::now());
    }

    fn sample_at(&self, reading: SensorReading, now: Instant) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if !inner.config.enabled {
            inner.state = ThermalState::Disabled;
            self.wake.notify_all();
            return;
        }
        if reading.failed || reading.temperature_c.is_none() {
            self.sensor_failures.fetch_add(1, Ordering::Relaxed);
            inner.consecutive_failures = inner.consecutive_failures.saturating_add(1);
            if inner.ever_monitored && inner.consecutive_failures >= 5 {
                if !matches!(inner.state, ThermalState::Paused | ThermalState::Cooling) {
                    self.begin_pause(&mut inner, now, "sensor_lost");
                }
                inner.state = ThermalState::Paused;
            } else if !inner.ever_monitored {
                inner.state = ThermalState::Unavailable;
            }
            self.wake.notify_all();
            return;
        }
        let temp = reading.temperature_c.unwrap();
        inner.ever_monitored = true;
        inner.consecutive_failures = 0;
        inner.current_temp_c = Some(temp);
        inner.sensor = reading.sensor;
        let prev = inner.maximum_temp_c.unwrap_or(temp);
        inner.maximum_temp_c = Some(prev.max(temp));
        let milli = (temp.max(0.0) * 1000.0) as u64;
        let _ = self
            .maximum_observed_milli_c
            .fetch_max(milli, Ordering::Relaxed);
        if matches!(inner.state, ThermalState::Paused | ThermalState::Cooling) {
            if temp <= inner.config.resume_temp_c {
                if let Some(start) = inner.pause_started.take() {
                    self.paused_ms
                        .fetch_add(start.elapsed().as_millis() as u64, Ordering::Relaxed);
                }
                inner.state = ThermalState::Monitoring;
                inner.pause_start_unix_ms = None;
                inner.affected_request = None;
                inner.hot_since = None;
                self.wake.notify_all();
            } else {
                inner.state = ThermalState::Cooling;
                self.wake.notify_all();
            }
            return;
        }
        if temp >= inner.config.max_temp_c {
            let started = inner.hot_since.get_or_insert(now);
            if now.duration_since(*started) >= Duration::from_secs(inner.config.max_temp_seconds) {
                self.begin_pause(&mut inner, now, "threshold");
                inner.state = ThermalState::Paused;
            } else {
                inner.state = ThermalState::ThresholdPending;
            }
        } else {
            inner.hot_since = None;
            inner.state = ThermalState::Monitoring;
        }
        self.wake.notify_all();
    }

    fn begin_pause(&self, inner: &mut Inner, now: Instant, reason: &str) {
        if inner.pause_started.is_none() {
            inner.pause_started = Some(now);
            inner.pause_start_unix_ms = Some(unix_now_ms());
            self.thermal_pauses.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                "thermal guard entered pause ({reason}); sensor={:?} temperature_c={:?}",
                inner.sensor,
                inner.current_temp_c
            );
        }
    }

    pub fn json(&self) -> Value {
        let inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return Value::Null,
        };
        let hot_duration = inner
            .hot_since
            .map(|s| s.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let hold_duration = inner
            .pause_started
            .map(|s| s.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let threshold_remaining = (inner.config.max_temp_seconds as f64 - hot_duration).max(0.0);
        json!({"state":inner.state.label(),"enabled":inner.config.enabled,"current_temperature_c":inner.current_temp_c,
            "maximum_temperature_c":inner.maximum_temp_c,"resume_temperature_c":inner.config.resume_temp_c,
            "maximum_temperature_threshold_c":inner.config.max_temp_c,"maximum_temperature_seconds":inner.config.max_temp_seconds,
            "hot_duration_seconds":hot_duration,"pause_start_time_ms":inner.pause_start_unix_ms,
            "threshold_remaining_seconds":threshold_remaining,
            "paused_seconds":hold_duration,"affected_request":inner.affected_request,"sensor":inner.sensor,
            "sensor_failures":self.sensor_failures.load(Ordering::Relaxed),"held_requests":self.held_requests.load(Ordering::Relaxed),
            "thermal_pauses":self.thermal_pauses.load(Ordering::Relaxed),"paused_seconds_total":self.paused_ms.load(Ordering::Relaxed) as f64 / 1000.0,
            "maximum_observed_temperature_c":self.maximum_observed_milli_c.load(Ordering::Relaxed) as f64 / 1000.0})
    }
}

fn unix_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard() -> ThermalGuard {
        ThermalGuard::new(ThermalConfig::default())
    }
    #[test]
    fn brief_spike_resets_without_pause() {
        let g = guard();
        g.sample(SensorReading {
            temperature_c: Some(90.0),
            sensor: Some("fake".into()),
            failed: false,
        });
        g.sample(SensorReading {
            temperature_c: Some(89.9),
            sensor: Some("fake".into()),
            failed: false,
        });
        assert_eq!(g.json()["state"], "monitoring");
    }

    #[test]
    fn sustained_threshold_pauses_and_resume_cools() {
        let g = guard();
        let base = Instant::now();
        for i in 0..=5 {
            g.sample_at(
                SensorReading {
                    temperature_c: Some(90.0),
                    sensor: Some("fake".into()),
                    failed: false,
                },
                base + Duration::from_secs(i),
            );
        }
        assert_eq!(g.json()["state"], "paused");
        let g2 = std::sync::Arc::new(g);
        let g3 = std::sync::Arc::clone(&g2);
        let t = std::thread::spawn(move || {
            let mut deadline = None;
            g3.checkpoint(7, &mut deadline).unwrap()
        });
        std::thread::sleep(Duration::from_millis(10));
        g2.sample(SensorReading {
            temperature_c: Some(82.0),
            sensor: Some("fake".into()),
            failed: false,
        });
        assert!(t.join().unwrap() >= 0.0);
        assert_eq!(g2.json()["state"], "monitoring");
    }

    #[test]
    fn five_failed_samples_fail_closed_after_monitoring() {
        let g = guard();
        g.sample(SensorReading {
            temperature_c: Some(70.0),
            sensor: Some("fake".into()),
            failed: false,
        });
        for _ in 0..5 {
            g.sample(SensorReading {
                failed: true,
                ..Default::default()
            });
        }
        assert_eq!(g.json()["state"], "paused");
    }

    #[test]
    fn disabled_does_not_hold() {
        let g = ThermalGuard::new(ThermalConfig {
            enabled: false,
            ..Default::default()
        });
        let mut deadline = None;
        assert!(g.checkpoint(1, &mut deadline).unwrap() < 100.0);
        assert_eq!(g.json()["state"], "disabled");
    }

    #[test]
    fn settings_change_does_not_release_hot_pause() {
        let g = guard();
        let base = Instant::now();
        for i in 0..=5 {
            g.sample_at(
                SensorReading {
                    temperature_c: Some(90.0),
                    sensor: Some("fake".into()),
                    failed: false,
                },
                base + Duration::from_secs(i),
            );
        }
        assert_eq!(g.json()["state"], "paused");
        g.update_config(ThermalConfig {
            max_temp_c: 92.0,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(g.json()["state"], "cooling");
        g.sample(SensorReading {
            temperature_c: Some(81.0),
            sensor: Some("fake".into()),
            failed: false,
        });
        assert_eq!(g.json()["state"], "monitoring");
    }
}
