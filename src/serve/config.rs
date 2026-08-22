//! Configuration for the HTTP server.
//!
//! The command line remains the source of truth for deployments that do not
//! use a file.  When `--config` is supplied, this module merges the JSON file
//! with explicit command-line overrides and keeps the effective configuration
//! in a small, serialisable value that the dashboard can validate and reload.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

const KEYS: &[&str] = &[
    "big",
    "model_dir",
    "big_drafter",
    "small",
    "embed",
    "big_port",
    "small_port",
    "embed_port",
    "max_seq",
    "mmproj",
    "mtmd_bridge",
    "vision_threads",
    "vision_min_tokens",
    "vision_max_tokens",
    "cpu_vision",
];

#[derive(Clone, Debug)]
pub struct ServeOverrides {
    pub config: Option<PathBuf>,
    pub big: Option<PathBuf>,
    pub model_dir: Option<PathBuf>,
    pub big_drafter: Option<PathBuf>,
    pub small: Option<PathBuf>,
    pub embed: Option<PathBuf>,
    pub big_port: Option<u16>,
    pub small_port: Option<u16>,
    pub embed_port: Option<u16>,
    pub max_seq: Option<usize>,
    pub mmproj: Option<PathBuf>,
    pub mtmd_bridge: Option<PathBuf>,
    pub vision_threads: Option<i32>,
    pub vision_min_tokens: Option<i32>,
    pub vision_max_tokens: Option<i32>,
    pub cpu_vision: bool,
}

#[derive(Clone, Debug)]
pub struct ServeConfig {
    pub big: PathBuf,
    pub model_dir: PathBuf,
    pub big_drafter: Option<PathBuf>,
    pub small: Option<PathBuf>,
    pub embed: Option<PathBuf>,
    pub big_port: u16,
    pub small_port: u16,
    pub embed_port: u16,
    pub max_seq: usize,
    pub mmproj: Option<PathBuf>,
    pub mtmd_bridge: Option<PathBuf>,
    pub vision_threads: i32,
    pub vision_min_tokens: i32,
    pub vision_max_tokens: i32,
    pub cpu_vision: bool,
    pub config_path: Option<PathBuf>,
    pub cli_locked: BTreeSet<String>,
}

impl ServeConfig {
    pub fn from_overrides(o: ServeOverrides) -> Result<Self, String> {
        let config_path = o.config.as_ref().map(|p| absolutize(p, None));
        let mut value = if let Some(path) = config_path.as_ref() {
            let text = fs::read_to_string(path)
                .map_err(|e| format!("cannot read config {}: {e}", path.display()))?;
            let value: Value = serde_json::from_str(&text)
                .map_err(|e| format!("invalid JSON in {}: {e}", path.display()))?;
            validate_keys(&value)?;
            value
        } else {
            Value::Object(Map::new())
        };
        let mut locked = BTreeSet::new();
        macro_rules! override_field {
            ($name:literal, $v:expr) => {
                if let Some(v) = $v {
                    value[$name] = path_or_value(v, config_path.as_deref());
                    locked.insert($name.into());
                }
            };
        }
        override_field!("big", o.big);
        override_field!("model_dir", o.model_dir);
        override_field!("big_drafter", o.big_drafter);
        override_field!("small", o.small);
        override_field!("embed", o.embed);
        if let Some(v) = o.big_port {
            value["big_port"] = Value::from(v);
            locked.insert("big_port".into());
        }
        if let Some(v) = o.small_port {
            value["small_port"] = Value::from(v);
            locked.insert("small_port".into());
        }
        if let Some(v) = o.embed_port {
            value["embed_port"] = Value::from(v);
            locked.insert("embed_port".into());
        }
        if let Some(v) = o.max_seq {
            value["max_seq"] = Value::from(v as u64);
            locked.insert("max_seq".into());
        }
        override_field!("mmproj", o.mmproj);
        override_field!("mtmd_bridge", o.mtmd_bridge);
        if let Some(v) = o.vision_threads {
            value["vision_threads"] = Value::from(v);
            locked.insert("vision_threads".into());
        }
        if let Some(v) = o.vision_min_tokens {
            value["vision_min_tokens"] = Value::from(v);
            locked.insert("vision_min_tokens".into());
        }
        if let Some(v) = o.vision_max_tokens {
            value["vision_max_tokens"] = Value::from(v);
            locked.insert("vision_max_tokens".into());
        }
        if o.cpu_vision {
            value["cpu_vision"] = Value::Bool(true);
            locked.insert("cpu_vision".into());
        }
        let mut cfg = Self::from_value(&value, config_path.clone())?;
        cfg.cli_locked = locked;
        Ok(cfg)
    }

    pub fn from_value(value: &Value, config_path: Option<PathBuf>) -> Result<Self, String> {
        validate_keys(value)?;
        let base = config_path.as_deref().and_then(Path::parent);
        let path = |key: &str| -> Result<Option<PathBuf>, String> {
            match value.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(s)) => Ok(Some(absolutize(Path::new(s), base))),
                Some(_) => Err(format!("{key} must be a string or null")),
            }
        };
        let required = |key: &str| -> Result<PathBuf, String> {
            path(key)?.ok_or_else(|| format!("missing required config field: {key}"))
        };
        let number = |key: &str, default: u64| -> Result<u64, String> {
            match value.get(key) {
                None => Ok(default),
                Some(v) => v
                    .as_u64()
                    .ok_or_else(|| format!("{key} must be a non-negative integer")),
            }
        };
        let signed = |key: &str, default: i32| -> Result<i32, String> {
            match value.get(key) {
                None => Ok(default),
                Some(v) => v
                    .as_i64()
                    .and_then(|n| i32::try_from(n).ok())
                    .ok_or_else(|| format!("{key} must be an integer")),
            }
        };
        let big = required("big")?;
        let model_dir = path("model_dir")?
            .unwrap_or_else(|| big.parent().unwrap_or(Path::new(".")).to_path_buf());
        let cfg = Self {
            big,
            model_dir,
            big_drafter: path("big_drafter")?,
            small: path("small")?,
            embed: path("embed")?,
            big_port: u16::try_from(number("big_port", 8080)?)
                .map_err(|_| "big_port must be 1..65535".to_string())?,
            small_port: u16::try_from(number("small_port", 8081)?)
                .map_err(|_| "small_port must be 1..65535".to_string())?,
            embed_port: u16::try_from(number("embed_port", 8082)?)
                .map_err(|_| "embed_port must be 1..65535".to_string())?,
            max_seq: usize::try_from(number("max_seq", 4096)?)
                .map_err(|_| "max_seq is too large".to_string())?,
            mmproj: path("mmproj")?,
            mtmd_bridge: path("mtmd_bridge")?,
            vision_threads: signed("vision_threads", 8)?,
            vision_min_tokens: signed("vision_min_tokens", -1)?,
            vision_max_tokens: signed("vision_max_tokens", -1)?,
            cpu_vision: value
                .get("cpu_vision")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            config_path,
            cli_locked: BTreeSet::new(),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.big_port == 0 || self.small_port == 0 || self.embed_port == 0 {
            return Err("ports must be between 1 and 65535".into());
        }
        if self.max_seq == 0 {
            return Err("max_seq must be positive".into());
        }
        match (&self.mmproj, &self.mtmd_bridge) {
            (None, None) => {},
            (Some(_), Some(_)) if self.vision_threads > 0 && self.vision_min_tokens != 0 && self.vision_max_tokens != 0 && !(self.vision_min_tokens > 0 && self.vision_max_tokens > 0 && self.vision_min_tokens > self.vision_max_tokens) => {},
            (Some(_), None) | (None, Some(_)) => return Err("vision requires both mmproj and mtmd_bridge".into()),
            _ => return Err("vision_threads must be positive and token limits must be -1 or positive with minimum <= maximum".into()),
        }
        Ok(())
    }

    pub fn json_value(&self) -> Value {
        let path = |p: &PathBuf| Value::String(p.to_string_lossy().into_owned());
        serde_json::json!({"big":path(&self.big),"model_dir":path(&self.model_dir),"big_drafter":self.big_drafter.as_ref().map(path),"small":self.small.as_ref().map(path),"embed":self.embed.as_ref().map(path),"big_port":self.big_port,"small_port":self.small_port,"embed_port":self.embed_port,"max_seq":self.max_seq,"mmproj":self.mmproj.as_ref().map(path),"mtmd_bridge":self.mtmd_bridge.as_ref().map(path),"vision_threads":self.vision_threads,"vision_min_tokens":self.vision_min_tokens,"vision_max_tokens":self.vision_max_tokens,"cpu_vision":self.cpu_vision})
    }

    pub fn apply_update(&self, update: &Value) -> Result<Self, String> {
        validate_keys(update)?;
        let mut merged = self.json_value();
        let Some(dst) = merged.as_object_mut() else {
            unreachable!()
        };
        let Some(src) = update.as_object() else {
            return Err("config update must be a JSON object".into());
        };
        for (k, v) in src {
            if self.cli_locked.contains(k) && dst.get(k) != Some(v) {
                return Err(format!(
                    "{k} is locked by an explicit command-line argument"
                ));
            }
            dst.insert(k.clone(), v.clone());
        }
        let mut next = Self::from_value(&merged, self.config_path.clone())?;
        next.cli_locked = self.cli_locked.clone();
        Ok(next)
    }

    pub fn restart_required_fields(&self, next: &Self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.big_port != next.big_port {
            out.push("big_port");
        }
        if self.small_port != next.small_port {
            out.push("small_port");
        }
        if self.embed_port != next.embed_port {
            out.push("embed_port");
        }
        out
    }

    pub fn persist(&self) -> Result<(), String> {
        let Some(path) = self.config_path.as_ref() else {
            return Err("server was not started with --config".into());
        };
        let tmp = path.with_extension("json.tmp");
        let text =
            serde_json::to_string_pretty(&self.json_value()).map_err(|e| e.to_string())? + "\n";
        fs::write(&tmp, text).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
        fs::rename(&tmp, path).map_err(|e| format!("cannot replace {}: {e}", path.display()))
    }
}

fn validate_keys(value: &Value) -> Result<(), String> {
    let Some(obj) = value.as_object() else {
        return Err("config must be a JSON object".into());
    };
    if let Some(key) = obj.keys().find(|k| !KEYS.contains(&k.as_str())) {
        return Err(format!("unknown config field: {key}"));
    }
    Ok(())
}

fn absolutize(path: &Path, base: Option<&Path>) -> PathBuf {
    let p = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.map(|b| b.join(path))
            .unwrap_or_else(|| path.to_path_buf())
    };
    fs::canonicalize(&p).unwrap_or(p)
}

fn path_or_value(path: PathBuf, config_path: Option<&Path>) -> Value {
    Value::String(
        absolutize(&path, config_path.and_then(Path::parent))
            .to_string_lossy()
            .into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Value {
        serde_json::json!({"big":"models/main.gguf","max_seq":8192,"vision_threads":4,"cpu_vision":false})
    }

    #[test]
    fn defaults_and_relative_paths_are_applied() {
        let cfg = ServeConfig::from_value(
            &base(),
            Some(PathBuf::from("C:/etc/reinstinct/server.json")),
        )
        .unwrap();
        assert_eq!(cfg.big_port, 8080);
        assert_eq!(cfg.max_seq, 8192);
        assert!(cfg.big.ends_with(Path::new("models/main.gguf")));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let value = serde_json::json!({"big":"model.gguf","batch_size":8});
        assert!(
            ServeConfig::from_value(&value, None)
                .unwrap_err()
                .contains("unknown config field")
        );
    }

    #[test]
    fn cli_locked_and_restart_fields_are_reported() {
        let mut cfg = ServeConfig::from_value(&base(), None).unwrap();
        cfg.cli_locked.insert("max_seq".into());
        let mut update = base();
        update["max_seq"] = Value::from(16384u64);
        assert!(cfg.apply_update(&update).unwrap_err().contains("locked"));
        let mut next = cfg.clone();
        next.big_port = 9000;
        assert_eq!(cfg.restart_required_fields(&next), vec!["big_port"]);
    }
}
