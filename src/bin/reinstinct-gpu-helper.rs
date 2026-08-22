//! Root-owned, deliberately boring GPU mutation helper.
//!
//! Install this binary root-owned with a mode-660 Unix socket owned by the
//! ReInstinct service group.  It accepts one JSON request per line and never
//! executes shell commands.  The server performs the user-facing validation;
//! this process repeats identity and range checks at the privilege boundary.

#[cfg(unix)]
mod unix_helper {
    use serde_json::{json, Value};
    use std::fs;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::{Path, PathBuf};

    const SYSFS: &str = "/sys/bus/pci/devices";

    pub fn run() -> Result<(), String> {
        let socket = std::env::args()
            .skip(1)
            .collect::<Vec<_>>()
            .first()
            .cloned()
            .unwrap_or_else(|| "/run/reinstinct-gpu-helper.sock".into());
        let socket = PathBuf::from(socket);
        if socket.exists() {
            fs::remove_file(&socket).map_err(|e| format!("remove old socket: {e}"))?;
        }
        let listener =
            UnixListener::bind(&socket).map_err(|e| format!("bind {}: {e}", socket.display()))?;
        for stream in listener.incoming() {
            match stream {
                Ok(mut stream) => {
                    let reply = handle(&mut stream);
                    let _ = writeln!(stream, "{}", reply);
                }
                Err(e) => eprintln!("helper accept error: {e}"),
            }
        }
        Ok(())
    }

    fn handle(stream: &mut UnixStream) -> Value {
        let mut line = String::new();
        if let Err(e) = BufReader::new(&mut *stream).read_line(&mut line) {
            return json!({"ok":false,"error":e.to_string()});
        }
        let request: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => return json!({"ok":false,"error":format!("invalid JSON: {e}")}),
        };
        if request["version"].as_u64() != Some(1) {
            return json!({"ok":false,"error":"unsupported helper protocol version"});
        }
        let Some(pci) = request["pci_address"].as_str() else {
            return json!({"ok":false,"error":"pci_address is required"});
        };
        let Some(action) = request["action"].as_str() else {
            return json!({"ok":false,"error":"action is required"});
        };
        if !exact_pci(pci) || pci.contains("..") {
            return json!({"ok":false,"error":"invalid exact PCI address"});
        }
        if !matches!(action, "set-power-limit" | "set-tuning" | "reset") {
            return json!({"ok":false,"error":"action is not allowlisted"});
        }
        let device = Path::new(SYSFS).join(pci);
        if !device.is_dir() {
            return json!({"ok":false,"error":"PCI device disappeared"});
        }
        let vendor = fs::read_to_string(device.join("vendor")).unwrap_or_default();
        if vendor.trim().to_ascii_lowercase() != "0x1002" {
            return json!({"ok":false,"error":"only AMD devices are permitted"});
        }
        let payload = request
            .get("payload")
            .cloned()
            .unwrap_or(Value::Object(Default::default()));
        let paths = mutation_paths(&device, action);
        if paths.is_empty() {
            return json!({"ok":false,"error":"driver reports no writable operation"});
        }
        // Roll back to the state immediately before this request. Reusing the
        // first-ever snapshot could unexpectedly undo later successful edits.
        let snapshot: Vec<_> = paths
            .iter()
            .filter_map(|p| Some((p.clone(), fs::read_to_string(p).ok()?)))
            .collect();
        let result = match action {
            "set-power-limit" => set_power(&paths, &payload),
            "set-tuning" => set_tuning(&paths, &payload),
            "reset" => fs::write(&paths[0], "1\n").map_err(|e| e.to_string()),
            _ => unreachable!(),
        };
        if let Err(error) = result {
            for (path, value) in snapshot {
                let _ = fs::write(path, value);
            }
            return json!({"ok":false,"error":format!("mutation rolled back: {error}")});
        }
        let readback = paths
            .iter()
            .filter_map(|p| {
                Some((
                    p.file_name()?.to_string_lossy().into_owned(),
                    Value::String(fs::read_to_string(p).ok()?),
                ))
            })
            .collect::<serde_json::Map<_, _>>();
        json!({"ok":true,"pci_address":pci,"action":action,"readback":readback})
    }

    fn mutation_paths(device: &Path, action: &str) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        let mut roots = vec![device.to_path_buf()];
        if let Ok(entries) = fs::read_dir(device.join("hwmon")) {
            roots.extend(entries.flatten().map(|e| e.path()));
        }
        for root in roots {
            let path = match action {
                "set-power-limit" => root.join("power1_cap"),
                "set-tuning" => root.join("power_dpm_force_performance_level"),
                "reset" => root.join("reset"),
                _ => continue,
            };
            if path.exists()
                && !fs::metadata(&path)
                    .map(|m| m.permissions().readonly())
                    .unwrap_or(true)
            {
                paths.push(path);
            }
        }
        paths
    }

    fn set_power(paths: &[PathBuf], payload: &Value) -> Result<(), String> {
        let watts = payload["watts"]
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("watts must be finite")?;
        let path = &paths[0];
        let parent = path.parent().ok_or("power control path has no parent")?;
        let min = fs::read_to_string(parent.join("power1_cap_min"))
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .unwrap_or(0.0);
        let max = fs::read_to_string(parent.join("power1_cap_max"))
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok());
        let microwatts = watts * 1_000_000.0;
        if microwatts < min || max.is_some_and(|v| microwatts > v) {
            return Err("watts outside driver-reported range".into());
        }
        let requested = microwatts as u64;
        fs::write(path, format!("{requested}\n")).map_err(|e| e.to_string())?;
        let applied = fs::read_to_string(path)
            .map_err(|e| format!("power readback failed: {e}"))?
            .trim()
            .parse::<u64>()
            .map_err(|e| format!("invalid power readback: {e}"))?;
        if applied != requested {
            return Err(format!(
                "power readback mismatch: requested {requested}, got {applied}"
            ));
        }
        Ok(())
    }

    fn set_tuning(paths: &[PathBuf], payload: &Value) -> Result<(), String> {
        let level = payload["performance_level"]
            .as_str()
            .ok_or("tuning requires performance_level")?;
        if !matches!(level, "auto" | "low" | "high" | "manual") {
            return Err("performance_level is not allowlisted".into());
        }
        fs::write(&paths[0], format!("{level}\n")).map_err(|e| e.to_string())?;
        let applied =
            fs::read_to_string(&paths[0]).map_err(|e| format!("tuning readback failed: {e}"))?;
        if applied.trim() != level {
            return Err(format!(
                "tuning readback mismatch: requested {level}, got {}",
                applied.trim()
            ));
        }
        Ok(())
    }

    fn exact_pci(s: &str) -> bool {
        let p: Vec<_> = s.split(':').collect();
        if p.len() != 3 {
            return false;
        }
        p[0].len() == 4
            && p[1].len() == 2
            && p[2].split('.').count() == 2
            && p[2].split('.').next().is_some_and(|x| x.len() == 2)
            && p[2].split('.').nth(1).is_some_and(|x| x.len() == 1)
    }
}

#[cfg(unix)]
fn main() {
    if let Err(e) = unix_helper::run() {
        eprintln!("reinstinct-gpu-helper: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(unix))]
fn main() {
    eprintln!("reinstinct-gpu-helper is supported only on Unix hosts");
    std::process::exit(1);
}
