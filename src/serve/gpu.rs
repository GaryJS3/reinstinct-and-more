//! Linux GPU inventory and the narrow protocol used by the privileged helper.
//!
//! Inventory is deliberately best-effort: PCI devices are discovered first,
//! and every optional sysfs file is read independently.  A broken hwmon node
//! therefore produces `null` for that field rather than taking down the API.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(test)]
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

#[derive(Clone, Debug, Default)]
pub struct SensorReading {
    pub temperature_c: Option<f64>,
    pub sensor: Option<String>,
    pub failed: bool,
}

#[derive(Clone, Debug)]
pub struct GpuInventory {
    root: PathBuf,
}

impl GpuInventory {
    pub fn host() -> Self {
        Self {
            root: PathBuf::from("/sys"),
        }
    }

    #[cfg(test)]
    fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn json(&self) -> Value {
        let (virtualization, gpus) = self.scan();
        json!({"virtualization": virtualization, "gpus": gpus})
    }

    pub fn inference_sensor(&self) -> SensorReading {
        let (_, gpus) = self.scan();
        // PCI ordering is not a reliable HIP ordinal on multi-GPU hosts. Use
        // the hottest valid AMD reading so the guard cannot silently monitor
        // a cooler sibling while inference heats another card.
        gpus.values()
            .filter(|gpu| gpu.get("vendor").and_then(Value::as_str) == Some("AMD"))
            .map(sensor_from_json)
            .filter(|reading| !reading.failed)
            .max_by(|a, b| {
                a.temperature_c
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(&b.temperature_c.unwrap_or(f64::NEG_INFINITY))
            })
            .unwrap_or(SensorReading {
                failed: true,
                ..Default::default()
            })
    }

    pub fn scan(&self) -> (Value, BTreeMap<String, Value>) {
        let virtualization = virtualization_evidence(&self.root);
        let mut devices = BTreeMap::new();
        let pci_root = self.root.join("bus/pci/devices");
        let mut hip_index = 0i64;
        let entries = fs::read_dir(&pci_root).ok().map(|entries| {
            let mut entries: Vec<_> = entries.flatten().collect();
            entries.sort_by_key(|entry| entry.file_name());
            entries
        });
        if let Some(entries) = entries {
            for entry in entries {
                let raw_address = entry.file_name().to_string_lossy().to_ascii_lowercase();
                let address = if raw_address.contains('_') {
                    let parts: Vec<_> = raw_address.split('_').collect();
                    if parts.len() == 4 {
                        format!("{}:{}:{}.{}", parts[0], parts[1], parts[2], parts[3])
                    } else {
                        raw_address.clone()
                    }
                } else {
                    raw_address.clone()
                };
                if !is_pci_address(&address) {
                    continue;
                }
                let path = entry.path();
                let vendor_id = read_trim(&path.join("vendor"));
                let device_id = read_trim(&path.join("device"));
                let class = read_trim(&path.join("class"));
                let drm = drm_mapping(&path);
                let driver = path
                    .join("driver")
                    .read_link()
                    .ok()
                    .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()));
                let is_gpu = class.as_deref().is_some_and(|c| c.starts_with("0x03"))
                    || !drm.is_empty()
                    || path.join("hwmon").exists();
                if !is_gpu {
                    continue;
                }
                let vendor = vendor_id.as_deref().map(vendor_name);
                let hip = if vendor == Some("AMD") {
                    let i = hip_index;
                    hip_index += 1;
                    Some(i)
                } else {
                    None
                };
                let hwmon = hwmon_json(&path);
                let memory = memory_json(&path, &drm);
                let pcie = pcie_json(&path);
                let controls = json!({
                    "helper_installed": helper_installed(),
                    "power_limit_writable": hwmon.get("power").and_then(|p| p.get("writable")).and_then(Value::as_bool).unwrap_or(false),
                    "tuning_writable": is_writable(&path.join("pp_od_clk_voltage")) || is_writable(&path.join("power_dpm_force_performance_level")),
                    "reset_supported": vendor == Some("AMD")
                });
                devices.insert(address.clone(), json!({
                    "pci_address": address,
                    "vendor_id": vendor_id,
                    "device_id": device_id,
                    "vendor": vendor,
                    "name": read_trim(&path.join("label")).or_else(|| driver.as_ref().map(|d| d.to_string())),
                    "driver": driver,
                    "hip_device_index": hip,
                    "drm": if drm.is_empty() { Value::Null } else { json!(drm) },
                    "vram": memory,
                    "pcie": pcie,
                    "power": hwmon.get("power").cloned().unwrap_or(Value::Null),
                    "temperatures": hwmon.get("temperatures").cloned().unwrap_or_else(|| json!([])),
                    "fan": hwmon.get("fan").cloned().unwrap_or(Value::Null),
                    "utilization": hwmon.get("utilization").cloned().unwrap_or(Value::Null),
                    "clocks": hwmon.get("clocks").cloned().unwrap_or(Value::Null),
                    "controls": controls
                }));
            }
        }
        (virtualization, devices)
    }
}

fn read_trim(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn parse_u64(path: &Path) -> Option<u64> {
    read_trim(path)?.parse().ok()
}
fn parse_f64(path: &Path) -> Option<f64> {
    read_trim(path)?.parse().ok()
}
fn is_writable(path: &Path) -> bool {
    path.exists()
        && fs::metadata(path)
            .map(|m| !m.permissions().readonly())
            .unwrap_or(false)
}

fn vendor_name(id: &str) -> &'static str {
    match id.trim_start_matches("0x").to_ascii_lowercase().as_str() {
        "1002" => "AMD",
        "10de" => "NVIDIA",
        "8086" => "Intel",
        _ => "Unknown",
    }
}

fn is_pci_address(s: &str) -> bool {
    let parts: Vec<_> = s.split(':').collect();
    parts.len() == 3
        && parts[0].len() == 4
        && parts[1].len() == 2
        && parts[2].split('.').count() == 2
        && parts[2].split('.').next().is_some_and(|x| x.len() == 2)
        && parts[2].split('.').nth(1).is_some_and(|x| x.len() == 1)
}

fn drm_mapping(path: &Path) -> Vec<String> {
    fs::read_dir(path.join("drm"))
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|s| s.starts_with("card") || s.starts_with("renderD"))
        .collect()
}

fn memory_json(path: &Path, drm: &[String]) -> Value {
    let drm_root = drm
        .iter()
        .find(|s| s.starts_with("card"))
        .map(|s| path.join("drm").join(s));
    let root = drm_root.as_deref().unwrap_or(path);
    let total = parse_u64(&root.join("device/mem_info_vram_total"))
        .or_else(|| parse_u64(&path.join("mem_info_vram_total")));
    let used = parse_u64(&root.join("device/mem_info_vram_used"))
        .or_else(|| parse_u64(&path.join("mem_info_vram_used")));
    json!({"total_bytes": total, "used_bytes": used, "free_bytes": total.zip(used).map(|(t,u)| t.saturating_sub(u))})
}

fn parse_link_speed(value: Option<String>) -> Option<(u32, f64)> {
    let lower = value?.to_ascii_lowercase();
    let generation = if lower.contains("2.5") {
        1
    } else if lower.contains("5") {
        2
    } else if lower.contains("8") {
        3
    } else if lower.contains("16") {
        4
    } else if lower.contains("32") {
        5
    } else if lower.contains("64") {
        6
    } else {
        return None;
    };
    let gt = match generation {
        1 => 2.5,
        2 => 5.0,
        3 => 8.0,
        4 => 16.0,
        5 => 32.0,
        _ => 64.0,
    };
    Some((generation, gt))
}

fn pcie_json(path: &Path) -> Value {
    let current = parse_link_speed(read_trim(&path.join("current_link_speed")));
    let maximum = parse_link_speed(read_trim(&path.join("max_link_speed")));
    let current_lanes = read_trim(&path.join("current_link_width"))
        .and_then(|s| s.trim_start_matches('x').parse::<u32>().ok());
    let max_lanes = read_trim(&path.join("max_link_width"))
        .and_then(|s| s.trim_start_matches('x').parse::<u32>().ok());
    let bandwidth = |link: Option<(u32, f64)>, lanes: Option<u32>| {
        link.zip(lanes)
            .map(|((_, gt), l)| gt * 128.0 / 130.0 * l as f64)
    };
    json!({"current_generation":current.map(|x| x.0),"maximum_generation":maximum.map(|x| x.0),
        "current_lanes":current_lanes,"maximum_lanes":max_lanes,
        "current_speed":read_trim(&path.join("current_link_speed")),"maximum_speed":read_trim(&path.join("max_link_speed")),
        "theoretical_gbps_per_direction":bandwidth(current,current_lanes),
        "maximum_theoretical_gbps_per_direction":bandwidth(maximum,max_lanes)})
}

fn hwmon_json(path: &Path) -> Value {
    let mut temperatures = Vec::new();
    let mut fan = Value::Null;
    let mut power = Value::Null;
    let mut utilization = Value::Null;
    let mut clocks = Value::Null;
    let mut hwmon_paths = Vec::new();
    if let Some(entries) = fs::read_dir(path.join("hwmon")).ok() {
        hwmon_paths.extend(entries.flatten().map(|e| e.path()));
    }
    if let Some(entries) = fs::read_dir(path.join("drm")).ok() {
        for card in entries
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("card"))
        {
            if let Some(hw) = fs::read_dir(card.path().join("device/hwmon")).ok() {
                hwmon_paths.extend(hw.flatten().map(|e| e.path()));
            }
        }
    }
    for h in hwmon_paths {
        let hw_name = h
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "hwmon".into());
        let name = read_trim(&h.join("name")).unwrap_or(hw_name);
        for entry in fs::read_dir(&h).ok().into_iter().flatten().flatten() {
            let filename = entry.file_name().to_string_lossy().into_owned();
            let p = entry.path();
            if let Some(prefix) = filename.strip_suffix("_input") {
                if filename.starts_with("temp") {
                    if let Some(raw) = parse_f64(&p) {
                        let label = read_trim(&h.join(format!("{prefix}_label")))
                            .unwrap_or_else(|| prefix.to_string());
                        temperatures.push(json!({"sensor":format!("{name}:{label}"),"label":label,"temperature_c":raw / 1000.0}));
                    }
                } else if filename.starts_with("fan") {
                    fan = json!({"rpm":parse_f64(&p),"sensor":name});
                } else if filename == "power1_input" {
                    if let Some(raw) = parse_f64(&p) {
                        power = json!({"watts":raw / 1_000_000.0,"limit_watts":parse_f64(&h.join("power1_cap")).map(|v| v / 1_000_000.0),"minimum_watts":parse_f64(&h.join("power1_cap_min")).map(|v| v / 1_000_000.0),"maximum_watts":parse_f64(&h.join("power1_cap_max")).map(|v| v / 1_000_000.0),"writable":is_writable(&h.join("power1_cap")),"sensor":name});
                    }
                }
            }
            if filename == "gpu_busy_percent" {
                utilization = json!({"percent":parse_f64(&p)});
            }
            if filename.contains("clock") && filename.ends_with("_input") {
                clocks = json!({"mhz":parse_f64(&p).map(|v| v / 1_000_000.0)});
            }
        }
    }
    json!({"temperatures":temperatures,"fan":fan,"power":power,"utilization":utilization,"clocks":clocks})
}

fn sensor_from_json(gpu: &Value) -> SensorReading {
    let temps = gpu
        .get("temperatures")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let chosen = temps
        .iter()
        .filter_map(|t| Some((t.get("temperature_c")?.as_f64()?, t)))
        .find(|(_, t)| {
            t.get("label").and_then(Value::as_str).is_some_and(|label| {
                let label = label.to_ascii_lowercase();
                label.contains("junction") || label.contains("hotspot")
            })
        })
        .or_else(|| {
            temps
                .iter()
                .filter_map(|t| Some((t.get("temperature_c")?.as_f64()?, t)))
                .max_by(|a, b| a.0.total_cmp(&b.0))
        });
    chosen
        .map(|(value, t)| SensorReading {
            temperature_c: Some(value),
            sensor: t.get("sensor").and_then(Value::as_str).map(str::to_string),
            failed: false,
        })
        .unwrap_or(SensorReading {
            failed: true,
            ..Default::default()
        })
}

fn virtualization_evidence(root: &Path) -> Value {
    let mut evidence = Vec::new();
    for path in [
        "class/dmi/id/product_name",
        "class/dmi/id/sys_vendor",
        "class/dmi/id/board_vendor",
    ] {
        if let Some(v) = read_trim(&root.join(path)) {
            evidence.push(format!("{path}={v}"));
        }
    }
    let product = evidence.join(" ").to_ascii_lowercase();
    let hypervisor = root.join("hypervisor/type").exists() || root.join("module/kvm").exists();
    if hypervisor {
        evidence.push("Linux hypervisor/KVM evidence".into());
    }
    let virtual_gpu = product.contains("virtio")
        || product.contains("qemu")
        || product.contains("virtual")
        || product.contains("vmware")
        || product.contains("microsoft corporation");
    let mode = if virtual_gpu && root.join("bus/pci/devices").exists() {
        "vm_likely_passthrough"
    } else if virtual_gpu {
        "vm_virtual_gpu"
    } else if hypervisor {
        "vm_likely_passthrough"
    } else if evidence.is_empty() {
        "unknown"
    } else {
        "bare_metal"
    };
    let confidence = if mode == "unknown" {
        0.0
    } else if virtual_gpu || hypervisor {
        0.9
    } else {
        0.65
    };
    json!({"mode":mode,"confidence":confidence,"evidence":evidence})
}

pub fn helper_installed() -> bool {
    std::env::var_os("REINSTINCT_GPU_HELPER_SOCKET")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/run/reinstinct-gpu-helper.sock"))
        .exists()
}

pub fn control(pci: &str, action: &str, payload: &Value) -> Result<Value, String> {
    if !is_pci_address(pci) {
        return Err("pci_address must be an exact PCI address such as 0000:03:00.0".into());
    }
    #[cfg(unix)]
    {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::UnixStream;
        let socket = std::env::var_os("REINSTINCT_GPU_HELPER_SOCKET")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/run/reinstinct-gpu-helper.sock"));
        let mut stream = UnixStream::connect(&socket)
            .map_err(|e| format!("GPU helper unavailable at {}: {e}", socket.display()))?;
        let request = json!({"version":1,"pci_address":pci,"action":action,"payload":payload});
        writeln!(stream, "{}", request).map_err(|e| e.to_string())?;
        let mut line = String::new();
        BufReader::new(stream)
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        let reply: Value =
            serde_json::from_str(&line).map_err(|e| format!("invalid helper response: {e}"))?;
        if reply.get("ok").and_then(Value::as_bool) == Some(true) {
            Ok(reply)
        } else {
            Err(reply
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("GPU helper rejected operation")
                .into())
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (pci, action, payload);
        Err("GPU helper is supported only on Unix hosts".into())
    }
}

#[cfg(test)]
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{create_dir_all, write};

    fn fake_tree() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "reinstinct-gpu-test-{}-{}",
            now_ms(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let pci = root.join("bus/pci/devices/0000_03_00_0");
        create_dir_all(pci.join("drm/card0/device/hwmon/hwmon0")).unwrap();
        write(pci.join("vendor"), "0x1002\n").unwrap();
        write(pci.join("device"), "0x66a1\n").unwrap();
        write(pci.join("class"), "0x030000\n").unwrap();
        write(pci.join("current_link_speed"), "8.0 GT/s PCIe\n").unwrap();
        write(pci.join("max_link_speed"), "16.0 GT/s PCIe\n").unwrap();
        write(pci.join("current_link_width"), "x8\n").unwrap();
        write(pci.join("max_link_width"), "x16\n").unwrap();
        write(
            pci.join("drm/card0/device/mem_info_vram_total"),
            "17179869184\n",
        )
        .unwrap();
        write(pci.join("drm/card0/device/mem_info_vram_used"), "1024\n").unwrap();
        write(pci.join("drm/card0/device/hwmon/hwmon0/name"), "amdgpu\n").unwrap();
        write(
            pci.join("drm/card0/device/hwmon/hwmon0/temp1_input"),
            "90000\n",
        )
        .unwrap();
        write(
            pci.join("drm/card0/device/hwmon/hwmon0/temp1_label"),
            "junction\n",
        )
        .unwrap();
        let second = root.join("bus/pci/devices/0000_04_00_0");
        create_dir_all(&second).unwrap();
        write(second.join("vendor"), "0x8086\n").unwrap();
        write(second.join("device"), "0x56a0\n").unwrap();
        write(second.join("class"), "0x030000\n").unwrap();
        root
    }

    #[test]
    fn inventory_is_keyed_by_pci_and_computes_theoretical_bandwidth() {
        let root = fake_tree();
        let inv = GpuInventory::new(root.clone());
        let json = inv.json();
        let gpu = &json["gpus"]["0000:03:00.0"];
        assert_eq!(gpu["vendor_id"], "0x1002");
        assert_eq!(gpu["pcie"]["current_generation"], 3);
        assert_eq!(gpu["pcie"]["current_lanes"], 8);
        assert!(
            gpu["pcie"]["theoretical_gbps_per_direction"]
                .as_f64()
                .unwrap()
                > 7.0
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn missing_sensor_is_a_failed_read_not_an_inventory_failure() {
        let root = fake_tree();
        let pci = root.join("bus/pci/devices/0000_03_00_0");
        fs::remove_file(pci.join("drm/card0/device/hwmon/hwmon0/temp1_input")).unwrap();
        assert!(GpuInventory::new(root.clone()).inference_sensor().failed);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn virtualization_evidence_and_non_amd_read_only_device_are_retained() {
        let root = fake_tree();
        create_dir_all(root.join("class/dmi/id")).unwrap();
        write(
            root.join("class/dmi/id/product_name"),
            "QEMU Virtual Machine\n",
        )
        .unwrap();
        let json = GpuInventory::new(root.clone()).json();
        assert_eq!(json["virtualization"]["mode"], "vm_likely_passthrough");
        assert_eq!(json["gpus"].as_object().unwrap().len(), 2);
        assert_eq!(json["gpus"]["0000:04:00.0"]["vendor"], "Intel");
        assert_eq!(
            json["gpus"]["0000:04:00.0"]["controls"]["power_limit_writable"],
            false
        );
        let _ = fs::remove_dir_all(root);
    }
}
