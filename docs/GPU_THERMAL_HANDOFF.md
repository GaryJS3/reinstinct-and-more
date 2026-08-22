# GPU dashboard, management, and thermal interlock handoff

## Implemented capability

- `GET /api/gpus` scans PCI devices below `/sys`, DRM links, hwmon, PCIe
  link files, and optional KFD/HIP-visible context. The response is keyed by
  exact PCI address and keeps unreadable fields as `null`.
- `/api/status.gpu` retains the existing primary-HIP shape and adds the
  inventory plus `thermal_guard` state. The dashboard has a collapsible GPU
  table and thermal-guard panel.
- Thermal settings are JSON-backed and live-applied:
  `gpu_thermal_guard_enabled=true`, `gpu_max_temp_c=90`,
  `gpu_max_temp_seconds=5`, and `gpu_resume_temp_c=82`.
- A one-second monitor uses junction/hotspot when present, otherwise the
  hottest labeled sensor. The worker checks the guard before prefill, before
  multimodal work, and before generated-token forward steps. Thermal hold
  time is excluded from generation timing and stored separately in run history.
- AMD mutations require an exact PCI identity, JSON, operation-specific
  `X-ReInstinct-Action`, driver-reported writability, and the structured
  `reinstinct-gpu-helper` Unix-socket protocol. The helper is allowlisted,
  range-bounded, serialized, readback-verified, and rollback-aware.

## Validation evidence

Local evidence currently includes:

- Rust fake-sysfs inventory tests for PCI-keyed devices, theoretical PCIe
  bandwidth, missing sensors, and reordered filesystem enumeration.
- Deterministic thermal state tests for brief spikes, sustained threshold,
  cooldown/resume, sensor-loss fail-closed behavior, and disabled state.
- `cargo check` passed.
- `cargo test --lib serve:: -- --nocapture` passed after the inventory and
  thermal state-machine fixes.
- C# HTTP contract suite builds and checks `/api/gpus`, thermal status fields,
  management OpenAPI paths, and exact-header behavior.

Live isolated MI50 evidence (ReInstinct `reinstinct-server.service`, port
8006) now includes:

- Release server and helper builds completed on the host. The helper is
  installed as `/usr/local/libexec/reinstinct-gpu-helper`, root-owned, with a
  `root:ai` mode-770 socket at `/run/reinstinct-gpu-helper.sock`.
- `/api/gpus` returned two PCI-keyed devices, including AMD `0000:01:00.0`,
  PCIe Gen4 x16, 252.06 theoretical GB/s per direction, junction 45 C, and
  VM-likely-passthrough evidence from QEMU/KVM.
- The thermal guard reported `monitoring`, junction telemetry, 45 C current,
  46 C maximum observed, zero sensor failures, and zero held requests.
- Live configuration validation rejected resume >= maximum, and disable /
  re-enable transitions returned `disabled` then `monitoring`. The persisted
  defaults are 90 C, 5 seconds, and 82 C.
- The C# HTTP contract suite passed against the deployed service, including
  text, streaming, JPEG, PNG, timeout recovery, GPU inventory, thermal
  configuration, OpenAPI, and mutation-header checks.
- Independent hwmon reads showed the same junction/edge sensor range and
  power-cap files used by the inventory (`temp3_input` and `power1_*`).

## Remaining deployment gates

The deployed validation did not force an unsafe real-GPU over-temperature
condition, so no live pause/resume timing or sensor-loss hold is claimed.
Those paths are covered by deterministic fake-temperature tests; a dedicated
isolated MI50 thermal test should be run only with an operator-approved,
reversible workload and independent hardware cutoff. Furnace remains outside
the scope and stayed unchanged.

For future deployments or upgrades, install the helper with a root-owned binary
and restricted socket, require authenticated reverse-proxy access, run
conservative capped workloads, watch junction temperature and
`power1_input`/`power1_cap`, and stop immediately on unsafe readings. Record API
capability, measured pause/resume behavior, limitations, and service state
separately.
