#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "usage: $0 OUTPUT_DIR COMMAND..." >&2
    exit 2
fi

output_dir="$1"
shift
mkdir -p "$output_dir"
hwmon="${REINSTINCT_HWMON:-/sys/class/drm/card1/device/hwmon/hwmon0}"
cutoff_millic="${REINSTINCT_JUNCTION_CUTOFF_MILLIC:-88000}"
telemetry="$output_dir/telemetry.csv"
printf 'timestamp,junction_c,memory_c,power_w\n' > "$telemetry"

"$@" > "$output_dir/stdout.log" 2> "$output_dir/stderr.log" &
command_pid=$!
while kill -0 "$command_pid" 2>/dev/null; do
    junction="$(<"$hwmon/temp2_input")"
    memory="$(<"$hwmon/temp3_input")"
    power="$(<"$hwmon/power1_input")"
    printf '%s,%d.%d,%d.%d,%d.%d\n' "$(date --iso-8601=seconds)" \
        "$((junction / 1000))" "$(((junction / 100) % 10))" \
        "$((memory / 1000))" "$(((memory / 100) % 10))" \
        "$((power / 1000000))" "$(((power / 100000) % 10))" >> "$telemetry"
    if (( junction >= cutoff_millic )); then
        printf 'junction reached %d.%d C\n' "$((junction / 1000))" "$(((junction / 100) % 10))" > "$output_dir/thermal-guard"
        kill -TERM "$command_pid" 2>/dev/null || true
        sleep 1
        kill -KILL "$command_pid" 2>/dev/null || true
        wait "$command_pid" 2>/dev/null || true
        exit 70
    fi
    sleep 0.25
done
wait "$command_pid"
