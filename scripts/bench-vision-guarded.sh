#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
    echo "usage: $0 PORT SERVER_PID OUTPUT_DIR IMAGE..." >&2
    exit 2
fi

port="$1"
server_pid="$2"
output_dir="$3"
shift 3
mkdir -p "$output_dir"

hwmon="${REINSTINCT_HWMON:-/sys/class/drm/card1/device/hwmon/hwmon0}"
cutoff_millic="${REINSTINCT_JUNCTION_CUTOFF_MILLIC:-88000}"
prompt='Describe this image in detail. Answer directly without explaining your reasoning.'
telemetry="$output_dir/telemetry.csv"
guard="$output_dir/thermal-guard"
printf 'timestamp,junction_c,memory_c,power_w\n' > "$telemetry"

monitor() {
    while kill -0 "$server_pid" 2>/dev/null; do
        junction="$(<"$hwmon/temp2_input")"
        memory="$(<"$hwmon/temp3_input")"
        power="$(<"$hwmon/power1_input")"
        printf '%s,%d.%d,%d.%d,%d.%d\n' "$(date --iso-8601=seconds)" \
            "$((junction / 1000))" "$(((junction / 100) % 10))" \
            "$((memory / 1000))" "$(((memory / 100) % 10))" \
            "$((power / 1000000))" "$(((power / 100000) % 10))" >> "$telemetry"
        if (( junction >= cutoff_millic )); then
            printf 'junction reached %d.%d C\n' "$((junction / 1000))" "$(((junction / 100) % 10))" > "$guard"
            kill -TERM "$server_pid" 2>/dev/null || true
            return
        fi
        sleep 0.25
    done
}

monitor &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

for image in "$@"; do
    name="$(basename "$image")"
    stem="${name%.*}"
    mime="image/jpeg"
    case "${name##*.}" in
        png|PNG) mime="image/png" ;;
    esac
    base64_file="$output_dir/$stem.base64"
    request_file="$output_dir/$stem-request.json"
    base64 -w0 "$image" > "$base64_file"
    jq -n --rawfile encoded "$base64_file" --arg mime "$mime" --arg prompt "$prompt" '{
      model: "big", temperature: 0, max_tokens: 64,
      messages: [{role: "user", content: [
        {type: "text", text: $prompt},
        {type: "image_url", image_url: {url: ("data:" + $mime + ";base64," + $encoded)}}
      ]}]
    }' > "$request_file"
    rm "$base64_file"
    curl --fail-with-body --silent --show-error --max-time 1800 \
        -H 'Content-Type: application/json' --data-binary "@$request_file" \
        "http://127.0.0.1:$port/v1/chat/completions" > "$output_dir/$stem-response.json"
    if [[ -f "$guard" ]]; then
        exit 70
    fi
done
