# ReInstinct API contract suite

This is a standalone C#/.NET 8 black-box suite. It uses only `HttpClient` and
`System.Text.Json`; it does not reference the Rust crate or any ReInstinct
internals.

Run it against an isolated server:

```powershell
dotnet run --project tests/api-contract -- --base-url http://127.0.0.1:8006
```

The suite exercises liveness, model discovery, text chat, SSE streaming and
usage, bounded run-history summaries/details, engine logs, network telemetry,
model catalog metadata, safe rejection of out-of-catalog switches, and throughput fields,
deterministic JPEG/PNG data URLs, OpenAI-shaped errors, the 8 MiB body limit,
timeout recovery, and client-disconnect recovery. A nonzero exit code means at
least one contract check failed.

The same command can target a named OpenAI-compatible reference service or
SDK proxy. Record the endpoint name, version, configuration, and result in
`docs/API_COMPATIBILITY_PLAN.md`; credentials are intentionally not accepted
as command-line arguments.
