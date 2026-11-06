# Dashboard browser regression checks

These checks load the exact generated dashboard in headless Microsoft Edge with mocked API responses. They do not contact or mutate a running inference service. Node.js, Rust, and Edge must be installed.

Run from the repository root in PowerShell:

```powershell
npm install --prefix target/dashboard-qa --no-package-lock --no-save playwright
$env:REINSTINCT_DASHBOARD_TEST_HTML = Join-Path (Get-Location) 'target/dashboard-review.html'
cargo test --lib serve::api::tests
$env:NODE_PATH = Join-Path (Get-Location) 'target/dashboard-qa/node_modules'
node tests/dashboard-review.cjs
```

The suite checks generated styles, GPU selection and power drafts across polling, independent configuration drafts, loaded-model controls, keyboard interaction, expired run handling, partial API failure, stale status, mobile overflow, and uncaught browser errors. A mobile screenshot is written to `target/dashboard-mobile.png`. Hardware behavior and real service mutations still require live validation.
