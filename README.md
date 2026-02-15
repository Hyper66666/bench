# Benchmarks

This directory stores Sengoo performance suites and benchmark outputs.

- `suites/runtime/`: runtime-oriented benchmark cases
- `suites/compile/`: full compile benchmark cases
- `suites/incremental/`: incremental compile benchmark cases
- `tests/`: smoke/e2e test inputs
- `scripts/`: perf gate and smoke scripts
- `baseline.json`: baseline and KPI target metadata
- `results/`: machine-readable benchmark run results

Use `sgc bench run|compile|incremental <suite>` to execute benchmark suites.

## Run Scripts

```bash
bash ./bench/scripts/perf-gate.sh --mode soft --sample ./bench/sample-regression.json
bash ./bench/scripts/e2e-smoke.sh
```

```powershell
powershell -File .\\bench\\scripts\\perf-gate.ps1 -Mode soft -Sample .\\bench\\sample-regression.json
powershell -File .\\bench\\scripts\\e2e-smoke.ps1
```

## Publish As Separate Repository

`bench/` is self-contained and can be pushed as an independent GitHub repo.

1. `cd bench`
2. `git init`
3. `git add .`
4. `git commit -m "bench: initial import"`
5. `git remote add origin <your-bench-repo-url>`
6. `git push -u origin main`

If Sengoo source is in another directory, set `SENGOO_ROOT` before running `scripts/e2e-smoke.*` or `cross_lang_bench.py`.
