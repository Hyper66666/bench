# Sengoo Bench

Independent benchmark/test repository for Sengoo.

Goal: keep performance tracking and test assets evolving independently from compiler core code.

## What You Get

- Runtime / full compile / incremental compile benchmark suites
- Perf gate scripts (soft/hard mode)
- E2E smoke scripts
- Cross-language benchmark harness (Sengoo vs C++/Rust/Python)

## Layout

- `suites/runtime/`
- `suites/compile/`
- `suites/incremental/`
- `tests/`
- `scripts/`
- `baseline.json`
- `results/`

## Requirements

- Rust toolchain (to build and run `sgc`)
- Python 3
- LLVM/Clang (for native backend paths)
- Sengoo main repository path available via `SENGOO_ROOT`

## Quick Usage

```bash
# run soft perf gate
bash ./scripts/perf-gate.sh --mode soft --sample ./sample-regression.json

# run smoke
bash ./scripts/e2e-smoke.sh

# run cross-language comparison
python ./cross_lang_bench.py
```

```powershell
powershell -File .\scripts\perf-gate.ps1 -Mode soft -Sample .\sample-regression.json
powershell -File .\scripts\e2e-smoke.ps1
python .\cross_lang_bench.py
```

## SENGOO_ROOT

When this repository is not located inside Sengoo source tree, set:

```bash
export SENGOO_ROOT=/path/to/Sengoo
```

```powershell
$env:SENGOO_ROOT = "C:\path\to\Sengoo"
```

Then run scripts from this repository root.
