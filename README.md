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

# run general scenario matrix (5 scenarios x 4 languages)
python ./scenario_matrix_bench.py
```

```powershell
powershell -File .\scripts\perf-gate.ps1 -Mode soft -Sample .\sample-regression.json
powershell -File .\scripts\e2e-smoke.ps1
python .\cross_lang_bench.py
python .\scenario_matrix_bench.py
```

## Latest Bench Reports (4 Languages)

### Legacy Cross-Language Suite

Report:

- `results/1771185060774-cross-language.json`

Overall comparison table:

| Metric | Sengoo | C++ | Rust | Python |
|---|---:|---:|---:|---:|
| Runtime p50 (ms) | 10.20 | 13.97 | 14.10 | 37.84 |
| Full compile avg (ms) | 862.21 | 1467.46 | 981.09 | 71.65 |
| Incremental before avg (ms) | 836.59 | 1798.03 | 1022.70 | 77.16 |
| Incremental after avg (ms) | 50.60 | 1740.37 | 1147.47 | 68.99 |
| Incremental reduction (%) | 93.95 | 3.21 | -12.20 | 10.59 |

### Scenario Matrix (5 Scenarios)

Report:

- `results/1771185238357-scenario-matrix.json`

Overall comparison table:

| Metric | Sengoo | C++ | Rust | Python |
|---|---:|---:|---:|---:|
| Runtime p50 avg (ms) | 8.92 | 8.55 | 8.59 | 45.14 |
| Full compile avg (ms) | 835.92 | 1669.41 | 972.98 | 67.48 |
| Incremental before avg (ms) | 841.96 | 1664.64 | 1039.68 | 67.25 |
| Incremental after avg (ms) | 33.71 | 1702.23 | 1088.19 | 65.52 |
| Incremental reduction avg (%) | 95.99 | -2.28 | -4.95 | 2.61 |

Scenario-level runtime p50 (ms):

| Scenario | Sengoo | C++ | Rust | Python |
|---|---:|---:|---:|---:|
| arith_loop | 12.56 | 8.65 | 8.38 | 44.36 |
| branch_mix | 7.79 | 8.75 | 8.76 | 46.28 |
| fn_call_hot | 7.71 | 8.13 | 7.87 | 45.04 |
| array_index | 7.88 | 8.17 | 8.15 | 47.55 |
| nested_loop | 8.67 | 9.04 | 9.78 | 42.49 |

Scenario-level full compile avg (ms):

| Scenario | Sengoo | C++ | Rust | Python |
|---|---:|---:|---:|---:|
| arith_loop | 846.67 | 1689.62 | 1008.01 | 67.38 |
| branch_mix | 828.09 | 1662.31 | 960.43 | 62.47 |
| fn_call_hot | 827.49 | 1647.67 | 969.11 | 72.84 |
| array_index | 845.99 | 1674.86 | 949.25 | 65.60 |
| nested_loop | 831.39 | 1672.59 | 978.11 | 69.11 |

Scenario-level incremental reduction (%):

| Scenario | Sengoo | C++ | Rust | Python |
|---|---:|---:|---:|---:|
| arith_loop | 96.52 | -3.56 | -9.27 | 7.61 |
| branch_mix | 94.39 | -2.72 | -12.92 | 0.36 |
| fn_call_hot | 96.61 | -2.17 | -4.22 | 2.90 |
| array_index | 96.22 | 0.64 | 0.79 | -3.47 |
| nested_loop | 96.22 | -3.59 | 0.86 | 5.64 |

Notes:

- Runtime numbers are execution-only (compile excluded).
- Incremental compile uses comment-only mutation for the second build.
- Python compile numbers are bytecode compile path (`py_compile`), not native binary compile.

## SENGOO_ROOT

When this repository is not located inside Sengoo source tree, set:

```bash
export SENGOO_ROOT=/path/to/Sengoo
```

```powershell
$env:SENGOO_ROOT = "C:\path\to\Sengoo"
```

Then run scripts from this repository root.
