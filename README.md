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

## Latest Scenario Matrix (4 Languages)

Report:

- `results/1771184915736-scenario-matrix.json`

Overall comparison table:

| Metric | Sengoo | C++ | Rust | Python |
|---|---:|---:|---:|---:|
| Runtime p50 avg (ms) | 9.07 | 9.62 | 8.57 | 47.04 |
| Full compile avg (ms) | 837.55 | 1716.57 | 1013.27 | 72.98 |
| Incremental before avg (ms) | 807.92 | 1692.82 | 1022.63 | 66.66 |
| Incremental after avg (ms) | 27.14 | 1674.30 | 1068.02 | 66.04 |
| Incremental reduction avg (%) | 96.64 | 1.08 | -4.52 | 0.92 |

Scenario-level runtime p50 (ms):

| Scenario | Sengoo | C++ | Rust | Python |
|---|---:|---:|---:|---:|
| arith_loop | 7.89 | 8.35 | 8.41 | 45.16 |
| branch_mix | 7.88 | 8.25 | 8.41 | 50.66 |
| fn_call_hot | 7.39 | 7.95 | 8.20 | 46.42 |
| array_index | 7.76 | 9.08 | 9.51 | 50.13 |
| nested_loop | 14.42 | 14.46 | 8.30 | 42.86 |

Scenario-level full compile avg (ms):

| Scenario | Sengoo | C++ | Rust | Python |
|---|---:|---:|---:|---:|
| arith_loop | 834.56 | 1703.47 | 1047.70 | 73.29 |
| branch_mix | 827.62 | 1808.28 | 1034.57 | 71.98 |
| fn_call_hot | 855.02 | 1684.48 | 992.27 | 71.08 |
| array_index | 825.41 | 1664.79 | 1002.01 | 70.84 |
| nested_loop | 845.13 | 1721.85 | 989.78 | 77.70 |

Scenario-level incremental reduction (%):

| Scenario | Sengoo | C++ | Rust | Python |
|---|---:|---:|---:|---:|
| arith_loop | 96.62 | -1.58 | -2.26 | 2.04 |
| branch_mix | 96.55 | 1.94 | -9.28 | -0.86 |
| fn_call_hot | 96.57 | 2.02 | -6.55 | 3.33 |
| array_index | 96.67 | 2.37 | -0.62 | -3.60 |
| nested_loop | 96.81 | 0.63 | -3.87 | 3.71 |

Notes:

- Runtime numbers are execution-only (compile excluded).
- Incremental compile currently uses comment-only mutation for the second build.
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
