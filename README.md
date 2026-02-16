# Sengoo Bench

Independent benchmark/test repository for Sengoo.

Goal: keep performance tracking and test assets evolving independently from compiler core code.

## What You Get

- Runtime / full compile / incremental compile benchmark suites
- Python interoperability benchmark suite (Sengoo Runtime / Rust / C++ / Python baseline)
- Bootstrap generality proof suite (feature-coverage + correctness evidence)
- Perf gate scripts (soft/hard mode)
- Advanced KPI gate script for real incremental / scale / link targets
- E2E smoke scripts
- Cross-language benchmark harness (Sengoo vs C++/Rust/Python)

## Layout

- `suites/runtime/`
- `suites/compile/`
- `suites/incremental/`
- `tests/`
- `scripts/`
- `baseline.json`
- `FRONTEND_BASELINE.md`
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

# run advanced pipeline benchmark
# (real incremental edits + 1k/10k/100k scale curve + fairness knobs + link share)
python ./advanced_pipeline_bench.py

# run Python interoperability benchmark
# (Sengoo Runtime PythonInterop vs Rust PyO3 vs C++ CPython API vs Python native)
python ./python_interop_bench.py

# run bootstrap generality proof
# (multi-capability scenario matrix with Python-reference correctness validation)
python ./bootstrap_generality_bench.py

# optional: include Sengoo daemon on/off comparison on real-incremental scenarios
python ./advanced_pipeline_bench.py --daemon-compare --daemon-addr 127.0.0.1:48768

# run advanced KPI gate (soft/hard)
python ./scripts/advanced-kpi-gate.py --mode soft --sample ./results/<latest-advanced-pipeline.json>

# run interop/bootstrap gate (soft/hard)
python ./scripts/interop-bootstrap-gate.py \
  --mode soft \
  --interop-sample ./results/<latest-python-interop.json> \
  --bootstrap-sample ./results/<latest-bootstrap-generality.json>
```

```powershell
powershell -File .\scripts\perf-gate.ps1 -Mode soft -Sample .\sample-regression.json
powershell -File .\scripts\e2e-smoke.ps1
python .\cross_lang_bench.py
python .\scenario_matrix_bench.py
python .\advanced_pipeline_bench.py
python .\python_interop_bench.py
python .\bootstrap_generality_bench.py
python .\advanced_pipeline_bench.py --daemon-compare --daemon-addr 127.0.0.1:48768
python .\scripts\advanced-kpi-gate.py --mode soft --sample .\results\<latest-advanced-pipeline.json>
python .\scripts\interop-bootstrap-gate.py --mode soft --interop-sample .\results\<latest-python-interop.json> --bootstrap-sample .\results\<latest-bootstrap-generality.json>
```

## New Bench Suites

### Python Interop Benchmark

- Script: `python_interop_bench.py`
- Output: `results/*-python-interop.json`
- Main metrics:
  - `init_avg_ms`: interpreter/module setup overhead
  - `loop_avg_ms`: repeated cross-language Python call overhead
  - `calls_per_sec`: effective call throughput
  - `loop_vs_python_native_pct`: relative cost against Python-native baseline

### Bootstrap Generality Proof

- Script: `bootstrap_generality_bench.py`
- Output: `results/*-bootstrap-generality.json`
- Main metrics:
  - scenario-level compile/runtime latency across capabilities
  - Python-reference correctness check for each scenario
  - `bootstrap_proof.status` (`pass`/`fail`) from explicit criteria

## Latest New-Suite Snapshots

### Python Interop (Latest)

Report:

- `results/1771230408116-python-interop.json`

| Runner | Init avg (ms) | Loop avg (ms) | Calls/s | vs Python native (%) |
|---|---:|---:|---:|---:|
| Python native | 0.26 | 2.18 | 9,156,502.94 | 0.00 |
| Sengoo Runtime (PythonInterop) | 0.31 | 2.67 | 7,503,958.34 | +22.02 |
| C++ (CPython C API) | 20.50 | 2.92 | 6,851,896.26 | +33.63 |
| Rust (PyO3) | 0.36 | 2.93 | 6,825,659.02 | +34.15 |

### Bootstrap Generality Proof (Latest)

Report:

- `results/1771230417893-bootstrap-generality.json`

Proof status:

- `pass` (`6/6` scenarios passed, `6` capability classes covered)

## Advanced KPI Targets

`scripts/advanced-kpi-gate.py` enforces the reference-profile targets:

- Real incremental (`loop_body_change`, `function_signature_change`, `add_new_function`): `after_avg_ms <= 200`
- Scale full build at `100000 LOC`: `e2e_avg_ms <= 2000`
- Scale stage budget at `100000 LOC`: `compile_frontend_llvm_avg_ms <= 300`
- Scale stage budget at `100000 LOC`: `codegen_obj_avg_ms <= 1500`
- Link time at `100000 LOC`: `link_avg_ms <= 500`
- Scale stage presence: `1000/10000/100000` all include Sengoo `frontend`, `codegen`, `link` metrics
- Reachability matrix presence: `all_reachable`, `half_reachable`, `library_entryless` profiles are required
- Reachability hard budget: `reachability_matrix/all_reachable/compile_frontend_llvm_avg_ms <= 300`
- Optional daemon regression guard: `daemon_after_avg_ms - oneshot_after_avg_ms <= 50` per scenario
- Optional phase-delta contract: report includes `phase_deltas` for incremental and scale blocks

Override budget thresholds:

```bash
python ./scripts/advanced-kpi-gate.py \
  --mode soft \
  --sample ./results/<latest-advanced-pipeline.json> \
  --max-frontend-100k-ms 300 \
  --max-codegen-100k-ms 1500 \
  --require-phase-deltas \
  --require-daemon-comparison
```

Nightly CI runs this gate in `hard` mode and fails the workflow when targets are violated.

Frontend refactor baseline is frozen in `FRONTEND_BASELINE.md` with source report
`results/1771218708000-advanced-pipeline.json`.

## Frontend Refactor Snapshot (Latest)

Reports:

- Baseline: `results/1771218708000-advanced-pipeline.json`
- Current: `results/1771220638259-advanced-pipeline.json`

Key deltas (ms):

| Metric | Baseline | Current | Delta |
|---|---:|---:|---:|
| `scale_curve/100000/frontend` | 645.14 | 459.38 | -185.76 |
| `scale_curve/100000/e2e` | 1149.44 | 971.25 | -178.19 |
| `reachability/all_reachable/frontend` | 908.54 | 670.34 | -238.21 |
| `reachability/half_reachable/frontend` | 800.16 | 562.45 | -237.72 |
| `reachability/library_entryless/frontend` | 910.73 | 653.65 | -257.08 |

Real incremental `after_avg_ms` (Sengoo):

| Scenario | Baseline | Current | Delta |
|---|---:|---:|---:|
| `loop_body_change` | 198.20 | 225.85 | +27.65 |
| `function_signature_change` | 180.89 | 212.23 | +31.34 |
| `add_new_function` | 200.31 | 211.50 | +11.18 |

## Latest Bench Reports (4 Languages)

### Advanced Pipeline (Real Incremental + Scale Curve + E2E Link)

Report:

- `results/1771197609597-advanced-pipeline.json`

Real incremental scenarios (`Before -> After`, ms):

| Scenario | Sengoo | C++ (PCH) | Rust (cargo incremental) | Python |
|---|---:|---:|---:|---:|
| loop_body_change | 852.97 -> 822.61 | 1361.78 -> 1220.94 | 1153.89 -> 1219.32 | 77.05 -> 73.19 |
| function_signature_change | 859.41 -> 861.07 | 1356.08 -> 1371.78 | 1118.73 -> 1191.29 | 80.35 -> 100.95 |
| add_new_function | 848.67 -> 885.52 | 1401.82 -> 1220.52 | 1093.13 -> 1235.07 | 99.28 -> 70.53 |

Scale curve: e2e build time (includes link where applicable):

| LOC | Sengoo (ms) | C++ (ms) | Rust (ms) | Python (ms) |
|---:|---:|---:|---:|---:|
| 1000 | 867.76 | 1220.54 | 1201.87 | 80.53 |
| 10000 | 1392.45 | 1250.38 | 1722.32 | 131.47 |
| 100000 | 6146.46 | 1715.47 | 6263.89 | 774.08 |

Link share:

| LOC | Sengoo link share (%) | C++ link share (%) |
|---:|---:|---:|
| 1000 | 77.50 | 60.90 |
| 10000 | 54.20 | 62.00 |
| 100000 | 12.67 | 44.73 |

Daemon on/off comparison (Sengoo real incremental, `--daemon-compare`):

| Scenario | One-shot after (ms) | Daemon after (ms) | Delta (ms) |
|---|---:|---:|---:|
| loop_body_change | 822.61 | 900.70 | 78.09 |
| function_signature_change | 861.07 | 849.27 | -11.80 |
| add_new_function | 885.52 | 915.49 | 29.97 |

Before/after deltas vs previous advanced report (`results/1771197260568-advanced-pipeline.json`):

| Metric | Previous (ms) | Current (ms) | Delta (ms) |
|---|---:|---:|---:|
| loop_body_change after | 896.18 | 822.61 | -73.57 |
| function_signature_change after | 898.27 | 861.07 | -37.20 |
| add_new_function after | 893.82 | 885.52 | -8.30 |
| 100k frontend | 922.19 | 933.12 | 10.94 |
| 100k codegen | 4116.29 | 4434.77 | 318.47 |
| 100k link | 730.10 | 778.57 | 48.47 |
| 100k e2e | 5768.58 | 6146.46 | 377.88 |

Advanced KPI gate (`python ./scripts/advanced-kpi-gate.py --mode soft --require-phase-deltas --require-daemon-comparison ...`) on this report:

- real incremental target (`<= 200ms`): FAIL (all 3 scenarios above target)
- 100k full build target (`<= 2000ms`): FAIL (`6146.46ms`)
- 100k frontend target (`<= 300ms`): FAIL (`933.12ms`)
- 100k codegen target (`<= 1500ms`): FAIL (`4434.77ms`)
- 100k link target (`<= 500ms`): FAIL (`778.57ms`)

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
