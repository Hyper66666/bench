# Sengoo Bench

Independent benchmark repository for Sengoo.  
Goal: keep performance and correctness measurements reproducible, comparable, and CI-gated.

## Scope

- Cross-language matrix: Sengoo vs C++ vs Rust vs Python
- Real incremental compile scenarios
- Scale curve: 1k / 10k / 100k / 1000k LOC
- Stage split and link-share analysis
- Compile peak-memory comparison
- Python interop benchmark
- Non-invasive reflection benchmark
- Bootstrap generality proof
- KPI gate scripts for CI
- Runnable demos and smoke checks

## Layout

```text
bench/
|-- scripts/
|-- suites/
|-- tests/
|-- demos/
|-- results/
|-- scenario_matrix_bench.py
|-- advanced_pipeline_bench.py
|-- python_interop_bench.py
|-- llm_scheduler_bench.py
|-- noninvasive_reflection_bench.py
`-- bootstrap_generality_bench.py
```

## Prerequisites

- Python 3.10+
- Rust toolchain (`cargo`, `rustc`)
- Sengoo source tree (`SENGOO_ROOT` or sibling layout)
- Recommended: LLVM/Clang (`clang`, `clang++`)

## Quick Start

### Clone

```bash
cd /path/to/workspace
git clone https://github.com/Hyper66666/Sengoo.git
git clone https://github.com/Hyper66666/bench.git
cd bench
```

```powershell
Set-Location C:\path\to\workspace
git clone https://github.com/Hyper66666/Sengoo.git
git clone https://github.com/Hyper66666/bench.git
Set-Location .\bench
```

If not sibling layout:

```bash
export SENGOO_ROOT=/absolute/path/to/Sengoo
```

```powershell
$env:SENGOO_ROOT = "C:\absolute\path\to\Sengoo"
```

### Environment check

```bash
python --version
cargo --version
clang --version
```

## Standard Run Flow

### 1) Smoke

```bash
bash ./scripts/e2e-smoke.sh
```

```powershell
powershell -File .\scripts\e2e-smoke.ps1
```

### 2) Core suites

```bash
python ./scenario_matrix_bench.py
python ./advanced_pipeline_bench.py
python ./python_interop_bench.py
python ./llm_scheduler_bench.py
python ./noninvasive_reflection_bench.py
python ./bootstrap_generality_bench.py
```

### LLM scheduler benchmark (prefill/decode orchestration focus)

```bash
python ./llm_scheduler_bench.py
```

This benchmark compares:
- Python scheduler + shared decode-step kernel
- Sengoo Runtime scheduler + same decode-step kernel

It emits two scenarios:
- `prefill_decode_orchestration_light_kernel` (orchestration-dominant, expected Sengoo advantage)
- `prefill_decode_orchestration_heavy_kernel` (compute-kernel heavier, expected parity)

Output:
- writes `results/*-llm-scheduler-bench.json`
- includes checksum parity, loop latency, and tokens/s gain

### 3) CI gates

```bash
python ./scripts/advanced-kpi-gate.py --mode soft --sample ./results/<advanced-report>.json --baseline-profile ./frontend-memory-baseline.json
python ./scripts/interop-bootstrap-gate.py --mode soft --interop-sample ./results/<interop-report>.json --bootstrap-sample ./results/<bootstrap-report>.json
python ./scripts/llm-scheduler-gate.py --mode soft --sample ./results/<llm-scheduler-report>.json
```

Use `--mode hard` in CI.

## Latest Snapshot (February 18, 2026)

Source reports:

- `results/1771185238357-scenario-matrix.json`
- `results/1771390773767-advanced-pipeline.json`
- `results/1771392747911-advanced-pipeline.json`
- `results/1771234431756-python-interop.json`
- `results/1771242399249-noninvasive-reflection-bench.json`
- `results/1771230417893-bootstrap-generality.json`
- `results/1771425334804-low-memory-e2e-1000k.json`

### Incremental reduction average

| Language | Incremental reduction avg |
|---|---:|
| Sengoo | 95.99% |
| C++ | -2.28% |
| Rust | -4.95% |
| Python | 2.61% |

### 10k-1000k e2e compile comparison

| LOC | Sengoo (ms) | C++ (ms) | Rust (ms) | Python (ms) |
|---|---:|---:|---:|---:|
| 10k | 372.28 | 693.01 | 2246.86 | 157.18 |
| 100k | 417.53 | 1074.84 | 6625.35 | 832.91 |
| 1000k | 1827.84 | 4883.70 | 54642.47 | 8283.46 |

### Compile peak RSS comparison (compile-stage)

| LOC | Sengoo (MB) | C++ (MB) | Rust (MB) | Python (MB) |
|---|---:|---:|---:|---:|
| 10k | 18.88 | 75.68 | 70.84 | 41.40 |
| 100k | 140.18 | 118.50 | 337.86 | 288.46 |
| 1000k | 1367.99 | 435.22 | 2681.55 | 2610.90 |

### Low-memory mode (1000k e2e, newly added)

| Mode | e2e avg (ms) | Peak RSS (MB) |
|---|---:|---:|
| Default (`sgc build`) | 2331.39 | 1418.61 |
| `--low-memory` | 1737.71 | 672.10 |

Benefits:

- e2e time reduction: 25.46%
- peak RSS reduction: 52.62%

Trade-offs:

- less incremental cache/session reuse
- single-thread frontend in low-memory mode
- lower MIR optimization cap

Enable:

```bash
sgc build your_file.sg --low-memory
sgc run your_file.sg --low-memory
```

## Result Files

Main outputs under `results/`:

- `*-scenario-matrix.json`
- `*-advanced-pipeline.json`
- `*-python-interop.json`
- `*-noninvasive-reflection-bench.json`
- `*-bootstrap-generality.json`
- `*-low-memory-e2e-1000k.json`

## Troubleshooting

- `cannot resolve Sengoo source root`: set `SENGOO_ROOT`.
- `clang++ not found`: install LLVM/Clang, or C++ runners may be skipped.
- timing variance: close heavy background processes and increase samples.

---

# 涓枃鐗?
杩欐槸 Sengoo 鐨勭嫭绔嬪熀鍑嗕粨搴擄紝鐢ㄤ簬鎶婃€ц兘涓庢纭€ф祴璇曞仛鎴愬彲澶嶇幇銆佸彲瀵规瘮銆佸彲鎺ュ叆 CI 鐨勬祦绋嬨€?
## 瑕嗙洊鍐呭

- 鍥涜瑷€瀵规瘮锛歋engoo / C++ / Rust / Python
- 鐪熷疄澧為噺鍦烘櫙锛堟敼寰幆浣撱€佹敼鍑芥暟绛惧悕銆佸姞鏂板嚱鏁帮級
- 瑙勬ā鏇茬嚎锛?k / 10k / 100k / 1000k LOC锛?- 闃舵鎷嗗垎涓庨摼鎺ュ崰姣?- 缂栬瘧宄板€煎唴瀛樺姣?- Python 浜掓搷浣滃熀鍑?- 闈炰镜鍏ュ紡鍙嶅皠鍩哄噯
- 鑷妇閫氱敤鎬ц瘉鏄?- CI 闂ㄧ鑴氭湰

## 蹇€熷紑濮?
```bash
cd /path/to/workspace
git clone https://github.com/Hyper66666/Sengoo.git
git clone https://github.com/Hyper66666/bench.git
cd bench
```

闈炲悓绾х洰褰曟椂鎵嬪姩璁剧疆锛?
```bash
export SENGOO_ROOT=/absolute/path/to/Sengoo
```

## 鎺ㄨ崘鎵ц娴佺▼

### 1锛夊啋鐑熸鏌?
```bash
bash ./scripts/e2e-smoke.sh
```

### 2锛夋牳蹇冨熀鍑?
```bash
python ./scenario_matrix_bench.py
python ./advanced_pipeline_bench.py
python ./python_interop_bench.py
python ./llm_scheduler_bench.py
python ./noninvasive_reflection_bench.py
python ./bootstrap_generality_bench.py
```

### 3锛夐棬绂?
```bash
python ./scripts/advanced-kpi-gate.py --mode soft --sample ./results/<advanced-report>.json --baseline-profile ./frontend-memory-baseline.json
python ./scripts/interop-bootstrap-gate.py --mode soft --interop-sample ./results/<interop-report>.json --bootstrap-sample ./results/<bootstrap-report>.json
python ./scripts/llm-scheduler-gate.py --mode soft --sample ./results/<llm-scheduler-report>.json
```

## 鏈€鏂板揩鐓э紙2026骞?鏈?8鏃ワ級

鎶ュ憡鏂囦欢锛?
- `results/1771185238357-scenario-matrix.json`
- `results/1771390773767-advanced-pipeline.json`
- `results/1771392747911-advanced-pipeline.json`
- `results/1771234431756-python-interop.json`
- `results/1771242399249-noninvasive-reflection-bench.json`
- `results/1771230417893-bootstrap-generality.json`
- `results/1771425334804-low-memory-e2e-1000k.json`

### 澧為噺鏀剁泭骞冲潎鍊?
| 璇█ | 澧為噺鏀剁泭骞冲潎鍊?|
|---|---:|
| Sengoo | 95.99% |
| C++ | -2.28% |
| Rust | -4.95% |
| Python | 2.61% |

### 10k-1000k e2e 缂栬瘧瀵规瘮

| LOC | Sengoo (ms) | C++ (ms) | Rust (ms) | Python (ms) |
|---|---:|---:|---:|---:|
| 10k | 372.28 | 693.01 | 2246.86 | 157.18 |
| 100k | 417.53 | 1074.84 | 6625.35 | 832.91 |
| 1000k | 1827.84 | 4883.70 | 54642.47 | 8283.46 |

### 缂栬瘧宄板€?RSS锛堜粎缂栬瘧闃舵锛?
| LOC | Sengoo (MB) | C++ (MB) | Rust (MB) | Python (MB) |
|---|---:|---:|---:|---:|
| 10k | 18.88 | 75.68 | 70.84 | 41.40 |
| 100k | 140.18 | 118.50 | 337.86 | 288.46 |
| 1000k | 1367.99 | 435.22 | 2681.55 | 2610.90 |

### 浣庡唴瀛樻ā寮忥紙鏂板锛?000k e2e锛?
| 妯″紡 | e2e 骞冲潎 (ms) | 宄板€?RSS (MB) |
|---|---:|---:|
| 榛樿锛坄sgc build`锛?| 2331.39 | 1418.61 |
| `--low-memory` | 1737.71 | 672.10 |

浼樺娍锛?
- e2e 鏃堕棿涓嬮檷 25.46%
- 宄板€?RSS 涓嬮檷 52.62%

鍓綔鐢細

- 澧為噺缂撳瓨/浼氳瘽澶嶇敤鑳藉姏浼氬噺寮?- 鍓嶇鍥哄畾鍗曠嚎绋?- MIR 浼樺寲涓婇檺涓嬮檷

寮€鍚柟寮忥細

```bash
sgc build your_file.sg --low-memory
sgc run your_file.sg --low-memory
```


