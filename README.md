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
python ./noninvasive_reflection_bench.py
python ./bootstrap_generality_bench.py
```

### 3) CI gates

```bash
python ./scripts/advanced-kpi-gate.py --mode soft --sample ./results/<advanced-report>.json --baseline-profile ./frontend-memory-baseline.json
python ./scripts/interop-bootstrap-gate.py --mode soft --interop-sample ./results/<interop-report>.json --bootstrap-sample ./results/<bootstrap-report>.json
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

# 中文版

这是 Sengoo 的独立基准仓库，用于把性能与正确性测试做成可复现、可对比、可接入 CI 的流程。

## 覆盖内容

- 四语言对比：Sengoo / C++ / Rust / Python
- 真实增量场景（改循环体、改函数签名、加新函数）
- 规模曲线（1k / 10k / 100k / 1000k LOC）
- 阶段拆分与链接占比
- 编译峰值内存对比
- Python 互操作基准
- 非侵入式反射基准
- 自举通用性证明
- CI 门禁脚本

## 快速开始

```bash
cd /path/to/workspace
git clone https://github.com/Hyper66666/Sengoo.git
git clone https://github.com/Hyper66666/bench.git
cd bench
```

非同级目录时手动设置：

```bash
export SENGOO_ROOT=/absolute/path/to/Sengoo
```

## 推荐执行流程

### 1）冒烟检查

```bash
bash ./scripts/e2e-smoke.sh
```

### 2）核心基准

```bash
python ./scenario_matrix_bench.py
python ./advanced_pipeline_bench.py
python ./python_interop_bench.py
python ./noninvasive_reflection_bench.py
python ./bootstrap_generality_bench.py
```

### 3）门禁

```bash
python ./scripts/advanced-kpi-gate.py --mode soft --sample ./results/<advanced-report>.json --baseline-profile ./frontend-memory-baseline.json
python ./scripts/interop-bootstrap-gate.py --mode soft --interop-sample ./results/<interop-report>.json --bootstrap-sample ./results/<bootstrap-report>.json
```

## 最新快照（2026年2月18日）

报告文件：

- `results/1771185238357-scenario-matrix.json`
- `results/1771390773767-advanced-pipeline.json`
- `results/1771392747911-advanced-pipeline.json`
- `results/1771234431756-python-interop.json`
- `results/1771242399249-noninvasive-reflection-bench.json`
- `results/1771230417893-bootstrap-generality.json`
- `results/1771425334804-low-memory-e2e-1000k.json`

### 增量收益平均值

| 语言 | 增量收益平均值 |
|---|---:|
| Sengoo | 95.99% |
| C++ | -2.28% |
| Rust | -4.95% |
| Python | 2.61% |

### 10k-1000k e2e 编译对比

| LOC | Sengoo (ms) | C++ (ms) | Rust (ms) | Python (ms) |
|---|---:|---:|---:|---:|
| 10k | 372.28 | 693.01 | 2246.86 | 157.18 |
| 100k | 417.53 | 1074.84 | 6625.35 | 832.91 |
| 1000k | 1827.84 | 4883.70 | 54642.47 | 8283.46 |

### 编译峰值 RSS（仅编译阶段）

| LOC | Sengoo (MB) | C++ (MB) | Rust (MB) | Python (MB) |
|---|---:|---:|---:|---:|
| 10k | 18.88 | 75.68 | 70.84 | 41.40 |
| 100k | 140.18 | 118.50 | 337.86 | 288.46 |
| 1000k | 1367.99 | 435.22 | 2681.55 | 2610.90 |

### 低内存模式（新增：1000k e2e）

| 模式 | e2e 平均 (ms) | 峰值 RSS (MB) |
|---|---:|---:|
| 默认（`sgc build`） | 2331.39 | 1418.61 |
| `--low-memory` | 1737.71 | 672.10 |

优势：

- e2e 时间下降 25.46%
- 峰值 RSS 下降 52.62%

副作用：

- 增量缓存/会话复用能力会减弱
- 前端固定单线程
- MIR 优化上限下降

开启方式：

```bash
sgc build your_file.sg --low-memory
sgc run your_file.sg --low-memory
```

