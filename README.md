# Sengoo Bench

Independent benchmark and verification repository for Sengoo.

This repo is designed for one purpose: **measure performance and correctness trends with reproducible scripts**.

## What This Repository Covers

- Cross-language performance matrix (Sengoo vs C++ vs Rust vs Python)
- Real incremental compile scenarios (loop-body change, signature change, add function)
- Scale curve (1k / 10k / 100k LOC)
- Link-share and phase split analysis
- Python interoperability benchmark (Sengoo Runtime / Rust PyO3 / C++ CPython API / Python native)
- Non-invasive reflection benchmark (disabled vs enabled-unused vs enabled-used)
- Bootstrap generality proof (multi-capability correctness + latency)
- CI gate scripts (soft/hard) for KPI enforcement
- E2E smoke scripts for baseline toolchain health

## Repository Layout

```text
bench/
|-- scripts/                       # perf/e2e/gate automation scripts
|-- suites/                        # suite assets and generated inputs
|-- tests/                         # sample programs and runnable examples
|   `-- python_interop_example.py  # direct Python interop example (new)
|-- results/                       # generated JSON reports (ignored by git)
|-- advanced_pipeline_bench.py
|-- scenario_matrix_bench.py
|-- python_interop_bench.py
|-- noninvasive_reflection_bench.py
|-- bootstrap_generality_bench.py
`-- README.md
```

## Prerequisites

Required:

- Python 3.10+
- Rust toolchain (`cargo`, `rustc`)
- Sengoo source tree available (set via `SENGOO_ROOT` or sibling layout)

Optional but recommended for full comparability:

- LLVM/Clang (`clang`, `clang++`)
- C++ toolchain runtime environment

## Clone and Reproduce (From Scratch)

### Option A: Sibling layout (recommended)

```bash
# parent dir can be any workspace root
cd /path/to/workspace
git clone https://github.com/Hyper66666/Sengoo.git
git clone https://github.com/Hyper66666/bench.git

# enter bench repo
cd bench
```

```powershell
Set-Location C:\path\to\workspace
git clone https://github.com/Hyper66666/Sengoo.git
git clone https://github.com/Hyper66666/bench.git
Set-Location .\bench
```

### Option B: Bench is not next to Sengoo

Set `SENGOO_ROOT` explicitly:

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

```powershell
python --version
cargo --version
clang --version
```

If `clang` is missing, C++-related runners may be skipped (interop benchmark still produces output for available runners).

## Standard Test Flow

Run from `bench/` root.

### 1) Toolchain smoke (fast sanity)

```bash
bash ./scripts/e2e-smoke.sh
```

```powershell
powershell -File .\scripts\e2e-smoke.ps1
```

This verifies:

- `sgc check` works
- LLVM IR emit works
- daemon happy-path works
- daemon fallback path works

### 2) Direct Python interop example (new)

This is a ready-to-run example under `tests/` that directly prints summary output:

```bash
python ./tests/python_interop_example.py
```

```powershell
python .\tests\python_interop_example.py
```

Optional quick tuning:

```bash
python ./tests/python_interop_example.py --calls 3000 --samples 2 --warmup 0
```

```powershell
python .\tests\python_interop_example.py --calls 3000 --samples 2 --warmup 0
```

The script:

- invokes `python_interop_bench.py`
- auto-detects the generated JSON report path
- prints a compact table (`loop_avg_ms`, `calls_per_sec`, `vs_python_native_pct`)

### 3) Core benchmark suites

```bash
python ./scenario_matrix_bench.py
python ./advanced_pipeline_bench.py
python ./python_interop_bench.py
python ./noninvasive_reflection_bench.py
python ./bootstrap_generality_bench.py
```

```powershell
python .\scenario_matrix_bench.py
python .\advanced_pipeline_bench.py
python .\python_interop_bench.py
python .\noninvasive_reflection_bench.py
python .\bootstrap_generality_bench.py
```

### 4) Gate checks (soft/hard)

Advanced KPI gate:

```bash
python ./scripts/advanced-kpi-gate.py --mode soft --sample ./results/<advanced-report>.json
```

Interop + bootstrap gate:

```bash
python ./scripts/interop-bootstrap-gate.py \
  --mode soft \
  --interop-sample ./results/<python-interop-report>.json \
  --bootstrap-sample ./results/<bootstrap-report>.json
```

PowerShell:

```powershell
python .\scripts\advanced-kpi-gate.py --mode soft --sample .\results\<advanced-report>.json
python .\scripts\interop-bootstrap-gate.py --mode soft --interop-sample .\results\<python-interop-report>.json --bootstrap-sample .\results\<bootstrap-report>.json
```

## Output Files and How to Read Them

Reports are written to `results/`:

- `*-scenario-matrix.json`
  - 5-scenario runtime/full/incremental matrix across 4 languages
- `*-advanced-pipeline.json`
  - real incremental edits + scale curve + link share + reachability matrix
- `*-python-interop.json`
  - Python boundary overhead and throughput comparison
- `*-noninvasive-reflection-bench.json`
  - non-invasive reflection overhead summary and threshold checks
- `*-bootstrap-generality.json`
  - capability coverage + correctness proof status

Quick interpretation checklist:

- Runtime competitiveness: compare `runtime_p50_avg_ms`
- Compile throughput: compare `full_compile_avg_ms`
- Incremental effectiveness: inspect `incremental_reduction_avg_pct`
- 100k scale bottleneck: check `scale_curve/100000/*`
- Python boundary cost: check `summary/ordered_by_loop_avg_ms`

## Latest Snapshot (Measured on February 16, 2026)

Source reports:

- `results/1771185238357-scenario-matrix.json`
- `results/1771228834821-advanced-pipeline.json`
- `results/1771230408116-python-interop.json`
- `results/1771230417893-bootstrap-generality.json`

Highlights:

- Scenario matrix incremental reduction avg:
  - Sengoo: `95.99%`
  - C++: `-2.28%`
  - Rust: `-4.95%`
  - Python: `2.61%`
- Advanced pipeline 100k e2e (Sengoo): `967.10ms`
- Python interop loop avg (ms):
  - Python native: `2.184`
  - Sengoo Runtime: `2.665`
  - C++ CPython API: `2.919`
  - Rust PyO3: `2.930`
- Bootstrap proof: `pass` (`6/6` scenarios passed)

## CI Gate Mapping

- `scripts/advanced-kpi-gate.py`
  - enforces real incremental + scale + stage budgets
- `scripts/interop-bootstrap-gate.py`
  - enforces interop competitiveness + bootstrap generality pass
- `scripts/e2e-smoke.sh` / `scripts/e2e-smoke.ps1`
  - validates baseline compile pipeline behavior

Use `--mode hard` in CI to fail the workflow when thresholds are violated.

## Troubleshooting

- `cannot resolve Sengoo source root`
  - set `SENGOO_ROOT` explicitly.
- `clang++ not found`
  - install LLVM/Clang or expect C++ runner to be skipped.
- cargo network failures
  - script will try offline build fallback when applicable.
- large variance in local timings
  - close background heavy processes and repeat with more samples.

---

## 中文版

# Sengoo Bench（中文说明）

这是 Sengoo 的独立基准与验证仓库，目标是让性能/正确性数据**可复现、可对比、可进 CI gate**。

## 这个仓库能做什么

- 四语言对比（Sengoo / C++ / Rust / Python）
- 真实增量场景（改循环体、改函数签名、加新函数）
- 规模曲线（1k / 10k / 100k LOC）
- 链接占比与阶段拆分分析
- Python 互操作基准（Sengoo Runtime / Rust PyO3 / C++ CPython API / Python 原生）
- Bootstrap 通用性证明（多能力覆盖 + 正确性）
- 软/硬门禁脚本（CI 可直接接入）
- 端到端 smoke 脚本（工具链健康检查）

## 目录结构

```text
bench/
|-- scripts/                       # perf/e2e/gate 脚本
|-- suites/                        # 基准套件资源
|-- tests/                         # 示例与可直接运行脚本
|   `-- python_interop_example.py  # Python 互操作直接示例（新增）
|-- results/                       # 结果 JSON（git 忽略）
|-- scenario_matrix_bench.py
|-- advanced_pipeline_bench.py
|-- python_interop_bench.py
|-- bootstrap_generality_bench.py
`-- README.md
```

## 依赖要求

必需：

- Python 3.10+
- Rust 工具链（`cargo` / `rustc`）
- 可访问 Sengoo 源码目录（通过 `SENGOO_ROOT` 或同级目录）

建议（用于完整对比）：

- LLVM/Clang（`clang` / `clang++`）
- C++ 运行环境

## clone 后如何复现

### 推荐方式：同级目录

```bash
cd /path/to/workspace
git clone https://github.com/Hyper66666/Sengoo.git
git clone https://github.com/Hyper66666/bench.git
cd bench
```

PowerShell:

```powershell
Set-Location C:\path\to\workspace
git clone https://github.com/Hyper66666/Sengoo.git
git clone https://github.com/Hyper66666/bench.git
Set-Location .\bench
```

### 非同级目录

手动设置 `SENGOO_ROOT`：

```bash
export SENGOO_ROOT=/absolute/path/to/Sengoo
```

```powershell
$env:SENGOO_ROOT = "C:\absolute\path\to\Sengoo"
```

### 环境检查

```bash
python --version
cargo --version
clang --version
```

## 推荐测试流程

### 1）先跑 smoke（快速确认环境）

```bash
bash ./scripts/e2e-smoke.sh
```

```powershell
powershell -File .\scripts\e2e-smoke.ps1
```

### 2）运行新增 Python 互操作示例（可直接出结果）

```bash
python ./tests/python_interop_example.py
```

```powershell
python .\tests\python_interop_example.py
```

快速参数：

```bash
python ./tests/python_interop_example.py --calls 3000 --samples 2 --warmup 0
```

该脚本会自动：

- 调用 `python_interop_bench.py`
- 识别本次生成的 JSON 报告路径
- 输出精简结果表（`loop_avg_ms`、`calls_per_sec`、`vs_python_native_pct`）

### 3）跑完整核心基准

```bash
python ./scenario_matrix_bench.py
python ./advanced_pipeline_bench.py
python ./python_interop_bench.py
python ./bootstrap_generality_bench.py
```

### 4）执行 gate

```bash
python ./scripts/advanced-kpi-gate.py --mode soft --sample ./results/<advanced-report>.json
python ./scripts/interop-bootstrap-gate.py --mode soft --interop-sample ./results/<python-interop-report>.json --bootstrap-sample ./results/<bootstrap-report>.json
```

## 结果怎么看

输出都在 `results/`：

- `*-scenario-matrix.json`：通用场景四语言矩阵
- `*-advanced-pipeline.json`：真实增量 + 规模曲线 + 链接占比
- `*-python-interop.json`：Python 边界开销与吞吐
- `*-bootstrap-generality.json`：能力覆盖和正确性证明

快速关注点：

- 运行时：`runtime_p50_avg_ms`
- 全量编译：`full_compile_avg_ms`
- 增量收益：`incremental_reduction_avg_pct`
- 100k 瓶颈：`scale_curve/100000/*`
- 互操作边界：`summary/ordered_by_loop_avg_ms`

## 最新快照（2026-02-16）

对应文件：

- `results/1771185238357-scenario-matrix.json`
- `results/1771228834821-advanced-pipeline.json`
- `results/1771230408116-python-interop.json`
- `results/1771230417893-bootstrap-generality.json`

关键数据：

- 增量平均收益：Sengoo `95.99%`
- 100k 端到端：Sengoo `967.10ms`
- 互操作 loop 平均：Sengoo Runtime `2.665ms`
- Bootstrap：`pass`（`6/6`）

## 常见问题

- `cannot resolve Sengoo source root`
  - 设置 `SENGOO_ROOT`。
- `clang++ not found`
  - 安装 Clang，或接受 C++ runner 被跳过。
- cargo 下载失败
  - 脚本会尽量走 offline fallback。
- 本地波动大
  - 关闭高负载进程并增加采样次数。
