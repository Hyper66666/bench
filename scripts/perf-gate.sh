#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="soft"
SAMPLE=""
BASELINE="${BENCH_ROOT}/baseline.json"

resolve_path() {
  local candidate="$1"
  if [[ -f "$candidate" ]]; then
    echo "$candidate"
    return
  fi
  if [[ -f "${BENCH_ROOT}/$candidate" ]]; then
    echo "${BENCH_ROOT}/$candidate"
    return
  fi
  echo "$candidate"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --sample)
      SAMPLE="${2:-}"
      shift 2
      ;;
    --baseline)
      BASELINE="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

SAMPLE="$(resolve_path "$SAMPLE")"
BASELINE="$(resolve_path "$BASELINE")"

if [[ "$MODE" != "soft" && "$MODE" != "hard" ]]; then
  echo "invalid --mode: $MODE (expected soft|hard)" >&2
  exit 2
fi
if [[ -z "$SAMPLE" ]]; then
  echo "--sample is required" >&2
  exit 2
fi
if [[ ! -f "$SAMPLE" ]]; then
  echo "sample report not found: $SAMPLE" >&2
  exit 2
fi
if [[ ! -f "$BASELINE" ]]; then
  echo "baseline file not found: $BASELINE" >&2
  exit 2
fi

python3 - "$MODE" "$SAMPLE" "$BASELINE" <<'PY'
import json
import sys

mode, sample_path, baseline_path = sys.argv[1], sys.argv[2], sys.argv[3]

with open(sample_path, "r", encoding="utf-8-sig") as f:
    report = json.load(f)
with open(baseline_path, "r", encoding="utf-8-sig") as f:
    baseline = json.load(f)

targets = baseline.get("targets", {})
cases = baseline.get("cases", {})

def fmt(v: float) -> str:
    return f"{v:.2f}"

checked = 0
summaries = []
violations = []

kind = report.get("kind")
suite = report.get("suite")
for case in report.get("cases", []):
    name = case.get("name", "<unknown>")
    key = f"{kind}/{suite}/{name}"
    base = cases.get(key)

    if kind == "runtime":
        if not base:
            continue
        curr = case.get("p50_ms")
        prev = base.get("p50_ms")
        target = targets.get("runtime_median_improvement_pct")
        if curr is None or prev is None:
            continue
        checked += 1
        improvement = ((prev - curr) / prev) * 100.0
        summaries.append(
            f"runtime/{name}: improvement={fmt(improvement)}% target={fmt(float(target))}%"
        )
        if target is not None and improvement < float(target):
            violations.append(
                f"runtime/{name} below target ({fmt(improvement)}% < {fmt(float(target))}%)"
            )

    elif kind == "compile":
        if not base:
            continue
        curr = case.get("total_ms")
        prev = base.get("total_ms")
        target = targets.get("full_compile_reduction_pct")
        if curr is None or prev is None:
            continue
        checked += 1
        reduction = ((prev - curr) / prev) * 100.0
        summaries.append(
            f"compile/{name}: reduction={fmt(reduction)}% target={fmt(float(target))}%"
        )
        if target is not None and reduction < float(target):
            violations.append(
                f"compile/{name} below target ({fmt(reduction)}% < {fmt(float(target))}%)"
            )

    elif kind == "incremental":
        before = case.get("before_ms")
        after = case.get("after_ms")
        target = targets.get("incremental_compile_reduction_pct")
        if before is None or after is None or before <= 0:
            continue
        checked += 1
        reduction = ((before - after) / before) * 100.0
        summaries.append(
            f"incremental/{name}: reduction={fmt(reduction)}% target={fmt(float(target))}%"
        )
        if target is not None and reduction < float(target):
            violations.append(
                f"incremental/{name} below target ({fmt(reduction)}% < {fmt(float(target))}%)"
            )

if checked == 0:
    print(f"no comparable benchmark metrics found in {sample_path}", file=sys.stderr)
    sys.exit(2)

print(f"perf-gate mode={mode} sample={sample_path} baseline={baseline_path}")
for line in summaries:
    print(f"  - {line}")

if not violations:
    print("perf-gate PASS")
    sys.exit(0)

print(f"perf-gate found {len(violations)} target violation(s):")
for line in violations:
    print(f"  ! {line}")

if mode == "hard":
    print("perf-gate HARD failure", file=sys.stderr)
    sys.exit(1)

print("perf-gate SOFT warning (not failing build)")
sys.exit(0)
PY
