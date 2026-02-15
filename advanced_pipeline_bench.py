#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


INCREMENTAL_ITERS = 3
SCALE_LOC_BUCKETS = [1000, 10000, 100000]
SCALE_ITERS_BY_LOC = {
    1000: 3,
    10000: 2,
    100000: 1,
}


INCREMENTAL_SCENARIOS = [
    "loop_body_change",
    "function_signature_change",
    "add_new_function",
]


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(math.ceil((len(ordered) - 1) * p))
    return ordered[idx]


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def require_tool(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise RuntimeError(f"required tool not found in PATH: {name}")
    return resolved


def run_checked(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def measure_command_ms(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> float:
    started = time.perf_counter()
    run_checked(cmd, cwd=cwd, env=env)
    return (time.perf_counter() - started) * 1000.0


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return
    path.write_text(content, encoding="utf-8", newline="\n")


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clear_pycache(path: Path) -> None:
    for pycache in path.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)


def exe_name(stem: str) -> str:
    return f"{stem}.exe" if sys.platform.startswith("win") else stem


def resolve_sengoo_root(bench_root: Path) -> Path:
    env_root = os.environ.get("SENGOO_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not (root / "Cargo.toml").exists():
            raise RuntimeError(f"SENGOO_ROOT does not look like Sengoo root: {root}")
        return root

    candidate = bench_root.parent
    if (candidate / "Cargo.toml").exists() and (candidate / "tools" / "sgc" / "src" / "main.rs").exists():
        return candidate

    raise RuntimeError("cannot resolve Sengoo source root; set SENGOO_ROOT")


def resolve_sgc_binary(sengoo_root: Path) -> Path:
    for candidate in [
        sengoo_root / "target" / "release" / exe_name("sgc"),
        sengoo_root / "target" / "debug" / exe_name("sgc"),
    ]:
        if candidate.exists():
            return candidate
    raise RuntimeError("sgc binary not found; run `cargo build -p sgc --release` first")


def render_incremental_sources_sengoo(scenario: str, mutated: bool) -> dict[str, str]:
    if scenario == "loop_body_change":
        body = "acc = acc + mix(i)" if not mutated else "acc = acc + mix(i) - (i % 3)"
        return {
            "main.sg": f"""def bump(x: i64) -> i64 {{
    x + 1
}}

def mix(x: i64) -> i64 {{
    bump(x) * 2
}}

def calc(n: i64) -> i64 {{
    let i = 0
    let acc = 0
    while i < n {{
        {body}
        i = i + 1
    }}
    acc
}}

def main() -> i64 {{
    print(calc(100000))
    0
}}
""",
        }
    if scenario == "function_signature_change":
        if not mutated:
            main = """def bump(x: i64) -> i64 {
    x + 1
}

def mix(x: i64) -> i64 {
    bump(x) * 2
}

def calc(n: i64) -> i64 {
    let i = 0
    let acc = 0
    while i < n {
        acc = acc + mix(i)
        i = i + 1
    }
    acc
}

def main() -> i64 {
    print(calc(100000))
    0
}
"""
        else:
            main = """def bump(x: i64, k: i64) -> i64 {
    x + k
}

def mix(x: i64, scale: i64) -> i64 {
    bump(x, 1) * scale
}

def calc(n: i64) -> i64 {
    let i = 0
    let acc = 0
    while i < n {
        acc = acc + mix(i, 2)
        i = i + 1
    }
    acc
}

def main() -> i64 {
    print(calc(100000))
    0
}
"""
        return {"main.sg": main}
    if scenario == "add_new_function":
        extra = ""
        use_extra = "acc = acc + mix(i)"
        if mutated:
            extra = """
def extra(v: i64) -> i64 {
    v + 3
}
"""
            use_extra = "acc = acc + mix(i) + extra(i % 5)"
        return {
            "main.sg": f"""def bump(x: i64) -> i64 {{
    x + 1
}}

def mix(x: i64) -> i64 {{
    bump(x) * 2
}}
{extra}
def calc(n: i64) -> i64 {{
    let i = 0
    let acc = 0
    while i < n {{
        {use_extra}
        i = i + 1
    }}
    acc
}}

def main() -> i64 {{
    print(calc(100000))
    0
}}
""",
        }
    raise ValueError(f"unknown scenario: {scenario}")


def render_incremental_sources_cpp(scenario: str, mutated: bool) -> dict[str, str]:
    if scenario == "loop_body_change":
        body = "acc += mix(i);" if not mutated else "acc += mix(i) - (i % 3);"
        return {
            "pch.hpp": """#pragma once
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
""",
            "math_util.hpp": """#pragma once
long long bump(long long x);
long long mix(long long x);
""",
            "math_util.cpp": """#include "math_util.hpp"
long long bump(long long x) { return x + 1; }
long long mix(long long x) { return bump(x) * 2; }
""",
            "main.cpp": f"""#include "math_util.hpp"
int main() {{
    long long acc = 0;
    for (long long i = 0; i < 100000; ++i) {{
        {body}
    }}
    std::cout << acc << "\\n";
    return 0;
}}
""",
        }
    if scenario == "function_signature_change":
        if not mutated:
            util_h = """#pragma once
long long bump(long long x);
long long mix(long long x);
"""
            util_cpp = """#include "math_util.hpp"
long long bump(long long x) { return x + 1; }
long long mix(long long x) { return bump(x) * 2; }
"""
            main_cpp = """#include "math_util.hpp"
int main() {
    long long acc = 0;
    for (long long i = 0; i < 100000; ++i) {
        acc += mix(i);
    }
    std::cout << acc << "\\n";
    return 0;
}
"""
        else:
            util_h = """#pragma once
long long bump(long long x, long long k);
long long mix(long long x, long long scale);
"""
            util_cpp = """#include "math_util.hpp"
long long bump(long long x, long long k) { return x + k; }
long long mix(long long x, long long scale) { return bump(x, 1) * scale; }
"""
            main_cpp = """#include "math_util.hpp"
int main() {
    long long acc = 0;
    for (long long i = 0; i < 100000; ++i) {
        acc += mix(i, 2);
    }
    std::cout << acc << "\\n";
    return 0;
}
"""
        return {
            "pch.hpp": """#pragma once
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
""",
            "math_util.hpp": util_h,
            "math_util.cpp": util_cpp,
            "main.cpp": main_cpp,
        }
    if scenario == "add_new_function":
        extra_fn = ""
        body = "acc += mix(i);"
        if mutated:
            extra_fn = """
static inline long long extra(long long v) {
    return v + 3;
}
"""
            body = "acc += mix(i) + extra(i % 5);"
        return {
            "pch.hpp": """#pragma once
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
""",
            "math_util.hpp": """#pragma once
long long bump(long long x);
long long mix(long long x);
""",
            "math_util.cpp": """#include "math_util.hpp"
long long bump(long long x) { return x + 1; }
long long mix(long long x) { return bump(x) * 2; }
""",
            "main.cpp": f"""#include "math_util.hpp"
{extra_fn}
int main() {{
    long long acc = 0;
    for (long long i = 0; i < 100000; ++i) {{
        {body}
    }}
    std::cout << acc << "\\n";
    return 0;
}}
""",
        }
    raise ValueError(f"unknown scenario: {scenario}")


def render_incremental_sources_rust(scenario: str, mutated: bool) -> dict[str, str]:
    cargo_toml = """[package]
name = "inc_case"
version = "0.1.0"
edition = "2021"

[workspace]
"""
    if scenario == "loop_body_change":
        body = "acc += util::mix(i);" if not mutated else "acc += util::mix(i) - (i % 3);"
        main_rs = f"""mod util;

fn main() {{
    let mut acc: i64 = 0;
    for i in 0..100000i64 {{
        {body}
    }}
    println!("{{}}", acc);
}}
"""
        util_rs = """pub fn bump(x: i64) -> i64 {
    x + 1
}

pub fn mix(x: i64) -> i64 {
    bump(x) * 2
}
"""
        return {"Cargo.toml": cargo_toml, "src/main.rs": main_rs, "src/util.rs": util_rs}
    if scenario == "function_signature_change":
        if not mutated:
            main_rs = """mod util;

fn main() {
    let mut acc: i64 = 0;
    for i in 0..100000i64 {
        acc += util::mix(i);
    }
    println!("{}", acc);
}
"""
            util_rs = """pub fn bump(x: i64) -> i64 {
    x + 1
}

pub fn mix(x: i64) -> i64 {
    bump(x) * 2
}
"""
        else:
            main_rs = """mod util;

fn main() {
    let mut acc: i64 = 0;
    for i in 0..100000i64 {
        acc += util::mix(i, 2);
    }
    println!("{}", acc);
}
"""
            util_rs = """pub fn bump(x: i64, k: i64) -> i64 {
    x + k
}

pub fn mix(x: i64, scale: i64) -> i64 {
    bump(x, 1) * scale
}
"""
        return {"Cargo.toml": cargo_toml, "src/main.rs": main_rs, "src/util.rs": util_rs}
    if scenario == "add_new_function":
        extra = ""
        body = "acc += util::mix(i);"
        if mutated:
            extra = """
#[inline(always)]
fn extra(v: i64) -> i64 {
    v + 3
}
"""
            body = "acc += util::mix(i) + extra(i % 5);"
        main_rs = f"""mod util;
{extra}
fn main() {{
    let mut acc: i64 = 0;
    for i in 0..100000i64 {{
        {body}
    }}
    println!("{{}}", acc);
}}
"""
        util_rs = """pub fn bump(x: i64) -> i64 {
    x + 1
}

pub fn mix(x: i64) -> i64 {
    bump(x) * 2
}
"""
        return {"Cargo.toml": cargo_toml, "src/main.rs": main_rs, "src/util.rs": util_rs}
    raise ValueError(f"unknown scenario: {scenario}")


def render_incremental_sources_python(scenario: str, mutated: bool) -> dict[str, str]:
    if scenario == "loop_body_change":
        body = "acc += mix(i)" if not mutated else "acc += mix(i) - (i % 3)"
        return {
            "util.py": """def bump(x: int) -> int:
    return x + 1

def mix(x: int) -> int:
    return bump(x) * 2
""",
            "main.py": f"""from util import mix

def main() -> None:
    acc = 0
    for i in range(100000):
        {body}
    print(acc)

if __name__ == "__main__":
    main()
""",
        }
    if scenario == "function_signature_change":
        if not mutated:
            util_py = """def bump(x: int) -> int:
    return x + 1

def mix(x: int) -> int:
    return bump(x) * 2
"""
            main_py = """from util import mix

def main() -> None:
    acc = 0
    for i in range(100000):
        acc += mix(i)
    print(acc)

if __name__ == "__main__":
    main()
"""
        else:
            util_py = """def bump(x: int, k: int) -> int:
    return x + k

def mix(x: int, scale: int) -> int:
    return bump(x, 1) * scale
"""
            main_py = """from util import mix

def main() -> None:
    acc = 0
    for i in range(100000):
        acc += mix(i, 2)
    print(acc)

if __name__ == "__main__":
    main()
"""
        return {"util.py": util_py, "main.py": main_py}
    if scenario == "add_new_function":
        extra = ""
        body = "acc += mix(i)"
        if mutated:
            extra = """
def extra(v: int) -> int:
    return v + 3
"""
            body = "acc += mix(i) + extra(i % 5)"
        return {
            "util.py": """def bump(x: int) -> int:
    return x + 1

def mix(x: int) -> int:
    return bump(x) * 2
""",
            "main.py": f"""from util import mix
{extra}
def main() -> None:
    acc = 0
    for i in range(100000):
        {body}
    print(acc)

if __name__ == "__main__":
    main()
""",
        }
    raise ValueError(f"unknown scenario: {scenario}")


def write_sources(base_dir: Path, file_map: dict[str, str]) -> None:
    for rel_path, content in file_map.items():
        write_file(base_dir / rel_path, content)


def build_cpp_pch(clangpp: str, cpp_dir: Path) -> None:
    pch_header = cpp_dir / "pch.hpp"
    pch_file = cpp_dir / "pch.hpp.pch"
    run_checked(
        [clangpp, "-std=c++20", "-O2", "-x", "c++-header", str(pch_header), "-o", str(pch_file)],
        cwd=cpp_dir,
    )


def cpp_full_build_ms(clangpp: str, cpp_dir: Path) -> float:
    main_obj = cpp_dir / "main.obj"
    util_obj = cpp_dir / "math_util.obj"
    out_exe = cpp_dir / exe_name("app")
    pch_file = cpp_dir / "pch.hpp.pch"
    started = time.perf_counter()
    run_checked(
        [clangpp, "-std=c++20", "-O2", "-include-pch", str(pch_file), "-c", str(cpp_dir / "math_util.cpp"), "-o", str(util_obj)],
        cwd=cpp_dir,
    )
    run_checked(
        [clangpp, "-std=c++20", "-O2", "-include-pch", str(pch_file), "-c", str(cpp_dir / "main.cpp"), "-o", str(main_obj)],
        cwd=cpp_dir,
    )
    run_checked([clangpp, str(main_obj), str(util_obj), "-o", str(out_exe)], cwd=cpp_dir)
    return (time.perf_counter() - started) * 1000.0


def cpp_incremental_build_ms(clangpp: str, cpp_dir: Path, changed_units: list[str]) -> float:
    main_obj = cpp_dir / "main.obj"
    util_obj = cpp_dir / "math_util.obj"
    out_exe = cpp_dir / exe_name("app")
    pch_file = cpp_dir / "pch.hpp.pch"
    started = time.perf_counter()
    if "math_util.cpp" in changed_units:
        run_checked(
            [clangpp, "-std=c++20", "-O2", "-include-pch", str(pch_file), "-c", str(cpp_dir / "math_util.cpp"), "-o", str(util_obj)],
            cwd=cpp_dir,
        )
    if "main.cpp" in changed_units:
        run_checked(
            [clangpp, "-std=c++20", "-O2", "-include-pch", str(pch_file), "-c", str(cpp_dir / "main.cpp"), "-o", str(main_obj)],
            cwd=cpp_dir,
        )
    run_checked([clangpp, str(main_obj), str(util_obj), "-o", str(out_exe)], cwd=cpp_dir)
    return (time.perf_counter() - started) * 1000.0


def measure_real_incremental(
    bench_root: Path,
    sgc_bin: Path,
    clangpp: str,
    cargo: str,
    py: str,
    sengoo_root: Path,
) -> dict[str, dict[str, Any]]:
    work = bench_root / ".advanced-work" / "real-incremental"
    clear_dir(work)
    results: dict[str, dict[str, Any]] = {}

    rust_env = {
        "CARGO_INCREMENTAL": "1",
    }

    for scenario in INCREMENTAL_SCENARIOS:
        results[scenario] = {}

        # Sengoo
        sg_dir = work / "sengoo" / scenario
        clear_dir(sg_dir)
        before_samples: list[float] = []
        after_samples: list[float] = []
        for _ in range(INCREMENTAL_ITERS):
            write_sources(sg_dir, render_incremental_sources_sengoo(scenario, mutated=False))
            before_samples.append(
                measure_command_ms(
                    [str(sgc_bin), "build", str(sg_dir / "main.sg"), "-O", "2", "--force-rebuild"],
                    cwd=sengoo_root,
                )
            )
            write_sources(sg_dir, render_incremental_sources_sengoo(scenario, mutated=True))
            after_samples.append(
                measure_command_ms(
                    [str(sgc_bin), "build", str(sg_dir / "main.sg"), "-O", "2"],
                    cwd=sengoo_root,
                )
            )
        bavg = average(before_samples) or 0.0
        aavg = average(after_samples) or 0.0
        results[scenario]["sengoo"] = {
            "before_samples_ms": before_samples,
            "after_samples_ms": after_samples,
            "before_avg_ms": bavg,
            "after_avg_ms": aavg,
            "reduction_pct": ((bavg - aavg) / bavg * 100.0) if bavg > 0 else None,
        }

        # C++ with PCH
        cpp_dir = work / "cpp" / scenario
        clear_dir(cpp_dir)
        before_samples = []
        after_samples = []
        changed_units = ["main.cpp"]
        if scenario == "function_signature_change":
            changed_units = ["math_util.cpp", "main.cpp"]
        for _ in range(INCREMENTAL_ITERS):
            write_sources(cpp_dir, render_incremental_sources_cpp(scenario, mutated=False))
            build_cpp_pch(clangpp, cpp_dir)
            before_samples.append(cpp_full_build_ms(clangpp, cpp_dir))
            write_sources(cpp_dir, render_incremental_sources_cpp(scenario, mutated=True))
            after_samples.append(cpp_incremental_build_ms(clangpp, cpp_dir, changed_units))
        bavg = average(before_samples) or 0.0
        aavg = average(after_samples) or 0.0
        results[scenario]["cpp"] = {
            "before_samples_ms": before_samples,
            "after_samples_ms": after_samples,
            "before_avg_ms": bavg,
            "after_avg_ms": aavg,
            "reduction_pct": ((bavg - aavg) / bavg * 100.0) if bavg > 0 else None,
            "fairness": {"pch_enabled": True},
        }

        # Rust (cargo incremental)
        rust_dir = work / "rust" / scenario
        clear_dir(rust_dir)
        before_samples = []
        after_samples = []
        for _ in range(INCREMENTAL_ITERS):
            write_sources(rust_dir, render_incremental_sources_rust(scenario, mutated=False))
            target_dir = rust_dir / "target"
            if target_dir.exists():
                shutil.rmtree(target_dir)
            before_samples.append(
                measure_command_ms([cargo, "build", "--quiet"], cwd=rust_dir, env=rust_env)
            )
            write_sources(rust_dir, render_incremental_sources_rust(scenario, mutated=True))
            after_samples.append(
                measure_command_ms([cargo, "build", "--quiet"], cwd=rust_dir, env=rust_env)
            )
        bavg = average(before_samples) or 0.0
        aavg = average(after_samples) or 0.0
        results[scenario]["rust"] = {
            "before_samples_ms": before_samples,
            "after_samples_ms": after_samples,
            "before_avg_ms": bavg,
            "after_avg_ms": aavg,
            "reduction_pct": ((bavg - aavg) / bavg * 100.0) if bavg > 0 else None,
            "fairness": {"cargo_incremental": True},
        }

        # Python
        py_dir = work / "python" / scenario
        clear_dir(py_dir)
        before_samples = []
        after_samples = []
        for _ in range(INCREMENTAL_ITERS):
            write_sources(py_dir, render_incremental_sources_python(scenario, mutated=False))
            clear_pycache(py_dir)
            before_samples.append(measure_command_ms([py, "-m", "compileall", "-q", str(py_dir)], cwd=py_dir))
            write_sources(py_dir, render_incremental_sources_python(scenario, mutated=True))
            after_samples.append(measure_command_ms([py, "-m", "compileall", "-q", str(py_dir)], cwd=py_dir))
        bavg = average(before_samples) or 0.0
        aavg = average(after_samples) or 0.0
        results[scenario]["python"] = {
            "before_samples_ms": before_samples,
            "after_samples_ms": after_samples,
            "before_avg_ms": bavg,
            "after_avg_ms": aavg,
            "reduction_pct": ((bavg - aavg) / bavg * 100.0) if bavg > 0 else None,
        }

    return results


def make_scale_source_sengoo(target_loc: int) -> str:
    lines = []
    fn_count = max(50, target_loc // 4)
    for i in range(fn_count):
        lines.append(f"def f{i}(x: i64) -> i64 {{")
        lines.append(f"    x + {i % 7}")
        lines.append("}")
        lines.append("")
    lines.extend(
        [
            "def main() -> i64 {",
            "    let acc = 0",
            "    let i = 0",
            "    while i < 1000 {",
            f"        acc = acc + f{fn_count - 1}(i)",
            "        i = i + 1",
            "    }",
            "    print(acc)",
            "    0",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def make_scale_source_cpp(target_loc: int) -> str:
    lines = ['#include "pch.hpp"', "", "static inline long long seed(long long x) { return x; }", ""]
    fn_count = max(50, target_loc // 3)
    for i in range(fn_count):
        lines.append(f"static inline long long f{i}(long long x) {{ return x + {i % 7}; }}")
    lines.extend(
        [
            "",
            "int main() {",
            "    long long acc = 0;",
            "    for (long long i = 0; i < 1000; ++i) {",
            f"        acc += f{fn_count - 1}(i);",
            "    }",
            '    std::cout << acc << "\\n";',
            "    return 0;",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def make_scale_source_rust(target_loc: int) -> str:
    lines = []
    fn_count = max(50, target_loc // 3)
    for i in range(fn_count):
        lines.append(f"#[inline(always)] fn f{i}(x: i64) -> i64 {{ x + {i % 7} }}")
    lines.extend(
        [
            "",
            "fn main() {",
            "    let mut acc: i64 = 0;",
            "    for i in 0..1000i64 {",
            f"        acc += f{fn_count - 1}(i);",
            "    }",
            '    println!("{}", acc);',
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def make_scale_source_python(target_loc: int) -> str:
    lines = []
    fn_count = max(50, target_loc // 3)
    for i in range(fn_count):
        lines.append(f"def f{i}(x: int) -> int:")
        lines.append(f"    return x + {i % 7}")
        lines.append("")
    lines.extend(
        [
            "def main() -> None:",
            "    acc = 0",
            "    for i in range(1000):",
            f"        acc += f{fn_count - 1}(i)",
            "    print(acc)",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]
    )
    return "\n".join(lines)


def measure_scale_curve(
    bench_root: Path,
    sgc_bin: Path,
    clangpp: str,
    cargo: str,
    py: str,
    sengoo_root: Path,
) -> dict[str, dict[str, Any]]:
    work = bench_root / ".advanced-work" / "scale-curve"
    clear_dir(work)
    results: dict[str, dict[str, Any]] = {}

    rust_env = {
        "CARGO_INCREMENTAL": "1",
    }
    runtime_c = sengoo_root / "tools" / "stdlib" / "runtime.c"
    if not runtime_c.exists():
        raise RuntimeError(f"runtime source not found: {runtime_c}")
    runtime_obj = work / "sengoo_runtime.obj"
    run_checked([clangpp, "-O2", "-x", "c", "-c", str(runtime_c), "-o", str(runtime_obj)], cwd=sengoo_root)

    for loc in SCALE_LOC_BUCKETS:
        iters = SCALE_ITERS_BY_LOC.get(loc, 1)
        loc_key = str(loc)
        results[loc_key] = {}

        # Sengoo (front-end LLVM + backend obj + link)
        sg_dir = work / "sengoo" / loc_key
        clear_dir(sg_dir)
        write_file(sg_dir / "main.sg", make_scale_source_sengoo(loc))
        front_samples: list[float] = []
        backend_samples: list[float] = []
        link_samples: list[float] = []
        e2e_samples: list[float] = []
        for _ in range(iters):
            ll = sg_dir / "main.ll"
            obj = sg_dir / "main.obj"
            exe = sg_dir / exe_name("main")
            front_ms = measure_command_ms(
                [str(sgc_bin), "build", str(sg_dir / "main.sg"), "-O", "2", "--emit-llvm", "-o", str(ll), "--force-rebuild"],
                cwd=sengoo_root,
            )
            backend_ms = measure_command_ms([clangpp, "-O2", "-x", "ir", "-c", str(ll), "-o", str(obj)], cwd=sg_dir)
            link_ms = measure_command_ms([clangpp, str(obj), str(runtime_obj), "-o", str(exe)], cwd=sg_dir)
            front_samples.append(front_ms)
            backend_samples.append(backend_ms)
            link_samples.append(link_ms)
            e2e_samples.append(front_ms + backend_ms + link_ms)
        results[loc_key]["sengoo"] = {
            "compile_frontend_llvm_avg_ms": average(front_samples),
            "codegen_obj_avg_ms": average(backend_samples),
            "link_avg_ms": average(link_samples),
            "e2e_avg_ms": average(e2e_samples),
            "e2e_p50_ms": percentile(e2e_samples, 0.50),
        }

        # C++ with PCH (compile obj + link)
        cpp_dir = work / "cpp" / loc_key
        clear_dir(cpp_dir)
        write_file(
            cpp_dir / "pch.hpp",
            """#pragma once
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
""",
        )
        write_file(cpp_dir / "main.cpp", make_scale_source_cpp(loc))
        build_cpp_pch(clangpp, cpp_dir)
        compile_samples = []
        link_samples = []
        e2e_samples = []
        for _ in range(iters):
            obj = cpp_dir / "main.obj"
            exe = cpp_dir / exe_name("main")
            compile_ms = measure_command_ms(
                [clangpp, "-std=c++20", "-O2", "-include-pch", str(cpp_dir / "pch.hpp.pch"), "-c", str(cpp_dir / "main.cpp"), "-o", str(obj)],
                cwd=cpp_dir,
            )
            link_ms = measure_command_ms([clangpp, str(obj), "-o", str(exe)], cwd=cpp_dir)
            compile_samples.append(compile_ms)
            link_samples.append(link_ms)
            e2e_samples.append(compile_ms + link_ms)
        results[loc_key]["cpp"] = {
            "compile_obj_avg_ms": average(compile_samples),
            "link_avg_ms": average(link_samples),
            "e2e_avg_ms": average(e2e_samples),
            "e2e_p50_ms": percentile(e2e_samples, 0.50),
            "fairness": {"pch_enabled": True},
        }

        # Rust (cargo build e2e includes link)
        rust_dir = work / "rust" / loc_key
        clear_dir(rust_dir)
        write_file(
            rust_dir / "Cargo.toml",
            """[package]
name = "scale_case"
version = "0.1.0"
edition = "2021"

[workspace]
""",
        )
        write_file(rust_dir / "src" / "main.rs", make_scale_source_rust(loc))
        e2e_samples = []
        for _ in range(iters):
            target_dir = rust_dir / "target"
            if target_dir.exists():
                shutil.rmtree(target_dir)
            e2e_samples.append(
                measure_command_ms([cargo, "build", "--quiet"], cwd=rust_dir, env=rust_env)
            )
        results[loc_key]["rust"] = {
            "e2e_avg_ms": average(e2e_samples),
            "e2e_p50_ms": percentile(e2e_samples, 0.50),
            "fairness": {"cargo_incremental": True},
            "note": "cargo build includes compile+link; split link-only timing is not isolated",
        }

        # Python (py_compile, no link stage)
        py_dir = work / "python" / loc_key
        clear_dir(py_dir)
        write_file(py_dir / "main.py", make_scale_source_python(loc))
        compile_samples = []
        for _ in range(iters):
            clear_pycache(py_dir)
            compile_samples.append(measure_command_ms([py, "-m", "py_compile", str(py_dir / "main.py")], cwd=py_dir))
        results[loc_key]["python"] = {
            "compile_avg_ms": average(compile_samples),
            "e2e_avg_ms": average(compile_samples),
            "e2e_p50_ms": percentile(compile_samples, 0.50),
            "note": "python has no native link stage in this benchmark",
        }

    return results


def print_incremental_tables(real_incremental: dict[str, dict[str, Any]]) -> None:
    langs = ["sengoo", "cpp", "rust", "python"]
    print("")
    print("Real Incremental Scenarios (before/after/reduction)")
    for scenario in INCREMENTAL_SCENARIOS:
        print("")
        print(f"Scenario: {scenario}")
        print("| Language | Before avg (ms) | After avg (ms) | Reduction (%) |")
        print("|---|---:|---:|---:|")
        for lang in langs:
            metrics = real_incremental[scenario][lang]
            reduction = metrics.get("reduction_pct")
            red_str = "n/a" if reduction is None else f"{reduction:.2f}"
            print(
                f"| {lang} | {metrics['before_avg_ms']:.2f} | {metrics['after_avg_ms']:.2f} | {red_str} |"
            )


def print_scale_tables(scale_curve: dict[str, dict[str, Any]]) -> None:
    langs = ["sengoo", "cpp", "rust", "python"]
    print("")
    print("Scale Curve: E2E Build Time (includes link where applicable)")
    print("| LOC | Sengoo (ms) | C++ (ms) | Rust (ms) | Python (ms) |")
    print("|---:|---:|---:|---:|---:|")
    for loc_key in [str(x) for x in SCALE_LOC_BUCKETS]:
        row = []
        for lang in langs:
            row.append(f"{scale_curve[loc_key][lang]['e2e_avg_ms']:.2f}")
        print(f"| {loc_key} | {' | '.join(row)} |")

    print("")
    print("Link Share (Sengoo/C++)")
    print("| LOC | Sengoo link share (%) | C++ link share (%) |")
    print("|---:|---:|---:|")
    for loc_key in [str(x) for x in SCALE_LOC_BUCKETS]:
        sg = scale_curve[loc_key]["sengoo"]
        cpp = scale_curve[loc_key]["cpp"]
        sg_share = (sg["link_avg_ms"] / sg["e2e_avg_ms"] * 100.0) if sg["e2e_avg_ms"] else 0.0
        cpp_share = (cpp["link_avg_ms"] / cpp["e2e_avg_ms"] * 100.0) if cpp["e2e_avg_ms"] else 0.0
        print(f"| {loc_key} | {sg_share:.2f} | {cpp_share:.2f} |")


def main() -> int:
    bench_root = Path(__file__).resolve().parent
    sengoo_root = resolve_sengoo_root(bench_root)
    results_dir = bench_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    sgc_bin = resolve_sgc_binary(sengoo_root)
    clangpp = require_tool("clang++")
    cargo = require_tool("cargo")
    py = sys.executable

    real_incremental = measure_real_incremental(bench_root, sgc_bin, clangpp, cargo, py, sengoo_root)
    scale_curve = measure_scale_curve(bench_root, sgc_bin, clangpp, cargo, py, sengoo_root)

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_unix_ms": now_unix_ms(),
        "config": {
            "incremental_iterations": INCREMENTAL_ITERS,
            "scale_loc_buckets": SCALE_LOC_BUCKETS,
            "scale_iterations_by_loc": SCALE_ITERS_BY_LOC,
        },
        "fairness": {
            "cpp": "precompiled header (PCH) enabled",
            "rust": "cargo incremental enabled (CARGO_INCREMENTAL=1)",
        },
        "real_incremental": real_incremental,
        "scale_curve": scale_curve,
        "notes": [
            "Scale curve e2e includes link time for compiled languages.",
            "Sengoo uses split timing: front-end LLVM emit + clang IR codegen + clang link.",
            "Rust uses cargo build e2e timing; split link-only timing is not isolated.",
        ],
    }

    out_path = results_dir / f"{now_unix_ms()}-advanced-pipeline.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

    print(f"Advanced bench report: {out_path}")
    print_incremental_tables(real_incremental)
    print_scale_tables(scale_curve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

