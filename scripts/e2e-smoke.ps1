$ErrorActionPreference = "Stop"

$benchRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if ($env:SENGOO_ROOT) {
    $projectRoot = Resolve-Path $env:SENGOO_ROOT
}
else {
    $projectRoot = Resolve-Path (Join-Path $benchRoot "..")
}
Set-Location $projectRoot

$sample = Join-Path $benchRoot "tests/hello_print.sg"
$outDir = Join-Path $benchRoot "tests/build/e2e"
$llvmOut = Join-Path $outDir "hello_print.ll"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if (-not (Test-Path -LiteralPath $sample)) {
    throw "[e2e] sample not found: $sample"
}

Write-Host "[e2e] checking sample program..."
cargo run -q -p sgc -- check $sample

Write-Host "[e2e] building llvm ir sample..."
cargo run -q -p sgc -- build $sample --emit-llvm --output $llvmOut

if (-not (Test-Path $llvmOut)) {
    throw "[e2e] expected LLVM IR output not found: $llvmOut"
}

Write-Host "[e2e] smoke test passed"
