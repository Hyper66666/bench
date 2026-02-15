param(
    [ValidateSet("soft", "hard")]
    [string]$Mode = "soft",
    [Parameter(Mandatory = $true)]
    [string]$Sample,
    [string]$Baseline = "baseline.json"
)

$ErrorActionPreference = "Stop"
$benchRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

function Resolve-InputPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )
    if (Test-Path -LiteralPath $PathValue) {
        return (Resolve-Path $PathValue).Path
    }
    $candidate = Join-Path $benchRoot $PathValue
    if (Test-Path -LiteralPath $candidate) {
        return (Resolve-Path $candidate).Path
    }
    return $PathValue
}

function Format-Number {
    param(
        [Parameter(Mandatory = $true)]
        [double]$Value
    )
    return [string]::Format("{0:N2}", $Value)
}

$samplePath = Resolve-InputPath -PathValue $Sample
$baselinePath = Resolve-InputPath -PathValue $Baseline

if (-not (Test-Path -LiteralPath $samplePath)) {
    Write-Error "sample report not found: $samplePath"
    exit 2
}
if (-not (Test-Path -LiteralPath $baselinePath)) {
    Write-Error "baseline file not found: $baselinePath"
    exit 2
}

$reportDoc = Get-Content -LiteralPath $samplePath -Raw | ConvertFrom-Json
$baselineDoc = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json

$baselineCaseByKey = @{}
if ($null -ne $baselineDoc -and $null -ne $baselineDoc.cases) {
    foreach ($entry in $baselineDoc.cases.PSObject.Properties) {
        $baselineCaseByKey[$entry.Name] = $entry.Value
    }
}

$targets = $baselineDoc.targets
$summaries = @()
$violations = @()
$checked = 0

$kind = [string]$reportDoc.kind
$suite = [string]$reportDoc.suite
$reportCases = @($reportDoc.cases)

foreach ($reportCase in $reportCases) {
    $caseName = [string]$reportCase.name
    $caseKey = "$kind/$suite/$caseName"
    $baselineCase = $baselineCaseByKey[$caseKey]

    switch -Exact ($kind) {
        "runtime" {
            if ($null -eq $baselineCase -or $null -eq $reportCase.p50_ms -or $null -eq $baselineCase.p50_ms) {
                continue
            }
            $checked++
            $improvement = (($baselineCase.p50_ms - $reportCase.p50_ms) / $baselineCase.p50_ms) * 100.0
            $target = $targets.runtime_median_improvement_pct
            $targetText = "n/a"
            if ($null -ne $target) {
                $targetText = "$(Format-Number -Value ([double]$target))%"
            }
            $summaries += "runtime/${caseName}: improvement=$(Format-Number -Value $improvement)% target=$targetText"
            if ($null -ne $target -and $improvement -lt [double]$target) {
                $violations += "runtime/$caseName below target ($(Format-Number -Value $improvement)% < $(Format-Number -Value ([double]$target))%)"
            }
        }
        "compile" {
            if ($null -eq $baselineCase -or $null -eq $reportCase.total_ms -or $null -eq $baselineCase.total_ms) {
                continue
            }
            $checked++
            $reduction = (($baselineCase.total_ms - $reportCase.total_ms) / $baselineCase.total_ms) * 100.0
            $target = $targets.full_compile_reduction_pct
            $targetText = "n/a"
            if ($null -ne $target) {
                $targetText = "$(Format-Number -Value ([double]$target))%"
            }
            $summaries += "compile/${caseName}: reduction=$(Format-Number -Value $reduction)% target=$targetText"
            if ($null -ne $target -and $reduction -lt [double]$target) {
                $violations += "compile/$caseName below target ($(Format-Number -Value $reduction)% < $(Format-Number -Value ([double]$target))%)"
            }
        }
        "incremental" {
            if ($null -eq $reportCase.before_ms -or $null -eq $reportCase.after_ms -or $reportCase.before_ms -le 0) {
                continue
            }
            $checked++
            $reduction = (($reportCase.before_ms - $reportCase.after_ms) / $reportCase.before_ms) * 100.0
            $target = $targets.incremental_compile_reduction_pct
            $targetText = "n/a"
            if ($null -ne $target) {
                $targetText = "$(Format-Number -Value ([double]$target))%"
            }
            $summaries += "incremental/${caseName}: reduction=$(Format-Number -Value $reduction)% target=$targetText"
            if ($null -ne $target -and $reduction -lt [double]$target) {
                $violations += "incremental/$caseName below target ($(Format-Number -Value $reduction)% < $(Format-Number -Value ([double]$target))%)"
            }
        }
        default {
            continue
        }
    }
}

if ($checked -eq 0) {
    Write-Error "no comparable benchmark metrics found in $samplePath"
    exit 2
}

Write-Output "perf-gate mode=$Mode sample=$samplePath baseline=$baselinePath"
foreach ($line in $summaries) {
    Write-Output "  - $line"
}

if ($violations.Count -eq 0) {
    Write-Output "perf-gate PASS"
    exit 0
}

Write-Output "perf-gate found $($violations.Count) target violation(s):"
foreach ($line in $violations) {
    Write-Output "  ! $line"
}

if ($Mode -eq "hard") {
    Write-Error "perf-gate HARD failure"
    exit 1
}

Write-Output "perf-gate SOFT warning (not failing build)"
exit 0
