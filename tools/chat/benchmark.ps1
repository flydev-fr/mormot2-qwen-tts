# Native inference throughput, after warm-up, with raw logs and CSV output.
# Build demos/chat/llmbench.dproj in Win64 Release before running this script.
[CmdletBinding()]
param(
    [string[]]$Model = @(
        'bin/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf',
        'bin/models/Qwen3-14B-Q4_K_M.gguf'
    ),
    [ValidateSet('cpu', 'vulkan', 'both')]
    [string]$Backend = 'both',
    [ValidateRange(1, 10)]
    [int]$Runs = 3,
    [ValidateRange(1, 8192)]
    [int]$PromptTokens = 512,
    [ValidateRange(1, 4096)]
    [int]$GeneratedTokens = 128,
    [ValidateRange(0, 128)]
    [int]$Threads = 0,
    [string]$OutputDirectory = ''
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$exe = Join-Path $repo 'bin/Win64-Release/llmbench.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    throw "Compile demos/chat/llmbench.dproj in Win64 Release first: $exe"
}
$modelPaths = foreach ($item in $Model) {
    $path = if ([IO.Path]::IsPathRooted($item)) { $item } else { Join-Path $repo $item }
    (Resolve-Path -LiteralPath $path).Path
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repo ('build/benchmarks/' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path

# Record only useful hardware/build information, without machine or user names.
$cpuName = $null
$gpuInfo = @()
$ramBytes = $null
try {
    $cpuName = [Microsoft.Win32.Registry]::GetValue(
        'HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0',
        'ProcessorNameString', $null)
    $gpuInfo = @(Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0*' -ErrorAction SilentlyContinue |
        Where-Object { $_.DriverDesc } | ForEach-Object {
            [pscustomobject]@{
                Name = $_.DriverDesc
                DriverVersion = $_.DriverVersion
                MemoryBytes = $_.'HardwareInformation.qwMemorySize'
            }
        })
} catch { Write-Verbose "Hardware registry information unavailable: $_" }
try {
    $ramBytes = (Get-CimInstance Win32_ComputerSystem -OperationTimeoutSec 3 -ErrorAction Stop).TotalPhysicalMemory
} catch { Write-Verbose 'RAM information unavailable.' }
$modelInfo = @($modelPaths | ForEach-Object {
    $modelItem = Get-Item -LiteralPath $_
    [pscustomobject]@{ Name = $modelItem.Name; Bytes = $modelItem.Length }
})
[ordered]@{
    RecordedAtUtc = [DateTime]::UtcNow.ToString('o')
    OS = [Environment]::OSVersion.VersionString
    Architecture = $env:PROCESSOR_ARCHITECTURE
    Cpu = $cpuName
    LogicalProcessors = [Environment]::ProcessorCount
    RamBytes = $ramBytes
    Gpus = $gpuInfo
    BenchmarkExeSha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
    Models = $modelInfo
    Runs = $Runs
    PromptTokens = $PromptTokens
    GeneratedTokens = $GeneratedTokens
    RequestedCpuThreads = $Threads
    VulkanKvF32 = ($env:LAI_KV_F32 -eq '1')
    RouteOnHost = ($env:LAI_ROUTE_HOST -eq '1')
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'system-info.json') -Encoding utf8

$modes = if ($Backend -eq 'both') { @('cpu', 'vulkan') } else { @($Backend) }
$results = [Collections.Generic.List[object]]::new()
$invariant = [Globalization.CultureInfo]::InvariantCulture

function Get-MedianText([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    $middle = [int][math]::Floor($sorted.Count / 2)
    $median = if ($sorted.Count % 2) { $sorted[$middle] } else { ($sorted[$middle - 1] + $sorted[$middle]) / 2 }
    return $median.ToString('F2', $invariant)
}

foreach ($path in $modelPaths) {
    $name = [IO.Path]::GetFileNameWithoutExtension($path)
    foreach ($mode in $modes) {
        for ($run = 1; $run -le $Runs; $run++) {
            $backendArg = if ($mode -eq 'cpu') { "$Threads" } else { 'vulkan' }
            $lines = & $exe $path $PromptTokens $GeneratedTokens $backendArg 2>&1
            $exitCode = $LASTEXITCODE
            $log = $lines -join "`n"
            $logPath = Join-Path $OutputDirectory "$name-$mode-$run.log"
            Set-Content -LiteralPath $logPath -Value $log -Encoding utf8
            if ($exitCode -ne 0) { throw "Benchmark failed ($exitCode): $logPath`n$log" }
            $prefill = [regex]::Match($log, 'prefill: \d+ tokens in (\d+) ms = ([0-9.]+) tokens/s')
            $decode = [regex]::Match($log, 'decode: \d+ tokens in (\d+) ms = ([0-9.]+) tokens/s')
            $threadMatch = [regex]::Match($log, 'backend: fast, (\d+) threads')
            if (-not $prefill.Success -or -not $decode.Success) {
                throw "Missing throughput measurements: $logPath"
            }
            $prefillRate = [double]::Parse($prefill.Groups[2].Value, $invariant)
            $decodeRate = [double]::Parse($decode.Groups[2].Value, $invariant)
            $row = [pscustomobject]@{
                Model = $name
                Backend = $mode
                Run = $run
                CpuThreads = $threadMatch.Groups[1].Value
                PromptTokens = $PromptTokens
                GeneratedTokens = $GeneratedTokens
                PrefillMs = [int]$prefill.Groups[1].Value
                PrefillTokensPerSecond = $prefillRate.ToString('R', $invariant)
                DecodeMs = [int]$decode.Groups[1].Value
                DecodeTokensPerSecond = $decodeRate.ToString('R', $invariant)
            }
            $results.Add($row)
            $results | Export-Csv -LiteralPath (Join-Path $OutputDirectory 'results.csv') -NoTypeInformation -Encoding utf8
            $summary = foreach ($group in ($results | Group-Object Model,Backend)) {
                $prefillValues = @($group.Group | ForEach-Object { [double]::Parse($_.PrefillTokensPerSecond, $invariant) })
                $decodeValues = @($group.Group | ForEach-Object { [double]::Parse($_.DecodeTokensPerSecond, $invariant) })
                [pscustomobject]@{
                    Model = $group.Group[0].Model
                    Backend = $group.Group[0].Backend
                    Runs = $group.Count
                    PrefillMedianTokensPerSecond = Get-MedianText $prefillValues
                    DecodeMedianTokensPerSecond = Get-MedianText $decodeValues
                }
            }
            $summary | Export-Csv -LiteralPath (Join-Path $OutputDirectory 'summary.csv') -NoTypeInformation -Encoding utf8
            Write-Host ("{0} {1} [{2}/{3}]: prefill {4:N1}, decode {5:N1} tokens/s" -f
                $name, $mode, $run, $Runs, $prefillRate, $decodeRate)
        }
    }
}
$summary | Format-Table | Out-Host
Write-Host "Raw logs, CSV and system-info.json: $OutputDirectory"
