# Build a fresh portable ZIP. The large model files are never included.
[CmdletBinding()]
param(
    [string]$MormotRoot = 'C:\Users\badb\Documents\Embarcadero\Studio\Libraries\mORMot2'
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$name = 'localchat-windows-x64-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$package = Join-Path $repo "bin/portable/$name"
if (Test-Path -LiteralPath $package) { throw "Package already exists: $package" }
foreach ($file in @('localchat.exe', 'llmbench.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $repo "bin/Win64-Release/$file"))) {
        throw "Build demos/chat/$([IO.Path]::GetFileNameWithoutExtension($file)).dproj in Win64 Release first."
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $MormotRoot 'src/mormot.defines.inc'))) {
    throw "mORMot source tree not found: $MormotRoot"
}
foreach ($dir in @('bin/Win64-Release', 'bin/models', 'tools/chat', 'licenses', 'reference', 'source/demos/chat')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $package $dir) | Out-Null
}
foreach ($file in @('localchat.exe', 'llmbench.exe')) {
    Copy-Item -LiteralPath (Join-Path $repo "bin/Win64-Release/$file") -Destination (Join-Path $package 'bin/Win64-Release')
}
foreach ($file in @('download-model.ps1', 'benchmark.ps1', 'portable.ps1')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $file) -Destination (Join-Path $package 'tools/chat')
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'PORTABLE-README.txt') -Destination (Join-Path $package 'LIRE-MOI.txt')
foreach ($file in @('LICENSE', 'NOTICE')) {
    Copy-Item -LiteralPath (Join-Path $repo $file) -Destination $package
    Copy-Item -LiteralPath (Join-Path $repo "external/mormot2-ai/$file") -Destination (Join-Path $package "licenses/mormot2-ai-$file")
}
Copy-Item -LiteralPath (Join-Path $MormotRoot 'LICENCE.md') -Destination (Join-Path $package 'licenses/mORMot2-LICENCE.md')
Get-ChildItem -LiteralPath (Join-Path $repo 'demos/chat') -File |
    Where-Object Extension -In @('.dpr', '.dproj') |
    Copy-Item -Destination (Join-Path $package 'source/demos/chat')

# Include the actual source snapshot, including embedded shader sources;
# no repository metadata, credentials, models or build caches are copied.
$trees = @(
    @{ From = (Join-Path $repo 'external/mormot2-ai/src'); To = 'source/external/mormot2-ai/src' },
    @{ From = (Join-Path $repo 'external/mormot2-ai/tools/vk'); To = 'source/external/mormot2-ai/tools/vk' },
    @{ From = (Join-Path $repo 'external/mormot2-ai/tools/cuda'); To = 'source/external/mormot2-ai/tools/cuda' },
    @{ From = (Join-Path $MormotRoot 'src'); To = 'source/mormot2/src' }
)
foreach ($tree in $trees) {
    $sourceRoot = (Resolve-Path -LiteralPath $tree.From).Path
    foreach ($file in (Get-ChildItem -LiteralPath $sourceRoot -File -Recurse)) {
        if ($file.Extension -notin @('.pas', '.inc', '.c', '.h', '.cpp', '.asm', '.comp', '.cu', '.sh', '.ps1')) { continue }
        $relative = $file.FullName.Substring($sourceRoot.Length).TrimStart([char[]]'\/')
        $target = Join-Path (Join-Path $package $tree.To) $relative
        New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($target)) | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $target
    }
}

foreach ($file in @('summary.csv', 'results.csv', 'environment.json')) {
    $reference = Join-Path $repo "build/benchmarks/qwen4b-vs14b/$file"
    if (Test-Path -LiteralPath $reference) {
        Copy-Item -LiteralPath $reference -Destination (Join-Path $package 'reference')
    }
}
$launchers = [ordered]@{
    'Download-Model.cmd' = '-Action Download'
    'Chat-GPU.cmd' = '-Action Chat -Backend vulkan'
    'Chat-CPU.cmd' = '-Action Chat -Backend cpu'
    'Benchmark.cmd' = '-Action Benchmark -Backend both'
    'Benchmark-CPU.cmd' = '-Action Benchmark -Backend cpu'
    'Benchmark-GPU.cmd' = '-Action Benchmark -Backend vulkan'
}
foreach ($launcher in $launchers.GetEnumerator()) {
    $body = @"
@echo off
setlocal
cd /d "%~dp0"
set "PSModulePath=%SystemRoot%\System32\WindowsPowerShell\v1.0\Modules"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\chat\portable.ps1" $($launcher.Value)
set "LOCALCHAT_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %LOCALCHAT_EXIT%
"@
    [IO.File]::WriteAllText((Join-Path $package $launcher.Key), ($body -replace '\r?\n', "`r`n"), [Text.Encoding]::ASCII)
}
$mormotRevision = & git -c "safe.directory=$($MormotRoot.Replace('\', '/'))" -C $MormotRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Unable to identify mORMot revision.' }
[ordered]@{
    CreatedAtUtc = [DateTime]::UtcNow.ToString('o')
    Platform = 'Windows x64, Delphi 13 Release'
    Mormot2AiRevision = (& git -C (Join-Path $repo 'external/mormot2-ai') rev-parse HEAD)
    Mormot2Revision = $mormotRevision
    LocalchatSha256 = (Get-FileHash (Join-Path $package 'bin/Win64-Release/localchat.exe')).Hash.ToLowerInvariant()
    BenchmarkSha256 = (Get-FileHash (Join-Path $package 'bin/Win64-Release/llmbench.exe')).Hash.ToLowerInvariant()
    LocalchatSourceSha256 = (Get-FileHash (Join-Path $repo 'demos/chat/localchat.dpr')).Hash.ToLowerInvariant()
    BenchmarkSourceSha256 = (Get-FileHash (Join-Path $repo 'demos/chat/llmbench.dpr')).Hash.ToLowerInvariant()
    ModelIncluded = $false
    ModelFile = 'Qwen3-14B-Q4_K_M.gguf'
    ModelSha256 = '5eaa0870bd81ed3b58a630a271234cfa604e43ffb3a19cd68e54a80dd9d52a66'
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $package 'build-info.json') -Encoding utf8

$zip = $package + '.zip'
Compress-Archive -LiteralPath $package -DestinationPath $zip -CompressionLevel Optimal
Write-Host "Portable folder: $package"
Write-Host "ZIP to share: $zip"
Write-Output $zip
