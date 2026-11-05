# Entry point shared by the double-click launchers in the portable archive.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Download', 'Chat', 'Benchmark')]
    [string]$Action,
    [ValidateSet('14B', '4B')]
    [string]$Model = '14B',
    [ValidateSet('cpu', 'vulkan', 'both')]
    [string]$Backend = 'vulkan',
    [ValidateRange(1, 10)]
    [int]$Runs = 3,
    [ValidateRange(1, 8192)]
    [int]$PromptTokens = 512,
    [ValidateRange(1, 4096)]
    [int]$GeneratedTokens = 128,
    [string]$Prompt = ''
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$modelName = if ($Model -eq '14B') { 'Qwen3-14B-Q4_K_M.gguf' } else { 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf' }
$expectedBytes = if ($Model -eq '14B') { 9001753984L } else { 2497281120L }
$modelPath = Join-Path $repo "bin/models/$modelName"
if ($Action -eq 'Download' -or
    -not (Test-Path -LiteralPath $modelPath) -or
    (Get-Item -LiteralPath $modelPath).Length -ne $expectedBytes) {
    $modelPath = & (Join-Path $PSScriptRoot 'download-model.ps1') -Model $Model
}
if ($Action -eq 'Download') { return }
if ($Action -eq 'Benchmark') {
    & (Join-Path $PSScriptRoot 'benchmark.ps1') -Model $modelPath -Backend $Backend `
        -Runs $Runs -PromptTokens $PromptTokens -GeneratedTokens $GeneratedTokens `
        -OutputDirectory (Join-Path $repo ('results/' + (Get-Date -Format 'yyyyMMdd-HHmmss')))
    return
}
if ($Backend -eq 'both') { throw 'For chat, choose cpu or vulkan.' }
$exe = Join-Path $repo 'bin/Win64-Release/localchat.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "Missing executable: $exe" }
$chatArgs = @('/model', $modelPath, "/$Backend")
if ($Prompt) { $chatArgs += @('/prompt', $Prompt) }
& $exe @chatArgs
if ($LASTEXITCODE -ne 0) { throw "Chat failed ($LASTEXITCODE). For a GPU error, try Chat-CPU.cmd." }
