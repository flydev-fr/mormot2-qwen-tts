# Pinned public GGUF download, resumable, with size and SHA-256 verification.
[CmdletBinding()]
param(
    [ValidateSet('14B', '4B')]
    [string]$Model = '14B',
    [string]$Destination = '',
    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if (-not $Destination) { $Destination = Join-Path $repo 'bin/models' }
$specs = @{
    '14B' = @{
        File = 'Qwen3-14B-Q4_K_M.gguf'
        Repo = 'unsloth/Qwen3-14B-GGUF'
        Revision = 'a04a82c4739b3ef5fa6da7d10261db2c67dd1985'
        Bytes = 9001753984L
        Sha256 = '5eaa0870bd81ed3b58a630a271234cfa604e43ffb3a19cd68e54a80dd9d52a66'
    }
    '4B' = @{
        File = 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
        Repo = 'unsloth/Qwen3-4B-Instruct-2507-GGUF'
        Revision = 'a06e946bb6b655725eafa393f4a9745d460374c9'
        Bytes = 2497281120L
        Sha256 = '3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597'
    }
}
$spec = $specs[$Model]
$target = [IO.Path]::GetFullPath((Join-Path $Destination $spec.File))

function Test-ModelFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -ne $spec.Bytes) { return $false }
    Write-Host "Checking SHA-256: $Path"
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $spec.Sha256
}

if (Test-ModelFile $target) {
    Write-Host 'Model verified.'
    Write-Output $target
    return
}
if ($VerifyOnly) { throw "Model missing or invalid: $target" }
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw 'curl.exe is required (included with Windows 10/11).'
}
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($target)) | Out-Null
$partial = $target + '.download'
if (Test-Path -LiteralPath $partial) {
    $partialBytes = (Get-Item -LiteralPath $partial).Length
    if ($partialBytes -gt $spec.Bytes -or
        ($partialBytes -eq $spec.Bytes -and -not (Test-ModelFile $partial))) {
        # This is only the temporary file created by this downloader.
        Remove-Item -LiteralPath $partial
    }
}
if (-not (Test-ModelFile $partial)) {
    $url = 'https://huggingface.co/{0}/resolve/{1}/{2}' -f $spec.Repo, $spec.Revision, $spec.File
    Write-Host ('Downloading {0} ({1:N2} GB). Rerun this script to resume.' -f $spec.File, ($spec.Bytes / 1e9))
    & curl.exe --fail --location --retry 5 --retry-delay 3 --connect-timeout 30 `
        --speed-limit 1024 --speed-time 120 --continue-at - --output $partial $url
    if ($LASTEXITCODE -ne 0) { throw "Download failed ($LASTEXITCODE). Rerun to resume: $partial" }
    if (-not (Test-ModelFile $partial)) {
        # A failed checksum must never be promoted to the final model path.
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial }
        throw 'Downloaded size or SHA-256 does not match. Rerun to download a fresh copy.'
    }
}
Move-Item -LiteralPath $partial -Destination $target -Force
Write-Host 'Download complete; size and SHA-256 verified.'
Write-Output $target
