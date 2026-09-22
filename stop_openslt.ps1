$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModalExe = "C:\Users\hoanghiworld\AppData\Local\Programs\Python\Python310\Scripts\modal.exe"

Set-Location $RepoRoot

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$appsJson = & $ModalExe app list --json
$apps = @()
if ($appsJson) {
    $apps = $appsJson | ConvertFrom-Json
}

$targetApps = @($apps | Where-Object { $_."Description" -like "openslt-realtime*" -and $_."State" -ne "stopped" })

if ($targetApps.Count -gt 0) {
    Write-Host "Stopping Modal realtime apps..." -ForegroundColor Yellow
    foreach ($app in $targetApps) {
        & $ModalExe app stop $app."App ID" | Out-Null
        Write-Host ("Stopped " + $app."App ID") -ForegroundColor DarkYellow
    }
} else {
    Write-Host "No running OpenSLT realtime Modal app found." -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Current Modal apps:" -ForegroundColor Cyan
& $ModalExe app list
