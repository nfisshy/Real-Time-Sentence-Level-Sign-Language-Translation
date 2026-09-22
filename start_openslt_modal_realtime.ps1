$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModalExe = "C:\Users\hoanghiworld\AppData\Local\Programs\Python\Python310\Scripts\modal.exe"

Set-Location $RepoRoot

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "Deploying Modal realtime app..." -ForegroundColor Cyan
& $ModalExe deploy "OpenSLT\modal_realtime_app.py"

Write-Host ""
Write-Host "Current Modal apps:" -ForegroundColor Cyan
& $ModalExe app list
