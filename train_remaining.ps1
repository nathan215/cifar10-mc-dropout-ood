# Train remaining modes (after 'all' is done). Run from project root:
#   .\train_remaining.ps1
# Uses same defaults as train.py; add flags as needed.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$modes = @("last_layer_strict", "last_block_plus_fc", "none")
foreach ($m in $modes) {
    Write-Host "========== Training mode: $m ==========" -ForegroundColor Cyan
    python train.py --mode $m --data-root ./data --out-dir ./checkpoints
    if ($LASTEXITCODE -ne 0) { throw "train.py failed for mode $m" }
}

Write-Host "All remaining modes finished." -ForegroundColor Green
