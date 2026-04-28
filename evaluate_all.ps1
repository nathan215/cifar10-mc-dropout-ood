# OOD eval for all five dropout modes (assumes training finished). Run from project root:
#   .\evaluate_all.ps1
# Checkpoints: ./checkpoints/resnet18_<mode>_best.pt (same as train.py).
# Outputs: ./checkpoints/ood_<mode>.csv and .json per mode.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$ckptDir = "./checkpoints"
$modes = @("all", "first_block", "last_layer_strict", "last_block_plus_fc") #"none"

# Optional: set to extra args for fog/gaussian_noise .pt diagnostics, e.g.:
#   $diagArgs = @("--save-diagnostics", "fog,gaussian_noise")
$diagArgs = @("--save-diagnostics", "fog,gaussian_noise")

foreach ($m in $modes) {
    $ckpt = Join-Path $ckptDir "resnet18_${m}_best.pt"
    if (-not (Test-Path $ckpt)) {
        throw "Missing checkpoint: $ckpt (train this mode first)"
    }
    Write-Host "========== OOD eval mode: $m ==========" -ForegroundColor Cyan
    $args = @(
        "evaluate_ood.py",
        "--checkpoint", $ckpt,
        "--mode", $m
    )
    if ($diagArgs) { $args += $diagArgs }
    python @args
    if ($LASTEXITCODE -ne 0) { throw "evaluate_ood.py failed for mode $m" }
}

Write-Host "All five modes evaluated." -ForegroundColor Green
