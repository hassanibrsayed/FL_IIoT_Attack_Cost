# Adaptive attacker, full run. Safe to stop (Ctrl+C) and re-run: finished runs are skipped.
# Every run is appended to adaptive_out\adaptive_results.csv the moment it finishes.
# Summary files (adaptive_cost.csv, adaptive_vs_static.csv, adaptive_cost.png) are written at the end;
# to rebuild them from a partial run:  add  --summary-only
$py = "C:\Users\hassa\anaconda3\python.exe"
$env:PYTHONPATH = Join-Path $PSScriptRoot "deps"
Set-Location $PSScriptRoot
& $py adaptive_attack.py --data "archive (8)/DNN-EdgeIIoT-dataset.csv" `
    --fl-rows 100000 --rounds 30 --seeds 10 --out adaptive_out 2>$null |
    Tee-Object -Append -FilePath adaptive_out.log
