# Norm-clipping defence vs. the adaptive attacker (fedavg/median/trimmed + clip).
# Strategies: boostn/revn are defence-aware (norm capped at the clipping bound).
# Safe to stop and re-run: finished runs are skipped. Outputs: defense_out\
$py = "C:\Users\hassa\anaconda3\python.exe"
$env:PYTHONPATH = Join-Path $PSScriptRoot "deps"
Set-Location $PSScriptRoot
& $py adaptive_attack.py --data "archive (8)/DNN-EdgeIIoT-dataset.csv" `
    --aggs fedavg_clip median_clip trimmed_clip `
    --strategies boostn revn lie1 lie25 boost1 `
    --static-breaks none --fl-rows 100000 --rounds 30 --seeds 5 --out defense_out 2>$null |
    Tee-Object -Append -FilePath defense_out.log
