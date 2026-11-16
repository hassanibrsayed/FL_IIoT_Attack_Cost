# Norm-clipping defence vs. the adaptive attacker on CIC-IoT2023 (5 seeds, bd criterion).
# Baselines for the clipped aggregators (m=0) are trained in-script. Safe to stop and re-run. Outputs: cic_defense\
$py = "C:\Users\hassa\anaconda3\python.exe"
$env:PYTHONPATH = Join-Path $PSScriptRoot "deps"
Set-Location $PSScriptRoot
& $py adaptive_attack.py --data ciciot_sample.csv --static none --static-breaks none --crit bd `
    --aggs fedavg_clip median_clip trimmed_clip `
    --strategies boostn revn lie1 lie25 boost1 `
    --fl-rows 100000 --rounds 30 --seeds 5 --out cic_defense 2>$null |
    Tee-Object -Append -FilePath cic_defense.log
