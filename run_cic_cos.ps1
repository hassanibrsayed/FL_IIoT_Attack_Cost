# Cosine defences (cosw, cosdrop) vs. the adaptive attacker on CIC-IoT2023 (5 seeds, bd criterion).
# Safe to stop and re-run: finished runs are skipped. Outputs: cic_cos\
$py = "C:\Users\hassa\anaconda3\python.exe"
$env:PYTHONPATH = Join-Path $PSScriptRoot "deps"
Set-Location $PSScriptRoot
& $py adaptive_attack.py --data ciciot_sample.csv --static none --static-breaks none --crit bd `
    --aggs cosw cosdrop `
    --strategies revn boostn blend5 blend8 revblend `
    --fl-rows 100000 --rounds 30 --seeds 5 --out cic_cos 2>$null |
    Tee-Object -Append -FilePath cic_cos.log
