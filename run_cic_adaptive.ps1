# Mini replication on CIC-IoT2023 (34 classes): adaptive attacker, 3 seeds, no defence.
# Detection criterion = balanced detection (recall + 1 - false alarm)/2 dropping 0.10 below its m=0 baseline,
# plus macro-F1 dropping 0.10 below baseline. Safe to stop and re-run (finished runs are skipped).
# Outputs: cic_adaptive\   Log: cic_adaptive.log
$py = "C:\Users\hassa\anaconda3\python.exe"
$env:PYTHONPATH = Join-Path $PSScriptRoot "deps"
Set-Location $PSScriptRoot
& $py adaptive_attack.py --data ciciot_sample.csv --static none --static-breaks none --crit bd `
    --aggs fedavg median trimmed --fl-rows 100000 --rounds 30 --seeds 10 --out cic_adaptive 2>$null |
    Tee-Object -Append -FilePath cic_adaptive.log

