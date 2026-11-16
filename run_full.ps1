# Full run v2: 30 rounds, 5 seeds, attack-recall break points, per-class leakage report.
# Results: poc_full2\   Log: poc_full2.log   (poc_full\ from v1 is kept untouched)
$py = "C:\Users\hassa\anaconda3\python.exe"
$env:PYTHONPATH = Join-Path $PSScriptRoot "deps"
Set-Location $PSScriptRoot
& $py poc_edge_iiot.py --data "archive (8)/DNN-EdgeIIoT-dataset.csv" --exp all `
    --max-rows 400000 --fl-rows 100000 --rounds 30 --seeds 5 --out poc_full2 2>$null |
    Tee-Object -FilePath poc_full2.log
