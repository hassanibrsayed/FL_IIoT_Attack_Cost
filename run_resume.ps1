# Resumes the interrupted full FL run in poc_full2\ (skips the runs already saved).
# Same arguments as run_full.ps1, but only the federated-learning experiment.
$py = "C:\Users\hassa\anaconda3\python.exe"
$env:PYTHONPATH = Join-Path $PSScriptRoot "deps"
Set-Location $PSScriptRoot
& $py poc_edge_iiot.py --data "archive (8)/DNN-EdgeIIoT-dataset.csv" --exp fl --resume `
    --fl-rows 100000 --rounds 30 --seeds 5 --out poc_full2 2>$null |
    Tee-Object -FilePath poc_full2_resume.log
