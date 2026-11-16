$py = "C:\Users\hassa\anaconda3\python.exe"
$env:PYTHONPATH = Join-Path $PSScriptRoot "deps"
Set-Location $PSScriptRoot
& $py poc_edge_iiot.py --data "archive (8)/DNN-EdgeIIoT-dataset.csv" --exp all --quick --out poc_quick 2>$null
