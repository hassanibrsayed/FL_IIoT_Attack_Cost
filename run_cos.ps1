# Direction-based defences (cosine) vs. the adaptive attacker.
#   cosw    = FLTrust-style: weight = ReLU(cosine to the median update), norms rescaled to the median norm
#   cosdrop = drop updates with cosine<=0 to the median update, rescale norms, trimmed mean
# Strategies are defence-aware (median norm, positive cosine with the honest direction).
# Safe to stop and re-run: finished runs are skipped. Outputs: cos_out\
$py = "C:\Users\hassa\anaconda3\python.exe"
$env:PYTHONPATH = Join-Path $PSScriptRoot "deps"
Set-Location $PSScriptRoot
& $py adaptive_attack.py --data "archive (8)/DNN-EdgeIIoT-dataset.csv" `
    --aggs cosw cosdrop `
    --strategies revn boostn blend5 blend8 revblend `
    --static-breaks none --fl-rows 100000 --rounds 30 --seeds 5 --out cos_out 2>$null |
    Tee-Object -Append -FilePath cos_out.log
