# Revision experiments, in order of priority. Every step is safe to stop and re-run: finished runs are skipped.
#   Step 1  mechanism: clean (m = 0) runs that log the spread of honest updates (disp_edge\, disp_cic\)
#   Step 2  heterogeneity-aware defences on CIC-IoT2023 (cic_hetdef\)  -- faster dataset first
#   Step 3  heterogeneity-aware defences on Edge-IIoTset (hetdef_out\)
#   Step 4  analysis: mechanism_analysis.py + summaries
# Defences: cclip = centred clipping (Karimireddy et al. 2021), bucket_cclip / bucket_median = bucketing with s = 2
# (Karimireddy et al. 2022) followed by centred clipping / coordinate-wise median.
# Attacker strategies: the base set plus the norm-matched ones that won against the earlier defences.
# Run a single step with:  .\run_revision.ps1 -Step 2
param([int]$Step = 0)
$py = "C:\Users\hassa\anaconda3\python.exe"
$env:PYTHONPATH = Join-Path $PSScriptRoot "deps"
Set-Location $PSScriptRoot
$edge = "archive (8)/DNN-EdgeIIoT-dataset.csv"
$strats = @("boostK", "rev3", "lie1", "lie25", "revn", "boostn", "revblend")
$aggs = @("cclip", "bucket_cclip", "bucket_median")

if ($Step -eq 0 -or $Step -eq 1) {
    & $py adaptive_attack.py --data $edge --dispersion --aggs median trimmed `
        --fl-rows 100000 --rounds 30 --seeds 10 --out disp_edge 2>$null | Tee-Object -Append -FilePath disp_edge.log
    & $py adaptive_attack.py --data ciciot_sample.csv --static none --dispersion --aggs median trimmed `
        --fl-rows 100000 --rounds 30 --seeds 10 --out disp_cic 2>$null | Tee-Object -Append -FilePath disp_cic.log
}
if ($Step -eq 0 -or $Step -eq 2) {
    & $py adaptive_attack.py --data ciciot_sample.csv --static none --static-breaks none --crit bd `
        --aggs $aggs --strategies $strats `
        --fl-rows 100000 --rounds 30 --seeds 5 --out cic_hetdef 2>$null | Tee-Object -Append -FilePath cic_hetdef.log
    & $py cic_winners_ci.py cic_hetdef
}
if ($Step -eq 0 -or $Step -eq 3) {
    & $py adaptive_attack.py --data $edge --static-breaks none `
        --aggs $aggs --strategies $strats `
        --fl-rows 100000 --rounds 30 --seeds 5 --out hetdef_out 2>$null | Tee-Object -Append -FilePath hetdef_out.log
}
if ($Step -eq 0 -or $Step -eq 4) {
    & $py mechanism_analysis.py
}
