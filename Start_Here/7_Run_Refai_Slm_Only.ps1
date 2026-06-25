$ErrorActionPreference = "Stop"

$StartHere = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageRoot = Split-Path -Parent $StartHere
$VenvPython = Join-Path $PackageRoot ".venv\Scripts\python.exe"
$Pipeline = Join-Path $PackageRoot "Code\Refai_V7_1_1_Slm_Local_Dating.py"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "The RefAI virtual environment is missing. Run .\4_Setup_Refai.ps1 first."
}

if (-not (Test-Path -LiteralPath $Pipeline)) {
    throw "Pipeline script not found: $Pipeline"
}

$env:REFAI_PROJECT_DIR = $PackageRoot
$env:REFAI_ENABLE_LLM_STRUCTURAL_SEGMENTATION = "false"
$env:REFAI_ENABLE_LLM_REVIEW = "false"
$env:REFAI_RUN_BATCH_PDFS = "true"

Write-Host "Starting RefAI SLM-only batch..." -ForegroundColor Cyan
& $VenvPython $Pipeline
if ($LASTEXITCODE -ne 0) {
    throw "RefAI SLM-only run failed with exit code $LASTEXITCODE."
}

Write-Host "Run completed. Check the newest folder in Output." -ForegroundColor Green
