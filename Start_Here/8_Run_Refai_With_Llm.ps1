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

if (-not $env:ANTHROPIC_API_KEY) {
    $SecureKey = Read-Host "Paste your Anthropic API key" -AsSecureString
    $Pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureKey)
    try {
        $env:ANTHROPIC_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($Pointer)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Pointer)
    }
}

if (-not $env:ANTHROPIC_API_KEY) {
    throw "No Anthropic API key was provided."
}

$env:REFAI_PROJECT_DIR = $PackageRoot
$env:REFAI_ENABLE_LLM_STRUCTURAL_SEGMENTATION = "true"
$env:REFAI_ENABLE_LLM_REVIEW = "true"
$env:REFAI_RUN_BATCH_PDFS = "true"

Write-Host "Starting RefAI SLM + LLM batch..." -ForegroundColor Cyan
$RunStartedAt = Get-Date
& $VenvPython $Pipeline
if ($LASTEXITCODE -ne 0) {
    throw "RefAI SLM + LLM run failed with exit code $LASTEXITCODE."
}

$LatestRun = Get-ChildItem -LiteralPath (Join-Path $PackageRoot "Output") -Directory -Filter "Run_*" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $LatestRun -or $LatestRun.LastWriteTime -lt $RunStartedAt.AddMinutes(-1)) {
    throw "No new RefAI run folder was found after the SLM + LLM command."
}
$SuccessMarker = Join-Path $LatestRun.FullName "RUN_COMPLETED_SUCCESSFULLY.txt"
if (-not (Test-Path -LiteralPath $SuccessMarker)) {
    $IncompleteMarker = Join-Path $LatestRun.FullName "RUN_INCOMPLETE.txt"
    if (Test-Path -LiteralPath $IncompleteMarker) {
        Get-Content -LiteralPath $IncompleteMarker | Write-Host -ForegroundColor Red
    }
    throw "The newest run has no RUN_COMPLETED_SUCCESSFULLY marker and must not be treated as valid output."
}

Write-Host ""
Get-Content -LiteralPath $SuccessMarker | Write-Host -ForegroundColor Green
Write-Host "Open Llm_Review_Outputs_02 for the LLM-reviewed results and Reviewed_Outputs_01 for the pre-LLM baseline." -ForegroundColor Cyan
