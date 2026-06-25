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
& $VenvPython $Pipeline
if ($LASTEXITCODE -ne 0) {
    throw "RefAI SLM + LLM run failed with exit code $LASTEXITCODE."
}

Write-Host "Run completed. Check the newest folder in Output." -ForegroundColor Green
