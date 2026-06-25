$ErrorActionPreference = "Stop"

$StartHere = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageRoot = Split-Path -Parent $StartHere
$VenvPython = Join-Path $PackageRoot ".venv\Scripts\python.exe"
$Notebook = Join-Path $PackageRoot "Code\Refai_V7.ipynb"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "The RefAI virtual environment is missing. Run .\4_Setup_Refai.ps1 first."
}

if (-not (Test-Path -LiteralPath $Notebook)) {
    throw "Notebook not found: $Notebook"
}

$env:REFAI_PROJECT_DIR = $PackageRoot

Write-Host "Opening JupyterLab from: $PackageRoot" -ForegroundColor Cyan
Write-Host "Open Code/Refai_V7.ipynb and select the RefAI kernel."

Push-Location $PackageRoot
try {
    & $VenvPython -m jupyter lab --notebook-dir="$PackageRoot"
} finally {
    Pop-Location
}
