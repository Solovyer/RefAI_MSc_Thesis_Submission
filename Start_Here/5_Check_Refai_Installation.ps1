$ErrorActionPreference = "Continue"

$StartHere = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageRoot = Split-Path -Parent $StartHere
$VenvPython = Join-Path $PackageRoot ".venv\Scripts\python.exe"
$Failures = 0
$Warnings = 0

function Pass($Message) {
    Write-Host "[PASS] $Message" -ForegroundColor Green
}

function Fail($Message) {
    $script:Failures++
    Write-Host "[FAIL] $Message" -ForegroundColor Red
}

function Warn($Message) {
    $script:Warnings++
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Check-Path($RelativePath, $Label) {
    $FullPath = Join-Path $PackageRoot $RelativePath
    if (Test-Path -LiteralPath $FullPath) {
        Pass "$Label found: $RelativePath"
    } else {
        Fail "$Label missing: $RelativePath"
    }
}

Write-Host ""
Write-Host "RefAI installation check" -ForegroundColor Cyan
Write-Host "Package root: $PackageRoot"
Write-Host ""

if (Test-Path -LiteralPath $VenvPython) {
    Pass "Virtual-environment Python found."
    $Version = & $VenvPython --version 2>&1
    Write-Host "       $Version"
} else {
    Fail "Virtual environment missing. Run 4_Setup_Refai.ps1."
}

Check-Path "Code\Refai_V7.ipynb" "Jupyter notebook"
Check-Path "Code\Refai_V7_1_1_Slm_Local_Dating.py" "Runnable Python pipeline"
Check-Path "Pdf" "PDF folder"
Check-Path "Reference_Docs\Excel_Docs" "Excel reference folder"
Check-Path "Reference_Docs\Excel_Docs\Refai_Custom_Aliases.xlsx" "Custom rulebook"
Check-Path "Templates\Output_Template_1979.xlsx" "Active output template"
Check-Path "Output" "Output folder"

$ExpectedPdfs = @(
    "Pdf\Test_1_1958_P284_287.pdf",
    "Pdf\Test_2_1979_3_P32_38.pdf",
    "Pdf\Test_3_1985_1_P32_41.pdf",
    "Pdf\Test_4_1966_5_P72_75.pdf",
    "Pdf\Test_5_1963_P345_348.pdf",
    "Pdf\Test_6_1973_4_P171_184.pdf",
    "Pdf\Test_7_1992_3_4_P45_48.pdf"
)

foreach ($Pdf in $ExpectedPdfs) {
    Check-Path $Pdf "Test PDF"
}

if (Test-Path -LiteralPath $VenvPython) {
    $ImportCode = @'
import importlib
modules = [
    "pandas", "openpyxl", "pdfplumber", "fitz", "PIL", "numpy",
    "requests", "torch", "transformers", "rapidocr_onnxruntime",
    "docx", "jupyterlab"
]
failed = []
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        failed.append(f"{module}: {exc}")
if failed:
    print("\n".join(failed))
    raise SystemExit(1)
print("All required imports succeeded.")
'@
    $ImportResult = $ImportCode | & $VenvPython -
    if ($LASTEXITCODE -eq 0) {
        Pass $ImportResult
    } else {
        Fail "One or more Python imports failed:"
        Write-Host $ImportResult
    }
}

if ($env:ANTHROPIC_API_KEY) {
    Pass "ANTHROPIC_API_KEY is set for this process."
} else {
    Warn "ANTHROPIC_API_KEY is not set. This is normal for SLM-only use."
}

Write-Host ""
Write-Host "Check complete: $Failures failure(s), $Warnings warning(s)."

if ($Failures -gt 0) {
    exit 1
}
exit 0
