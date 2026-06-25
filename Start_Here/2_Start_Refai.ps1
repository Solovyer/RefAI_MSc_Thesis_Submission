$ErrorActionPreference = "Stop"

$StartHere = Split-Path -Parent $MyInvocation.MyCommand.Path

function Run-Script($Name) {
    $Script = Join-Path $StartHere $Name
    if (-not (Test-Path -LiteralPath $Script)) {
        throw "Startup script not found: $Script"
    }
    & $Script
}

while ($true) {
    Write-Host ""
    Write-Host "====================================" -ForegroundColor DarkCyan
    Write-Host " RefAI Startup Menu" -ForegroundColor Cyan
    Write-Host "====================================" -ForegroundColor DarkCyan
    Write-Host "1. Run step 4: install or update RefAI environment"
    Write-Host "2. Run step 5: check installation"
    Write-Host "3. Run step 6: open JupyterLab"
    Write-Host "4. Run step 7: SLM-only batch"
    Write-Host "5. Run step 8: SLM + LLM batch"
    Write-Host "6. Open step 1: Start Here guide"
    Write-Host "0. Exit"
    Write-Host ""

    $Choice = Read-Host "Select an option"

    switch ($Choice) {
        "1" { Run-Script "4_Setup_Refai.ps1" }
        "2" { Run-Script "5_Check_Refai_Installation.ps1" }
        "3" { Run-Script "6_Launch_Refai_Jupyter.ps1" }
        "4" { Run-Script "7_Run_Refai_Slm_Only.ps1" }
        "5" { Run-Script "8_Run_Refai_With_Llm.ps1" }
        "6" { Start-Process (Join-Path $StartHere "1_Start_Here.md") }
        "0" { break }
        default { Write-Host "Unknown option. Choose 0-6." -ForegroundColor Yellow }
    }

    if ($Choice -eq "0") {
        break
    }
}
