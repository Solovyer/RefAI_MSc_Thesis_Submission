$ErrorActionPreference = "Stop"

$StartHere = Split-Path -Parent $MyInvocation.MyCommand.Path

function Run-Script($Name) {
    $Script = Join-Path $StartHere $Name
    if (-not (Test-Path -LiteralPath $Script)) {
        throw "Startup script not found: $Script"
    }
    try {
        $global:LASTEXITCODE = 0
        & $Script
        if ($LASTEXITCODE -ne 0) {
            throw "Step ended with exit code $LASTEXITCODE."
        }
    } catch {
        Write-Host ""
        Write-Host "The selected RefAI step did not finish successfully." -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        Write-Host "The window will remain open so the error can be read." -ForegroundColor Yellow
        [void](Read-Host "Press Enter to return to the startup menu")
    }
}

while ($true) {
    Write-Host ""
    Write-Host "====================================" -ForegroundColor DarkCyan
    Write-Host " RefAI Startup Menu" -ForegroundColor Cyan
    Write-Host "====================================" -ForegroundColor DarkCyan
    Write-Host "First use: choose 1, then 2." -ForegroundColor Yellow
    Write-Host "Later use: reopen this file and choose 4 or 5." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. First installation or repair"
    Write-Host "2. Verify the installation"
    Write-Host "3. Open JupyterLab"
    Write-Host "4. Run SLM-only batch (no API key)"
    Write-Host "5. Run SLM + LLM batch (API key required)"
    Write-Host "6. Open the written Start Here guide"
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
        Write-Host "Closing RefAI. To use it again, right-click 2_Start_Refai.ps1 and select Run with PowerShell."
        break
    }
}
