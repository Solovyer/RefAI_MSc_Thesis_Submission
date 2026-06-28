$ErrorActionPreference = "Stop"

$StartHere = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageRoot = Split-Path -Parent $StartHere
$VenvDir = Join-Path $PackageRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $StartHere "3_Requirements.txt"

function Find-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($Version in @("3.12", "3.11", "3.10")) {
            try {
                $null = & py "-$Version" -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    return @("py", "-$Version")
                }
            } catch {}
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        try {
            $VersionSupported = & python -c "import sys; print('yes' if (3, 10) <= sys.version_info[:2] <= (3, 12) else 'no')" 2>$null
            if ($LASTEXITCODE -eq 0) {
                if (($VersionSupported | Select-Object -Last 1).Trim() -eq "yes") {
                    return @("python")
                }
            }
        } catch {}
    }

    throw "RefAI requires 64-bit Python 3.10, 3.11 or 3.12 (3.12 recommended). Python 3.13 and 3.14 are not currently compatible with the OCR dependency. Install Python 3.12 and reopen PowerShell."
}

Write-Host ""
Write-Host "RefAI setup" -ForegroundColor Cyan
Write-Host "Package root: $PackageRoot"

if (-not (Test-Path -LiteralPath $Requirements)) {
    throw "Requirements file not found: $Requirements"
}

$PythonCommand = @(Find-Python)

$CreateVenv = -not (Test-Path -LiteralPath $VenvPython)

if (-not $CreateVenv) {
    $VenvVersionSupported = & $VenvPython -c "import sys; print('yes' if (3, 10) <= sys.version_info[:2] <= (3, 12) else 'no')" 2>$null
    if ($LASTEXITCODE -ne 0 -or ($VenvVersionSupported | Select-Object -Last 1).Trim() -ne "yes") {
        Write-Host "Existing .venv uses an unsupported Python version; recreating it..." -ForegroundColor Yellow
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
        $CreateVenv = $true
    }
}

if ($CreateVenv) {
    Write-Host "Creating virtual environment in .venv..." -ForegroundColor Yellow
    if ($PythonCommand.Count -eq 2) {
        & $PythonCommand[0] $PythonCommand[1] -m venv $VenvDir
    } else {
        & $PythonCommand[0] -m venv $VenvDir
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual-environment creation failed."
    }
} else {
    Write-Host "Existing .venv found; reusing it." -ForegroundColor Green
}

Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip wheel "setuptools<82" --retries 10 --timeout 120
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed."
}

Write-Host "Installing RefAI requirements..." -ForegroundColor Yellow
& $VenvPython -m pip install -r $Requirements --retries 10 --timeout 120
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

Write-Host "Registering Jupyter kernel..." -ForegroundColor Yellow
& $VenvPython -m ipykernel install --user --name refai --display-name "RefAI"
if ($LASTEXITCODE -ne 0) {
    throw "Jupyter-kernel registration failed."
}

Write-Host ""
Write-Host "Running installation check..." -ForegroundColor Cyan
& (Join-Path $StartHere "5_Check_Refai_Installation.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "Setup completed, but the installation check reported errors."
}

Write-Host ""
Write-Host "RefAI setup completed." -ForegroundColor Green
Write-Host "Next step: .\6_Launch_Refai_Jupyter.ps1"
