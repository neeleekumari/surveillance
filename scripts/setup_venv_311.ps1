# Creates and activates a Python 3.11 virtual environment, then installs 3.11-compatible requirements
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_venv_311.ps1

$ErrorActionPreference = 'Stop'

# Check that the py launcher can find Python 3.11
$pyList = & py -0p 2>$null
if ($LASTEXITCODE -ne 0 -or -not ($pyList -match '3\.11')) {
    Write-Host 'Python 3.11 not found by the py launcher.' -ForegroundColor Yellow
    Write-Host 'Please install Python 3.11 (64-bit) from https://www.python.org/downloads/release/python-3110/' -ForegroundColor Yellow
    exit 1
}

# Create venv
Write-Host 'Creating .venv with Python 3.11...' -ForegroundColor Cyan
& py -3.11 -m venv .venv

# Activate venv
$venvActivate = Join-Path (Get-Location) '.venv\Scripts\Activate.ps1'
if (-not (Test-Path $venvActivate)) {
    Write-Host 'Activation script not found. Venv creation may have failed.' -ForegroundColor Red
    exit 1
}
. $venvActivate

# Upgrade pip
Write-Host 'Upgrading pip...' -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install requirements (311-locked set)
Write-Host 'Installing requirements-311.txt...' -ForegroundColor Cyan
pip install -r requirements-311.txt

Write-Host 'Done! Virtual environment is active. To deactivate: deactivate' -ForegroundColor Green
