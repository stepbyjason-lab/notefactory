# NoteFactory public setup (PowerShell)
# Usage: scripts/setup.ps1

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
Set-Location $rootDir

$python = @("python", "py", "python3") |
    Where-Object { Get-Command $_ -ErrorAction SilentlyContinue } |
    Select-Object -First 1
if (-not $python) {
    throw "Python 3.10+ was not found. Install Python, then run this script again."
}

if (-not (Test-Path -LiteralPath ".venv")) {
    & $python -m venv .venv
}

$venvPython = Join-Path ".venv" "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    $venvPython = Join-Path ".venv" "bin/python"
}
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Could not find Python inside .venv."
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

if (-not (Test-Path -LiteralPath ".env.local")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env.local"
    Write-Host "Created .env.local from .env.example. Add GEMINI_API_KEY before generating a note."
} else {
    Write-Host "Kept existing .env.local."
}

Write-Host "Setup complete. Run: .venv\Scripts\python.exe note_pipe.py <sipher.json> --out notes_out"
