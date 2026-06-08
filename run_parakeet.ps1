$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$main = Join-Path $projectRoot "main.py"

if (-not (Test-Path $python)) {
    Write-Error "Project virtual environment not found at $python"
}

& $python $main
