# run-local.ps1
# Launch the Streamlit app from the project's Python 3.13 venv, by full path.
# Does NOT rely on PATH or an activated environment, so a stray `.venv` (3.11)
# can't shadow it.

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$py = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Error "venv not found at $py`nCreate it:  python -m venv venv ; .\venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

$ver = & $py -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($ver -notin @('3.12', '3.13')) {
    Write-Error "venv Python is $ver; this project needs 3.12 or 3.13. Rebuild venv with a supported Python."
    exit 1
}
Write-Host "Using $py (Python $ver)"

# Open the browser locally rather than run headless.
$env:STREAMLIT_SERVER_HEADLESS = 'false'

& $py -m streamlit run predictions.py @args
