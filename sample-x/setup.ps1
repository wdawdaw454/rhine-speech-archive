param([string]$Python = '')
$ErrorActionPreference = 'Stop'

function Find-Python {
    param([string]$Requested)
    if ($Requested) {
        if (!(Test-Path -LiteralPath $Requested)) { throw "Specified Python does not exist: $Requested" }
        $candidate = (Resolve-Path -LiteralPath $Requested).Path
        & $candidate -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
        if ($LASTEXITCODE -ne 0) { throw 'Sample-X requires Python 3.11.' }
        return $candidate
    }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $probe = & py.exe -3.11 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $probe) { return $probe.Trim() }
    }
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command) {
        & $command.Source -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
        if ($LASTEXITCODE -eq 0) { return $command.Source }
    }
    throw 'Python 3.11 was not found. Install it, or pass -Python <path-to-python.exe>.'
}

$python = Find-Python $Python
$venvPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE) { throw 'Failed to create the isolated SampleX environment.' }
}
& $venvPython -m pip install --disable-pip-version-check --cache-dir (Join-Path $PSScriptRoot '.pip-cache') -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE) { throw 'Failed to install SampleX dependencies.' }
