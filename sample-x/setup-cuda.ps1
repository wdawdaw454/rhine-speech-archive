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
$cudaPython = Join-Path $PSScriptRoot '.venv-cuda/Scripts/python.exe'
if (!(Test-Path -LiteralPath $cudaPython)) {
    & $Python -m venv (Join-Path $PSScriptRoot '.venv-cuda')
    if ($LASTEXITCODE) { throw 'Failed to create isolated CUDA environment.' }
}
& $cudaPython -m pip install --disable-pip-version-check --cache-dir (Join-Path $PSScriptRoot '.pip-cache') -r (Join-Path $PSScriptRoot 'requirements-cuda.txt')
if ($LASTEXITCODE) { throw 'Failed to install CUDA environment dependencies.' }
& $cudaPython -m pip install --disable-pip-version-check --cache-dir (Join-Path $PSScriptRoot '.pip-cache') 'torch==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE) { throw 'Failed to install pinned CUDA PyTorch. CPU remains available.' }
& $cudaPython -m pip check
if ($LASTEXITCODE) { throw 'CUDA dependency check failed.' }
