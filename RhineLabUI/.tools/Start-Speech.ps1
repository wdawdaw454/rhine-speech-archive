param([switch]$NoBrowser, [switch]$Stop)
$ErrorActionPreference = 'Stop'
$frontRoot = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent $frontRoot
$backendRoot = Join-Path $workspace 'latest_stage'
$url = 'http://127.0.0.1:8765/'

function Resolve-SpeechPython {
    if ($env:SPEECH_PYTHON -and (Test-Path -LiteralPath $env:SPEECH_PYTHON)) { return $env:SPEECH_PYTHON }
    $candidates = @(
        (Join-Path $backendRoot '.venv-asr\Scripts\pythonw.exe'),
        (Join-Path $backendRoot '.venv\Scripts\pythonw.exe'),
        (Join-Path $backendRoot '.venv-asr\Scripts\python.exe'),
        (Join-Path $backendRoot '.venv\Scripts\python.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    foreach ($name in @('pythonw.exe', 'python.exe')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) { return $command.Source }
    }
    throw 'Python was not found. Run the setup command in the repository root, or set SPEECH_PYTHON.'
}

try {
    $status = $null
    try { $status = Invoke-RestMethod ($url + 'api/status') -TimeoutSec 2 } catch { }
    if ($Stop) {
        if ($status) {
            if ($status.external_busy -or $status.state -in @('loading','starting','listening','stopping','transcribing','enrolling','enroll_recording')) {
                throw 'A speech task is active. Stop it in the speech page before closing the backend.'
            }
            Invoke-RestMethod ($url + 'api/shutdown') -Method Post -ContentType 'application/json' -Body '{}' | Out-Null
        }
        exit 0
    }
    if (!(Test-Path -LiteralPath (Join-Path $frontRoot 'dist/index.html'))) { throw 'Speech frontend build is missing.' }
    if (!$status) {
        $python = Resolve-SpeechPython
        $env:OMP_NUM_THREADS = '4'
        $env:TQDM_DISABLE = '1'
        $env:PYTHONIOENCODING = 'utf-8'
        $entry = Join-Path $backendRoot 'scripts/run_dictation_web.py'
        Start-Process -FilePath $python -ArgumentList @(('"' + $entry + '"')) -WorkingDirectory $backendRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $PSScriptRoot 'speech-backend.log') -RedirectStandardError (Join-Path $PSScriptRoot 'speech-backend-error.log') | Out-Null
    }
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            $response = Invoke-WebRequest ($url + 'rhine/') -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200 -and $response.Content.Contains('RHINE LAB')) { $ready = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    if (!$ready) { throw 'Backend did not serve /rhine/. If an older backend is running, close it from its page and launch again.' }
    Write-Host ('Speech terminal: ' + $url + 'rhine/')
    if (!$NoBrowser) { Start-Process ($url + 'rhine/') }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
