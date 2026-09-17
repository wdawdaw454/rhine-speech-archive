param(
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

function Find-Python {
    param([string]$Requested)
    if ($Requested) {
        if (!(Test-Path -LiteralPath $Requested)) { throw "指定的 Python 不存在：$Requested" }
        return (Resolve-Path -LiteralPath $Requested).Path
    }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $probe = & py.exe -3 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $probe) { return $probe.Trim() }
    }
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    throw '未找到 Python 3。请从 python.org 安装 Python 3.10+，或用 -Python 参数指定 python.exe。'
}

function Invoke-Step {
    param([string]$Name, [scriptblock]$Action)
    Write-Host ("==> " + $Name) -ForegroundColor Cyan
    & $Action
}

$python = Find-Python $Python
$backend = Join-Path $root 'latest_stage'
$venv = Join-Path $backend '.venv-asr'
$venvPython = Join-Path $venv 'Scripts\python.exe'

Write-Host ("Python: " + $python)

if (!(Test-Path -LiteralPath $venvPython)) {
    Invoke-Step '创建后端虚拟环境' {
        & $python -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw '虚拟环境创建失败' }
    }
}

Invoke-Step '安装后端依赖（首次约数 GB，含 PyTorch）' {
    & $venvPython -m pip install --disable-pip-version-check -r (Join-Path $backend 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw '依赖安装失败' }
}

Invoke-Step '下载轻量模型（FSMN-VAD / CAM++）' {
    & $venvPython (Join-Path $backend 'scripts\prepare_target_models.py')
    if ($LASTEXITCODE -ne 0) { throw '模型下载失败' }
}

$int8Bundle = Join-Path $backend 'models\sensevoice_small_int8_bundle'
$onnxBundle = Join-Path $backend 'models\sensevoice_small_onnx_bundle'
if (!(Test-Path -LiteralPath (Join-Path $int8Bundle 'model.onnx'))) {
    $snapshot = ''
    Invoke-Step '下载 SenseVoiceSmall（约 900 MB）' {
        $script:snapshot = (& $venvPython -c "from modelscope import snapshot_download; print(snapshot_download('iic/SenseVoiceSmall'))" | Select-Object -Last 1).Trim()
        if ($LASTEXITCODE -ne 0) { throw 'SenseVoiceSmall 下载失败' }
    }
    if (!(Test-Path -LiteralPath (Join-Path $onnxBundle 'model.onnx'))) {
        Invoke-Step '导出 SenseVoice ONNX' {
            & $venvPython (Join-Path $backend 'scripts\export_sensevoice_onnx.py') --source-model-dir $snapshot --output-dir $onnxBundle
            if ($LASTEXITCODE -ne 0) { throw 'ONNX 导出失败' }
        }
    }
    Invoke-Step '生成 INT8 量化模型（约 230 MB）' {
        & $venvPython (Join-Path $backend 'scripts\quantize_sensevoice_onnx.py') --bundle-dir $onnxBundle --output-dir $int8Bundle
        if ($LASTEXITCODE -ne 0) { throw 'INT8 量化失败' }
    }
} else {
    Write-Host '==> SenseVoice INT8 模型已就绪，跳过导出'
}

$dist = Join-Path $root 'RhineLabUI\dist\index.html'
if (!(Test-Path -LiteralPath $dist)) {
    $node = Get-Command node.exe -ErrorAction SilentlyContinue
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (!$node -or !$npm) { throw '前端构建产物缺失，且未找到 Node.js。请安装 Node.js 22.12+ 后重新运行本脚本。' }
    Invoke-Step '安装前端依赖并构建' {
        Push-Location (Join-Path $root 'RhineLabUI')
        try {
            & npm.cmd ci
            if ($LASTEXITCODE -ne 0) { throw 'npm ci 失败' }
            & npm.cmd run build
            if ($LASTEXITCODE -ne 0) { throw '前端构建失败' }
        } finally { Pop-Location }
    }
} else {
    Write-Host '==> 前端构建产物已就绪'
}

Write-Host ''
Write-Host '初始化完成。双击「启动莱茵语音工作台.cmd」即可启动。' -ForegroundColor Green
Write-Host '可选模型（Fun-ASR-Nano / Qwen3-ASR / MOSS / 样品-X）请参考 docs/USAGE.md 按需安装。'
