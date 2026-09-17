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
Write-Host '首次使用可在网页的「模型管理」档案中按需安装模型；样品-X模型资产需按 sample-x/README.md 手动准备。'
