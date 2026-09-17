param(
    [string]$PackageIndex = 'https://pypi.tuna.tsinghua.edu.cn/simple',
    [string]$TorchIndex = 'https://mirror.sjtu.edu.cn/pytorch-wheels/cu128'
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$environmentDirectory = Join-Path $projectDirectory '.venv-moss'
if (-not (Test-Path (Join-Path $environmentDirectory 'Scripts/python.exe'))) {
    py -3.11 -m venv $environmentDirectory
    if ($LASTEXITCODE -ne 0) { throw 'MOSS 虚拟环境创建失败' }
}
$mossPython = Join-Path $environmentDirectory 'Scripts/python.exe'
& $mossPython -m pip install --upgrade pip --index-url $PackageIndex
if ($LASTEXITCODE -ne 0) { throw 'MOSS pip 更新失败' }
& $mossPython (Join-Path $PSScriptRoot 'download_moss_torch.py')
if ($LASTEXITCODE -ne 0) { throw 'MOSS PyTorch 下载或校验失败（已下载分段会保留供续传）' }
$torchWheel = Join-Path $projectDirectory '.cache/moss-wheels/torch-2.8.0+cu128-cp311-cp311-win_amd64.whl'
& $mossPython -m pip install $torchWheel --index-url $PackageIndex --timeout 120 --retries 10
if ($LASTEXITCODE -ne 0) { throw 'MOSS PyTorch 安装失败' }
& $mossPython -m pip install torchaudio==2.8.0 --index-url $TorchIndex --timeout 120 --retries 10
if ($LASTEXITCODE -ne 0) { throw 'MOSS PyTorch 安装失败' }
& $mossPython -m pip install -r (Join-Path $projectDirectory 'requirements-moss.txt') --index-url $PackageIndex --timeout 120 --retries 10
if ($LASTEXITCODE -ne 0) { throw 'MOSS 依赖安装失败' }
& $mossPython (Join-Path $PSScriptRoot 'prepare_moss_models.py')
if ($LASTEXITCODE -ne 0) { throw 'MOSS 模型下载失败' }
Write-Output 'MOSS 已准备好。重启网页服务，在 非实时 → 多人会议转写 中加载模型。'
