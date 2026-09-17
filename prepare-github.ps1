param(
    [string]$Destination = '',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$rootFull = [System.IO.Path]::GetFullPath($root)

if (!$Destination) {
    $parent = Split-Path -Parent $rootFull
    $repoName = Split-Path -Leaf $rootFull
    $Destination = Join-Path $parent ($repoName + '-github')
}

if ([System.IO.Path]::IsPathRooted($Destination)) {
    $target = [System.IO.Path]::GetFullPath($Destination)
} else {
    $target = [System.IO.Path]::GetFullPath((Join-Path $rootFull $Destination))
}
$targetMarker = $target.TrimEnd('\') + '\'
$rootMarker = $rootFull.TrimEnd('\') + '\'

if ($target -eq $rootFull -or $targetMarker.StartsWith($rootMarker, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "导出目录必须在原工作区外：$target"
}

if (Test-Path -LiteralPath $target) {
    if (!$Force) {
        throw "目标目录已存在：$target。确认可以覆盖时加 -Force 重新运行。"
    }
    if ($target -eq $rootFull -or $targetMarker.StartsWith($rootMarker, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "拒绝删除原工作区内部的目录：$target"
    }
    Remove-Item -LiteralPath $target -Recurse -Force
}

New-Item -ItemType Directory -Path $target | Out-Null
$global:copiedFiles = 0
$global:copiedBytes = [int64]0

function Copy-ExplicitFiles {
    param(
        [string[]]$Names,
        [string]$SourceRoot,
        [string]$TargetRoot
    )

    foreach ($name in $Names) {
        $source = Join-Path $SourceRoot $name
        if (!(Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "缺少应发布文件：$source"
        }
        $destination = Join-Path $TargetRoot $name
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
        $script:copiedFiles++
        $script:copiedBytes += (Get-Item -LiteralPath $source).Length
    }
}

function Copy-PublishTree {
    param(
        [string]$SourceDir,
        [string]$TargetDir,
        [string]$IgnorePrefix = ''
    )

    $temporaryGit = Join-Path ([System.IO.Path]::GetTempPath()) ("prepare-github-" + [System.Guid]::NewGuid().ToString('N') + ".git")
    $temporaryIgnore = Join-Path $temporaryGit "root-excludes"
    try {
        & git init --bare --quiet $temporaryGit
        if ($LASTEXITCODE -ne 0) { throw '无法创建临时 Git 元数据，请确认已安装 Git。' }
        & git --git-dir=$temporaryGit config core.quotepath false
        if ($LASTEXITCODE -ne 0) { throw '无法配置临时 Git。' }

        $scopedRules = New-Object System.Collections.Generic.List[string]
        $scopedRules.Add('.git/')
        foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $rootFull '.gitignore'))) {
            if (!$IgnorePrefix) {
                $scopedRules.Add($line)
                continue
            }
            $prefixWithSlash = $IgnorePrefix + '/'
            if ($line.StartsWith($prefixWithSlash, [System.StringComparison]::OrdinalIgnoreCase)) {
                $scopedRules.Add('/' + $line.Substring($prefixWithSlash.Length))
            } else {
                $scopedRules.Add($line)
            }
        }
        [System.IO.File]::WriteAllLines($temporaryIgnore, $scopedRules)

        Push-Location $SourceDir
        try {
            $files = & git --git-dir=$temporaryGit --work-tree=$SourceDir ls-files --others --exclude-standard --exclude-from=$temporaryIgnore
            if ($LASTEXITCODE -ne 0) { throw "无法枚举可发布文件：$SourceDir" }
        } finally {
            Pop-Location
        }

        New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
        foreach ($relativePath in $files) {
            if (!$relativePath) { continue }
            $source = Join-Path $SourceDir $relativePath
            if (!(Test-Path -LiteralPath $source -PathType Leaf)) { continue }

            $destination = Join-Path $TargetDir $relativePath
            New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
            Copy-Item -LiteralPath $source -Destination $destination
            $script:copiedFiles++
            $script:copiedBytes += (Get-Item -LiteralPath $source).Length
        }
    } finally {
        if (Test-Path -LiteralPath $temporaryGit) {
            Remove-Item -LiteralPath $temporaryGit -Recurse -Force
        }
    }
}

$rootFiles = @(
    '.gitignore',
    'LICENSE',
    'README.md',
    'prepare-github.ps1',
    'setup-workbench.ps1',
    '初始化莱茵语音工作台.cmd',
    '启动莱茵语音工作台.cmd'
)
Copy-ExplicitFiles -Names $rootFiles -SourceRoot $rootFull -TargetRoot $target

# PROJECT_BRIEF.md 是内部研发过程稿，不进入公开发布目录。
$publicDocs = @('USAGE.md', 'OPEN-SOURCE.md', 'GITHUB-CHECKLIST.md')
foreach ($document in $publicDocs) {
    Copy-ExplicitFiles -Names @($document) -SourceRoot (Join-Path $rootFull 'docs') -TargetRoot (Join-Path $target 'docs')
}

Copy-PublishTree -SourceDir (Join-Path $rootFull 'RhineLabUI') -TargetDir (Join-Path $target 'RhineLabUI') -IgnorePrefix 'RhineLabUI'
Copy-PublishTree -SourceDir (Join-Path $rootFull 'latest_stage') -TargetDir (Join-Path $target 'latest_stage') -IgnorePrefix 'latest_stage'
Copy-PublishTree -SourceDir (Join-Path $rootFull 'sample-x') -TargetDir (Join-Path $target 'sample-x') -IgnorePrefix 'sample-x'

& git init --quiet --initial-branch=main $target
if ($LASTEXITCODE -ne 0) { throw '无法在导出目录初始化 Git。' }
& git -C $target config core.quotepath false
if ($LASTEXITCODE -ne 0) { throw '无法配置发布仓库 Git。' }
& git -C $target config core.autocrlf false
if ($LASTEXITCODE -ne 0) { throw '无法配置发布仓库行尾策略。' }

Push-Location $target
try {
    & git add -A
    if ($LASTEXITCODE -ne 0) { throw 'git add 失败，请查看上方错误。' }

    $forbidden = & git ls-files | Select-String -Pattern '(^|/)(outputs|speaker_profiles|runs|portable-models|node_modules|\.venv[^/]*)(/|$)|\.log$'
    if ($forbidden) {
        $forbidden | ForEach-Object { Write-Host ("禁止入库文件：" + $_.Line) -ForegroundColor Red }
        throw '发布目录包含用户数据、权重、虚拟环境或日志。'
    }

    $legacyNames = & git grep --cached -ilE 'doubao|豆包|liteasr' -- . ':(exclude)docs/GITHUB-CHECKLIST.md' ':(exclude)prepare-github.ps1'
    if ($LASTEXITCODE -eq 0 -and $legacyNames) {
        $legacyNames | ForEach-Object { Write-Host ("旧模型名残留：" + $_) -ForegroundColor Red }
        throw '发布目录仍包含旧模型名。'
    }
    if ($LASTEXITCODE -gt 1) { throw '旧模型名检查失败。' }

    $targetGit = [System.IO.Path]::GetFullPath((Join-Path $target '.git'))
    $nestedGit = Get-ChildItem -LiteralPath $target -Force -Recurse -Directory -Filter .git |
        Where-Object { [System.IO.Path]::GetFullPath($_.FullName) -ne $targetGit } |
        Select-Object -First 1
    if ($nestedGit) {
        throw ("发现嵌套 Git 元数据：" + $nestedGit.FullName)
    }

    $trackedCount = (& git ls-files | Measure-Object).Count
    $sizeText = if ($global:copiedBytes -ge 1GB) {
        [math]::Round($global:copiedBytes / 1GB, 2).ToString() + ' GB'
    } elseif ($global:copiedBytes -ge 1MB) {
        [math]::Round($global:copiedBytes / 1MB, 2).ToString() + ' MB'
    } else {
        [math]::Round($global:copiedBytes / 1KB, 2).ToString() + ' KB'
    }

    Write-Host ''
    Write-Host ("发布目录：" + $target) -ForegroundColor Green
    Write-Host ("复制文件：" + $global:copiedFiles + "；Git 跟踪：" + $trackedCount + "；体积：" + $sizeText)
    Write-Host '预检查通过：敏感目录、旧模型名、嵌套 Git 均未发现。'
    Write-Host ''
    Write-Host '下一步：'
    Write-Host '  1. cd <发布目录>'
    Write-Host '  2. git commit -m "Initial release"'
    Write-Host '  3. git remote add origin <GitHub仓库地址>'
    Write-Host '  4. git push -u origin main'
} finally {
    Pop-Location
}
