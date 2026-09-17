# GitHub 发布前清单

上传前请逐项确认。目标是：代码可入库，权重、用户数据与日志不入库。

## 1. 必须排除的内容

| 路径 | 体积 / 风险 | 处理 |
| --- | --- | --- |
| `latest_stage/.venv*`、`sample-x/.venv*` | 数 GB 虚拟环境 | 已忽略 |
| `latest_stage/models/` | 约 2 GB 模型权重 | 已忽略，用脚本下载 |
| `latest_stage/data/`、`Funasr_tests/` | 本地 WAV、实验数据与临时测试 | 已整体忽略 |
| `latest_stage/docs/`、`latest_stage/README.md` | 历史研发记录 | 已忽略，公开说明统一在根目录 `docs/` |
| `latest_stage/run_windows*.bat`、`setup_windows_gpu_models.bat` | 旧版本机路径启动/安装入口 | 已忽略，统一使用根目录初始化与启动脚本 |
| `sample-x/runtime/portable-models/` | 339 MB 样品-X转换权重，许可不明 | 已忽略，**严禁上传** |
| `latest_stage/outputs/` | 用户录音、识别结果、声纹 JSON | 已忽略，属敏感数据 |
| `sample-x/runs/`、`logs/`、`benchmarks/` | 会话录音与运行记录 | 已忽略 |
| `sample-x` 的 benchmark / CUDA 验收 / 依赖真实录音分段的脚本 | 本机实验与运行数据派生物 | 已忽略，仅保留服务、运行时与确定性测试 |
| `RhineLabUI/node_modules/` | 103 MB 依赖 | 已忽略 |
| `RhineLabUI/art/`、`reference/`、`wallpaper/` | 原始三维工程、参考资料与壁纸工程 | 已忽略；公开仓库保留源码、公共资源与 `dist` |
| `RhineLabUI/dist/` | 约 42 MB 构建产物 | **保留入库**：让克隆者不必安装 Node.js 也能开箱运行；重新构建后需提交更新 |
| `RhineLabUI/verification/` | 内部截图与验收记录 | 已忽略；仅保留 `boot-lettering/webfont-sources.json` 供构建脚本校验 |
| `remote_stage/`、`.remote_original/`、`.patchbase/`、`.tmp_qwen3_asr/`、`.codex-temp/` | 历史工作区与临时目录 | 已忽略 |
| `*.log`、`__pycache__/`、`.pytest_cache/` 等缓存 | 运行垃圾 | 已忽略 |

## 2. 敏感数据检查

上传前在仓库根目录执行：

```powershell
git status --short
git ls-files | Select-String -Pattern 'outputs|speaker_profiles|sample-x/runs|portable-models|\.venv|node_modules|\.log$'
```

第二条命令的期望输出为空。特别确认 `latest_stage/outputs/speaker_profiles/browser_target.json`（真实声纹）与 `sample-x/runs/`（真实录音）不在其中。

## 3. 许可与合规

- 自有代码：根目录已提供 MIT `LICENSE`，与 RhineLabUI 保持一致。
- 样品-X Sample-X 权重：不可再分发，保持忽略；README 已声明需要自行准备。
- 《明日方舟》/“莱茵生命”元素：非官方同人使用，README 已声明版权与免责；不要移除该声明。
- MiSans 字体：按其官方协议分发 webfont 分片，保留 `NOTICE.txt` 与许可 PDF。
- 模型权重一律通过 ModelScope / Hugging Face 官方渠道下载，不在仓库内二次分发。

## 4. 代码可移植性状态

1. 已完成：`RhineLabUI/.tools/Start-Speech.ps1` 按仓库相对路径自动探测 `latest_stage/.venv-asr` / `.venv`，并支持 `SPEECH_PYTHON`。
2. 已完成：根目录提供 `初始化莱茵语音工作台.cmd` 与 `setup-workbench.ps1`；`RhineLabUI/dist` 随仓库提交。
3. `latest_stage` 的部分历史实验文档可能引用服务器环境路径：可在 README 标注“历史记录，不代表当前运行方式”，或移入 `docs/history/`。

## 5. 嵌套 Git 仓库处理

当前 `latest_stage/` 内部有自己的 `.git`，其 remote 指向内网仓库，且包含大量未提交改动。若直接在根目录 `git init && git add .`，Git 会把 `latest_stage` 记录成 gitlink，而不是加入后端源码；公开克隆者将无法开箱运行。

推荐直接在当前工作区根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\prepare-github.ps1
```

脚本会按 `.gitignore` 导出允许入库的文件到同级 `ASR_Voiceprint-github` 目录，在那里初始化全新的 Git 仓库，并自动检查敏感目录、旧模型名和嵌套 Git。重复导出时加 `-Force`。这不会改动 `latest_stage/.git` 及其内网历史。

如果需要手动处理，也可以临时移走嵌套仓库元数据：把 `latest_stage/.git` 重命名到仓库外备份位置（不要删除），根目录初始化并提交后再决定是否移回。操作前先确认内网仓库没有正在进行的任务。

不建议做成 submodule：除非 `latest_stage` 也发布为独立公开仓库，否则额外克隆步骤和内网 remote 会破坏“开箱即跑”。

## 6. 发布前检查命令

当前根目录尚未初始化 Git，`git status` 报 “not a git repository” 属于预期状态。初始化发布仓库后执行：

```powershell
git status --short
git ls-files | Select-String -Pattern 'outputs|speaker_profiles|sample-x/runs|portable-models|\.venv|node_modules|\.log$'
```

第一条确认没有误加大文件或本地目录，第二条期望为空。随后可用 `git grep -inE "doubao|豆包|liteasr"` 检查公开文件集中没有旧模型名；历史归档与运行记录已被 `.gitignore` 排除，本地存在不影响发布。

## 7. 建议的仓库结构

公开仓库建议只保留：

```text
启动莱茵语音工作台.cmd
初始化莱茵语音工作台.cmd
README.md
.gitignore
LICENSE
setup-workbench.ps1
prepare-github.ps1
docs/                   # USAGE.md、OPEN-SOURCE.md、GITHUB-CHECKLIST.md
RhineLabUI/             # 前端源码与 dist（不含 node_modules）
latest_stage/           # 后端源码、脚本、测试（不含模型、输出、venv）
sample-x/                 # 样品-X服务源码（不含权重、venv、runs）
```

首次提交前用 `git status` 复查没有误加的大文件；若误提交过权重，仅删除文件不够，需重写历史后再公开。
