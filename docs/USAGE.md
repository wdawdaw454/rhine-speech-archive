# 使用说明

## 1. 前置条件

| 组件 | 要求 | 说明 |
| --- | --- | --- |
| 操作系统 | Windows 10/11 | 电脑音频采集依赖 WASAPI 回环；其他系统可跑后端但没有回环输入 |
| Python | 3.10+，建议 3.11 | 后端与样品-X服务使用独立虚拟环境 |
| Node.js | 22.12+ | 仅构建前端时需要，日常启动不需要常驻 Node 服务 |
| GPU | NVIDIA，可选 | Qwen3-ASR、Fun-ASR-Nano、MOSS、样品-X CUDA 模式受益；无 GPU 自动回退 CPU |
| 浏览器 | Chrome / Edge 等现代浏览器 | 需要 WebGL 2；样品-X档案会请求麦克风或屏幕共享授权 |

磁盘预算参考：核心 SenseVoice ONNX 模型约 230 MB；MOSS 会议模型约 2 GB；Qwen3-ASR 1.7B 与样品-X权重另计。虚拟环境安装 PyTorch 后体积较大，请预留足够空间。

## 2. 环境搭建

推荐直接双击仓库根目录的 `初始化莱茵语音工作台.cmd`。它会完成本节的主环境创建、依赖安装、核心模型下载、SenseVoice 导出与量化，并检查前端产物；需要自定义 Python 时可执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-workbench.ps1 -Python C:\Path\To\python.exe
```

以下手动步骤适合开发、排查或按需安装可选模型。除特别说明外，均在仓库根目录的 PowerShell 中执行。

### 2.1 后端主环境

```powershell
cd latest_stage
python -m venv .venv-asr
.venv-asr\Scripts\pip install -r requirements.txt
```

下载目标说话人与实时模式所需的轻量模型（FSMN-VAD、CAM++）：

```powershell
.venv-asr\Scripts\python scripts\prepare_target_models.py
```

模型默认下载到当前用户 `~\.cache\modelscope\models\`，运行时读取本机缓存，不重复下载。

### 2.2 SenseVoice ONNX 模型

仓库不包含模型权重。如需重建 `models/sensevoice_small_int8_bundle`：

```powershell
# 先确保 ModelScope 缓存中已有 iic/SenseVoiceSmall
.venv-asr\Scripts\python scripts\export_sensevoice_onnx.py
.venv-asr\Scripts\python scripts\quantize_sensevoice_onnx.py
```

### 2.3 可选：MOSS 会议模型

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_moss_windows.ps1
```

该脚本会创建独立的 `.venv-moss`（Python 3.11 + PyTorch CUDA），并下载固定 revision 的官方模型到 `models/moss_transcribe_diarize`，逐文件校验 SHA-256。

### 2.4 可选：样品-X Sample-X 档案

样品-X档案需要 `sample-x/` 目录下的本地模型文件（Sample-X v3.2.1 转换权重）。这些权重不在仓库中，需要按 `sample-x/README.md` 的说明自行准备，然后创建独立环境：

```powershell
cd ..\sample-x
powershell -ExecutionPolicy Bypass -File setup.ps1 -Python <Python3.11路径>
# 需要 CUDA 加速时：
powershell -ExecutionPolicy Bypass -File setup-cuda.ps1
```

没有 CUDA 环境时，启动器会使用 CPU（MNN）路径；界面也可手动选择“仅 CPU”。

### 2.5 构建前端（修改源码后需要）

仓库已包含 `RhineLabUI\dist\` 正式构建产物，普通使用不需要安装 Node.js。修改 `RhineLabUI` 源码后重建：

```powershell
cd ..\RhineLabUI
npm ci
npm run build
```

构建产物输出到 `dist/`，后端启动时会检查 `dist/index.html` 是否存在。日常使用不需要启动 Vite 开发服务；开发调试可用 `npm run dev`（默认 5174 端口，`/api` 代理到 8765）。


## 3. 启动与停止

- **启动**：双击仓库根目录的 `启动莱茵语音工作台.cmd`。若后端未运行，会以隐藏窗口启动；就绪后自动打开 <http://127.0.0.1:8765/rhine/>。
- **停止**：在页面中结束所有识别任务后，运行 `RhineLabUI\停止莱茵服务.cmd`（对应 `Start-Speech.ps1 -Stop`）。有任务运行时会拒绝退出，请先在页面点击停止。
- **重复启动**：启动器会检测 `127.0.0.1:8765` 是否已有服务，直接复用，不会重复开进程。
- **原语音界面**：<http://127.0.0.1:8765/> 保留旧版表单式工作台；`/rhine/` 是三维档案版。

后端日志位于 `RhineLabUI\.tools\speech-backend.log` 与 `speech-backend-error.log`，排查启动失败时先看这两个文件。

## 4. 界面操作

1. 首次进入会播放入场动画，按 `Enter` / `Esc` 或点击 `ENTER SYSTEM` 跳过。
2. `←` / `→` 切换功能栏（实时普通、目标说话人、非实时、会议、样品-X），`↑` / `↓` 在栏内翻阅档案。
3. 点击或按 `Enter` 打开 **ACCESS FILE**：左侧仍是可旋转、可拆解的玻璃档案，右侧是该功能的操作面板、音频来源与结果页签。
4. `/` 或点击 **ARCHIVE INDEX** 打开检索，可按编号、功能名或模型名搜索。
5. `Esc` 返回档案阵列；收藏与设置存储在浏览器本地。

每个档案一次只允许一个任务。运行中切换档案不会自动开始新任务；关闭页面不会停止后端录音，离开前请先点击停止。

## 5. 各功能操作要点

### 5.1 实时 / 非实时普通识别（X-001、X-002、X-003、X-005、X-006）

1. 进入档案后点击 **加载模型**，等待“当前模型已就绪”。
2. 选择输入来源：麦克风、电脑音频或单个 WAV。
3. 点击开始，实时模式会在页面显示灰色预览，句尾或停止后定稿；预览可能被修订，以最终文本为准。
4. 完成后可复制、导出 TXT、回放或下载标准化的 16 kHz WAV。

Fun-ASR-Nano 实时模式首字约 1.5–2.5 秒；Qwen3-ASR 1.7B 首次加载可能超过一分钟，且需要较大显存。

### 5.2 目标说话人识别（X-007）

1. 先在 X-009 注册目标声纹（见 5.4）。
2. 进入 X-007，点击 **加载模型**（FSMN-VAD + CAM++ + SenseVoice，全程 CPU）。
3. 选择输入来源后开始录音。系统按约 1.5 秒窗口做声纹验证，只有通过验证的音频会送入 ASR。
4. 页面显示最近一次匹配分数；阈值默认 0.45。误收他人时调高，目标漏字时先检查注册质量再调低。

相似度是余弦距离门槛，不是概率。该功能是声纹验证而非声源分离，重叠讲话、窗口内换人仍可能出错；完整输入 WAV（包括被过滤者的声音）会保存在输出目录。

### 5.3 多人会议转写（X-008）

1. 加载 MOSS 模型后，选择输入来源并开始采集或上传 WAV。
2. 停止后整段录音一次性联合推理，按说话人编号与时间戳分段显示。
3. 支持复制全文、导出 TXT / JSON / SRT；输出目录额外保存 `meeting_segments.json` 与 `meeting.srt`。

S01、S02 只表示本次录音中的匿名说话人，不能跨录音对应身份。长录音耗时与显存随输入、输出增长，显存不足时请缩短录音或关闭其他 GPU 任务。

### 5.4 声纹注册与管理（X-009）

1. 点击 **加载声纹组件**，输入目标名称。
2. 选择输入来源后录制 3–30 秒清晰单人语音（建议安静环境 10–20 秒），或上传注册 WAV。
3. 注册成功后声纹特征保存在 `latest_stage\outputs\speaker_profiles\browser_target.json`；原音频不保存。
4. 覆盖与删除需要确认；删除后需重新注册才能恢复目标识别。

### 5.5 样品-X Sample-X（X-010）

1. 进入档案后点击 **加载样品-X引擎**，等待引擎就绪（首次需要几十秒）。
2. 选择音源：浏览器麦克风、屏幕共享音频或 WAV 重放；浏览器会弹出相应授权。
3. 端点检测按最长 20 秒完整分段推理，草稿可修订、定稿不改写；长文本自动跟随末尾，向上阅读时不强制拉回。
4. 结果保存在 `sample-x\runs\<会话编号>\`，支持复制、导出 TXT、下载 WAV 与 JSON。

默认“自动”模式优先 CUDA FP32 解码 / 输出层 + MNN CPU 编码，异常时回退 CPU 重算；面板会显示实际后端与回退原因。关闭页面会中断会话并标记未完整完成。

## 6. 输出文件位置

| 功能 | 输出目录 | 内容 |
| --- | --- | --- |
| 普通识别 / 目标说话人 / 会议 | `latest_stage\outputs\phase5_16\browser_ui\<时间戳>\` | `recording.wav`、`transcript.txt`、`events.jsonl`、`summary.json`，会议另有 `meeting.srt` 等 |
| 声纹特征 | `latest_stage\outputs\speaker_profiles\` | `browser_target.json` |
| 样品-X Sample-X | `sample-x\runs\<会话编号>\` | WAV、事件 JSONL、统计 JSON |
| 后端日志 | `RhineLabUI\.tools\` | `speech-backend.log`、`speech-backend-error.log` |

输出目录包含原始录音与声纹，属于敏感数据，请妥善保管；`.gitignore` 已将这些目录排除在 Git 之外。

## 7. 常见问题

- **启动器提示 Python 环境缺失**：先运行 `初始化莱茵语音工作台.cmd`；仍失败时可设置 `SPEECH_PYTHON` 为后端虚拟环境中的 `python.exe`。
- **提示 Speech frontend build is missing**：`RhineLabUI\dist` 缺失，重新下载完整仓库，或在 `RhineLabUI` 执行 `npm ci && npm run build`。
- **提示 Backend did not serve /rhine/**：后端启动失败，查看 `speech-backend-error.log`；若旧版 8765 服务仍在运行，先从其页面关闭再重试。
- **页面无声音 / 识别为空**：实时模式需要先点击开始并等到“正在聆听”；浏览器音频需要一次用户交互后才会播放界面音效，不影响识别。
- **CUDA 报错或回退 CPU**：确认显卡驱动与 PyTorch CUDA 版本匹配；样品-X档案可在界面选择“仅 CPU”先跑通，再切回自动重试。
- **换麦克风 / 扬声器无效**：普通档案使用 Windows 默认设备，请修改系统默认输入 / 输出；样品-X档案在浏览器授权中选择设备。

## 8. 开发与测试

```powershell
# 后端测试（latest_stage 目录）
.venv-asr\Scripts\python -m pytest -q tests\test_samplex_service.py tests\test_dictation_web.py

# 前端类型检查与语音档案回归（RhineLabUI 目录）
npm run build
node --experimental-strip-types --test scripts/check-speech.mjs

# 样品-X服务测试（sample-x 目录）
.venv\Scripts\python test_backend.py
# tokenizer 回归需要先按 sample-x/README.md 补齐本地模型资产；缺失时会跳过
.venv\Scripts\python runtime\test_tokenizer.py
```
