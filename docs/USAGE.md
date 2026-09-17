# 使用说明

## 1. 前置条件

| 组件 | 要求 | 说明 |
| --- | --- | --- |
| 操作系统 | Windows 10/11 | 电脑音频采集依赖 WASAPI 回环 |
| Python | 3.10+，建议 3.11 | 主后端与样品-X 服务使用独立虚拟环境 |
| Node.js | 22.12+ | 仅重新构建前端时需要 |
| GPU | NVIDIA，可选 | Qwen3-ASR、Fun-ASR-Nano、MOSS、样品-X CUDA 模式受益 |
| 浏览器 | Chrome / Edge 等现代浏览器 | 需要 WebGL 2；样品-X 档案会请求麦克风或屏幕共享授权 |

磁盘空间参考：SenseVoice ONNX 模型约 230 MB；MOSS 会议模型约 2 GB；Qwen3-ASR 与样品-X 模型资产另计。安装 PyTorch 后虚拟环境体积较大。初始化只安装运行环境，不预下载这些识别模型。

## 2. 环境搭建

推荐双击仓库根目录的 `初始化莱茵语音工作台.cmd`。它会创建主虚拟环境、安装依赖，并检查前端构建产物。需要指定 Python 时可执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-workbench.ps1 -Python C:\Path\To\python.exe
```

以下手动步骤适合开发、排查或修复模型缓存。除特别说明外，均在仓库根目录的 PowerShell 中执行。

### 2.1 后端主环境

```powershell
cd latest_stage
python -m venv .venv-asr
.venv-asr\Scripts\pip install -r requirements.txt
```

## 3. 模型管理（X-011）

启动工作台后进入 X-011「模型管理」，页面会显示每个模型关联的功能、安装状态、来源许可、设备与体积要求、本机位置，并提供安装 / 卸载按钮。

- **SenseVoice 实时引擎**：自动下载 SenseVoiceSmall，导出 ONNX 并生成 INT8 推理包。
- **FSMN-VAD / CAM++**：目标说话人识别、声纹注册和 Fun-ASR-Nano 实时模式所需的轻量组件。
- **Fun-ASR-Nano / Qwen3-ASR**：普通识别的可选大模型，按档案需要安装。
- **MOSS-Transcribe-Diarize**：会议转写模型及独立运行环境。
- **样品-X**：模型资产不随仓库分发，需按说明手动准备；本页可卸载已放入的资产和独立环境。
- **Silero VAD**：随仓库内置，用于样品-X端点检测，无需在线安装。

模型操作会先释放当前已加载引擎；识别任务运行中会拒绝安装或卸载。卸载只删除页面声明的模型缓存、生成文件或专用运行环境，不会删除 `outputs` 中的录音、转写结果和声纹数据。样品-X模型资产没有自动下载入口，避免误取得未随仓库授权分发的权重。

## 4. 手动模型与构建

如需在命令行下载目标说话人与实时模式所需的轻量模型：

```powershell
.venv-asr\Scripts\python scripts\prepare_target_models.py
```

模型默认下载到当前用户 `~\.cache\modelscope\models\`，运行时会复用本机缓存。

### 4.1 SenseVoice ONNX 模型

如需手动重建 `models/sensevoice_small_int8_bundle`：

```powershell
# 先确保 ModelScope 缓存中已有 iic/SenseVoiceSmall
.venv-asr\Scripts\python scripts\export_sensevoice_onnx.py
.venv-asr\Scripts\python scripts\quantize_sensevoice_onnx.py
```

### 4.2 可选：MOSS 会议模型

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_moss_windows.ps1
```

该脚本会创建独立的 `.venv-moss`，下载固定 revision 的官方模型，并逐文件校验 SHA-256。

### 4.3 可选：样品-X Sample-X 档案

样品-X 档案需要按 [sample-x/README.md](../sample-x/README.md) 准备本地模型资产。准备好后创建独立环境：

```powershell
cd ..\sample-x
powershell -ExecutionPolicy Bypass -File setup.ps1 -Python <Python3.11路径>

# 需要 CUDA 加速时：
powershell -ExecutionPolicy Bypass -File setup-cuda.ps1
```

没有 CUDA 环境时，启动器会使用 CPU（MNN）路径；界面也可以手动选择“仅 CPU”。

### 4.4 构建前端

仓库已包含 `RhineLabUI\dist\` 构建产物，普通使用不需要安装 Node.js。修改前端源码后重建：

```powershell
cd ..\RhineLabUI
npm ci
npm run build
```

后端启动时会检查 `dist/index.html` 是否存在。开发调试可用 `npm run dev`，默认 5174 端口，`/api` 会代理到 8765。

## 5. 启动与停止

- **启动**：双击仓库根目录的 `启动莱茵语音工作台.cmd`，浏览器会打开三维档案界面。
- **停止**：结束所有识别任务后，运行 `RhineLabUI\停止莱茵服务.cmd`。有任务运行时会拒绝退出，请先在页面点击停止。
- **重复启动**：启动器会复用已就绪的后端服务，不会重复启动进程。
- **界面入口**：`/rhine/` 是三维档案版；根路径保留旧版表单式工作台。

后端日志位于 `RhineLabUI\.tools\speech-backend.log` 与 `speech-backend-error.log`，排查启动失败时优先查看。

## 6. 界面操作

1. 首次进入会播放入场动画，按 `Enter` / `Esc` 或点击 `ENTER SYSTEM` 跳过。
2. `←` / `→` 切换功能栏，`↑` / `↓` 在栏内翻阅档案。
3. 点击或按 `Enter` 打开 **ACCESS FILE**：左侧是可旋转、可拆解的玻璃档案，右侧是该功能的操作面板、音频来源与结果页签。
4. `/` 或点击 **ARCHIVE INDEX** 打开检索，可按编号、功能名或模型名搜索。
5. `Esc` 返回档案阵列；收藏与设置存储在浏览器本地。

每个档案一次只允许一个任务。运行中切换档案不会自动开始新任务；关闭页面不会停止后端录音，离开前请先点击停止。

## 7. 功能操作要点

### 7.1 实时 / 非实时普通识别（X-001、X-002、X-003、X-005、X-006）

1. 进入档案后点击 **加载模型**，等待“当前模型已就绪”。
2. 选择输入来源：麦克风、电脑音频或单个 WAV。
3. 点击开始。实时模式会显示灰色预览，句尾或停止后定稿；预览可能被修订，以最终文本为准。
4. 完成后可复制、导出 TXT、回放或下载标准化的 16 kHz WAV。

Fun-ASR-Nano 实时模式首字约 1.5–2.5 秒；Qwen3-ASR 1.7B 首次加载可能超过一分钟，且需要较大显存。

### 7.2 目标说话人识别（X-007）

1. 先在 X-009 注册目标声纹。
2. 进入 X-007，点击 **加载模型**。
3. 选择输入来源后开始录音。系统按约 1.5 秒窗口做声纹验证，只有通过验证的音频会送入 ASR。
4. 页面显示最近一次匹配分数；阈值默认 0.45。误收他人时调高，目标漏字时先检查注册质量再调低。

相似度是余弦距离门槛，不是概率。该功能是声纹验证而非声源分离，重叠讲话、窗口内换人仍可能出错。

### 7.3 多人会议转写（X-008）

1. 加载 MOSS 模型后，选择输入来源并开始采集或上传 WAV。
2. 停止后整段录音一次性联合推理，按说话人编号与时间戳分段显示。
3. 支持复制全文、导出 TXT / JSON / SRT。

S01、S02 只表示本次录音中的匿名说话人，不能跨录音对应身份。长录音耗时与显存随输入、输出增长，显存不足时请缩短录音或关闭其他 GPU 任务。

### 7.4 声纹注册与管理（X-009）

1. 点击 **加载声纹组件**，输入目标名称。
2. 选择输入来源后录制 3–30 秒清晰单人语音，建议安静环境 10–20 秒；也可以上传注册 WAV。
3. 注册成功后即可在 X-007 中选择该目标。
4. 覆盖与删除需要确认；删除后需重新注册才能恢复目标识别。

### 7.5 样品-X Sample-X（X-010）

1. 进入档案后点击 **加载样品-X引擎**，等待引擎就绪，首次需要几十秒。
2. 选择音源：浏览器麦克风、屏幕共享音频或 WAV 重放；浏览器会弹出相应授权。
3. 端点检测按最长 20 秒完整分段推理，草稿可修订、定稿不改写；长文本自动跟随末尾，向上阅读时不强制拉回。
4. 支持复制、导出 TXT、下载 WAV 与 JSON。

默认“自动”模式优先 CUDA FP32 解码 / 输出层 + MNN CPU 编码，异常时回退 CPU 重算；面板会显示实际后端与回退原因。关闭页面会中断会话并标记未完整完成。

## 8. 结果与数据管理

- 普通识别与目标说话人识别支持复制、导出 TXT 和下载录音。
- 会议转写支持 TXT、JSON 与 SRT，包含分段说话人与时间戳。
- 声纹注册只保存推理所需特征，不保留注册音频。
- 导出的音频、文本和声纹相关数据可能包含个人信息，请根据实际使用场景妥善保管。

## 9. 常见问题

- **页面提示模型未安装**：进入 X-011「模型管理」安装对应档案的模型；无需重新初始化或重启工作台。

- **启动器提示 Python 环境缺失**：先运行 `初始化莱茵语音工作台.cmd`；仍失败时可设置 `SPEECH_PYTHON` 为后端虚拟环境中的 `python.exe`。
- **提示 Speech frontend build is missing**：重新获取完整仓库，或在 `RhineLabUI` 执行 `npm ci && npm run build`。
- **提示 Backend did not serve /rhine/**：查看 `speech-backend-error.log`；若旧版服务仍在运行，先停止后重试。
- **页面无声音 / 识别为空**：实时模式需要先点击开始并等到“正在聆听”；浏览器音频需要一次用户交互后才会播放界面音效，不影响识别。
- **CUDA 报错或回退 CPU**：确认显卡驱动与 PyTorch CUDA 版本匹配；样品-X 档案可在界面选择“仅 CPU”先跑通，再切回自动重试。
- **换麦克风 / 扬声器无效**：普通档案使用 Windows 默认设备，请修改系统默认输入 / 输出；样品-X 档案在浏览器授权中选择设备。

## 10. 开发与测试

```powershell
# 后端测试（latest_stage 目录）
.venv-asr\Scripts\python -m pytest -q tests\test_samplex_service.py tests\test_dictation_web.py

# 前端类型检查与语音档案回归（RhineLabUI 目录）
npm run build
node --experimental-strip-types --test scripts/check-speech.mjs

# 样品-X 服务测试（sample-x 目录）
.venv\Scripts\python test_backend.py
.venv\Scripts\python runtime\test_tokenizer.py
```
