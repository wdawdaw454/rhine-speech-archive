# 莱茵语音工作台 · Rhine Lab Speech Workbench

把本地语音识别装进一个可以操作的三维档案终端。

![莱茵生命终端：由透明档案盒构成的三维阵列](RhineLabUI/docs/media/archive.jpg)

莱茵语音工作台是一个运行在 Windows 本机的中文语音识别工作台：双击 `启动莱茵语音工作台.cmd`，浏览器会打开一个《明日方舟》莱茵生命风格的三维档案界面，每份“玻璃档案”对应一种识别能力——实时听写、非实时转写、目标说话人过滤、多人会议转写、声纹注册管理，以及样品-X Sample-X 准实时识别。

所有音频采集与推理都在本机 `127.0.0.1` 服务内完成，不需要云端 API Key，也不会把录音上传到任何服务器。界面基于开源项目 [RhineLabUI](https://github.com/LBEILC/RhineLabUI) 的三维终端改造，保留原入场动画、档案阵列、玻璃展开、360° 查看与检索收藏，仅将档案内容替换为语音功能。

## 功能总览

| 档案 | 功能 | 引擎 / 模型 | 适用输入 |
| --- | --- | --- | --- |
| X-001 | 非实时普通识别 | SenseVoice Small（ONNX INT8 / CPU） | 麦克风、电脑音频、WAV |
| X-002 | 非实时普通识别 | Fun-ASR-Nano（PyTorch / GPU 优先） | 麦克风、电脑音频、WAV |
| X-003 | 非实时普通识别 | Qwen3-ASR 1.7B（多语言，显存要求较高） | 麦克风、电脑音频、WAV |
| X-005 | 实时普通识别 | SenseVoice Small（约 0.8 秒分块重识别） | 麦克风、电脑音频、WAV |
| X-006 | 实时普通识别 | Fun-ASR-Nano（VAD 分段，预览可修订） | 麦克风、电脑音频、WAV |
| X-007 | 实时目标说话人识别 | FSMN-VAD + CAM++ + SenseVoice | 麦克风、电脑音频、WAV |
| X-008 | 多人会议转写 | MOSS-Transcribe-Diarize（说话人分离 + 时间戳） | 麦克风、电脑音频、WAV |
| X-009 | 声纹注册、查看与删除 | CAM++ 声纹特征 | 麦克风、电脑音频、WAV |
| X-010 | 实时普通识别（准实时） | 样品-X Sample-X v3.2.1 · MNN / 可选 CUDA | 浏览器麦克风、屏幕共享、WAV 重放 |

输入能力说明：

- **麦克风**：使用 Windows 默认输入设备，由后端采集，无需浏览器麦克风授权。
- **电脑音频**：通过 WASAPI 回环采集系统默认播放设备，无需开启“立体声混音”。
- **WAV 文件**：单文件最长 10 分钟、最大 64 MB（样品-X档案为 100 MB），自动重采样为 16 kHz 单声道。
- **实时采集上限**：麦克风 / 电脑音频单次最多 120 秒；声纹注册最多 30 秒。

识别结果可复制、导出 TXT；会议档案额外支持 JSON 与 SRT 导出，并保留分段说话人与时间戳。每次任务的录音、事件与统计会保存在本机输出目录，详见 [使用说明](docs/USAGE.md)。

## 快速开始

当前仓库面向 **Windows 10/11 + NVIDIA GPU（可选）**。日常使用只需要两步：

1. 双击 `初始化莱茵语音工作台.cmd`：创建 `latest_stage\.venv-asr`，安装依赖，下载 / 导出核心模型，并检查前端产物。
2. 双击 `启动莱茵语音工作台.cmd`：启动本机服务并自动打开 <http://127.0.0.1:8765/rhine/>。

首次启动会经过原版入场动画，按 `Enter` / `Esc` 或点击 `ENTER SYSTEM` 可跳过。初始化脚本也可以在 PowerShell 中执行：`powershell -ExecutionPolicy Bypass -File .\setup-workbench.ps1`。

完整的环境搭建（含 Qwen3-ASR、MOSS 会议模型、样品-X CUDA 环境的按需安装）、每个档案的操作步骤和常见问题，见 [docs/USAGE.md](docs/USAGE.md)。

启动器会优先使用 `latest_stage\.venv-asr`，也支持用 `SPEECH_PYTHON` 环境变量显式指定解释器。若修改了前端源码，进入 `RhineLabUI` 执行 `npm ci && npm run build` 重建 `dist`。

## 目录结构

```text
.
├─ 初始化莱茵语音工作台.cmd # 一键初始化入口
├─ 启动莱茵语音工作台.cmd   # 一键启动入口
├─ setup-workbench.ps1    # 初始化脚本
├─ prepare-github.ps1     # 导出干净 GitHub 发布目录
├─ RhineLabUI/             # 三维档案前端（TypeScript + Three.js + Vite）
│  ├─ src/speech.ts        # 档案右侧的识别操作面板
│  ├─ src/sample-x.ts        # 样品-X Sample-X 档案的浏览器采集与结果管理
│  ├─ src/speech-catalog.ts# 九份功能档案与模型的绑定
│  ├─ dist/                # 正式构建产物（随仓库提供，开箱可跑）
│  └─ .tools/Start-Speech.ps1
├─ latest_stage/           # Python 识别后端与模型调度（默认端口 8765）
│  ├─ src/web/             # HTTP 服务、任务互斥、声纹、会议与样品-X调度
│  ├─ src/models/          # SenseVoice ONNX、Fun-ASR-Nano、Qwen3-ASR、MOSS
│  ├─ src/pipeline/        # 实时 / 非实时 / 目标说话人流水线
│  ├─ scripts/             # 模型下载、导出、冒烟测试
│  └─ tests/               # 后端回归测试
└─ sample-x/                 # 样品-X Sample-X 独立推理服务（默认端口 8877）
   ├─ runtime/             # MNN 推理代码与本地权重（权重不入库）
   ├─ server.py            # 本机识别服务
   └─ pipeline.py          # 分段、VAD 与调度实现
```

## 技术栈

- **前端**：TypeScript、Three.js、Vite；界面与三维模型来自 [RhineLabUI](https://github.com/LBEILC/RhineLabUI)。
- **后端**：Python 标准库 `http.server`（无额外 Web 框架）、NumPy、FunASR、ONNX Runtime、ModelScope、soundfile、soxr、PyAudioWPatch。
- **模型**：SenseVoice Small、Fun-ASR-Nano、Qwen3-ASR 1.7B、FSMN-VAD、CAM++、MOSS-Transcribe-Diarize、Silero VAD、样品-X Sample-X。
- **推理设备**：优先 CUDA（NVIDIA GPU），不可用时自动回退 CPU；SenseVoice ONNX INT8 与声纹组件全程 CPU。

## 隐私与边界

- 服务仅监听 `127.0.0.1`，不提供公网访问、多用户隔离或鉴权，请勿直接暴露到局域网。
- 录音、识别文本、声纹特征保存在本机 `latest_stage/outputs/`、`sample-x/runs/` 等目录；声纹属于敏感数据，请勿提交到 Git 或分享。
- 目标说话人是**声纹验证**，不是声源分离，也不可用于身份认证；重叠讲话、相似声音与噪声仍可能误收或漏字。
- 会议转写的 S01/S02 等编号只在单次录音内有效，不能跨录音对应到同一个人。

## 版权与致谢

- 本仓库自有代码采用 MIT 许可证（与 RhineLabUI 一致）。
- 三维终端、配乐与界面基于 [RhineLabUI](https://github.com/LBEILC/RhineLabUI)；《明日方舟》及“莱茵生命”相关名称与美术元素版权归 **上海鹰角网络科技有限公司** 所有，本项目为非官方同人复刻，不得用于商业用途。
- 各识别模型与开源库的来源、许可证与再分发说明，统一整理在 [docs/OPEN-SOURCE.md](docs/OPEN-SOURCE.md)。其中样品-X Sample-X 转换权重**不允许再分发**，仓库已通过 `.gitignore` 排除。
