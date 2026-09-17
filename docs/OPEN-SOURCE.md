# 开源引用与模型来源

本项目站在多个开源项目与开放模型的肩膀上。以下按“识别模型 / 前端 / Python 依赖 / 字体与素材”分类列出，许可证以各上游项目发布为准。

## 1. 识别模型与语音组件

| 名称 | 来源 | 本项目用途 | 许可证 / 说明 |
| --- | --- | --- | --- |
| SenseVoice Small | [FunASR / ModelScope `iic/SenseVoiceSmall`](https://modelscope.cn/models/iic/SenseVoiceSmall) | X-001、X-005 实时与非实时普通识别；导出为 ONNX INT8 本地推理 | FunASR 生态模型；权重不入库 |
| Fun-ASR-Nano | [FunAudioLLM `Fun-ASR-Nano-2512`](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) | X-002、X-006 中英日及中文方言识别 | 读取 ModelScope 本地缓存；权重不入库 |
| Qwen3-ASR 1.7B | [Qwen `Qwen/Qwen3-ASR-1.7B`](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B) | X-003 非实时多语言识别 | 通义千问 ASR 模型；权重不入库 |
| FSMN-VAD | [ModelScope `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch) | 语音活动检测，服务于实时分段与声纹窗口 | FunASR 生态模型 |
| CAM++ | [ModelScope `iic/speech_campplus_sv_zh-cn_16k-common`](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common) | X-007 / X-009 目标说话人声纹注册与验证 | FunASR 生态模型 |
| MOSS-Transcribe-Diarize | [GitHub `OpenMOSS/MOSS-Transcribe-Diarize`](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) · [Hugging Face 模型](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) | X-008 多人会议转写与说话人分离 | 模型 Apache-2.0；约 2 GB，权重不入库，用脚本校验下载 |
| 样品-X Sample-X v3.2.1 | 本地转换权重（`sample-x/runtime/portable-models`） | X-010 准实时识别 | **权重不可再分发，已从 Git 排除**；推理代码为本项目自研 |
| Silero VAD | [GitHub `snakers4/silero-vad`](https://github.com/snakers4/silero-vad) | 样品-X档案的端点检测（`sample-x/silero_vad.onnx`） | MIT |
| 标点恢复模型 | [ModelScope `iic/punc_ct-transformer_cn-en-common-vocab471067-large`](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large) | 历史目标说话人流水线的标点恢复 | FunASR 生态模型 |

## 2. 前端与工具链

| 名称 | 来源 | 用途 | 许可证 |
| --- | --- | --- | --- |
| [RhineLabUI](https://github.com/LBEILC/RhineLabUI) | 同作者开源项目 | 三维档案界面、入场动画、玻璃档案模型与交互 | MIT（界面代码）；《明日方舟》元素为非官方同人使用 |
| [Three.js](https://github.com/mrdoob/three.js) | mrdoob 等 | 三维场景渲染 | MIT |
| [Vite](https://github.com/vitejs/vite) | Vue 团队 | 前端构建 | MIT |
| [TypeScript](https://github.com/microsoft/TypeScript) | Microsoft | 前端语言 | Apache-2.0 |
| [@kitlangton/rolling-number](https://github.com/kitlangton/rolling-number) | Kit Langton | 滚动数字 / 文字效果 | MIT |
| [Prettier](https://github.com/prettier/prettier) | Prettier 团队 | 代码格式化 | MIT |
| [Node.js](https://nodejs.org/) | OpenJS Foundation | 构建工具链运行时 | MIT（依据其 LICENSE 文件） |
| [Blender](https://www.blender.org/) | Blender Foundation | 档案三维模型制作 | GPL-3.0（仅用于创作，不随仓库分发软件本体） |

## 3. Python 依赖

| 名称 | 用途 | 许可证 |
| --- | --- | --- |
| [FunASR](https://github.com/modelscope/FunASR) | 语音识别框架与模型加载 | MIT |
| [ModelScope](https://github.com/modelscope/modelscope) | 模型下载与缓存 | Apache-2.0 |
| [NumPy](https://github.com/numpy/numpy) | 音频矩阵计算 | BSD-3-Clause |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | SenseVoice ONNX 推理 | MIT |
| [ONNX](https://github.com/onnx/onnx) | 模型导出与量化 | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | Fun-ASR-Nano、Qwen3-ASR、MOSS 推理 | BSD-3-Clause |
| [Transformers](https://github.com/huggingface/transformers) | MOSS 会议模型 | Apache-2.0 |
| [MNN](https://github.com/alibaba/MNN) | 样品-X Sample-X CPU 编码 | Apache-2.0 |
| [soundfile](https://github.com/bastibe/python-soundfile) | WAV 读写 | BSD-3-Clause |
| [soxr](https://github.com/dofuuz/python-soxr) | 流式重采样到 16 kHz | LGPL-2.1（动态链接） |
| [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) | Windows WASAPI 回环采集 | MIT |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) | 麦克风采集 | MIT |
| [PyYAML](https://github.com/yaml/pyyaml) | 配置解析 | MIT |
| [pydantic](https://github.com/pydantic/pydantic) | 配置模型 | MIT |
| [tqdm](https://github.com/tqdm/tqdm) | 进度条 | MIT |
| [zhconv](https://github.com/gumblex/zhconv) | 简繁转换 | Apache-2.0 |
| [cn2an](https://github.com/Ailln/cn2an) | 中文数字规范化 | MIT |
| [editdistance](https://github.com/aflc/editdistance) | CER 评估 | MIT |
| [pytest](https://github.com/pytest-dev/pytest) | 回归测试 | MIT |

## 4. 字体与素材

- **MiSans 字体**：版权所有 © 2020-2025 北京小米移动软件有限公司，依据其官方许可协议使用；webfont 分片来自 [misans-webfont](https://github.com/mobeicanyue/misans-webfont)。许可证文件见 `RhineLabUI/dist/fonts/MiSans-license.pdf` 与 `NOTICE.txt`。
- **原创配乐「观测室」**：本项目原创程序编配，随仓库 MIT 许可证授权；生成脚本见 `RhineLabUI/scripts/render-audio.mjs`。
- **逐字输入短音**：提取自《明日方舟》特别映像「莱茵生命：访问」，原音权利归原作者所有，不纳入 MIT 授权范围。
- **《明日方舟》相关名称与美术元素**：版权归上海鹰角网络科技有限公司所有，本项目为非官方同人复刻，不用于商业用途。

## 5. 不可再分发内容

以下内容已通过 `.gitignore` 排除，请勿提交到公开仓库：

- `sample-x/runtime/portable-models/`、`decoded/`、`cuda-graphs/`：样品-X Sample-X 转换权重与设备相关缓存。
- `latest_stage/models/`：MOSS、SenseVoice 等模型权重（体积大，且再分发前需逐一核对上游许可）。
- `latest_stage/outputs/`、`sample-x/runs/`：用户录音、识别结果与声纹特征。
- 各虚拟环境、日志、缓存与 `node_modules`。
