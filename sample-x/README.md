# Sample-X · 样品-X 本地识别服务

样品-X 是语音工作台 X-010 档案使用的**可选**本地准实时识别引擎。主后端（`latest_stage`，端口 8765）会按需启动本目录中的独立服务（端口 8877），两者只监听 `127.0.0.1`。

> **模型不随仓库分发。** `runtime/portable-models/` 中的推理权重体积较大且许可不支持再分发，已被 `.gitignore` 排除。克隆仓库后，X-010 档案在补齐模型前无法加载；其余档案（SenseVoice、Fun-ASR-Nano、Qwen3-ASR、MOSS、声纹）不受影响。

## 准备模型

除下列权重外，还需从本地模型包恢复 `runtime/decoded/asr/` 中的 tokenizer 与特征配置。目录结构固定，文件名区分大小写：

```text
portable-models/
├─ embedding.npy
├─ no_stream/
│  ├─ decoder_full.mnn
│  └─ decoder_full.mnn.weight
└─ stream/
   ├─ encoder.mnn
   ├─ encoder.mnn.weight
   ├─ logit.mnn
   └─ logit.mnn.weight
decoded/asr/
├─ feature_extractor/
│  ├─ cmvn.txt
│  └─ fbank.conf
└─ token/
   ├─ add_special_tokens.txt
   ├─ merges.txt
   └─ vocab.txt
```

如需 CUDA 加速，还要在 `runtime/cuda-graphs/` 放入 `decoder.json` 与 `logit.json` 架构元数据；缺失时自动使用 CPU（MNN）路径。

## 安装环境

```powershell
# CPU / MNN 环境（Python 3.11）
powershell -ExecutionPolicy Bypass -File setup.ps1 -Python C:\Path\To\Python311\python.exe

# 可选：CUDA FP32 解码 / 输出层环境
powershell -ExecutionPolicy Bypass -File setup-cuda.ps1 -Python <Python3.11路径>
```

两个脚本分别创建 `.venv/` 与 `.venv-cuda/`，互不影响。启动时优先使用 `.venv-cuda`（校验通过才启用），否则回退 `.venv`；也可用环境变量 `SAMPLEX_PYTHON` 指定解释器。

## 行为与限制

- 浏览器 `getUserMedia → AudioContext(16k) → AudioWorklet` 采集，每 100ms 发送 PCM16；电脑音频使用浏览器共享，WAV 按真实时长重放。
- Silero VAD 6.2.1 做端点检测，完整分段最长 20 秒；草稿可修订，定稿不改写，不做短句拼接。
- 自动模式优先 CUDA FP32 解码 / 输出层 + MNN CPU 编码，异常时回退 CPU 重算同段音频；面板显示实际后端、设备与回退原因。
- KV 缓存只在单次完整音频推理内复用，**不是跨音频块的流式解码**；CPU 长句可能数秒才更新。
- 每次会话的 WAV、事件 JSONL 与统计保存在 `runs/<会话编号>/`，该目录包含真实录音，已从 Git 排除。
- 与其他档案共用互斥锁；关闭页面会中断会话并标记未完成。

## 验证

```powershell
& .venv\Scripts\python.exe test_backend.py
# 补齐 runtime/decoded/asr 后运行；未准备模型资产时会跳过
& .venv\Scripts\python.exe runtime\test_tokenizer.py

# 主后端集成测试（在 latest_stage 目录）
python -m pytest -q tests\test_samplex_service.py
```
