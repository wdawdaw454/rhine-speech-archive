const elements = {
  connection: document.querySelector("#connection"),
  model: document.querySelector("#model-select"),
  mode: document.querySelector("#mode-select"),
  recognitionType: document.querySelector("#recognition-type"),
  recognitionTypeField: document.querySelector("#recognition-type-field"),
  speakerPanel: document.querySelector("#speaker-panel"),
  targetModeHint: document.querySelector("#target-mode-hint"),
  loadModel: document.querySelector("#load-model-button"),
  modelReady: document.querySelector("#model-ready"),
  source: document.querySelector("#source-select"),
  wavField: document.querySelector("#wav-field"),
  wavFile: document.querySelector("#wav-file"),
  captureHint: document.querySelector("#capture-hint"),
  modeDescription: document.querySelector("#mode-description"),
  audioLevel: document.querySelector("#audio-level"),
  recordingPanel: document.querySelector("#recording-panel"),
  recordingAudio: document.querySelector("#recording-audio"),
  recordingDownload: document.querySelector("#recording-download"),
  modelDescription: document.querySelector("#model-description"),
  nanoHint: document.querySelector("#nano-hint"),
  realtimeTiming: document.querySelector("#realtime-timing"),
  meetingHint: document.querySelector("#meeting-hint"),
  meetingSegments: document.querySelector("#meeting-segments"),
  meetingJson: document.querySelector("#meeting-json"),
  meetingSrt: document.querySelector("#meeting-srt"),
  device: document.querySelector("#device-name"),
  stateOrb: document.querySelector("#state-orb"),
  stateText: document.querySelector("#state-text"),
  loadTiming: document.querySelector("#load-timing"),
  elapsed: document.querySelector("#elapsed"),
  transcript: document.querySelector("#transcript"),
  empty: document.querySelector("#empty-text"),
  committed: document.querySelector("#committed-text"),
  partial: document.querySelector("#partial-text"),
  error: document.querySelector("#error-message"),
  record: document.querySelector("#record-button"),
  recordLabel: document.querySelector("#record-label"),
  clear: document.querySelector("#clear-button"),
  copy: document.querySelector("#copy-button"),
  shutdown: document.querySelector("#shutdown-button"),
  toast: document.querySelector("#toast"),
};

let currentState = "idle";
let toastTimer;
let models = [];
let pendingAction = false;
let lastRecordingUrl = null;
let pollInFlight = false;
let latestStatus = null;
let lastMeetingSegments = "";
const inputDevices = {};
const speaker = Object.fromEntries(["target-load", "speaker-name", "enroll-mic", "enroll-file", "enroll-upload",
  "speaker-forget", "speaker-threshold", "speaker-summary", "speaker-live", "enroll-source-hint"].map(id => [id, document.getElementById(id)]));

function isTargetMode() {
  return elements.mode.value === "streaming" && elements.recognitionType.value === "target";
}

function isNanoRealtime() {
  return elements.mode.value === "streaming" && elements.model.value === "fun-asr-nano";
}

function isMeetingMode() {
  return elements.mode.value === "offline" && elements.recognitionType.value === "meeting";
}

function renderRecognitionTypes(preferred = elements.recognitionType.value) {
  const choices = elements.mode.value === "offline"
    ? [["normal", "普通识别"], ["meeting", "多人会议转写"]]
    : [["normal", "普通识别"], ["target", "目标说话人识别"]];
  elements.recognitionType.replaceChildren();
  for (const [value, label] of choices) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    elements.recognitionType.append(option);
  }
  elements.recognitionType.value = choices.some(([value]) => value === preferred) ? preferred : "normal";
}

function meetingSpeakerColor(speaker) {
  // Derive color from the ID, not row order: returning speakers keep their color.
  const match = /^S(\d{1,4})$/.exec(speaker);
  if (!match) return "var(--text)";
  const id = Number(match[1]);
  const palette = ["#56e0c2", "#82b7ff", "#f5ce76", "#f69fc7",
    "#b8a2ff", "#75d7ed", "#ffb18a", "#b7df83"];
  return palette[id - 1] || `hsl(${((id + 1) * 137.508 % 360).toFixed(3)} 72% 74%)`;
}

function renderMeetingSegments(status) {
  const segments = status.meeting_segments || [];
  const key = JSON.stringify(segments);
  if (key !== lastMeetingSegments) {
    lastMeetingSegments = key;
    elements.meetingSegments.replaceChildren();
    for (const segment of segments) {
      const row = document.createElement("article");
      row.className = "meeting-segment";
      row.style.setProperty("--speaker-color", meetingSpeakerColor(segment.speaker));
      const heading = document.createElement("div");
      heading.className = "meeting-segment-heading";
      const label = document.createElement("span");
      label.className = "meeting-speaker";
      label.textContent = segment.speaker;
      const timing = document.createElement("span");
      timing.textContent = `${formatElapsed(segment.start)} – ${formatElapsed(segment.end)}（${segment.start.toFixed(2)}–${segment.end.toFixed(2)} 秒）`;
      heading.append(label, timing);
      const content = document.createElement("p");
      content.textContent = segment.text;
      row.append(heading, content);
      elements.meetingSegments.append(row);
    }
  }
  elements.meetingSegments.hidden = !segments.length;
  elements.committed.hidden = Boolean(segments.length);
  for (const [kind, link] of [["json", elements.meetingJson], ["srt", elements.meetingSrt]]) {
    if (kind === "srt") link.textContent = "下载 SRT";
    const url = status.meeting_exports?.[kind];
    link.hidden = !url;
    if (url) link.href = url;
    else link.removeAttribute("href");
  }
}

function refreshControls() {
  if (latestStatus) render(latestStatus);
}

async function loadInputDevice() {
  const source = elements.source.value;
  if (source === "wav") return refreshControls();
  try {
    inputDevices[source] = await request(`/api/device?source=${source}`);
  } catch (error) {
    inputDevices[source] = {name: source === "system" ? "默认电脑音频输出" : "默认麦克风", error: error.message};
  }
  refreshControls();
}

function selectedModelAvailable() {
  return Boolean(models.find((item) => item.id === elements.model.value)?.available);
}

function renderModelDescription() {
  const model = models.find((item) => item.id === elements.model.value);
  elements.modelDescription.textContent = isTargetMode()
    ? "准实时目标识别 · SenseVoice ONNX INT8 / CPU · 每个已验证窗口合并重识别，预览可修订。"
    : model?.description || "";
}

function toast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => elements.toast.classList.remove("show"), 1800);
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}

function formatElapsed(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const remainder = String(total % 60).padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function formatSeconds(value) {
  return `${Number(value || 0).toFixed(2)}s`;
}

function renderLoadTiming(status) {
  const timing = status.load_timings || {};
  if (!Object.keys(timing).length) {
    elements.loadTiming.textContent = status.state === "loading" ? "正在记录分阶段加载耗时…" : "";
    return;
  }
  if (timing.cache_hit) {
    elements.loadTiming.textContent = `模型缓存命中 · 准备 ${formatSeconds(timing.backend_prepare_seconds)}`;
    return;
  }
  const parts = [`总计 ${formatSeconds(timing.backend_prepare_seconds)}`];
  if (timing.cpu_weight_load_seconds != null) {
    parts.push(`CPU 权重 ${formatSeconds(timing.cpu_weight_load_seconds)}`);
  }
  if (timing.device_transfer_seconds != null) {
    parts.push(`GPU 转移 ${formatSeconds(timing.device_transfer_seconds)}`);
  }
  if (timing.tokenizer_init_seconds != null) {
    parts.push(`分词器 ${formatSeconds(timing.tokenizer_init_seconds)}`);
  }
  elements.loadTiming.textContent = parts.join(" · ");
}

function render(status) {
  latestStatus = status;
  currentState = status.state;
  const active = ["loading", "starting", "listening", "stopping", "transcribing", "enrolling", "enroll_recording"].includes(status.state);
  const meetingTranscribing = status.recognition_type === "meeting" && status.mode === "offline" && status.state === "transcribing";
  const canToggle = meetingTranscribing || !["loading", "starting", "stopping", "transcribing", "enrolling", "enroll_recording"].includes(status.state);
  const hasText = Boolean(status.committed_text || status.partial_text);

  elements.connection.classList.add("online");
  elements.connection.lastElementChild.textContent = "本地服务已连接";
  elements.stateText.textContent = status.message;
  renderLoadTiming(status);
  elements.stateOrb.classList.toggle("active", status.state === "listening");
  elements.elapsed.textContent = formatElapsed(status.recording_seconds);
  elements.audioLevel.value = status.audio_level || 0;
  const transcribingSession = ["starting", "listening", "stopping", "transcribing"].includes(status.state);
  const sessionType = status.recognition_type || (status.target_only ? "target" : "normal");
  if (transcribingSession && status.mode && (elements.mode.value !== status.mode || elements.recognitionType.value !== sessionType)) {
    elements.mode.value = status.mode;
    renderRecognitionTypes(sessionType);
    renderModelOptions();
  }
  if (transcribingSession && status.model_id) elements.model.value = status.model_id;
  if ((transcribingSession || status.state === "enroll_recording") && status.source) elements.source.value = status.source;
  renderModelDescription();
  const targetEnabled = isTargetMode();
  const nanoRealtime = isNanoRealtime();
  elements.nanoHint.hidden = !nanoRealtime;
  elements.meetingHint.hidden = !isMeetingMode();
  elements.meetingHint.textContent = "MOSS 非实时多人会议：停止录音后整段分析，说话人为匿名编号，无需注册声纹；保留重叠时间戳。麦克风/电脑音频单次最多 120 秒，WAV 最多 10 分钟；重叠讲话可能漏字或分错说话人。";
  const metrics = status.realtime_metrics || {};
  const timingParts = [];
  if (metrics.backend_seconds != null) timingParts.push(`最近推理 ${formatSeconds(metrics.backend_seconds)}`);
  if (metrics.final_delay_seconds != null) timingParts.push(`最近句尾 ${formatSeconds(metrics.final_delay_seconds)}（至后端定稿，不含页面轮询）`);
  elements.realtimeTiming.hidden = !nanoRealtime || status.model_id !== "fun-asr-nano" || status.mode !== "streaming" || !timingParts.length;
  elements.realtimeTiming.textContent = timingParts.join(" · ");
  elements.recognitionTypeField.hidden = false;
  elements.speakerPanel.hidden = !targetEnabled;
  elements.targetModeHint.textContent = "FSMN-VAD + CAM++ + SenseVoice（准实时，全 CPU）。每约 1.5 秒验证一次声纹，通过后合并重识别；预览允许修订，句尾定稿，最长约 12 秒分段。首字约 2–2.5 秒为初步预算，非保证；不匹配或证据不足时结束上一段并清理上下文。短于 0.6 秒的片段可能跳过，不分离重叠声音。";
  speaker["speaker-live"].hidden = !targetEnabled;
  const targetReady = status.target_ready && status.speaker_profile;
  speaker["speaker-summary"].textContent = status.speaker_profile
    ? `${status.speaker_profile.name} · ${status.speaker_profile.speech_seconds}s 有效语音` : "未注册";
  speaker["target-load"].disabled = pendingAction || active || status.target_ready;
  speaker["target-load"].textContent = status.target_ready ? "声纹组件已就绪" : "加载声纹组件";
  for (const id of ["speaker-name", "enroll-file", "speaker-threshold"]) speaker[id].disabled = pendingAction || active;
  elements.recognitionType.disabled = pendingAction || active;
  speaker["enroll-mic"].disabled = pendingAction || (active && status.state !== "enroll_recording") || !status.target_ready
    || (elements.source.value === "wav" && status.state !== "enroll_recording");
  speaker["enroll-mic"].textContent = status.state === "enroll_recording" ? "停止并注册" : "录音注册";
  speaker["enroll-source-hint"].textContent = elements.source.value === "system"
    ? "录音注册来源：电脑音频（默认耳机／扬声器），不会打开麦克风。点击后仅播放目标说话人的声音，避免其他人声、音乐和系统提示音。最多 30 秒，可提前停止并注册。"
    : elements.source.value === "wav"
      ? "当前输入来源为 WAV 文件：请在下方“注册 WAV”选择 3–30 秒音频，然后点击“上传注册”；不会打开录音设备。"
      : "录音注册来源：默认麦克风，与上方“输入来源”一致。点击后请目标说话人单独讲话，最多 30 秒，可提前停止并注册。";
  speaker["enroll-upload"].disabled = pendingAction || active || !status.target_ready || !speaker["enroll-file"].files[0];
  speaker["speaker-forget"].disabled = pendingAction || active || !status.speaker_profile;
  const labels = {target: "匹配目标", non_target: "其他声音，已跳过", pending: "等待足够语音"};
  speaker["speaker-live"].textContent = targetEnabled
    ? (!targetReady ? "目标模式：请加载声纹组件并注册声音" : `目标：${status.speaker_profile.name} · ${labels[status.speaker_decision] || "等待验证"}${status.speaker_score == null ? "" : ` · 相似度 ${status.speaker_score.toFixed(3)}（非概率）`}`)
    : "普通转写：不筛选说话人";
  const ready = status.loaded_model_id === elements.model.value;
  const fileInput = elements.source.value === "wav";
  const systemInput = elements.source.value === "system";
  const inputDevice = inputDevices[elements.source.value];
  elements.device.textContent = fileInput ? "WAV 文件输入" : systemInput && active && status.capture_device
    ? status.capture_device : inputDevice?.name || (systemInput ? "默认电脑音频输出" : "默认麦克风");
  const selectedFile = elements.wavFile.files[0];
  elements.wavField.hidden = !fileInput;
  elements.captureHint.textContent = fileInput
    ? (nanoRealtime ? "先加载模型，再上传单个 WAV（最多 64 MB / 10 分钟）。上传完成后按真实时间输入模型（不自动播放声音），可停止回放并等待已输入音频定稿；不会打开麦克风。"
      : "先加载模型，再选择单个 WAV 并开始转写。最多 64 MB / 10 分钟；流式模式按音频分块处理，不等待实时播放。")
    : systemInput
      ? `通过 WASAPI 采集当前默认耳机／扬声器的混合声音，不打开麦克风。开始采集后播放浏览器或软件音频，每次最多 120 秒；切换输出设备后需停止再开始。系统提示音也会录入，请仅采集有权使用的音频。${inputDevice?.error ? ` 设备检查：${inputDevice.error}` : ""}`
      : "先加载模型，再开始录音。由本地服务采集默认麦克风，等到正在聆听或正在录音再说话，每次最多 120 秒。";
  elements.modelReady.textContent = status.state === "loading" ? "正在加载…" : ready ? "当前模型已就绪" : "当前模型未加载";
  elements.modelReady.classList.toggle("ready", ready);
  elements.loadModel.textContent = status.state === "loading" ? "正在加载…" : ready ? "模型已加载" : "加载模型";
  elements.loadModel.disabled = pendingAction || active || ready || !selectedModelAvailable();
  if (status.recording_url !== lastRecordingUrl) {
    lastRecordingUrl = status.recording_url || null;
    elements.recordingPanel.hidden = !lastRecordingUrl;
    if (lastRecordingUrl) {
      elements.recordingAudio.src = lastRecordingUrl;
      elements.recordingDownload.href = lastRecordingUrl;
    } else {
      elements.recordingAudio.pause();
      elements.recordingAudio.removeAttribute("src");
    }
  }
  elements.committed.textContent = status.committed_text;
  elements.partial.textContent = status.partial_text;
  renderMeetingSegments(status);
  elements.empty.hidden = hasText;
  if (!hasText) {
    elements.empty.textContent =
      status.state === "loading" ? "正在加载模型，此操作不会打开麦克风。" :
      status.state === "starting" ? "正在准备音频输入…" :
      status.state === "enroll_recording" ? (status.source === "system"
        ? "正在采集电脑音频注册：请仅播放目标说话人的声音，完成后点击“停止并注册”。"
        : "麦克风注册录音中：请目标说话人单独讲话，完成后点击“停止并注册”。") :
      status.state === "enrolling" ? "正在提取声纹，此步骤不生成转写文本。" :
      status.state === "listening" ? (status.source === "wav" ? "正在实时输入 WAV，等待 VAD 检测语音并生成预览…" : status.source === "system" ? (status.mode === "offline" ? "正在采集电脑音频，点击停止后生成转写文本。" : "正在采集电脑音频，请在默认输出设备播放需要识别的声音。") : status.mode === "offline" ? "正在录音，点击停止后生成转写文本。" : "正在聆听，请开始说话。") :
      status.state === "transcribing" ? (meetingTranscribing ? "正在整段生成会议转写及说话人时间戳，可取消；取消后需重新加载 MOSS。" : status.source === "wav" ? `正在转写：${status.filename}` : "录音已结束，模型正在生成文本…") :
      status.state === "stopping" ? "正在结束录音，请稍候…" :
      !ready ? "请先点击“加载模型”，加载完成后再开始。" :
      targetEnabled && !targetReady ? "请加载声纹组件并注册目标声音，然后开始识别。" :
      fileInput ? "选择单个 WAV 文件，然后点击“开始转写”。" : systemInput ? "模型已就绪，点击“开始采集电脑音频”，然后播放需要转写的声音。" : "模型已就绪，点击开始录音后说话。";
  }
  elements.error.textContent = status.error || "";
  elements.model.disabled = active || pendingAction;
  elements.mode.disabled = active || pendingAction;
  elements.source.disabled = active || pendingAction;
  elements.wavFile.disabled = active || pendingAction;
  elements.clear.disabled = active || pendingAction;
  elements.record.disabled = pendingAction || !canToggle || (!active && (!ready || (targetEnabled && !targetReady) || (fileInput && !selectedFile)));
  elements.record.classList.toggle("stop", status.state === "listening" || meetingTranscribing);
  elements.recordLabel.textContent =
    pendingAction && fileInput ? "正在提交…" :
    status.state === "starting" ? "正在准备…" :
    status.state === "transcribing" ? (meetingTranscribing ? "取消会议转写" : "正在转写…") :
    status.state === "stopping" ? "正在停止…" :
    status.state === "listening" ? (status.source === "wav" ? "停止回放并定稿" : status.source === "system" ? "停止采集并定稿" : "停止录音") : fileInput ? (nanoRealtime ? "开始实时转写" : "开始转写") : systemInput ? "开始采集电脑音频" : "开始录音";
  elements.transcript?.scrollTo?.({ top: elements.transcript.scrollHeight, behavior: "smooth" });
}

async function loadModels() {
  ({ models } = await request("/api/models"));
  renderModelOptions();
  refreshControls();
}

function renderModelOptions() {
  const previous = elements.model.value;
  renderRecognitionTypes();
  const recognitionType = elements.recognitionType.value;
  const compatible = models.filter((model) => model.modes.includes(elements.mode.value)
    && model.recognition_types.includes(recognitionType));
  elements.model.replaceChildren();
  for (const model of compatible) {
    const option = document.createElement("option");
    option.value = model.id;
    const label = isTargetMode() ? model.target_name || `FSMN-VAD + CAM++ + ${model.name}` : model.name;
    option.textContent = model.available ? label : `${label}（文件缺失）`;
    option.disabled = !model.available;
    elements.model.append(option);
  }
  const chosen = compatible.find((model) => model.id === previous && model.available) || compatible.find((model) => model.available);
  if (chosen) elements.model.value = chosen.id;
  elements.modeDescription.textContent = elements.mode.value === "offline"
    ? "先采集麦克风或电脑音频，停止后转写；也支持 WAV 文件和多人会议转写。" : "边说或边播放电脑音频，边显示文本；可选择普通识别或目标说话人识别。";
  renderModelDescription();
}

elements.mode.addEventListener("change", () => { renderModelOptions(); refreshControls(); });
elements.recognitionType.addEventListener("change", () => { renderModelOptions(); refreshControls(); });
elements.source.addEventListener("change", () => { refreshControls(); loadInputDevice(); });
elements.wavFile.addEventListener("change", () => {
  const file = elements.wavFile.files[0];
  if (file && (!file.name.toLowerCase().endsWith(".wav") || !file.size || file.size > 64 * 1024 * 1024)) {
    toast("请选择非空 WAV 文件，大小不能超过 64 MB");
    elements.wavFile.value = "";
  }
  refreshControls();
});

elements.model.addEventListener("change", () => {
  renderModelDescription();
  refreshControls();
});

async function poll() {
  if (pollInFlight || pendingAction) return;
  pollInFlight = true;
  try {
    render(await request("/api/status"));
  } catch (_) {
    elements.connection.classList.remove("online");
    elements.connection.lastElementChild.textContent = "本地服务未连接";
    elements.record.disabled = true;
    elements.loadModel.disabled = true;
  } finally {
    pollInFlight = false;
  }
}

elements.loadModel.addEventListener("click", async () => {
  if (pendingAction) return;
  pendingAction = true;
  refreshControls();
  try {
    render(await request("/api/load-model", {method: "POST", body: JSON.stringify({
      model_id: elements.model.value, mode: elements.mode.value,
    })}));
  } catch (error) {
    toast(error.message);
  } finally {
    pendingAction = false;
    refreshControls();
    poll();
  }
});

elements.record.addEventListener("click", async () => {
  if (pendingAction) return;
  pendingAction = true;
  refreshControls();
  try {
    if (currentState === "listening" || (currentState === "transcribing" && latestStatus?.recognition_type === "meeting")) {
      render(await request("/api/stop", { method: "POST", body: "{}" }));
    } else if (elements.source.value === "wav") {
      const file = elements.wavFile.files[0];
      if (!file) throw new Error("请先选择 WAV 文件");
      const query = new URLSearchParams({model_id: elements.model.value, mode: elements.mode.value, filename: file.name,
        recognition_type: elements.recognitionType.value,
        target_only: isTargetMode(), speaker_threshold: speaker["speaker-threshold"].value});
      render(await request(`/api/transcribe-file?${query}`, {
        method: "POST", headers: {"Content-Type": "audio/wav"}, body: file,
      }));
    } else {
      // Do not accidentally capture this page's previous recording playback.
      elements.recordingAudio.pause();
      render(await request("/api/start", {
        method: "POST",
        body: JSON.stringify({ model_id: elements.model.value, mode: elements.mode.value,
          recognition_type: elements.recognitionType.value,
          source: elements.source.value,
          target_only: isTargetMode(), speaker_threshold: Number(speaker["speaker-threshold"].value) }),
      }));
    }
  } catch (error) {
    toast(error.message);
  } finally {
    pendingAction = false;
    refreshControls();
    poll();
  }
});

elements.clear.addEventListener("click", async () => {
  try {
    render(await request("/api/clear", { method: "POST", body: "{}" }));
  } catch (error) {
    toast(error.message);
  }
});

elements.copy.addEventListener("click", async () => {
  const text = [elements.committed.textContent, elements.partial.textContent]
    .filter(Boolean).join("\n");
  if (!text) return toast("暂无可复制文本");
  try {
    await navigator.clipboard.writeText(text);
    toast("文本已复制");
  } catch (_) {
    toast("浏览器未允许复制");
  }
});

elements.shutdown.addEventListener("click", async () => {
  if (!window.confirm("关闭本地转写服务？")) return;
  try {
    await request("/api/shutdown", { method: "POST", body: "{}" });
    toast("本地服务已关闭");
  } catch (error) {
    toast(error.message);
  }
});

async function speakerAction(path, options = {}) {
  if (pendingAction) return;
  pendingAction = true;
  refreshControls();
  try { render(await request(path, {method: "POST", body: "{}", ...options})); }
  catch (error) { toast(error.message); }
  finally { pendingAction = false; refreshControls(); poll(); }
}
speaker["enroll-file"].addEventListener("change", refreshControls);
speaker["target-load"].addEventListener("click", () => speakerAction("/api/target/load"));
speaker["enroll-mic"].addEventListener("click", () => {
  if (currentState === "enroll_recording") return speakerAction("/api/stop");
  if (elements.source.value === "wav") return toast("WAV 输入请使用下方“注册 WAV”和“上传注册”");
  if (latestStatus?.speaker_profile && !window.confirm("重新注册将替换当前声纹，继续？")) return;
  elements.recordingAudio.pause();
  speakerAction("/api/target/enroll-recording", {body: JSON.stringify({
    name: speaker["speaker-name"].value, source: elements.source.value,
  })});
});
speaker["enroll-upload"].addEventListener("click", () => {
  const file = speaker["enroll-file"].files[0];
  if (!file || !file.name.toLowerCase().endsWith(".wav") || file.size > 64 * 1024 * 1024) return toast("请选择 3–30 秒 WAV，大小不超过 64 MB");
  if (latestStatus?.speaker_profile && !window.confirm("重新注册将替换当前声纹，继续？")) return;
  speakerAction(`/api/target/enroll-file?${new URLSearchParams({name: speaker["speaker-name"].value})}`,
    {headers: {"Content-Type": "audio/wav"}, body: file});
});
speaker["speaker-forget"].addEventListener("click", () => {
  if (window.confirm("永久删除本机保存的目标声纹？之后需要重新注册。")) speakerAction("/api/target/forget");
});

Promise.all([
  loadModels(),
  loadInputDevice(),
  poll(),
]).catch((error) => toast(error.message));

setInterval(poll, 350);
