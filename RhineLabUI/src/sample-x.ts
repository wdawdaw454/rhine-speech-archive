/** A-flow: browser DSP -> 16k PCM16/100ms -> original Silero/prefix pipeline. */
type Run = {
  ws?: WebSocket; stream?: MediaStream; ctx?: AudioContext; node?: AudioWorkletNode;
  sourceNode?: MediaStreamAudioSourceNode; timer?: ReturnType<typeof setTimeout>; timeout?: ReturnType<typeof setTimeout>;
  ready: boolean; stopping: boolean; finished: boolean; settled: boolean; sent: number;
  audio?: Int16Array; position: number; finals: Map<number, string>; partial: string;
  replayStarted?: number;
};
type Runtime = { backend?: string; backend_id?: string; device?: string; fallback_reason?: string };
interface ManagedModel { id: string; name: string; installed: boolean }
type Health = Runtime & { ready: boolean; loading: boolean; busy: boolean; message?: string };
const service = 'http://127.0.0.1:8877';

export async function decodeSampleXWav(bytes: ArrayBuffer): Promise<Int16Array> {
  const ascii = (start: number, end: number) => new TextDecoder('ascii').decode(bytes.slice(start, end));
  if (ascii(0, 4) !== 'RIFF' || ascii(8, 12) !== 'WAVE') throw Error('请选择标准 WAV 音频');
  const view = new DataView(bytes); let format: number[] | undefined, payload: ArrayBuffer | undefined;
  for (let pos = 12; pos + 8 <= bytes.byteLength;) {
    const tag = ascii(pos, pos + 4), size = view.getUint32(pos + 4, true), end = pos + 8 + size;
    if (end > bytes.byteLength) throw Error('WAV 文件不完整');
    if (tag === 'fmt ' && size >= 16) format = [view.getUint16(pos + 8, true), view.getUint16(pos + 10, true), view.getUint32(pos + 12, true), view.getUint16(pos + 22, true)];
    if (tag === 'data') payload = bytes.slice(pos + 8, end);
    pos = end + size % 2;
  }
  if (format?.join('/') === '1/1/16000/16' && payload) {
    if (!payload.byteLength || payload.byteLength % 2) throw Error('WAV 没有完整音频采样');
    return new Int16Array(payload);
  }
  const ctx = new AudioContext({ sampleRate: 16000 });
  try {
    const audio = await ctx.decodeAudioData(bytes.slice(0));
    if (!audio.length || audio.sampleRate !== 16000) throw Error('无法转换为 16kHz 音频');
    const out = new Int16Array(audio.length);
    for (let i = 0; i < out.length; i++) {
      let x = 0; for (let c = 0; c < audio.numberOfChannels; c++) x += audio.getChannelData(c)[i] / audio.numberOfChannels;
      out[i] = Math.round(Math.max(-1, Math.min(1, x)) * 32767);
    }
    return out;
  } finally { await ctx.close(); }
}

export class SampleXArchiveControls {
  readonly root = document.createElement('section');
  private active?: Run;
  private health: Health = { ready: false, loading: false, busy: false };
  private otherBusy = false;
  private connected = false;
  private chosenAudio?: Int16Array;
  private chosenName = '';
  private fileGeneration = 0;
  private reading = false;
  private loading = false;
  private opened = false;
  private modelInstalled = true;
  private modelStatusChecked = false;
  private modelStatusRefreshedAt = 0;
  constructor(private mute: (muted: boolean) => void, private notify: (message: string) => void, private manageModels: () => void) {
    this.root.id = 'sample-x-archive'; this.root.className = 'speech-archive';
    this.root.setAttribute('aria-label', '样品-X A 原前端流程');
    this.root.innerHTML = `
      <div class="speech-tabs" role="tablist" aria-label="样品-X功能详情">
        <button id="sample-x-tab-controls" role="tab" aria-selected="true" aria-controls="sample-x-controls" data-panel="controls">01 <span>使用功能</span></button>
        <button id="sample-x-tab-results" role="tab" aria-selected="false" aria-controls="sample-x-results" data-panel="results">02 <span>识别结果</span></button>
        <button id="sample-x-tab-info" role="tab" aria-selected="false" aria-controls="sample-x-info" data-panel="info">03 <span>功能说明</span></button>
      </div>
      <div class="speech-status-line"><span id="sample-x-state" role="status">请先加载样品-X引擎</span></div>
      <p id="sample-x-error" class="speech-error" role="alert" hidden></p>
      <p id="sample-x-fallback" class="speech-note" role="status" hidden></p>
      <section id="sample-x-controls" role="tabpanel" aria-labelledby="sample-x-tab-controls">
        <label class="speech-field"><span>推理后端（下次开始生效）</span><select id="sample-x-backend"><option value="auto" selected>自动 · CUDA 优先，异常回退 CPU</option><option value="cpu">仅 CPU</option></select></label>
        <label class="speech-field"><span>INPUT / 音频来源</span><select id="sample-x-source"><option value="microphone">麦克风</option><option value="computer">电脑音频</option><option value="wav">WAV 文件</option></select></label>
        <label class="speech-field" id="sample-x-device-field"><span>浏览器麦克风</span><select id="sample-x-device"><option value="">系统默认麦克风</option></select></label>
        <label class="speech-field"><span>草稿请求间隔</span><select id="sample-x-interval"><option value="0.5">0.5 秒</option><option value="0.8" selected>0.8 秒（默认）</option><option value="1">1 秒</option><option value="1.28">1.28 秒</option><option value="2">2 秒</option></select></label>
        <div id="sample-x-file-field" hidden><label class="speech-file"><span>选择 WAV 文件 ↗</span><input id="sample-x-file" type="file" accept=".wav,audio/wav" aria-label="选择样品-X WAV 文件"/><small id="sample-x-file-name">未选择文件 · 最大 100 MB</small></label><button id="sample-x-sample" class="speech-secondary">使用示例录音 ↗</button></div>
        <p id="sample-x-capture-note" class="speech-note"></p>
        <p id="sample-x-runtime" class="speech-note">Sample-X_v3.2.1 · 后端尚未加载</p>
        <p id="sample-x-model-note" class="speech-note"></p>
        <div class="speech-engine-actions"><button id="sample-x-open-models" class="speech-secondary" hidden>前往模型管理 ↗</button><button id="sample-x-load" class="speech-secondary">加载样品-X引擎 ↗</button></div>
      </section>
      <section id="sample-x-results" role="tabpanel" aria-labelledby="sample-x-tab-results" hidden>
        <p id="sample-x-meter" class="speech-note">本档案尚未开始转写</p>
        <progress id="sample-x-progress" max="1" value="0" hidden aria-label="WAV 重放进度"></progress>
        <div id="sample-x-transcript" class="speech-transcript" tabindex="0" aria-label="样品-X转写文本"><span id="sample-x-final"></span><span id="sample-x-partial"></span></div>
        <p id="sample-x-metrics" class="speech-note"></p>
        <div class="speech-output-actions"><button id="sample-x-copy" disabled>复制文本 ↗</button><button id="sample-x-export" disabled>导出 TXT ↓</button><a id="sample-x-report" hidden download>识别记录 JSON ↓</a></div>
        <div id="sample-x-recording" class="speech-recording" hidden><audio id="sample-x-audio" controls preload="none" aria-label="样品-X 本次录音"></audio><a id="sample-x-download" download>保存录音 ↓</a></div>
      </section>
      <section id="sample-x-info" role="tabpanel" aria-labelledby="sample-x-tab-info" hidden>
        <p class="speech-note">沿用 A 的浏览器降噪、回声消除、自动增益和 16kHz PCM16 / 100ms 采集。准确率优先：恢复原停顿规则与最多 20 秒的完整分段，不再提前切短句、不重叠拼接或改写定稿。</p>
        <p class="speech-note">草稿间隔是最低请求间隔，不是固定出字承诺；繁忙时根据实际推理耗时降低草稿频率，只保留最新待处理草稿，完整定稿优先。已完成草稿立即显示。长文本默认跟随最新结果，向上滚动阅读时不强制拉回。</p>
        <p class="speech-note">本地离线样品-X模型通过累积音频重复推理更新草稿，不是原厂流式解码；草稿会修订。仅移植 A，不包含 B、云端 API 或自动输入其他软件。</p>
        <p class="speech-note">自动模式优先使用 CUDA FP32 解码器／输出层及每段独立 KV 缓存，编码器仍为 MNN CPU。CUDA 加载或推理失败时改用 CPU，重算同一段音频，不跳过定稿。回退后保持 CPU，切换到“仅 CPU”运行一次再切回自动可重试 CUDA。实际后端及回退原因会显示在本面板和识别记录中。</p>
        <p class="speech-note">录音及记录保存在本项目 sample-x/runs。切换档案不停止采集；关闭页面会中断并标记未完成。模型独立运行，不改动原 A/B 服务。</p>
        <details class="speech-diagnostics"><summary>运行信息</summary><pre id="sample-x-diagnostics"></pre></details>
      </section>
      <div class="speech-control-footer"><button id="sample-x-start" class="speech-primary" disabled>开始听写 →</button><p class="speech-note">停顿自动定稿，继续聆听；结束时保留不足 100ms 的尾包。</p></div>
      <div id="sample-x-connection" class="speech-connection">LOCAL / 尚未加载</div>`;
    this.root.addEventListener('click', e => { const tab = (e.target as Element).closest<HTMLElement>('[data-panel]'); if (tab) this.panel(tab.dataset.panel!); });
    this.root.addEventListener('keydown', e => {
      const tab = (e.target as Element).closest<HTMLElement>('[data-panel]');
      if (!tab || !['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
      e.preventDefault(); e.stopPropagation();
      const names = ['controls', 'results', 'info'], name = names[(names.indexOf(tab.dataset.panel!) + (e.key === 'ArrowRight' ? 1 : 2)) % 3];
      this.panel(name); this.el('tab-' + name).focus();
    });
    this.el('source').addEventListener('change', () => this.render());
    this.el('load').addEventListener('click', () => void this.load());
    this.el('open-models').addEventListener('click', () => this.manageModels());
    this.el('start').addEventListener('click', () => this.active ? this.stop(this.active) : void this.begin());
    this.el('file').addEventListener('change', () => void this.file());
    this.el('sample').addEventListener('click', () => void this.file(true));
    this.el('copy').addEventListener('click', () => { void navigator.clipboard.writeText(this.text()).then(() => this.notify('文本已复制'), () => this.error('请选中文本手动复制')); });
    this.el('export').addEventListener('click', () => {
      const url = URL.createObjectURL(new Blob([this.text()], { type: 'text/plain;charset=utf-8' }));
      const link = document.createElement('a'); link.href = url; link.download = '样品-X-A-转写.txt'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    window.addEventListener('beforeunload', e => { if (this.active) { e.preventDefault(); e.returnValue = ''; } });
    window.addEventListener('pagehide', () => { if (this.active) this.cleanup(this.active); });
    this.panel('controls'); this.render();
  }
  open(destination: HTMLElement) {
    destination.append(this.root);
    if (!this.opened) { this.opened = true; void this.devices(); void this.poll(); }
  }
  private el<T extends HTMLElement = HTMLElement>(name: string) { return this.root.querySelector<T>('#sample-x-' + name)!; }
  private value(name: string) { return this.el<HTMLSelectElement>(name).value; }
  private text() { return [this.el('final').textContent, this.el('partial').textContent].filter(Boolean).join('\n'); }
  private message(text: string) { this.el('state').textContent = text; }
  private error(error: unknown) { this.el('error').hidden = false; this.el('error').textContent = error instanceof Error ? error.message : String(error); }
  private panel(name: string) {
    for (const key of ['controls', 'results', 'info']) {
      this.el(key).hidden = key !== name; this.el('tab-' + key).setAttribute('aria-selected', String(key === name)); this.el('tab-' + key).tabIndex = key === name ? 0 : -1;
    }
  }
  private async devices() {
    try {
      const select = this.el<HTMLSelectElement>('device'), previous = select.value;
      const devices = await navigator.mediaDevices.enumerateDevices();
      select.replaceChildren(new Option('系统默认麦克风', ''));
      for (const d of devices) if (d.kind === 'audioinput' && d.deviceId) select.add(new Option(d.label || '麦克风（授权后显示名称）', d.deviceId));
      if ([...select.options].some(o => o.value === previous)) select.value = previous;
    } catch { /* The default remains available until permission is granted. */ }
  }
  private async request<T>(path: string, post = false): Promise<T> {
    const response = await fetch(path, { cache: 'no-store', signal: AbortSignal.timeout(5000), ...(post ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' } : {}) });
    const data = await response.json(); if (!response.ok) throw Error(data.error || '本地服务请求失败'); return data;
  }
  private async load() {
    this.loading = true; this.el('error').hidden = true; this.render();
    try { this.health = await this.request<Health>('/api/sample-x/load', true); this.message(this.health.message || '正在加载…'); }
    catch (e) { this.error(e); } finally { this.loading = false; this.render(); }
  }
  private async poll() {
    try {
      const [health, other] = await Promise.all([this.request<Health>('/api/sample-x/status'), this.request<{ state: string }>('/api/status')]);
      this.health = health; this.otherBusy = ['loading', 'starting', 'listening', 'stopping', 'transcribing', 'enrolling', 'enroll_recording'].includes(other.state); this.connected = true;
      await this.refreshModelStatus();
      if (!this.active && !this.text()) this.message(health.message || (health.ready ? '样品-X引擎已就绪' : this.modelStatusChecked && !this.modelInstalled ? '样品-X模型尚未准备完成' : '请先加载样品-X引擎'));
    } catch { this.connected = false; }
    this.render(); setTimeout(() => void this.poll(), 1200);
  }
  private async refreshModelStatus() {
    if (this.health.ready || Date.now() - this.modelStatusRefreshedAt < 5000) return;
    this.modelStatusRefreshedAt = Date.now();
    try {
      const status = await this.request<{ models: ManagedModel[] }>('/api/model-manager');
      const model = status.models.find(item => item.id === 'sample-x');
      if (model) { this.modelInstalled = model.installed; this.modelStatusChecked = true; }
    } catch { /* Engine status remains useful when the manager API is briefly unavailable. */ }
  }
  private render() {
    const source = this.value('source'), running = !!this.active, blocked = running || this.reading;
    for (const id of ['backend', 'source', 'device', 'interval', 'file', 'sample']) this.el<HTMLButtonElement>(id).disabled = blocked;
    this.el<HTMLButtonElement>('sample').disabled = blocked || !this.health.ready;
    this.el('file-field').hidden = source !== 'wav'; this.el('device-field').hidden = source !== 'microphone';
    this.el('capture-note').textContent = source === 'wav' ? '按录音时长重放，不播放声音；兼容的 PCM16/16kHz/单声道 WAV 保持采样不变。' : source === 'computer' ? '在浏览器弹窗中勾选共享音频；未选择音频时不会开始识别。' : 'A 默认开启浏览器回声消除、降噪、自动增益；仅在点击开始后申请录音权限。';
    this.el<HTMLButtonElement>('load').disabled = running || this.loading || this.health.loading || this.health.ready || !this.connected || this.otherBusy;
    this.el('load').textContent = this.loading || this.health.loading ? '正在加载样品-X…' : this.health.ready ? '样品-X引擎已就绪 ✓' : '加载样品-X引擎 ↗';
    this.el<HTMLButtonElement>('start').disabled = running ? !!this.active?.stopping : !this.connected || !this.health.ready || this.health.busy || this.otherBusy || this.reading || (source === 'wav' && !this.chosenAudio);
    this.el('start').textContent = running ? this.active?.stopping ? '正在保留尾音并定稿…' : this.active?.ready ? '结束并定稿 ■' : '取消准备 ■' : this.otherBusy || this.health.busy ? '另一任务正在运行…' : source === 'wav' ? '开始文件识别 →' : '开始听写 →';
    this.el('start').classList.toggle('is-recording', running);
    const backend = this.health.backend_id === 'cuda-hybrid' ? 'CUDA + CPU' : this.health.backend_id === 'cpu' ? 'CPU' : '待确认';
    this.el('connection').textContent = !this.connected ? '本地后端未连接 · 请使用启动入口' : this.health.ready ? `LOCAL ${backend} / 样品-X A · 已连接` : 'LOCAL / 样品-X A · 尚未就绪';
    this.el('runtime').textContent = `Sample-X_v3.2.1 · ${this.health.backend || '后端尚未加载'}${this.health.device ? ' · ' + this.health.device : ''}`;
    const modelMissing = this.modelStatusChecked && !this.modelInstalled && !this.health.ready;
    this.el('model-note').textContent = modelMissing
      ? '样品-X模型尚未准备完成。请先在 X-011「模型管理」查看手动准备说明。'
      : '';
    this.el<HTMLButtonElement>('open-models').hidden = !modelMissing;
    this.el('fallback').hidden = !this.health.fallback_reason;
    this.el('fallback').textContent = this.health.fallback_reason ? `已回退 CPU · ${this.health.fallback_reason}` : '';
    for (const id of ['copy', 'export']) this.el<HTMLButtonElement>(id).disabled = !this.text();
  }
  private async file(sample = false) {
    const generation = ++this.fileGeneration; this.chosenAudio = undefined; this.reading = true; this.render(); this.el('error').hidden = true;
    try {
      const file = this.el<HTMLInputElement>('file').files?.[0];
      if (!sample && !file) { this.el('file-name').textContent = '未选择文件'; return; }
      if (file && !sample && (file.size > 100 * 1024 * 1024 || !file.name.toLowerCase().endsWith('.wav'))) throw Error('请选择不超过 100 MB 的 WAV');
      const bytes = sample ? await (await fetch(service + '/sample.wav')).arrayBuffer() : await file!.arrayBuffer();
      const audio = await decodeSampleXWav(bytes); if (generation !== this.fileGeneration) return;
      this.chosenAudio = audio; this.chosenName = sample ? '示例中文录音' : file!.name;
      this.el('file-name').textContent = `${this.chosenName} · ${(audio.length / 16000).toFixed(2)} 秒`;
    } catch (e) { this.error(e); } finally { if (generation === this.fileGeneration) { this.reading = false; this.render(); } }
  }
  private closeCapture(s: Run) {
    s.stream?.getTracks().forEach(t => { t.onended = null; t.stop(); }); s.stream = undefined;
    s.node?.disconnect(); s.sourceNode?.disconnect(); void s.ctx?.close().catch(() => {}); s.ctx = undefined; s.node = undefined;
  }
  private cleanup(s: Run) {
    clearTimeout(s.timer); clearTimeout(s.timeout); this.closeCapture(s);
    if (s.ws && s.ws.readyState < WebSocket.CLOSING) s.ws.close();
    if (this.active === s) this.active = undefined;
    this.mute(false); this.el<HTMLAudioElement>('audio').inert = false; this.render();
  }
  private fail(s: Run, error: unknown) {
    if (s.settled) return; s.settled = true; this.error(error); this.message('未完整完成 · 已保留收到的文字'); this.cleanup(s);
  }
  private send(s: Run, pcm: ArrayBuffer) {
    if (this.active !== s || s.finished) return;
    if (s.ws?.readyState !== WebSocket.OPEN) return this.fail(s, '识别连接已断开');
    if (s.ws.bufferedAmount > 320000) return this.fail(s, '音频发送积压，结果可能不完整');
    s.sent += pcm.byteLength / 2; s.ws.send(pcm);
  }
  private finish(s: Run) {
    if (s.finished || s.settled) return;
    if (s.ws?.readyState !== WebSocket.OPEN) return this.fail(s, '定稿连接已断开');
    s.finished = true; clearTimeout(s.timeout); s.ws.send(JSON.stringify({ type: 'finish', samples_sent: s.sent })); this.closeCapture(s);
    s.timeout = setTimeout(() => this.fail(s, '定稿等待超时；当前结果可能不完整'), 120000);
  }
  private stop(s: Run) {
    if (s.stopping) return; s.stopping = true; clearTimeout(s.timer); this.render();
    if (!s.ready) return this.fail(s, '已取消准备');
    this.message('正在定稿，保留最后一段音频…');
    if (s.node) { s.node.port.postMessage('stop'); s.timeout = setTimeout(() => this.fail(s, '采集停止超时'), 5000); } else this.finish(s);
  }
  private frame(s: Run) {
    if (this.active !== s || s.stopping || !s.audio) return;
    const end = Math.min(s.position + 1600, s.audio.length); this.send(s, s.audio.slice(s.position, end).buffer); s.position = end;
    this.el<HTMLProgressElement>('progress').value = s.position / s.audio.length;
    if (s.position === s.audio.length) this.stop(s);
    else {
      // Absolute audio deadlines avoid accumulating one timer/render delay per
      // packet over a long WAV. Audio bytes and ordering remain unchanged.
      const deadline = (s.replayStarted ?? performance.now()) + Math.min(s.position + 1600, s.audio.length) / 16;
      s.timer = setTimeout(() => this.frame(s), Math.max(0, deadline - performance.now()));
    }
  }
  private draw(s: Run) {
    const transcript = this.el('transcript');
    const follow = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 48;
    const final = [...s.finals.entries()].sort((a, b) => a[0] - b[0]).map(x => x[1]).filter(Boolean).join('\n');
    const partial = s.partial ? '\n' + s.partial : '';
    if (this.el('final').textContent !== final) this.el('final').textContent = final;
    if (this.el('partial').textContent !== partial) this.el('partial').textContent = partial;
    if (follow) transcript.scrollTop = transcript.scrollHeight;
    this.render();
  }
  private async begin() {
    if (this.active) return;
    const source = this.value('source'); if (source === 'wav' && !this.chosenAudio) return;
    const s: Run = { ready: false, stopping: false, finished: false, settled: false, sent: 0, position: 0, finals: new Map(), partial: '', audio: this.chosenAudio };
    this.active = s; this.mute(true); this.el<HTMLAudioElement>('audio').pause(); this.el<HTMLAudioElement>('audio').inert = true;
    this.el('error').hidden = true; this.el('recording').hidden = true; this.el('report').hidden = true; this.el('metrics').textContent = '';
    this.el('progress').hidden = source !== 'wav'; this.el<HTMLProgressElement>('progress').value = 0; this.draw(s); this.panel('results');
    this.message('正在准备音频，请按浏览器提示授权…'); s.timeout = setTimeout(() => this.fail(s, '音频准备超时'), 60000);
    try {
      let settings = {};
      if (source !== 'wav') {
        const displayOptions = { video: true, audio: { suppressLocalAudioPlayback: false }, systemAudio: 'include' } as DisplayMediaStreamOptions;
        s.stream = source === 'computer' ? await navigator.mediaDevices.getDisplayMedia(displayOptions) : await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true, ...(this.value('device') ? { deviceId: { exact: this.value('device') } } : {}) } });
        if (this.active !== s) { this.closeCapture(s); return; }
        const track = s.stream.getAudioTracks().find(t => t.readyState === 'live'); if (!track) throw Error('未获得音频，请勾选共享音频后重试');
        settings = { ...track.getSettings(), device_label: track.label }; void this.devices();
        s.stream.getTracks().forEach(t => { t.onended = () => this.stop(s); });
        s.ctx = new AudioContext({ sampleRate: 16000 }); if (s.ctx.sampleRate !== 16000) throw Error('浏览器不支持 16kHz 音频');
        await s.ctx.audioWorklet.addModule(import.meta.env.BASE_URL + 'sample-x-capture.js');
        if (this.active !== s) { this.closeCapture(s); return; }
        await s.ctx.resume();
      }
      if (this.active !== s) return;
      clearTimeout(s.timeout); s.ws = new WebSocket('ws://127.0.0.1:8877/ws/asr'); s.timeout = setTimeout(() => this.fail(s, '连接或切换样品-X后端超时'), 60000);
      s.ws.onopen = () => { if (this.active === s) s.ws!.send(JSON.stringify({ type: 'start', version: 'a', backend: this.value('backend') || 'auto', source: source === 'wav' ? 'file' : 'browser', interval: Number(this.value('interval')), capture_settings: settings, input_name: source === 'wav' ? this.chosenName : source, capture_api: source === 'wav' ? 'WAV replay' : 'Web Audio', browser_dsp: source === 'microphone' ? true : null })); };
      s.ws.onerror = () => this.fail(s, '样品-X识别连接失败'); s.ws.onclose = () => { if (!s.settled) this.fail(s, '连接中断，未定稿内容可能不完整'); };
      s.ws.onmessage = e => {
        if (this.active !== s) return;
        try {
          const m = JSON.parse(e.data);
          if (m.runtime) this.health = { ...this.health, ...m.runtime };
          if (m.type === 'ready') {
            s.ready = true; clearTimeout(s.timeout); this.render();
            if (source === 'wav') { this.message('按录音时长重放，正在识别…'); s.replayStarted = performance.now(); s.timer = setTimeout(() => this.frame(s), Math.min(100, s.audio!.length / 16)); }
            else {
              s.sourceNode = s.ctx!.createMediaStreamSource(new MediaStream(s.stream!.getAudioTracks())); s.node = new AudioWorkletNode(s.ctx!, 'capture');
              s.node.onprocessorerror = () => this.fail(s, '浏览器音频处理器错误');
              s.node.port.onmessage = event => { if (this.active !== s) return; if (event.data === 'flushed') { clearTimeout(s.timeout); this.finish(s); } else this.send(s, event.data); };
              s.sourceNode.connect(s.node); s.node.connect(s.ctx!.destination); this.message('正在聆听，停顿后自动分段…');
            }
          } else if (m.type === 'partial') { s.partial = m.text; this.draw(s); }
          else if (m.type === 'final') { for (const revision of m.revisions || []) s.finals.set(revision.segment_id, revision.text); s.finals.set(m.segment_id, m.text); s.partial = ''; this.draw(s); }
          else if (m.type === 'meter') this.el('meter').textContent = `已接收 ${m.audio_seconds.toFixed(1)} 秒 · ${m.peak_dbfs <= -170 ? '数字静音，请检查音源' : `平均 ${m.rms_dbfs} dBFS · 削波 ${m.clip_percent}%`}`;
          else if (m.type === 'error') this.fail(s, m.message);
          else if (m.type === 'done') {
            s.partial = ''; this.draw(s); const r = m.report;
            if (!/^[0-9a-f]{32}$/.test(r.run_id)) throw Error('识别记录编号无效');
            const audio = `${service}/runs/${r.run_id}/input.wav`;
            this.el<HTMLAudioElement>('audio').src = audio; this.el<HTMLAnchorElement>('download').href = audio;
            this.el<HTMLAnchorElement>('report').href = `${service}/runs/${r.run_id}/summary.json`;
            this.el('recording').hidden = false; this.el('report').hidden = false;
            this.el('metrics').textContent = `音频 ${r.audio_seconds.toFixed(2)}s · 首字 ${r.first_text_seconds == null ? '无' : r.first_text_seconds.toFixed(2) + 's'}（含输入等待）· 累计处理 ${r.compute_seconds.toFixed(2)}s`;
            this.el('diagnostics').textContent = JSON.stringify(r, null, 2); this.message('已完成，录音和记录保存在本机'); s.settled = true; this.cleanup(s);
          }
        } catch (e) { this.fail(s, e); }
      };
    } catch (e) { this.fail(s, e); }
  }
}
