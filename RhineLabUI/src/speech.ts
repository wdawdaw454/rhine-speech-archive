import { escapeHtml as esc } from './html';
import './speech.css';
import { requiredModelIds, speechArchives } from './speech-catalog';

type Feature = 'live' | 'record' | 'target' | 'meeting' | 'voice' | 'models';
type Source = 'microphone' | 'system' | 'wav';
interface Model { id: string; name: string; description: string; device: string; available: boolean; modes: string[]; recognition_types: string[] }
interface ManagedModel { id: string; name: string; installed: boolean }
interface Status {
  external_busy?: boolean;
  state: string; message: string; error: string | null; model_id: string; loaded_model_id: string | null;
  source: Source; mode: string; recognition_type: string; recording_seconds: number; audio_level: number;
  committed_text: string; partial_text: string; recording_url: string | null; run_id: string | null;
  target_ready: boolean; speaker_profile: { name: string; speech_seconds: number } | null;
  speaker_decision: string; speaker_score: number | null; capture_device: string;
  meeting_segments: { speaker: string; start: number; end: number; text: string }[];
  meeting_exports: Record<string, string>; load_timings: Record<string, unknown>;
  realtime_metrics: Record<string, unknown>;
  run_context?: { model_id: string; mode: string; recognition_type: string; source: Source } | null;
}
export const features: Record<Feature, { title: string; code: string; mode: string; type: string; hint: string }> = {
  live: { title: '实时转写', code: 'LIVE TRANSCRIPTION', mode: 'streaming', type: 'normal', hint: '边说边显示。也可采集电脑音频，或按实时节奏输入 WAV。' },
  record: { title: '录音转写', code: 'AUDIO TRANSCRIPTION', mode: 'offline', type: 'normal', hint: '先录音，停止后整段识别；也可直接转写 WAV 文件。' },
  target: { title: '目标说话人', code: 'TARGET SPEAKER', mode: 'streaming', type: 'target', hint: '仅转写与已注册声纹匹配的声音。使用 SenseVoice，预览可修订。' },
  meeting: { title: '会议转写', code: 'MEETING TRANSCRIPTION', mode: 'offline', type: 'meeting', hint: '整段生成文字、说话人编号与时间戳。编号不代表真实身份。' },
  voice: { title: '声纹管理', code: 'VOICE ENROLLMENT', mode: 'streaming', type: 'target', hint: '注册 3–30 秒清晰单人语音。声纹仅保存在本机，重新注册会替换原声纹。' },
  models: { title: '模型管理', code: 'MODEL REGISTRY', mode: 'streaming', type: 'normal', hint: '集中管理本地识别模型。' },
};
const activeStates = new Set(['loading', 'starting', 'listening', 'stopping', 'transcribing', 'enrolling', 'enroll_recording']);
const captureStates = new Set(['starting', 'listening', 'stopping', 'enroll_recording']);
const stamp = (n: number) => `${Math.floor((n || 0) / 60).toString().padStart(2, '0')}:${Math.floor((n || 0) % 60).toString().padStart(2, '0')}`;
const modelNames: Record<string, string> = {
  'sensevoice-realtime': 'SenseVoice 实时引擎',
  'sensevoice-small': 'SenseVoice Small',
  'fsmn-vad': 'FSMN-VAD',
  'cam-plus': 'CAM++ 声纹模型',
  'fun-asr-nano': 'Fun-ASR-Nano',
  'qwen3-asr': 'Qwen3-ASR 1.7B',
  'moss-transcribe-diarize': 'MOSS-Transcribe-Diarize',
};

export class SpeechArchiveControls {
  readonly root = document.createElement('section');
  private feature: Feature = 'record';
  private fixedModel = 'sensevoice-small';
  private archiveId = '';
  private results = new Map<string, Status>();
  private panel = 'controls';
  private models: Model[] = [];
  private managedModels: ManagedModel[] = [];
  private status?: Status;
  private connected = false;
  private pending = false;
  private polling = false;
  private generation = 0;
  private modelMemory = new Map<Feature, string>();
  private previousSegments = '';
  private recordingUrl = '';
  private deviceRequest = 0;
  private modelsRefreshedAt = 0;
  private managedModelsRefreshedAt = 0;
  constructor(private mute: (muted: boolean) => void, private notify: (message: string) => void, private manageVoice: () => void, private manageModels: () => void) {
    this.root.id = 'speech-archive';
    this.root.className = 'speech-archive';
    this.root.setAttribute('aria-label', '档案功能操作');
    this.root.innerHTML = `
      <div class="speech-tabs" role="tablist" aria-label="功能详情">
        <button id="speech-tab-controls" role="tab" aria-selected="true" aria-controls="speech-controls" data-speech-panel="controls">01 <span>使用功能</span></button>
        <button id="speech-tab-results" role="tab" aria-selected="false" aria-controls="speech-results" data-speech-panel="results">02 <span>识别结果</span></button>
        <button id="speech-tab-info" role="tab" aria-selected="false" aria-controls="speech-info" data-speech-panel="info">03 <span>功能说明</span></button>
      </div>
      <div class="speech-status-line"><span id="speech-state" role="status">连接本地后端…</span><span id="speech-time">00:00</span></div>
      <p id="speech-error" class="speech-error" role="alert" hidden></p>
      <section id="speech-controls" class="speech-controls" role="tabpanel" aria-labelledby="speech-tab-controls">
        <div hidden><span id="speech-index"></span><span id="speech-code"></span><h2 id="speech-title"></h2><p id="speech-hint"></p></div>
        <label class="speech-field"><span>INPUT / 音频来源</span><select id="speech-source"><option value="microphone">麦克风</option><option value="system">电脑音频</option><option value="wav">WAV 文件</option></select></label>
        <p class="speech-note" id="speech-device">默认麦克风</p>
        <label class="speech-file" id="speech-file-field" hidden><span>选择 WAV 文件 ↗</span><input id="speech-file" type="file" accept=".wav,audio/wav" aria-label="选择 WAV 文件"/><small id="speech-file-name"></small></label>
        <p class="speech-note speech-model-note" id="speech-model-note"><span id="speech-model-note-text">正在检查本档案所需模型…</span><button id="speech-open-models" hidden>前往模型管理 ↗</button></p>
        <section id="speech-engine"><select id="speech-model" aria-label="档案绑定引擎" hidden></select><button class="speech-secondary" id="speech-load">加载引擎 ↗</button></section>
        <section id="speech-target" hidden><div class="speech-profile"><span>VOICEPRINT / 当前声纹</span><strong id="speech-profile">未注册</strong><button id="speech-manage">打开声纹管理档案 ↗</button></div><button class="speech-secondary" id="speech-target-load">加载声纹组件 ↗</button><label class="speech-field" id="speech-threshold-field"><span>匹配阈值</span><input id="speech-threshold" type="number" min="0.10" max="0.95" step="0.01" value="0.45"/></label><p class="speech-note" id="speech-target-note">阈值越高越严格。短语音可能跳过，不分离重叠声音；相似度不是概率。</p></section>
        <section id="speech-enrollment" hidden><label class="speech-field"><span>声纹名称</span><input id="speech-name" maxlength="40" value="主讲人" autocomplete="off"/></label><p class="speech-note">录音注册最多 30 秒，停止后提取声纹。WAV 注册使用上方文件。</p><button class="speech-danger" id="speech-forget">删除已保存声纹</button></section>
      </section>
      <section id="speech-results" class="speech-results" role="tabpanel" aria-labelledby="speech-tab-results" hidden>
        <p id="speech-session" class="speech-note">本档案尚未开始转写</p><meter id="speech-level" min="0" max="1" value="0" aria-label="音频输入音量"></meter>
        <p id="speech-decision" class="speech-note" hidden></p>
        <div class="speech-transcript" id="speech-transcript" tabindex="0" aria-label="转写文本"><div id="speech-empty"><span>AWAITING AUDIO INPUT</span><p>本档案暂无转写结果。</p></div><div id="speech-text"><span id="speech-committed"></span><span id="speech-partial"></span></div><div id="speech-segments" hidden></div></div>
        <div class="speech-output-actions"><button id="speech-copy" disabled>复制文本 ↗</button><button id="speech-export" disabled>导出 TXT ↓</button><button id="speech-clear" disabled>清空文本</button><a id="speech-json" hidden download>会议 JSON ↓</a><a id="speech-srt" hidden download>字幕 SRT ↓</a></div>
        <div class="speech-recording" id="speech-recording" hidden><audio id="speech-audio" controls preload="none" aria-label="本次录音回放"></audio><a id="speech-download" download="录音.wav">保存录音 ↓</a></div>
      </section>
      <section id="speech-info" role="tabpanel" aria-labelledby="speech-tab-info" hidden><ol id="speech-archive-notes"></ol><details class="speech-diagnostics"><summary>运行信息</summary><pre id="speech-diagnostics"></pre><button id="speech-shutdown">关闭本地后端</button><a href="/" target="_blank" rel="noopener">原版转写界面 ↗</a></details></section>
      <div class="speech-control-footer"><button class="speech-primary" id="speech-start" disabled>开始识别 →</button><p class="speech-note" id="speech-limit"></p></div>
      <div class="speech-connection" id="speech-connection" role="status">正在连接本地后端</div>`;
    this.root.addEventListener('click', e => {
      const button = (e.target as Element).closest<HTMLButtonElement>('[data-speech-panel]');
      if (button) this.showPanel(button.dataset.speechPanel!);
    });
    this.root.addEventListener('keydown', e => {
      const button = (e.target as Element).closest<HTMLButtonElement>('[data-speech-panel]');
      if (!button || !['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
      e.preventDefault(); e.stopPropagation();
      const panels = this.feature === 'voice' ? ['controls', 'info'] : ['controls', 'results', 'info'];
      this.showPanel(panels[(panels.indexOf(this.panel) + (e.key === 'ArrowRight' ? 1 : panels.length - 1)) % panels.length]);
      this.el('tab-' + this.panel).focus();
    });
    this.el<HTMLSelectElement>('source').addEventListener('change', () => { this.render(); void this.device(); });
    this.el<HTMLSelectElement>('model').addEventListener('change', () => { this.modelMemory.set(this.feature, this.value('model')); this.render(); });
    this.el<HTMLInputElement>('file').addEventListener('change', () => {
      try { if (this.file()) this.validateFile(); } catch (e) { this.el<HTMLInputElement>('file').value = ''; this.error(e); }
      this.render();
    });
    this.el('load').addEventListener('click', () => void this.act('/api/load-model', { model_id: this.value('model'), mode: features[this.feature].mode }));
    this.el('open-models').addEventListener('click', () => this.manageModels());
    this.el('target-load').addEventListener('click', () => void this.act('/api/target/load'));
    this.el('start').addEventListener('click', () => void this.start());
    this.el('manage').addEventListener('click', () => this.manageVoice());
    this.el('forget').addEventListener('click', () => {
      if (confirm('永久删除本机保存的目标声纹？之后需要重新注册。')) void this.act('/api/target/forget');
    });
    this.el('clear').addEventListener('click', () => void this.act('/api/clear'));
    this.el('copy').addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(this.text()); this.notice('文本已复制'); } catch { this.error('浏览器未允许复制，可选中右侧文本手动复制。'); }
    });
    this.el('export').addEventListener('click', () => {
      const url = URL.createObjectURL(new Blob([this.text()], { type: 'text/plain;charset=utf-8' }));
      const link = document.createElement('a'); link.href = url; link.download = `转写-${this.result()?.run_id || Date.now()}.txt`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    this.el('shutdown').addEventListener('click', () => {
      if (confirm('关闭本地语音后端？所有界面将断开连接；下次请双击启动入口。')) void this.act('/api/shutdown');
    });
    this.options(); this.render();
    void this.poll();
    window.addEventListener('beforeunload', e => { if (this.busy()) { e.preventDefault(); e.returnValue = ''; } });
  }
  private el<T extends HTMLElement = HTMLElement>(id: string) { return this.root.querySelector<T>(`#speech-${id}`)!; }
  private value(id: string) { return this.el<HTMLInputElement | HTMLSelectElement>(id).value; }
  private file() { return this.el<HTMLInputElement>('file').files?.[0]; }
  private requiredModels() { return requiredModelIds[this.archiveId] || []; }
  private requiredModelName(id: string) { return this.managedModels.find(model => model.id === id)?.name || modelNames[id] || id; }
  private busy() { return !!this.status && (activeStates.has(this.status.state) || !!this.status.external_busy); }
  private key(feature = this.feature, model = this.fixedModel) { return `${features[feature].mode}/${features[feature].type}/${model}`; }
  private result() { return this.results.get(this.key()); }
  private text() { const result = this.result(); return [result?.committed_text, result?.partial_text].filter(Boolean).join('\n'); }
  private ownsSession() {
    const s = this.status; if (!s) return false;
    if (s.state.startsWith('enroll')) return this.feature === 'voice';
    const f = features[this.feature];
    return this.feature !== 'voice' && s.model_id === this.fixedModel && s.mode === f.mode && s.recognition_type === f.type;
  }
  private notice(message: string) { this.notify(message); }
  private error(error: unknown) { this.el('error').hidden = false; this.el('error').textContent = error instanceof Error ? error.message : String(error); }
  private validateFile() {
    const file = this.file();
    if (!file || !file.name.toLowerCase().endsWith('.wav') || !file.size || file.size > 64 * 1024 * 1024) throw new Error('请选择非空 WAV 文件，大小不超过 64 MB。');
    return file;
  }
  openArchive(archive: (typeof speechArchives)[number], destination: HTMLElement) {
    if (this.archiveId !== archive.id) {
      this.archiveId = archive.id; this.feature = archive.feature; this.fixedModel = archive.model;
      this.el<HTMLSelectElement>('source').value = archive.feature === 'meeting' ? 'wav' : 'microphone';
      this.options(); this.showPanel('controls'); this.el('error').hidden = true;
      this.el('archive-notes').innerHTML = archive.findings.map(text => `<li>${esc(text)}</li>`).join('');
      void this.device();
    }
    destination.append(this.root); this.render();
  }
  private showPanel(panel: string) {
    this.panel = panel;
    for (const name of ['controls', 'results', 'info']) {
      this.el(name).hidden = name !== panel;
      this.el('tab-' + name).setAttribute('aria-selected', String(name === panel));
      this.el('tab-' + name).tabIndex = name === panel ? 0 : -1;
    }
  }
  private options() {
    const f = features[this.feature];
    const list = this.models.filter(m => m.id === this.fixedModel && m.modes.includes(f.mode) && m.recognition_types.includes(f.type));
    this.el('model').innerHTML = list.map(m => `<option value="${esc(m.id)}" ${m.available ? '' : 'disabled'}>${esc(m.name)}${m.available ? '' : '（文件缺失）'}</option>`).join('');
    const previous = this.modelMemory.get(this.feature);
    this.el<HTMLSelectElement>('model').value = list.find(m => m.id === previous && m.available)?.id || list.find(m => m.available)?.id || '';
  }
  private async request<T>(path: string, options: RequestInit = {}): Promise<T> {
    const response = await fetch(path, { cache: 'no-store', ...options, signal: AbortSignal.timeout(options.method ? 30000 : 5000) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `本地服务返回 ${response.status}`);
    return data as T;
  }
  private async device() {
    const source = this.value('source'); const sequence = ++this.deviceRequest;
    if (source === 'wav') { this.el('device').textContent = '文件仅发送给本机后端，不上传云端。'; return; }
    try {
      const data = await this.request<{ name: string; available: boolean; error?: string }>(`/api/device?source=${source}`);
      if (sequence === this.deviceRequest) this.el('device').textContent = data.available === false ? data.error || '该输入设备不可用' : data.name;
    } catch { if (sequence === this.deviceRequest) this.el('device').textContent = '暂时无法读取输入设备'; }
  }
  private async act(path: string, body: unknown = {}, file?: File) {
    if (this.pending || !this.connected) return;
    this.pending = true; this.generation++; this.el('error').hidden = true; this.render();
    try {
      const result = await this.request<Status>(path, { method: 'POST', headers: { 'Content-Type': file ? 'audio/wav' : 'application/json' }, body: file || JSON.stringify(body) });
      if (result.state) this.accept(result);
      if (path === '/api/shutdown') { this.connected = false; this.notice('本地后端已关闭'); }
    } catch (e) { this.error(e); }
    finally { this.pending = false; this.render(); }
  }
  private async start() {
    if (this.pending || !this.connected) return;
    const s = this.status;
    if (this.ownsSession() && s && (s.state === 'listening' || s.state === 'enroll_recording' || (s.state === 'transcribing' && s.recognition_type === 'meeting'))) return this.act('/api/stop');
    if (this.busy()) return;
    const source = this.value('source');
    try {
      const file = source === 'wav' ? this.validateFile() : undefined;
      if (this.feature === 'voice') {
        if (s?.speaker_profile && !confirm('重新注册将替换当前声纹，继续？')) return;
        this.el<HTMLAudioElement>('audio').pause(); this.mute(true);
        if (file) await this.act(`/api/target/enroll-file?${new URLSearchParams({ name: this.value('name') })}`, {}, file);
        else await this.act('/api/target/enroll-recording', { name: this.value('name'), source });
      } else {
        const threshold = Number(this.value('threshold'));
        if (this.feature === 'target' && (!Number.isFinite(threshold) || threshold < 0.10 || threshold > 0.95)) throw new Error('匹配阈值必须为 0.10–0.95。');
        const f = features[this.feature];
        const body = { model_id: this.value('model'), mode: f.mode, recognition_type: f.type, target_only: this.feature === 'target', speaker_threshold: threshold, source };
        this.el<HTMLAudioElement>('audio').pause(); this.mute(true);
        if (file) {
          const query = new URLSearchParams(Object.entries({ ...body, filename: file.name }).map(([k, v]) => [k, String(v)]));
          await this.act(`/api/transcribe-file?${query}`, {}, file);
        } else await this.act('/api/start', body);
      }
    } catch (e) { this.error(e); }
    finally { this.mute(!!this.status && captureStates.has(this.status.state)); if (this.feature !== 'voice' && this.ownsSession()) this.showPanel('results'); }
  }
  private accept(status: Status) {
    this.status = status;
    if (status.run_id && !status.state.startsWith('enroll') && status.state !== 'loading') {
      const result = status.run_context ? { ...status, ...status.run_context } : status;
      this.results.set(`${result.mode}/${result.recognition_type}/${result.model_id}`, result);
    }
    this.mute(captureStates.has(status.state));
    if (status.error) this.error(status.error);
    this.render();
  }
  private async poll() {
    if (!this.pending && !this.polling) {
      this.polling = true; const generation = this.generation;
      try {
        if (!this.models.length || !this.managedModels.length || Date.now() - this.modelsRefreshedAt > 5000) {
          const [engineStatus, managerStatus] = await Promise.all([
            this.request<{ models: Model[] }>('/api/models'),
            this.request<{ models: ManagedModel[] }>('/api/model-manager'),
          ]);
          this.models = engineStatus.models;
          this.managedModels = managerStatus.models;
          this.modelsRefreshedAt = this.managedModelsRefreshedAt = Date.now(); this.options();
        }
        const status = await this.request<Status>('/api/status');
        if (generation === this.generation) { this.connected = true; this.accept(status); }
      } catch { if (generation === this.generation) { this.connected = false; this.mute(true); this.render(); } }
      finally { this.polling = false; }
    }
    setTimeout(() => void this.poll(), this.connected ? 500 : 1800);
  }
  private render() {
    const f = features[this.feature], s = this.status;
    const r = this.result();
    const busy = this.busy(), blocked = this.pending || busy || !this.connected;
    const isVoice = this.feature === 'voice', target = isVoice || this.feature === 'target';
    const model = this.models.find(m => m.id === this.value('model'));
    const missingModels = this.requiredModels().filter(id => {
      const engine = this.models.find(item => item.id === id);
      if (engine) return !engine.available;
      const managed = this.managedModels.find(item => item.id === id);
      return Boolean(managed && !managed.installed);
    });
    const missingModelNames = missingModels.map(id => this.requiredModelName(id));
    const modelNoteText = this.el('model-note-text');
    const openModels = this.el<HTMLButtonElement>('open-models');
    const ready = !!model?.available && model.id === s?.loaded_model_id;
    if (this.ownsSession() && s && captureStates.has(s.state)) this.el<HTMLSelectElement>('source').value = s.source;
    this.el('title').textContent = f.title; this.el('code').textContent = f.code;
    this.el('index').textContent = String(Object.keys(features).indexOf(this.feature) + 1).padStart(2, '0');
    this.el('hint').textContent = f.hint;
    this.root.querySelectorAll<HTMLButtonElement>('[data-speech-feature]').forEach(b => { b.setAttribute('aria-pressed', String(b.dataset.speechFeature === this.feature)); b.disabled = blocked && (busy || this.pending); });
    this.el('connection').textContent = this.connected ? 'LOCAL BACKEND / 已连接' : '本地后端未连接 · 正在重试';
    this.el('connection').dataset.connected = String(this.connected);
    this.el('tab-results').hidden = isVoice;
    this.el('engine').hidden = isVoice; this.el('target').hidden = !target;
    this.el('threshold-field').hidden = isVoice; this.el('target-note').hidden = isVoice;
    this.el('manage').hidden = isVoice; this.el('enrollment').hidden = !isVoice;
    this.el('file-field').hidden = this.value('source') !== 'wav';
    this.el('file-name').textContent = this.file()?.name || `未选择文件 · 最大 64 MB / ${isVoice ? '3–30 秒' : '10 分钟'}`;
    this.el('limit').textContent = isVoice ? '声纹文件保存在本机，不自动删除已有注册。' : '麦克风 / 电脑音频最长 120 秒；WAV 最长 10 分钟。';
    for (const id of ['source', 'model', 'file', 'threshold', 'name']) this.el<HTMLInputElement>(id).disabled = blocked;
    this.el<HTMLButtonElement>('load').disabled = blocked || ready || !model?.available;
    this.el('load').textContent = s?.state === 'loading' ? '正在加载…' : ready ? '引擎已就绪 ✓' : '加载引擎 ↗';
    if (missingModels.length) {
      modelNoteText.textContent = `所需模型未安装：${missingModelNames.join('、')}。请先在 X-011「模型管理」安装。`;
      openModels.hidden = false;
    } else {
      openModels.hidden = true;
      modelNoteText.textContent = model
        ? `${model.description} · ${model.device === 'cpu' ? 'CPU' : 'GPU'}`
        : isVoice
          ? '声纹组件使用 FSMN-VAD + CAM++ · CPU'
          : this.models.length && this.managedModels.length
            ? '本档案模型已就绪'
            : '正在检查本档案所需模型…';
    }
    this.el('profile').textContent = s?.speaker_profile ? `${s.speaker_profile.name} · ${s.speaker_profile.speech_seconds}s 有效语音` : '尚未注册';
    this.el<HTMLButtonElement>('target-load').disabled = blocked || !!s?.target_ready;
    this.el('target-load').textContent = s?.target_ready ? '声纹组件已就绪 ✓' : '加载声纹组件 ↗';
    this.el<HTMLButtonElement>('forget').disabled = blocked || !s?.speaker_profile;
    this.el<HTMLButtonElement>('manage').disabled = blocked;
    const stoppable = this.ownsSession() && (s?.state === 'listening' || s?.state === 'enroll_recording' || (s?.state === 'transcribing' && s.recognition_type === 'meeting'));
    const canStart = isVoice ? !!s?.target_ready : ready && (!target || !!(s?.target_ready && s.speaker_profile));
    this.el<HTMLButtonElement>('start').disabled = this.pending || !this.connected || (busy ? !stoppable : !canStart || (this.value('source') === 'wav' && !this.file()));
    this.el('start').textContent = this.pending ? '正在提交…' : s?.state === 'enroll_recording' ? '停止并注册 →' : stoppable ? (s?.state === 'transcribing' ? '取消会议转写 ■' : '停止并定稿 ■') : busy ? '处理中…' : isVoice ? (this.value('source') === 'wav' ? '上传 WAV 注册 →' : '开始录音注册 →') : this.value('source') === 'wav' ? '开始文件转写 →' : this.value('source') === 'system' ? '开始采集电脑音频 →' : '开始录音 →';
    this.el('start').classList.toggle('is-recording', !!stoppable);
    if (!s) return;
    this.el('state').textContent = !this.connected ? '连接中断，恢复连接后同步。' : busy && !this.ownsSession() && s.state !== 'loading' ? '另一档案正在执行：' + s.message : !busy && !this.ownsSession() ? (missingModels.length ? '请先在 X-011 安装所需模型' : isVoice ? (/声纹|注册/.test(s.message) ? s.message : s.target_ready ? '声纹组件已就绪' : '请加载声纹组件') : ready ? '本档案引擎已就绪' : '请先加载本档案引擎') : s.message;
    this.el('time').textContent = stamp(this.ownsSession() ? s.recording_seconds : r?.recording_seconds || 0);
    if (s.external_busy) this.el('state').textContent = '样品-X档案正在识别，请先在 X-010 结束并定稿';
    this.el<HTMLMeterElement>('level').value = s.audio_level || 0;
    this.el('session').textContent = r?.run_id ? `${this.models.find(m => m.id === r.model_id)?.name || r.model_id} / ${r.recognition_type === 'meeting' ? '会议' : r.recognition_type === 'target' ? '目标说话人' : r.mode === 'streaming' ? '实时' : '整段'} / ${r.source === 'wav' ? 'WAV' : r.source === 'system' ? '电脑音频' : '麦克风'}` : '本档案尚未开始转写';
    this.el('decision').hidden = !target || isVoice;
    this.el('decision').textContent = `${({ target: '匹配目标', non_target: '其他声音，已跳过', pending: '等待足够语音' } as Record<string, string>)[r?.speaker_decision || ""] || '等待验证'}${r?.speaker_score == null ? '' : ` · 相似度 ${r?.speaker_score.toFixed(3)}（非概率）`}`;
    const transcript = this.el('transcript'); const atBottom = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 70;
    if (this.el('committed').textContent !== (r?.committed_text || '')) this.el('committed').textContent = r?.committed_text || '';
    const partial = r?.partial_text ? `\n${r?.partial_text}` : '';
    if (this.el('partial').textContent !== partial) this.el('partial').textContent = partial;
    const segments = r?.meeting_segments || []; const hasSegments = segments.length > 0;
    this.el('empty').hidden = !!this.text() || hasSegments;
    this.el('text').hidden = hasSegments; this.el('segments').hidden = !hasSegments;
    const signature = JSON.stringify(segments);
    if (signature !== this.previousSegments) {
      this.previousSegments = signature;
      this.el('segments').innerHTML = segments.map(seg => `<article class="speech-segment" style="--speaker-hue:${(Number(seg.speaker.replace(/\D/g, '')) || 1) * 67 % 360}"><header><strong>${esc(seg.speaker)}</strong><time>${stamp(seg.start)} — ${stamp(seg.end)}</time></header><p>${esc(seg.text)}</p></article>`).join('');
    }
    if (atBottom) transcript.scrollTop = transcript.scrollHeight;
    for (const id of ['copy', 'export']) this.el<HTMLButtonElement>(id).disabled = !this.text();
    this.el<HTMLButtonElement>('clear').disabled = blocked || !this.text() || r?.run_id !== s.run_id;
    this.el<HTMLButtonElement>('shutdown').disabled = this.pending || !this.connected;
    for (const kind of ['json', 'srt']) { const a = this.el<HTMLAnchorElement>(kind); const url = r?.meeting_exports?.[kind]; a.hidden = !url || r?.run_id !== s.run_id; if (url?.startsWith('/api/meeting-export?')) a.href = url; else a.removeAttribute('href'); }
    const url = r?.recording_url?.startsWith('/api/recording?') ? r?.recording_url : '';
    if (url !== this.recordingUrl) {
      this.recordingUrl = url;
      const audio = this.el<HTMLAudioElement>('audio'); audio.pause();
      if (url) { audio.src = url; this.el<HTMLAnchorElement>('download').href = url; } else { audio.removeAttribute('src'); audio.load(); this.el('download').removeAttribute('href'); }
    }
    this.el('recording').hidden = !url || r?.run_id !== s.run_id;
    if (captureStates.has(s.state)) { this.el<HTMLAudioElement>('audio').pause(); this.el<HTMLAudioElement>('audio').inert = true; } else this.el<HTMLAudioElement>('audio').inert = false;
    this.el('diagnostics').textContent = JSON.stringify({ state: s.state, run: s.run_id, engine: s.loaded_model_id, capture: s.capture_device, load: s.load_timings, realtime: s.realtime_metrics }, null, 2);
  }
}
