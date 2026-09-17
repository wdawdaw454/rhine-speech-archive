import { escapeHtml as esc } from './html';
import './model-manager.css';

interface ManagedModel {
  id: string;
  name: string;
  description: string;
  features: string[];
  source: string;
  source_url: string;
  license: string;
  device: string;
  estimated_size: string;
  install_mode: string;
  manual: string | null;
  installed: boolean;
}

interface ManagerOperation {
  id: string;
  action: string;
  state: string;
  message: string;
  started_at: number;
  log: string[];
  error: string | null;
}

interface ManagerStatus {
  models: ManagedModel[];
  operation: ManagerOperation | null;
}

const actionText: Record<string, string> = {
  install: '安装',
  uninstall: '卸载',
};

export class ModelManagerControls {
  readonly root = document.createElement('section');
  private models: ManagedModel[] = [];
  private operation?: ManagerOperation | null;
  private connected = false;
  private pending = '';
  private request = 0;
  private notified = '';

  constructor(private notify: (message: string) => void) {
    this.root.id = 'model-manager';
    this.root.className = 'model-manager';
    this.root.setAttribute('aria-label', '模型管理');
    this.root.innerHTML = `
      <div class="model-manager-head">
        <div>
          <span class="model-manager-kicker">LOCAL MODEL REGISTRY</span>
          <p>所有识别档案共享这里的本地模型。未安装模型时工作台仍可启动。</p>
        </div>
        <button id="model-manager-refresh" class="model-refresh">刷新状态 ↻</button>
      </div>
      <p id="model-manager-message" class="model-message" role="status">正在读取本地模型状态…</p>
      <div id="model-manager-list" class="model-list"></div>
      <details id="model-manager-operation" class="model-operation" hidden>
        <summary><span id="model-manager-operation-title">模型操作</span><span id="model-manager-operation-state"></span></summary>
        <pre id="model-manager-log"></pre>
      </details>
      <aside class="model-storage" aria-label="模型存放说明">
        <h3>模型存放说明</h3>
        <ul>
          <li><strong>ModelScope 模型</strong>保存在当前用户目录的 <code>%USERPROFILE%\\.cache\\modelscope\\models\\</code>。</li>
          <li><strong>SenseVoice 实时引擎</strong>的生成包保存在仓库的 <code>latest_stage\\models\\</code>。</li>
          <li><strong>MOSS 会议模型</strong>和独立环境保存在 <code>latest_stage\\models\\</code> 与 <code>latest_stage\\.venv-moss\\</code>。</li>
          <li><strong>样品-X</strong>的手动资产和独立环境保存在仓库的 <code>sample-x\\</code> 目录内。</li>
          <li><strong>Silero VAD</strong>已随仓库内置在 <code>sample-x\\</code> 目录内。</li>
        </ul>
      </aside>`;
    this.root.querySelector('#model-manager-refresh')!.addEventListener('click', () => void this.refresh());
    this.root.addEventListener('click', event => {
      const install = (event.target as Element).closest<HTMLButtonElement>('[data-model-install]');
      if (install) void this.act('install', install.dataset.modelInstall!);
      const uninstall = (event.target as Element).closest<HTMLButtonElement>('[data-model-uninstall]');
      if (uninstall) void this.act('uninstall', uninstall.dataset.modelUninstall!);
    });
    this.render();
    void this.refresh();
    this.schedule();
  }

  open(destination: HTMLElement) {
    destination.append(this.root);
    void this.refresh();
    this.render();
  }

  private el<T extends HTMLElement = HTMLElement>(id: string) {
    return this.root.querySelector<T>(`#model-manager-${id}`)!;
  }

  private async requestApi<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(path, {
      cache: 'no-store',
      ...init,
      headers: init?.body ? { 'Content-Type': 'application/json' } : undefined,
      signal: AbortSignal.timeout(init?.method ? 15000 : 10000),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `本地服务返回 ${response.status}`);
    return data as T;
  }

  private async refresh() {
    const sequence = ++this.request;
    try {
      const status = await this.requestApi<ManagerStatus>('/api/model-manager');
      if (sequence !== this.request) return;
      this.models = status.models;
      this.operation = status.operation;
      this.connected = true;
      const operation = this.operation;
      const finished = operation?.state === 'complete' || operation?.state === 'error';
      if (finished && this.notified !== operation.id + operation.action + operation.started_at) {
        this.notified = operation.id + operation.action + operation.started_at;
        this.notify(operation.message);
      }
    } catch {
      if (sequence === this.request) this.connected = false;
    } finally {
      if (sequence === this.request) this.render();
    }
  }

  private async act(action: 'install' | 'uninstall', id: string) {
    if (this.pending || this.operation?.state === 'running') return;
    const model = this.models.find(item => item.id === id);
    if (!model) return;
    if (action === 'uninstall' && !confirm(`确定卸载「${model.name}」？模型缓存、生成文件或专用运行环境会从本机删除；声纹、录音和转写结果不会删除。`)) return;
    this.pending = id;
    try {
      const status = await this.requestApi<ManagerStatus>(`/api/model-manager/${action}`, {
        method: 'POST',
        body: JSON.stringify({ id }),
      });
      this.models = status.models;
      this.operation = status.operation;
      this.notify(`正在${actionText[action]} ${model.name}`);
    } catch (error) {
      this.notify(error instanceof Error ? error.message : String(error));
    } finally {
      this.pending = '';
      this.render();
    }
  }

  private schedule() {
    setTimeout(() => {
      void this.refresh().finally(() => this.schedule());
    }, this.operation?.state === 'running' ? 1200 : 4000);
  }

  private render() {
    this.el('message').textContent = this.connected
      ? (this.operation?.state === 'running' ? this.operation.message : '模型状态已同步。安装或卸载期间可以继续浏览工作台。')
      : '本地后端未连接 · 正在重试';
    this.el('message').dataset.connected = String(this.connected);

    const list = this.el('list');
    if (!list.children.length) {
      list.innerHTML = this.models.map(model => this.card(model)).join('');
    } else {
      this.models.forEach(model => {
        const card = list.querySelector<HTMLElement>(`[data-model-card="${model.id}"]`);
        if (!card) return;
        card.dataset.installed = String(model.installed);
        card.querySelector<HTMLElement>('.model-state')!.textContent = model.installed ? '已安装' : '未安装';
        const install = card.querySelector<HTMLButtonElement>('[data-model-install]');
        if (install) install.disabled = this.busy();
        const uninstall = card.querySelector<HTMLButtonElement>('[data-model-uninstall]');
        if (uninstall) uninstall.disabled = this.busy() || !model.installed;
      });
    }

    const operation = this.operation;
    const details = this.el<HTMLDetailsElement>('operation');
    details.hidden = !operation;
    if (operation) {
      const model = this.models.find(item => item.id === operation.id);
      this.el('operation-title').textContent = `${actionText[operation.action] || operation.action} · ${model?.name || operation.id}`;
      this.el('operation-state').textContent = operation.state === 'running' ? operation.message : `${operation.message}${operation.error ? `：${operation.error}` : ''}`;
      this.el('log').textContent = operation.log.join('\n') || '暂无命令输出。下载进度通常显示在命令行工具内部，完成后会更新状态。';
      if (operation.state === 'running') details.open = true;
    }
    this.el<HTMLButtonElement>('refresh').disabled = this.operation?.state === 'running';
  }

  private busy() {
    return Boolean(this.pending) || this.operation?.state === 'running';
  }

  private card(model: ManagedModel) {
    const busy = this.busy();
    const install = model.install_mode === 'manual'
      ? `<a class="model-install manual" href="${esc(model.source_url)}" target="_blank" rel="noopener">查看手动准备说明 ↗</a>`
      : model.install_mode === 'bundled'
        ? '<span class="model-bundled">已随仓库内置 ✓</span>'
        : `<button class="model-install" data-model-install="${esc(model.id)}" ${busy ? 'disabled' : ''}>${model.installed ? '重新校验 / 补齐 ↻' : '安装模型 ↓'}</button>`;
    const uninstall = model.install_mode === 'bundled'
      ? ''
      : `<button class="model-uninstall" data-model-uninstall="${esc(model.id)}" ${busy || !model.installed ? 'disabled' : ''}>卸载模型 ✕</button>`;
    return `<article class="model-card" data-model-card="${esc(model.id)}" data-installed="${model.installed}">
      <header>
        <div><h3>${esc(model.name)}</h3><p>${esc(model.description)}</p></div>
        <span class="model-state">${model.installed ? '已安装' : '未安装'}</span>
      </header>
      <dl>
        <div><dt>关联功能</dt><dd>${model.features.map(feature => `<span>${esc(feature)}</span>`).join('')}</dd></div>
        <div><dt>来源 / 许可</dt><dd><a href="${esc(model.source_url)}" target="_blank" rel="noopener">${esc(model.source)}</a><small>${esc(model.license)}</small></dd></div>
        <div><dt>设备 / 空间</dt><dd>${esc(model.device)}<small>${esc(model.estimated_size)}</small></dd></div>
      </dl>
      ${model.manual ? `<p class="model-manual">${esc(model.manual)}</p>` : ''}
      <footer>${install}${uninstall}</footer>
    </article>`;
  }
}
