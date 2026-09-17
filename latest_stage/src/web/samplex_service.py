"""Lazy local sidecar with a shared admission lock for all speech archives."""
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
from threading import RLock
from urllib.request import urlopen

ACTIVE = {'loading', 'starting', 'listening', 'stopping', 'transcribing', 'enrolling', 'enroll_recording'}


class SampleXService:
    def __init__(self, workspace: Path, controller, port=8765):
        self.root = workspace / 'sample-x'
        self.controller = controller
        self.port = port
        self.lock = RLock()
        self.process = None
        self.claimed = False
        self.token = secrets.token_urlsafe(32)

    def occupied(self):
        if self.process is not None and self.process.poll() is not None:
            self.claimed = False
        return self.claimed

    def claim(self, token):
        if not secrets.compare_digest(token, self.token):
            raise ValueError('无效的样品-X服务凭据')
        if self.occupied() or self.controller.snapshot()['state'] in ACTIVE:
            raise RuntimeError('另一识别任务正在运行，请先结束它')
        self.claimed = True

    def release(self, token):
        if not secrets.compare_digest(token, self.token):
            raise ValueError('无效的样品-X服务凭据')
        self.claimed = False

    def status(self):
        if self.process is None:
            return {'ready': False, 'loading': False, 'busy': False, 'message': '请先加载样品-X引擎'}
        if self.process.poll() is not None:
            self.claimed = False
            return {'ready': False, 'loading': False, 'busy': False, 'message': '样品-X服务已退出，请重新加载；详情见 sample-x/logs/service.log'}
        try:
            with urlopen('http://127.0.0.1:8877/health', timeout=.5) as response:
                return json.load(response)
        except Exception:
            return {'ready': False, 'loading': True, 'busy': self.claimed, 'message': '正在加载样品-X模型…'}

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return self.status()
        cuda_python = self.root / '.venv-cuda/Scripts/python.exe'
        cuda_installed = cuda_python.is_file() and (self.root / '.venv-cuda/Lib/site-packages/torch/__init__.py').is_file()
        default_python = cuda_python if cuda_installed else self.root / '.venv/Scripts/python.exe'
        python = Path(os.environ.get('SAMPLEX_PYTHON', str(default_python)))
        if not python.is_file():
            raise FileNotFoundError('样品-X独立环境缺失，请运行 sample-x/setup.ps1；不会修改原 ASR 环境')
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', 8877)) == 0:
                raise RuntimeError('8877 端口已被占用，未修改该服务')
        logs = self.root / 'logs'
        logs.mkdir(exist_ok=True)
        env = dict(os.environ, SAMPLEX_PARENT=str(os.getpid()), SAMPLEX_TOKEN=self.token,
                   SAMPLEX_MAIN_PORT=str(self.port), PYTHONIOENCODING='utf-8')
        with (logs / 'service.log').open('ab') as log:
            self.process = subprocess.Popen([str(python), '-u', str(self.root / 'server.py')],
                cwd=self.root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return {'ready': False, 'loading': True, 'busy': False, 'message': '正在加载样品-X模型…'}
