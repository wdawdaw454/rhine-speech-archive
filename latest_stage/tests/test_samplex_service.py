from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from src.web.samplex_service import SampleXService, ACTIVE


def service(tmp_path):
    controller = Mock()
    controller.snapshot.return_value = {'state': 'idle'}
    return SampleXService(tmp_path, controller)


def test_admission_exclusive_and_requires_private_token(tmp_path):
    model = service(tmp_path)
    with pytest.raises(ValueError):
        model.claim('untrusted')
    model.claim(model.token)
    assert model.occupied()
    with pytest.raises(RuntimeError):
        model.claim(model.token)
    with pytest.raises(ValueError):
        model.release('untrusted')
    assert model.occupied()
    model.release(model.token)
    assert not model.occupied()


@pytest.mark.parametrize('state', sorted(ACTIVE))
def test_admission_rejects_all_other_active_features(tmp_path, state):
    model = service(tmp_path)
    model.controller.snapshot.return_value = {'state': state}
    with pytest.raises(RuntimeError):
        model.claim(model.token)
    assert not model.occupied()


def test_dead_sidecar_releases_reservation(tmp_path):
    model = service(tmp_path)
    model.claim(model.token)
    model.process = Mock()
    model.process.poll.return_value = 1
    assert not model.occupied()
    assert not model.status()['ready']


def test_start_requires_project_local_environment(tmp_path, monkeypatch):
    monkeypatch.delenv('SAMPLEX_PYTHON', raising=False)
    model = service(tmp_path)
    with pytest.raises(FileNotFoundError, match='独立环境'):
        model.start()
    assert model.process is None


def test_start_uses_local_code_env_models_and_hidden_process(tmp_path, monkeypatch):
    monkeypatch.delenv('SAMPLEX_PYTHON', raising=False)
    model = service(tmp_path)
    python = model.root / '.venv/Scripts/python.exe'
    python.parent.mkdir(parents=True)
    python.touch()
    with patch('src.web.samplex_service.socket.socket') as socket, patch('src.web.samplex_service.subprocess.Popen') as process:
        socket.return_value.__enter__.return_value.connect_ex.return_value = 1
        assert model.start()['loading']
        args, kwargs = process.call_args
        assert args[0] == [str(python), '-u', str(model.root / 'server.py')]
        assert kwargs['cwd'] == model.root
        assert kwargs['env']['SAMPLEX_TOKEN'] == model.token
        assert 'Qwen3_ASR_Remote' not in ' '.join(args[0])


def test_http_reservation_blocks_legacy_mutations(tmp_path):
    import json
    from http.server import ThreadingHTTPServer
    from http.client import HTTPConnection
    from threading import Thread
    from src.web.dictation_server import DictationController, make_handler
    controller = DictationController(project_root=tmp_path, models=[])
    model = SampleXService(tmp_path, controller)
    with patch('src.web.dictation_server.SampleXService', return_value=model):
        handler = make_handler(controller, tmp_path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = HTTPConnection(*server.server_address, timeout=5)
    def post(path, body):
        conn.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'})
        response = conn.getresponse()
        result = response.status, json.loads(response.read())
        return result
    try:
        assert post('/api/sample-x/claim', {'token': model.token})[0] == 200
        for endpoint in ('start', 'load-model', 'target/enroll-recording', 'target/forget', 'shutdown', 'clear'):
            assert post('/api/' + endpoint, {})[0] == 409
        conn.request('GET', '/api/status')
        assert json.loads(conn.getresponse().read())['external_busy']
        assert post('/api/sample-x/release', {'token': 'bad'})[0] == 400
        assert post('/api/sample-x/release', {'token': model.token})[0] == 200
        assert post('/api/clear', {})[0] == 200
    finally:
        conn.close(); server.shutdown(); server.server_close(); controller.shutdown(); thread.join(3)


def test_start_prefers_complete_project_cuda_environment(tmp_path, monkeypatch):
    monkeypatch.delenv('SAMPLEX_PYTHON', raising=False)
    model = service(tmp_path)
    python = model.root / '.venv-cuda/Scripts/python.exe'
    torch = model.root / '.venv-cuda/Lib/site-packages/torch/__init__.py'
    for path in [python, torch]:
        path.parent.mkdir(parents=True, exist_ok=True); path.touch()
    with patch('src.web.samplex_service.socket.socket') as socket, patch('src.web.samplex_service.subprocess.Popen') as process:
        socket.return_value.__enter__.return_value.connect_ex.return_value = 1
        model.start()
    assert process.call_args.args[0][0] == str(python)


def test_partial_cuda_install_still_uses_cpu_environment(tmp_path, monkeypatch):
    monkeypatch.delenv('SAMPLEX_PYTHON', raising=False)
    model = service(tmp_path)
    for folder in ['.venv', '.venv-cuda']:
        python = model.root / folder / 'Scripts/python.exe'
        python.parent.mkdir(parents=True); python.touch()
    with patch('src.web.samplex_service.socket.socket') as socket, patch('src.web.samplex_service.subprocess.Popen') as process:
        socket.return_value.__enter__.return_value.connect_ex.return_value = 1
        model.start()
    assert process.call_args.args[0][0] == str(model.root / '.venv/Scripts/python.exe')
