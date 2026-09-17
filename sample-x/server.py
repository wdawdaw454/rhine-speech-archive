"""Rhine A-only browser PCM service; never opens a native microphone."""
import asyncio
import ctypes
import json
import logging
import multiprocessing
import os
from pathlib import Path
import re
from aiohttp import ClientSession, WSMsgType, web
from model_pool import ModelPool
from pipeline import LiveSession

ROOT = Path(__file__).resolve().parent
PORT = 8877
MAIN_PORT = int(os.environ.get('SAMPLEX_MAIN_PORT', '8765'))
ORIGINS = {f'http://{host}:{port}' for host in ('127.0.0.1', 'localhost') for port in (MAIN_PORT, 5174)}


@web.middleware
async def local_only(request, handler):
    if request.host not in {f'127.0.0.1:{PORT}', f'localhost:{PORT}'}:
        raise web.HTTPForbidden(text='Invalid host')
    origin = request.headers.get('Origin')
    if origin and origin not in ORIGINS:
        raise web.HTTPForbidden(text='Invalid origin')
    response = await handler(request)
    if origin:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Vary'] = 'Origin'
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


async def admission(action):
    async with ClientSession() as client:
        async with client.post(f'http://127.0.0.1:{MAIN_PORT}/api/sample-x/{action}',
                json={'token': os.environ['SAMPLEX_TOKEN']}) as response:
            result = await response.json()
            if response.status != 200:
                raise RuntimeError(result.get('error', '本地后端未允许开始'))


async def health(request):
    return web.json_response({'ready': request.app['engine'].ready, 'loading': not request.app['engine'].ready, 'busy': request.app['state']['busy'],
        'service': 'rhine-sample-x', **request.app['engine'].runtime})


async def artifact(request):
    run, name = request.match_info['run'], request.match_info['name']
    if not re.fullmatch('[0-9a-f]{32}', run) or name not in {'input.wav', 'summary.json', 'events.jsonl'}:
        raise web.HTTPNotFound()
    path = ROOT / 'runs' / run / name
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={'Content-Disposition': f'attachment; filename="{run}-{name}"'})


async def recognize(request):
    ws = web.WebSocketResponse(max_msg_size=32768, heartbeat=20)
    await ws.prepare(request)
    if request.app['state']['busy']:
        await ws.send_json({'type': 'error', 'message': '样品-X正在识别，请先结束当前会话'})
        await ws.close()
        return ws
    request.app['state']['busy'] = True
    session = receive = None
    claimed = False
    try:
        await admission('claim')
        claimed = True
        config = await asyncio.wait_for(ws.receive_json(), 10)
        interval = float(config.get('interval', .8))
        if config.get('type') != 'start' or config.get('version') != 'a' or config.get('source') not in {'browser', 'file'}:
            raise ValueError('本档案只支持 A 原前端流程')
        if interval not in {.5, .8, 1., 1.28, 2.}:
            raise ValueError('无效的文字更新间隔')
        runtime = await request.app['engine'].configure(config.get('backend', 'auto'))
        session = LiveSession('a', interval, None, ws.send_json, config,
                              decoder=request.app['engine'].decode, runtime=runtime)
        await ws.send_json({'type': 'ready', 'run_id': session.id, 'version': 'a', 'sample_rate': 16000, 'runtime': runtime})
        receive = asyncio.create_task(ws.receive())
        while True:
            finished, _ = await asyncio.wait({receive, session.work}, timeout=.3, return_when=asyncio.FIRST_COMPLETED)
            if session.work in finished:
                await session.work
                raise RuntimeError('识别任务提前结束')
            if receive in finished:
                message = receive.result()
                if message.type == WSMsgType.BINARY:
                    session.push(message.data)
                elif message.type == WSMsgType.TEXT:
                    command = json.loads(message.data)
                    if command.get('type') != 'finish':
                        raise ValueError('只支持 finish 命令')
                    if int(command.get('samples_sent', -1)) != session.received:
                        raise ValueError('发送与接收的采样数量不一致')
                    report = await session.finish()
                    await ws.send_json({'type': 'done', 'report': report})
                    break
                else:
                    break
                receive = asyncio.create_task(ws.receive())
            await ws.send_json(session.meter())
    except Exception as exc:
        logging.exception('A session failed')
        if session:
            session.error = str(exc)
        if not ws.closed:
            try:
                await ws.send_json({'type': 'error', 'message': str(exc)})
            except ConnectionError:
                pass
    finally:
        if receive and not receive.done():
            receive.cancel()
        if session and not session.closed:
            await session.abort()
        if claimed:
            try:
                await admission('release')
            except Exception:
                logging.exception('Admission release failed')
        request.app['state']['busy'] = False
        await ws.close()
    return ws


async def main():
    # The child exits gracefully with its owning backend, including model process.
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    parent = kernel.OpenProcess(0x00100000, False, int(os.environ['SAMPLEX_PARENT']))
    if not parent:
        raise RuntimeError('Parent backend is unavailable')
    engine = ModelPool()
    runner = None
    try:
        await engine.configure('auto')
        app = web.Application(middlewares=[local_only], client_max_size=32768)
        app.update(engine=engine, state={'busy': False})
        app.router.add_get('/health', health)
        app.router.add_get('/ws/asr', recognize)
        app.router.add_get('/runs/{run}/{name}', artifact)
        app.router.add_get('/sample.wav', lambda _: web.FileResponse(ROOT / 'fixtures/asr_zh.wav'))
        runner = web.AppRunner(app, shutdown_timeout=5, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', PORT).start()
        while kernel.WaitForSingleObject(parent, 0) == 258:
            await asyncio.sleep(1)
    finally:
        if runner:
            await runner.cleanup()
        engine.close()
        kernel.CloseHandle(parent)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
