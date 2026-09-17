"""Model-process lifecycle and recovery, independent of HTTP transport."""
import asyncio
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import logging
import multiprocessing
import time
from engine_worker import initialize, ready, decode

class ModelPool:
    def __init__(self):
        self.pool = None
        self.preference = None
        self.runtime = {}
        self.ready = False

    async def _open(self, preference, reason=''):
        self.ready = False
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        # Use the sidecar's own interpreter. Cross-venv set_executable would
        # carry the parent's sys.path into a different environment on spawn.
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                                        initializer=initialize, initargs=(preference, reason))
        self.runtime = await asyncio.get_running_loop().run_in_executor(self.pool, ready)
        self.ready = True

    async def configure(self, preference='auto'):
        if preference not in {'auto', 'cpu'}:
            raise ValueError('无效的样品-X后端选择')
        if self.ready and self.pool is not None and self.preference == preference:
            return dict(self.runtime)
        try:
            await self._open(preference)
            if preference == 'auto' and self.runtime.get('fallback_reason'):
                # Drop the failed CUDA interpreter too, so driver allocations
                # cannot linger while the fallback continues serving on CPU.
                await self._open(preference, self.runtime['fallback_reason'])
        except (BrokenProcessPool, OSError) as exc:
            if preference != 'auto':
                raise
            logging.exception('CUDA model process failed during startup')
            await self._open(preference, f'CUDA 模型进程启动失败，使用 CPU：{type(exc).__name__}')
        self.preference = preference
        return dict(self.runtime)

    async def decode(self, pcm):
        start = time.perf_counter()
        try:
            result = await asyncio.get_running_loop().run_in_executor(self.pool, decode, pcm)
        except BrokenProcessPool:
            self.ready = False
            if self.runtime.get('backend_id') != 'cuda-hybrid':
                raise
            logging.exception('CUDA model process exited; retrying this audio on CPU')
            await self._open(self.preference, 'CUDA 模型进程异常退出，已用新 CPU 工作进程重算')
            result = await asyncio.get_running_loop().run_in_executor(self.pool, decode, pcm)
            result['processing_seconds'] = time.perf_counter() - start
        if self.runtime.get('backend_id') == 'cuda-hybrid' and result['runtime'].get('backend_id') == 'cpu':
            await self._open(self.preference, result['runtime']['fallback_reason'])
            result['processing_seconds'] = time.perf_counter() - start
            result['runtime'] = dict(self.runtime)
        self.runtime = dict(result['runtime'])
        return result

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
