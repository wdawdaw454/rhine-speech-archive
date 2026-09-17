"""Deterministic failure-injection tests; no GPU/environment mutations."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import AsyncMock, Mock, patch
import unittest
import numpy as np
import engine_worker as worker
from model_pool import ModelPool


class WorkerTests(unittest.TestCase):
    def tearDown(self):
        worker.asr, worker.runtime = None, {}

    def test_initialization_failure_warms_cpu_and_exposes_reason(self):
        cpu = Mock()
        with patch.object(worker, 'cuda_model', side_effect=RuntimeError('no GPU')), patch.object(worker, 'cpu_model', return_value=cpu):
            worker.initialize()
        self.assertEqual(worker.ready()['backend_id'], 'cpu')
        self.assertIn('no GPU', worker.ready()['fallback_reason'])
        cpu.transcribe.assert_called_once()

    def test_explicit_cpu_never_imports_or_initializes_cuda(self):
        with patch.object(worker, 'cuda_model') as cuda, patch.object(worker, 'cpu_model', return_value=Mock()):
            worker.initialize('cpu')
        cuda.assert_not_called()
        self.assertEqual(worker.ready()['requested_backend'], 'cpu')
        self.assertFalse(worker.ready()['fallback_reason'])

    def test_cuda_failure_retries_identical_pcm_once_then_stays_cpu(self):
        audio = np.arange(512, dtype='f4')
        gpu = Mock(); gpu.transcribe.side_effect = RuntimeError('out of memory')
        cpu = Mock(); cpu.transcribe.return_value = {'text': 'same audio', 'processing_seconds': .01}
        worker.asr = gpu; worker.runtime = {'backend_id': 'cuda-hybrid'}
        with patch.object(worker, 'cpu_model', return_value=cpu):
            result = worker.decode(audio)
            second = worker.decode(audio)
        self.assertIs(cpu.transcribe.call_args_list[0].args[0], audio)
        self.assertEqual(gpu.transcribe.call_count, 1)
        self.assertEqual(cpu.transcribe.call_count, 2)
        self.assertEqual(result['runtime']['backend_id'], 'cpu')
        self.assertIn('out of memory', second['runtime']['fallback_reason'])

    def test_cpu_failure_propagates_instead_of_fake_completion(self):
        worker.asr = Mock(); worker.asr.transcribe.side_effect = RuntimeError('CPU failed')
        worker.runtime = {'backend_id': 'cpu'}
        with self.assertRaisesRegex(RuntimeError, 'CPU failed'):
            worker.decode(np.zeros(512, 'f4'))


class PoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_preference_does_not_retry_failed_cuda_each_utterance(self):
        manager = ModelPool()
        manager.pool = Mock(); manager.preference = 'auto'
        manager.ready = True
        manager.runtime = {'backend_id': 'cpu', 'fallback_reason': 'previous failure'}
        with patch.object(manager, '_open', new_callable=AsyncMock) as open_pool:
            result = await manager.configure('auto')
        open_pool.assert_not_awaited()
        self.assertEqual(result['backend_id'], 'cpu')

    async def test_dead_cuda_worker_restarts_cpu_and_retries_same_pcm(self):
        manager = ModelPool(); manager.preference = 'auto'
        manager.runtime = {'backend_id': 'cuda-hybrid'}
        audio = np.arange(512, dtype='f4')
        result = {'text': 'recovered', 'processing_seconds': .1, 'runtime': {'backend_id': 'cpu'}}
        with ThreadPoolExecutor(1) as executor:
            manager.pool = executor
            async def reopen(preference, reason):
                self.assertEqual(preference, 'auto')
                self.assertIn('CPU', reason)
                manager.runtime = {'backend_id': 'cpu'}
            with patch('model_pool.decode', side_effect=[BrokenProcessPool('dead'), result]) as decode, patch.object(manager, '_open', side_effect=reopen):
                actual = await manager.decode(audio)
            self.assertEqual(decode.call_count, 2)
            self.assertIs(decode.call_args_list[0].args[0], decode.call_args_list[1].args[0])
        self.assertEqual(actual['text'], 'recovered')
        self.assertEqual(manager.runtime['backend_id'], 'cpu')

    async def test_unknown_backend_rejected_before_pool_creation(self):
        manager = ModelPool()
        with self.assertRaises(ValueError):
            await manager.configure('untrusted')
        self.assertIsNone(manager.pool)


if __name__ == '__main__':
    unittest.main(verbosity=2)
