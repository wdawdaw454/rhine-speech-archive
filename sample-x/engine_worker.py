"""Single-owner model process; CUDA preferred, lossless sticky CPU fallback."""
import gc
import logging
import time
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime'))
asr = None
runtime = {}


def cpu_model():
    from transcribe_native import NativeRecognizer
    return NativeRecognizer(threads=4)


def cuda_model():
    from cuda_runtime import recognizer_class
    return recognizer_class()(threads=4)


def use_cpu(reason='', requested='auto'):
    global asr, runtime
    asr = None
    gc.collect()
    # Do not require another CUDA call: a device fault may poison its context.
    asr = cpu_model()
    runtime = {'model': 'Sample-X_v3.2.1', 'backend': 'Windows CPU / MNN 3.6.1',
               'backend_id': 'cpu', 'requested_backend': requested, 'threads': 4,
               'device': 'CPU', 'fallback_reason': reason}


def initialize(preference='auto', fallback_reason=''):
    global asr, runtime
    from transcribe_native import read_audio
    if preference not in {'auto', 'cpu'}:
        raise ValueError('Unsupported backend preference')
    if preference == 'auto' and not fallback_reason:
        failure = ''
        try:
            asr = cuda_model()
            asr.transcribe(read_audio(ROOT / 'fixtures/asr_zh.wav'))
            import torch
            runtime = {'model': 'Sample-X_v3.2.1', 'backend': 'CUDA FP32 decoder + MNN CPU encoder',
                       'backend_id': 'cuda-hybrid', 'requested_backend': preference, 'threads': 4,
                       'device': torch.cuda.get_device_name(0), 'fallback_reason': ''}
            return
        except Exception as exc:
            failure = f'CUDA 初始化失败：{type(exc).__name__}: {exc}'
            logging.warning('%s; falling back to CPU', failure)
        use_cpu(failure, preference)
    else:
        use_cpu(fallback_reason, preference)
    asr.transcribe(read_audio(ROOT / 'fixtures/asr_zh.wav'))


def decode(pcm):
    start = time.perf_counter()
    failure = ''
    try:
        result = asr.transcribe(pcm)
    except Exception as exc:
        if runtime.get('backend_id') != 'cuda-hybrid':
            raise
        failure = f'CUDA 推理失败，已改用 CPU 重算：{type(exc).__name__}: {exc}'
    if failure:
        logging.warning('%s', failure)
        use_cpu(failure)
        result = asr.transcribe(pcm)
        result['processing_seconds'] = time.perf_counter() - start
    return {**result, 'runtime': ready()}


def ready():
    return dict(runtime)
