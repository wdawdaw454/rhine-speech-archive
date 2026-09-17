"""Windows default-render-endpoint capture, never microphone fallback.

WASAPI loopback records the mix sent to the selected output at session start.
Capture callbacks keep running during slow ASR; a streaming resampler preserves
filter state and flushes its delayed tail on stop.
"""
from queue import Empty, Full, Queue
from threading import Event
import time

import numpy as np


def _library():
    try:
        import pyaudiowpatch
    except ImportError as exc:
        raise RuntimeError("电脑音频需要 Windows 回环组件，请在运行环境安装 PyAudioWPatch") from exc
    return pyaudiowpatch


def system_audio_device():
    """Read endpoint metadata only; does not start recording or change routing."""
    try:
        pa = _library()
        with pa.PyAudio() as manager:
            device = manager.get_default_wasapi_loopback()
            if not device.get("isLoopbackDevice") or device.get("maxInputChannels", 0) < 1:
                raise RuntimeError("默认输出设备不支持 WASAPI 回环")
            return {"available": True, "name": device["name"], "error": None}
    except Exception as exc:
        return {"available": False, "name": "默认电脑音频输出", "error": str(exc)}


class SystemAudioStream:
    def __init__(self, *, sample_rate=16000, blocksize=1600, stop_event=None,
                 duration=None, max_queue_seconds=30.0, stall_timeout=3.0):
        if sample_rate <= 0 or blocksize <= 0 or max_queue_seconds <= 0 or stall_timeout <= 0:
            raise ValueError("invalid system audio capture parameters")
        if duration is not None and duration <= 0:
            raise ValueError("duration must be positive")
        self.sample_rate, self.blocksize = int(sample_rate), int(blocksize)
        self.stop_event = stop_event or Event()
        self.duration = duration
        self.max_queue_seconds, self.stall_timeout = max_queue_seconds, stall_timeout
        self.statuses = []
        self.device_name = "默认电脑音频输出"
        self.native_sample_rate = self.native_channels = None
        self._iterator = None

    def __iter__(self):
        if self._iterator is not None:
            raise RuntimeError("电脑音频会话不能重复打开")
        self._iterator = self._capture()
        return self._iterator

    def close(self):
        if self._iterator is not None:
            self._iterator.close()

    def _capture(self):
        if self.stop_event.is_set():
            return
        pa = _library()
        import soxr

        with pa.PyAudio() as manager:
            try:
                output = manager.get_default_wasapi_device(d_out=True)
                device = manager.get_wasapi_loopback_analogue_by_dict(output)
                if not device.get("isLoopbackDevice") or device.get("maxInputChannels", 0) < 1:
                    raise ValueError("没有可用的默认回环输入")
                self.device_name = device["name"]
                rate = self.native_sample_rate = int(device["defaultSampleRate"])
                channels = self.native_channels = int(device["maxInputChannels"])
                if not 8000 <= rate <= 192000 or not 1 <= channels <= 32:
                    raise ValueError("不支持的输出设备采样率或声道数")
                output_rate = int(output["defaultSampleRate"])
                output_channels = int(output["maxOutputChannels"])
                if not 8000 <= output_rate <= 192000 or not 1 <= output_channels <= 32:
                    raise ValueError("不支持的默认播放设备格式")
            except Exception as exc:
                raise RuntimeError(f"无法获取电脑音频：{exc}；请连接耳机／扬声器并设为默认输出") from exc
            native_block = max(1, round(rate * self.blocksize / self.sample_rate))
            queue = Queue(maxsize=max(1, int(np.ceil(self.max_queue_seconds * rate / native_block))))
            errors = []
            captured = 0
            limit = None if self.duration is None else round(self.duration * rate)
            last_callback = time.monotonic()
            previous_adc_end = None

            def callback(data, frame_count, info, status):
                nonlocal captured, last_callback, previous_adc_end
                if self.stop_event.is_set():
                    return None, pa.paComplete
                try:
                    if status:
                        self.statuses.append(f"loopback-portaudio-status-{status}")
                        raise RuntimeError("电脑音频采集异常或丢帧，请降低负载或重新连接输出设备")
                    adc = info.get("input_buffer_adc_time", 0)
                    if adc and previous_adc_end is not None and adc - previous_adc_end > .5:
                        self.statuses.append("loopback-clock-gap")
                        raise RuntimeError("电脑音频设备时钟中断，请停止后重新开始")
                    if adc:
                        previous_adc_end = adc + frame_count / rate
                    wav = np.frombuffer(data, dtype=np.float32)
                    if wav.size != frame_count * channels or not np.isfinite(wav).all():
                        raise RuntimeError("电脑音频返回了无效采样")
                    # Downmix before resampling; keep each native sample once.
                    mono = wav.reshape(-1, channels).mean(axis=1).astype(np.float32)
                    if limit is not None:
                        mono = mono[:max(0, limit - captured)]
                    if mono.size:
                        queue.put_nowait(mono)
                        captured += mono.size
                    last_callback = time.monotonic()
                    return None, pa.paComplete if limit is not None and captured >= limit else pa.paContinue
                except Full:
                    self.statuses.append("capture-queue-overflow")
                    errors.append(RuntimeError("电脑音频积压超过缓冲上限，已停止，不能保证完整转写"))
                except Exception as exc:
                    errors.append(exc)
                return None, pa.paAbort

            resampler = soxr.ResampleStream(rate, self.sample_rate, 1, dtype="float32", quality="HQ")
            pending = np.empty(0, np.float32)
            consumed = 0
            stream = keepalive = None
            try:
                # Some WASAPI endpoints stop producing loopback packets when
                # no application is rendering. A shared-mode digital-silence
                # stream keeps the SAME endpoint's clock alive, so VAD receives
                # real silence and can finalize after playback stops. It emits
                # no audible signal and never changes volume or device routing.
                def silence(_data, frame_count, _info, _status):
                    return bytes(frame_count * output_channels * 4), (
                        pa.paComplete if self.stop_event.is_set() else pa.paContinue)
                keepalive = manager.open(format=pa.paFloat32, channels=output_channels, rate=output_rate,
                                         frames_per_buffer=max(1, round(output_rate * self.blocksize / self.sample_rate)),
                                         output=True, output_device_index=output["index"], stream_callback=silence)
                stream = manager.open(format=pa.paFloat32, channels=channels, rate=rate,
                                      frames_per_buffer=native_block, input=True,
                                      input_device_index=device["index"], stream_callback=callback)
                while True:
                    if errors:
                        raise errors[0]
                    if self.stop_event.is_set() and queue.empty():
                        break
                    try:
                        native = queue.get(timeout=.10)
                    except Empty:
                        if not stream.is_active():
                            if self.stop_event.is_set() or limit is not None and captured >= limit:
                                break
                            raise RuntimeError("电脑音频输出设备已停止或断开，请重新开始采集")
                        if time.monotonic() - last_callback > self.stall_timeout:
                            raise RuntimeError("电脑音频设备长时间未返回数据，请确认默认输出设备后重新开始")
                        continue
                    pending = np.concatenate((pending, resampler.resample_chunk(native)))
                    while len(pending) >= self.blocksize:
                        block, pending = pending[:self.blocksize], pending[self.blocksize:]
                        yield consumed / self.sample_rate, block
                        consumed += len(block)
                # Flush delayed resampling samples, including short final blocks.
                pending = np.concatenate((pending, resampler.resample_chunk(np.empty(0, np.float32), last=True)))
                while len(pending):
                    block, pending = pending[:self.blocksize], pending[self.blocksize:]
                    yield consumed / self.sample_rate, block
                    consumed += len(block)
            except OSError as exc:
                raise RuntimeError(f"无法采集电脑音频：{exc}；检查默认输出设备，切换设备后需重新开始") from exc
            finally:
                def close_native(native):
                    if native is not None:
                        try:
                            if native.is_active():
                                native.stop_stream()
                        finally:
                            native.close()
                try:
                    close_native(stream)
                finally:
                    close_native(keepalive)
