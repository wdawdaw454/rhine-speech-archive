"""Continuous capture/endpointing with coalesced drafts and lossless final jobs."""
import asyncio
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import uuid
import wave
import numpy as np
from vad import VoiceProbability,Endpoint
from engine_worker import decode

ROOT=Path(__file__).resolve().parent
# Accuracy first: the original A endpoint and complete <=20s utterances.
# Draft scheduling must NEVER change the samples used for a final decode.
A_MAX_SAMPLES=320000

@dataclass
class Job:
    segment:int
    audio:np.ndarray
    kind:str
    reason:str
    start_sample:int
    last_voice_sample:int
    queued:float

class LiveSession:
    def __init__(self,version,interval,pool,emit,metadata=None,decoder=None,runtime=None):
        self.version=version
        self.interval=interval
        self.pool=pool
        self.decoder=decoder
        self.runtime=dict(runtime or {})
        self.emit=emit
        self.vad=VoiceProbability()
        self.endpoint=Endpoint(version)
        self.id=uuid.uuid4().hex
        self.folder=ROOT/'runs'/self.id
        self.folder.mkdir(parents=True)
        self.wav=wave.open(str(self.folder/'input.wav'),'wb')
        self.wav.setparams((1,2,16000,0,'NONE','not compressed'))
        self.log=(self.folder/'events.jsonl').open('w',encoding='utf8')
        self.metadata=metadata or {}
        self.started=time.perf_counter()
        self.audio_zero=None
        self.first_packet=None
        self.previous_packet=None
        self.max_packet_gap=0.
        self.received=0
        self.processed=0
        self.pending=np.zeros(0,np.float32)
        self.preroll=deque(maxlen=8 if version=='a' else 16)
        self.segment=[]
        self.segment_samples=0
        self.segment_start=0
        self.segment_id=0
        self.last_voice_sample=0
        self.next_update=int(interval*16000)
        self.jobs=deque()
        self.wake=asyncio.Event()
        self.finished=False
        self.closed=False
        self.cancelled=False
        self.final_requested=set()
        self.completed=[]
        self.first_text=None
        self.compute_seconds=0.
        self.coalesced=0
        self.hasher=hashlib.sha256()
        self.square_sum=0.
        self.peak=0
        self.clipped=0
        self.error=None
        self.last_meter=0.
        self.last_decoded=None
        self.decode_jobs=[]
        self.draft_interval=interval
        self.next_draft_time=0.
        self.work=asyncio.create_task(self._work())

    async def _send(self,event):
        event.update(run_id=self.id,elapsed_seconds=round(time.perf_counter()-self.started,4))
        self.log.write(json.dumps(event,ensure_ascii=False)+'\n')
        self.log.flush()
        await self.emit(event)

    def push(self,data,stamp=None):
        if self.finished:raise ValueError('输入已结束')
        if not data or len(data)%2 or len(data)>32000:raise ValueError('音频包必须为 2～32000 字节 PCM16')
        stamp=stamp or time.perf_counter()
        if self.first_packet is None:
            self.first_packet=stamp
            self.audio_zero=stamp-len(data)/32000
        if self.previous_packet is not None:self.max_packet_gap=max(self.max_packet_gap,stamp-self.previous_packet)
        self.previous_packet=stamp
        self.wav.writeframesraw(data)
        self.hasher.update(data)
        pcm=np.frombuffer(data,'<i2').astype(np.float32)
        self.received+=len(pcm)
        self.square_sum+=float(np.dot(pcm.astype(np.float64),pcm.astype(np.float64)))
        self.peak=max(self.peak,int(np.max(np.abs(pcm))))
        self.clipped+=int(np.count_nonzero(np.abs(pcm)>=32700))
        self.pending=np.concatenate([self.pending,pcm])
        while len(self.pending)>=512:
            frame,self.pending=self.pending[:512],self.pending[512:]
            self.processed+=512
            probability=self.vad(frame)
            boundary=self.endpoint.push(probability)
            if probability>=0.5:self.last_voice_sample=self.processed
            if not self.segment_samples:
                self.preroll.append(frame.copy())
                if boundary!='start' and not self.endpoint.active:continue
                self.segment=list(self.preroll)
                self.segment_samples=sum(map(len,self.segment))
                self.segment_start=self.processed-self.segment_samples
                self.preroll.clear()
                self.next_update=int(self.interval*16000)
            else:
                self.segment.append(frame.copy())
                self.segment_samples+=512
            if boundary=='end':self._commit('silence')
            elif self.segment_samples>=A_MAX_SAMPLES:self._commit('segment_limit')
            elif self.segment_samples>=self.next_update:
                self.next_update=self.segment_samples+int(self.draft_interval*16000)
                # B keeps the latest draft during trailing silence, then corrects once.
                if self.version=='a' or self.processed-self.last_voice_sample<5120:
                    self._queue('partial','update')

    def meter(self):
        rms=np.sqrt(self.square_sum/max(1,self.received))/32768
        return {'type':'meter','audio_seconds':self.received/16000,'rms_dbfs':round(20*np.log10(max(rms,1e-9)),1),
                'peak_dbfs':round(20*np.log10(max(self.peak/32768,1e-9)),1),
                'clip_percent':round(100*self.clipped/max(1,self.received),3),
                'queue_jobs':len(self.jobs),'speech':self.endpoint.active,'max_packet_gap_ms':round(1000*self.max_packet_gap,1),
                'draft_interval_seconds':round(self.draft_interval,3)}

    def _queue(self,kind,reason):
        if kind=='partial' and any(job.kind=='final' for job in self.jobs):
            # Do not spend a pending final's budget on a newer draft. Capture
            # continues and the final still receives every original sample.
            self.coalesced+=1
            return
        audio=np.concatenate(self.segment)
        if kind=='final' and self.version=='b' and reason=='silence':
            # Keep 160 ms after the last voiced frame; the 2 s endpoint wait is not speech.
            length=max(400,min(len(audio),self.last_voice_sample-self.segment_start+2560))
            audio=audio[:length]
        job=Job(self.segment_id,audio,kind,reason,self.segment_start,
                min(self.last_voice_sample,self.segment_start+len(audio)),time.perf_counter())
        if kind=='final':self.final_requested.add(self.segment_id)
        for previous in list(self.jobs):
            if previous.kind=='partial':
                self.jobs.remove(previous)
                self.coalesced+=1
        self.jobs.append(job)
        self.wake.set()

    def _commit(self,reason):
        if self.segment_samples:self._queue('final',reason)
        self.segment_id+=1
        self.segment=[]
        self.segment_samples=0
        self.preroll.clear()

    async def _work(self):
        try:
            while True:
                if not self.jobs:
                    if self.finished:return
                    self.wake.clear()
                    await self.wake.wait()
                    continue
                # Finals remain in chronological order and always win over
                # speculative work. A completed in-flight draft stays visible.
                job=next((item for item in self.jobs if item.kind=='final'),self.jobs[0])
                delay=self.next_draft_time-time.perf_counter()
                if job.kind=='partial' and delay>0:
                    self.wake.clear()
                    try:await asyncio.wait_for(self.wake.wait(),delay)
                    except asyncio.TimeoutError:pass
                    continue
                self.jobs.remove(job)
                start=time.perf_counter()
                cached=self.last_decoded
                reused=bool(cached and cached[0]==job.segment and np.array_equal(cached[1],job.audio))
                if reused:result=cached[2]
                else:
                    result=await self.decoder(job.audio) if self.decoder else await asyncio.get_running_loop().run_in_executor(self.pool,decode,job.audio)
                    self.compute_seconds+=result['processing_seconds']
                    self.last_decoded=(job.segment,job.audio,result)
                if 'runtime' in result:self.runtime=dict(result['runtime'])
                self.decode_jobs.append({'segment_id':job.segment,'kind':job.kind,'samples':len(job.audio),
                    'processing_ms':0. if reused else round(result['processing_seconds']*1000,2),
                    'queue_ms':round((start-job.queued)*1000,2),'reused':reused})
                if self.cancelled:return
                if not reused:
                    # Rate-limit only speculative updates. Complete utterances
                    # and their boundaries are independent of CPU/GPU speed.
                    self.draft_interval=max(self.interval,min(8.,result['processing_seconds']*1.25))
                self.next_draft_time=start+self.draft_interval
                # FIFO guarantees this is still before the segment's final.
                # Do not hide a completed draft just because a final is queued:
                # doing so creates a long blank interval at segment boundaries.
                text=result['text']
                if result['text'] and self.first_text is None:self.first_text=time.perf_counter()-(self.audio_zero or self.started)
                event={'type':job.kind,'segment_id':job.segment,'text':text,'raw_text':result['text'],'reason':job.reason,
                       'runtime':dict(self.runtime),
                       'overlap_seconds':0.,'overlap_matched':False,'revisions':[],
                       'draft_interval_seconds':round(self.draft_interval,3),
                       'audio_seconds':self.received/16000,'segment_audio_seconds':len(job.audio)/16000,
                       'segment_start_seconds':job.start_sample/16000,
                       'processing_ms':0. if reused else round(result['processing_seconds']*1000,2),
                       'reused':reused,
                       'queue_ms':round((start-job.queued)*1000,2),
                       'capture_to_result_seconds':round(time.perf_counter()-(self.audio_zero or self.started),4)}
                if job.kind=='final':
                    event['voice_end_to_final_seconds']=round(time.perf_counter()-((self.audio_zero or self.started)+job.last_voice_sample/16000),4)
                    self.completed.append(event)
                await self._send(event)
        except Exception as exc:
            self.error=str(exc)
            self.finished=True
            raise

    async def finish(self):
        if not self.finished:
            if self.segment_samples and len(self.pending):
                self.segment.append(self.pending.copy())
                self.segment_samples+=len(self.pending)
            self.pending=np.zeros(0,np.float32)
            self._commit('finish')
            self.finished=True
            self.wake.set()
        await self.work
        return self.save('complete')

    async def abort(self):
        self.cancelled=True
        self.finished=True
        self.jobs.clear()
        self.wake.set()
        # Await a currently running decode before another client can acquire the model.
        try:await self.work
        except Exception:pass
        return self.save('incomplete')

    def save(self,status):
        report={'run_id':self.id,'version':self.version,'status':status,'metadata':self.metadata,
                'model':'Sample-X_v3.2.1','decoder':'complete audio contexts; backend details in runtime/segments',
                'runtime':dict(self.runtime),
                'vad':'Silero 6.2.1 ONNX; original vendor VAD is not used',
                'text':'\n'.join(e['text'] for e in self.completed if e['text']),
                'audio_seconds':self.received/16000,'pcm_sha256':self.hasher.hexdigest(),
                'first_text_seconds':self.first_text,'compute_seconds':self.compute_seconds,
                'wall_seconds':time.perf_counter()-self.started,'coalesced_drafts':self.coalesced,
                'scheduler':'accuracy-first-v3' if self.version=='a' else 'original-b',
                'max_decode_seconds':A_MAX_SAMPLES/16000 if self.version=='a' else 20,
                'decode_jobs':self.decode_jobs,
                'audio':self.meter(),'segments':self.completed,'error':self.error}
        if not self.closed:
            self.wav.close()
            self.log.close()
            self.closed=True
        (self.folder/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        return report
