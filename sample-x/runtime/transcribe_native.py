"""Local Windows CPU transcription using the downloaded SampleX model.

No Android runtime, network access, microphone capture, or remote service.
The Python API retains loaded models across short utterances.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import wave

from native_runtime import ROOT, Session, Tokenizer, embedding_table, np
import kaldi_native_fbank as knf

MODEL_SHA256='3a46e175f322822f368d970580271888383cc571996776754a17af669b48867d'

def read_audio(source: Path) -> np.ndarray:
    def read(path):
        with wave.open(str(path),'rb') as f:
            if (f.getnchannels(),f.getsampwidth(),f.getframerate(),f.getcomptype())!=(1,2,16000,'NONE'):
                raise ValueError('Audio requires PCM16 mono 16 kHz')
            return np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)
    if not source.is_file():raise FileNotFoundError(source)
    try:return read(source)
    except (wave.Error,ValueError):
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:raise ValueError('Install FFmpeg or supply PCM16 mono 16 kHz WAV')
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='llf-native-',dir=ROOT/'tmp') as temp:
            output=Path(temp)/'audio.wav'
            subprocess.run([ffmpeg,'-nostdin','-v','error','-i',str(source),'-ac','1','-ar','16000','-c:a','pcm_s16le',str(output)],check=True,capture_output=True)
            return read(output)

class NativeRecognizer:
    """Single-user short-utterance recognizer. Input is PCM16-scale float audio.

    A call is a complete utterance, not a continuation of the previous call.
    The model's original non-stream decoder is used with full causal attention.
    """
    def __init__(self,threads=4,max_tokens=200):
        start=time.perf_counter()
        self.threads=threads;self.max_tokens=max_tokens
        self.embeddings=embedding_table();self.tokenizer=Tokenizer()
        self.encoder=Session(ROOT/'portable-models/stream/encoder.mnn',threads)
        self.decoder=Session(ROOT/'portable-models/no_stream/decoder_full.mnn',threads)
        self.head=Session(ROOT/'portable-models/stream/logit.mnn',threads)
        self.cmvn=np.loadtxt(ROOT/'decoded/asr/feature_extractor/cmvn.txt').astype(np.float32)
        self.prefix=self.embeddings[self.tokenizer.encode('<audio>')][None]
        self.suffix=self.embeddings[self.tokenizer.encode('</audio>')][None]
        self.load_seconds=time.perf_counter()-start

    def transcribe(self,pcm):
        pcm=np.asarray(pcm,dtype=np.float32)
        if pcm.ndim!=1 or not np.isfinite(pcm).all():raise ValueError('Expected finite mono audio samples')
        if len(pcm)>320000:raise ValueError('Native entry currently supports utterances up to 20 seconds')
        duration=len(pcm)/16000
        start=time.perf_counter()
        metrics={'feature_seconds':0.,'encoder_seconds':0.,'decode_seconds':0.}
        # Suppress digital silence, not a replacement for the original neural VAD.
        if len(pcm)<400 or float(np.sqrt(np.mean(pcm*pcm)))<8:
            return {'text':'','tokens':[],'audio_seconds':duration,'processing_seconds':time.perf_counter()-start,'silence_skipped':True,**metrics}
        opts=knf.FbankOptions();opts.frame_opts.dither=0;opts.frame_opts.snip_edges=True;opts.mel_opts.num_bins=80
        fb=knf.OnlineFbank(opts);fb.accept_waveform(16000,pcm.tolist());fb.input_finished()
        feat=np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)])
        feat=((feat-self.cmvn[:,0])*self.cmvn[:,1]).astype(np.float32)
        feat=np.pad(feat,((0,(-len(feat))%32),(0,0)))
        metrics['feature_seconds']=time.perf_counter()-start
        mark=time.perf_counter()
        audio=self.encoder(fbank=feat[None],mask=np.ones((1,len(feat)),np.int32))['encoder_out']
        metrics['encoder_seconds']=time.perf_counter()-mark
        context=np.concatenate([self.prefix,audio,self.suffix],axis=1)
        ids=[];mark=time.perf_counter();first_token=None
        for step in range(self.max_tokens):
            n=context.shape[1]
            hidden=self.decoder(input_embedding=context,position_ids=np.arange(n,dtype=np.int32)[None],mask=np.triu(np.full((1,1,n,n),-1e9,np.float32),1))['last_hidden_state']
            logits=self.head(hidden_state=hidden)['lm_logits'].reshape(-1)
            if not np.isfinite(logits).all():raise RuntimeError('Non-finite decoder output')
            token=int(logits.argmax())
            if first_token is None:first_token=time.perf_counter()-start
            if token==2:break
            ids.append(token)
            context=np.concatenate([context,self.embeddings[[token]][None]],axis=1)
        else:raise RuntimeError(f'Decoder did not finish within {self.max_tokens} tokens')
        metrics['decode_seconds']=time.perf_counter()-mark
        return {'text':self.tokenizer.decode(ids),'tokens':ids,'audio_seconds':duration,'processing_seconds':time.perf_counter()-start,'first_token_seconds':first_token,'silence_skipped':False,**metrics}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio',type=Path);parser.add_argument('-o','--output',type=Path)
    parser.add_argument('--threads',type=int,default=4,choices=range(1,25))
    parser.add_argument('--repeat',type=int,default=1,help='Repeat in one process to measure warm performance')
    parser.add_argument('--paced',action='store_true',help='Simulate real-time file arrival before utterance decoding')
    args=parser.parse_args()
    if args.repeat<1:parser.error('--repeat must be positive')
    wall=time.perf_counter();pcm=read_audio(args.audio)
    recognizer=NativeRecognizer(threads=args.threads)
    records=[]
    for iteration in range(args.repeat):
        begin=time.perf_counter();pacing=0.
        if args.paced:
            time.sleep(len(pcm)/16000);pacing=time.perf_counter()-begin
        result=recognizer.transcribe(pcm)
        result.update(iteration=iteration+1,input_pacing_seconds=pacing,arrival_to_final_seconds=time.perf_counter()-begin)
        result['real_time_factor']=result['processing_seconds']/result['audio_seconds'] if result['audio_seconds'] else 0.
        records.append(result)
    report={'backend':'Windows CPU / MNN 3.6.1','model':'Sample-X_v3.2.1','model_sha256':MODEL_SHA256,'decoder':'non_stream, full causal context','threads':args.threads,'source':str(args.audio.resolve()),'load_seconds':recognizer.load_seconds,'wall_seconds':time.perf_counter()-wall,'runs':records,'text':records[-1]['text']}
    output=args.output or ROOT/'results'/f'{args.audio.stem}-native.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
