"""Windows CPU helpers for the locally decoded model (experimental)."""
from pathlib import Path
import sys, time
import numpy as np
import MNN

ROOT=Path(__file__).resolve().parent
MODEL_ROOT=Path(__file__).resolve().parents[2]/'models'/'sample-x'
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')

class Session:
    def __init__(self,path,threads=4):
        self.model=MNN.Interpreter(str(path))
        self.session=self.model.createSession({'backend':'CPU','numThread':threads,'precision':'high'})
        self.inputs=self.model.getSessionInputAll(self.session)
    def __call__(self,**values):
        changed=False
        for name,value in values.items():
            value=np.ascontiguousarray(value)
            if tuple(self.inputs[name].getShape())!=value.shape:
                self.model.resizeTensor(self.inputs[name],value.shape);changed=True
        if changed:self.model.resizeSession(self.session)
        for name,value in values.items():
            value=np.ascontiguousarray(value)
            tensor=MNN.Tensor(value.shape,MNN.Halide_Type_Float if value.dtype==np.float32 else MNN.Halide_Type_Int,value,MNN.Tensor_DimensionType_Caffe)
            self.inputs[name].copyFrom(tensor)
        code=self.model.runSession(self.session)
        if code:raise RuntimeError(f'MNN runSession failed: {code}')
        return {k:v.getNumpyData().copy() for k,v in self.model.getSessionOutputAll(self.session).items()}

def embedding_table():
    cache=MODEL_ROOT/'portable-models/embedding.npy'
    if cache.exists():return np.load(cache,mmap_mode='r')
    model=Session(MODEL_ROOT/'portable-models/stream/logit.mnn')
    zero=model(hidden_state=np.zeros((1,1,768),np.float32))['lm_logits'][0,0]
    table=np.empty((64000,768),np.float32)
    identity=np.eye(768,dtype=np.float32)
    for start in range(0,768,32):
        output=model(hidden_state=identity[None,start:start+32])['lm_logits'][0]
        table[:,start:start+32]=(output-zero).T
        print('embedding',start+32,flush=True)
    np.save(cache,table)
    return table

class Tokenizer:
    def __init__(self):
        path=MODEL_ROOT/'decoded/asr/token'
        lines=(path/'vocab.txt').read_text(encoding='utf8').splitlines()
        self.vocab={lines[i]:int(lines[i+1]) for i in range(0,len(lines),2)}
        self.inverse={v:k for k,v in self.vocab.items()}
        special=(path/'add_special_tokens.txt').read_text(encoding='utf8').splitlines()
        # Added tokens are literal Unicode, including Chinese numerals 17..29.
        # They do not use the GPT-2 byte alphabet used by ordinary BPE tokens.
        self.special={int(special[i+1]):special[i] for i in range(0,len(special),2)}
        bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
        for b in range(256):
            if b not in bs:bs.append(b);cs.append(256+n);n+=1
        self.enc=dict(zip(bs,map(chr,cs)));self.dec={v:k for k,v in self.enc.items()}
        merges=(path/'merges.txt').read_text(encoding='utf8').splitlines()
        self.ranks={tuple(s.split()):i for i,s in enumerate(merges) if len(s.split())==2}
    def encode(self,text):
        word=tuple(self.enc[b] for b in text.encode())
        while len(word)>1:
            pairs=set(zip(word,word[1:]));pair=min(pairs,key=lambda p:self.ranks.get(p,10**12))
            if pair not in self.ranks:break
            result=[];i=0
            while i<len(word):
                if i+1<len(word) and (word[i],word[i+1])==pair:result.append(word[i]+word[i+1]);i+=2
                else:result.append(word[i]);i+=1
            word=tuple(result)
        return [self.vocab[w] for w in word]
    def decode(self,ids):
        chunks=[];pending=bytearray()
        for token in map(int,ids):
            if token in self.special:
                if pending:
                    chunks.append(pending.decode('utf8','replace'));pending.clear()
                chunks.append(self.special[token])
            else:
                pending.extend(self.dec[c] for c in self.inverse[token])
        if pending:chunks.append(pending.decode('utf8','replace'))
        return ''.join(chunks)

if __name__=='__main__':
    t=time.perf_counter();e=embedding_table();print(e.shape,e.min(),e.max(),time.perf_counter()-t)
    tok=Tokenizer();print('prompt',tok.encode('<audio>'),tok.encode('</audio>'))
