"""CPU Silero VAD, with endpoint policies kept separate from acoustic scores."""
from collections import deque
from pathlib import Path
import numpy as np
import onnxruntime as ort

class VoiceProbability:
    def __init__(self):
        options=ort.SessionOptions()
        options.intra_op_num_threads=1
        options.inter_op_num_threads=1
        self.model=ort.InferenceSession(str(Path(__file__).with_name('silero_vad.onnx')),
                                       sess_options=options,providers=['CPUExecutionProvider'])
        self.state=np.zeros((2,1,128),np.float32)
        self.context=np.zeros((1,64),np.float32)

    def __call__(self,pcm):
        x=np.concatenate([self.context,pcm[None].astype(np.float32)/32768],axis=1)
        probability,self.state=self.model.run(None,{'input':x,'state':self.state,'sr':np.array(16000,np.int64)})
        self.context=x[:,-64:]
        return float(probability[0,0])

class Endpoint:
    """A reproduces the existing Silero endpoint. B transfers Android time windows.

    B uses Silero probabilities, NOT the vendor's proprietary VAD network.
    Time-window fractions use 32 ms frames, documented as an approximation.
    """
    def __init__(self,version):
        self.version=version
        self.active=False
        self.quiet_start=None
        self.samples=0
        self.start_window=deque(maxlen=15)  # Android config: 480 ms, >=50% voice
        self.end_window=deque(maxlen=65)    # Android config: 2080 ms, >=92.5% silence

    def push(self,probability):
        self.samples+=512
        voiced=probability>=0.5
        if self.version=='a':
            if voiced:self.quiet_start=None
            if voiced and not self.active:
                self.active=True
                return 'start'
            if probability<0.35 and self.active:
                if self.quiet_start is None:self.quiet_start=self.samples
                if self.samples-self.quiet_start>=9600:
                    self.active=False
                    self.quiet_start=None
                    return 'end'
        else:
            self.start_window.append(voiced)
            if not self.active:
                if sum(self.start_window)>=8:
                    self.active=True
                    self.end_window.clear()
                    return 'start'
            else:
                self.end_window.append(not voiced)
                if len(self.end_window)==65 and sum(self.end_window)/65>=0.925:
                    self.active=False
                    self.start_window.clear()
                    self.end_window.clear()
                    return 'end'
        return None
