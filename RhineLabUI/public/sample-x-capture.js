// Same 100 ms PCM framing as the original Qwen frontend.
class Capture extends AudioWorkletProcessor {
  constructor(){super();this.buffer=new Int16Array(1600);this.offset=0;this.active=true;
    this.port.onmessage=e=>{if(e.data==='stop'&&this.active){this.active=false;if(this.offset)this.port.postMessage(this.buffer.slice(0,this.offset).buffer);this.port.postMessage('flushed');}};
  }
  process(inputs){const channels=inputs[0];if(channels?.length&&this.active)for(let i=0;i<channels[0].length;i++){
    let sample=0;for(const channel of channels)sample+=channel[i]/channels.length;
    this.buffer[this.offset++]=Math.round(Math.max(-1,Math.min(1,sample))*32767);
    if(this.offset===1600){this.port.postMessage(this.buffer.buffer);this.buffer=new Int16Array(1600);this.offset=0;}
  }return true;}
}
registerProcessor('capture',Capture);
