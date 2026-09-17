"""CUDA decoder, same local weights, per-utterance KV cache.

The encoder remains on MNN CPU. Fallback is managed by engine_worker.
IDST weight decoding follows MNN 3.6.1 ConvolutionCommon.cpp (Apache-2.0).
No reference text participates in inference.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent


class Weights:
    def __init__(self, graph, model):
        metadata = json.loads(graph.read_text(encoding='utf8'))
        self.ops = metadata['oplists']
        self.raw = Path(str(model) + '.weight').read_bytes()
        if hashlib.sha256(model.read_bytes()).hexdigest() != metadata['model_sha256'] or hashlib.sha256(self.raw).hexdigest() != metadata['weights_sha256']:
            raise RuntimeError('CUDA model assets differ from the validated architecture')

    def floats(self, offset, size):
        return np.frombuffer(self.raw, '<f4', size // 4, offset).copy()

    def norm(self, op):
        offset, gamma, beta = op['main']['external']
        return self.floats(offset, gamma), self.floats(offset + gamma, beta)

    def linear(self, op):
        main = op['main']
        offset, size, alpha_size, bias_size, _ = main['external']
        quant = main['quanParameter']
        assert quant['type'] == 1 and not quant['has_scaleInt']
        ndim = self.raw[offset]
        dim_dtype = '<i4' if quant['shapeInt32'] else '<u2'
        shape = np.frombuffer(self.raw, dim_dtype, ndim, offset + 1)
        pos = offset + 1 + ndim * np.dtype(dim_dtype).itemsize
        count = self.raw[pos] or 256
        values = np.sort(np.frombuffer(self.raw, 'i1', count, pos + 1))
        pos += 1 + count
        bits = max(1, (count - 1).bit_length())
        length = int(np.prod(shape))
        packed = np.frombuffer(self.raw, 'u1', offset + size - pos, pos)
        indices = np.unpackbits(packed, bitorder='big')[:length * bits].reshape(-1, bits)
        indices = indices @ (1 << np.arange(bits - 1, -1, -1))
        weight = values[indices].astype(np.float32)
        alpha = self.floats(offset + size, alpha_size)
        if quant['readType']:
            alpha = alpha.reshape(-1, 2)
            clamp = quant['aMin'] or -128
            if clamp < 0:
                alpha[:, 0] -= clamp * alpha[:, 1]
            alpha *= quant['quantScale']
            weight = weight.reshape(len(alpha), -1) * alpha[:, 1, None] + alpha[:, 0, None]
        else:
            weight = weight.reshape(len(alpha), -1) * (alpha * quant['quantScale'])[:, None]
        common = main['common']
        assert weight.shape == (common['outputCount'], common['inputCount'])
        return weight, self.floats(offset + size + alpha_size, bias_size)


class CudaDecoder:
    def __init__(self, device='cuda', gelu='tanh'):
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; no silent CPU fallback')
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device, self.gelu = device, gelu
        data = Weights(ROOT / 'runtime/cuda-graphs/decoder.json', ROOT / 'runtime/portable-models/no_stream/decoder_full.mnn')
        convert = lambda pair: tuple(torch.from_numpy(v).to(device) for v in pair)
        norms = [convert(data.norm(o)) for o in data.ops if o['type'] == 'LayerNorm']
        linears = [convert(data.linear(o)) for o in data.ops if o['type'] == 'Convolution']
        assert len(norms) == 25 and len(linears) == 72
        self.layers = [(norms[2*i:2*i+2], linears[6*i:6*i+6]) for i in range(12)]
        self.final_norm = norms[-1]
        trig = []
        for name in ['precompute_gpt2_rope_cos_even', 'precompute_gpt2_rope_sin_even']:
            op = next(o for o in data.ops if o['name'] == name)
            trig.append(torch.from_numpy(data.floats(*op['main']['external']).reshape(op['main']['dims'])).to(device))
        self.cos, self.sin = trig
        self.reset()

    def reset(self):
        self.cache = [None] * 12
        self.length = 0

    @torch.inference_mode()
    def forward(self, x, cached=True):
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        start = self.length if cached else 0
        n = x.shape[1]
        cos = self.cos[start:start+n][None, :, None, :]
        sin = self.sin[start:start+n][None, :, None, :]
        def rope(t):
            a, b = t.reshape(1, n, 12, 64).chunk(2, dim=-1)
            return torch.cat((a*cos-b*sin, b*cos+a*sin), dim=-1).transpose(1, 2)
        for i, (norms, linear) in enumerate(self.layers):
            y = F.layer_norm(x, (768,), *norms[0], eps=1e-5)
            q, k = rope(F.linear(y, *linear[0])), rope(F.linear(y, *linear[1]))
            v = F.linear(y, *linear[2]).reshape(1, n, 12, 64).transpose(1, 2)
            if cached and self.cache[i] is not None:
                k, v = torch.cat((self.cache[i][0], k), dim=2), torch.cat((self.cache[i][1], v), dim=2)
            if cached:
                self.cache[i] = k, v
            # Incremental decoding has one query and all past keys; it must not
            # use PyTorch's upper-left aligned causal mask for unequal lengths.
            assert start == 0 or n == 1
            y = F.scaled_dot_product_attention(q, k, v, is_causal=(start == 0))
            x = x + F.linear(y.transpose(1, 2).reshape(1, n, 768), *linear[3])
            y = F.layer_norm(x, (768,), *norms[1], eps=1e-5)
            x = x + F.linear(F.gelu(F.linear(y, *linear[4]), approximate=self.gelu), *linear[5])
        if cached:
            self.length += n
        return F.layer_norm(x[:, -1:], (768,), *self.final_norm, eps=1e-5)

    def __call__(self, *, input_embedding, position_ids, mask):
        n = input_embedding.shape[1]
        if n != self.length + 1:
            self.reset()
        x = input_embedding if not self.length else input_embedding[:, -1:]
        return {'last_hidden_state': self.forward(x)}


class CudaHead:
    def __init__(self, device='cuda'):
        data = Weights(ROOT / 'runtime/cuda-graphs/logit.json', ROOT / 'runtime/portable-models/stream/logit.mnn')
        op = next(o for o in data.ops if o['type'] == 'Convolution')
        self.weight, self.bias = (torch.from_numpy(v).to(device) for v in data.linear(op))

    @torch.inference_mode()
    def __call__(self, *, hidden_state):
        return {'lm_logits': F.linear(hidden_state, self.weight, self.bias).cpu().numpy()}


def recognizer_class():
    from transcribe_native import NativeRecognizer
    import native_runtime
    import time

    class HybridRecognizer(NativeRecognizer):
        def __init__(self, threads=4, max_tokens=200):
            start = time.perf_counter()
            self.threads, self.max_tokens = threads, max_tokens
            self.embeddings, self.tokenizer = native_runtime.embedding_table(), native_runtime.Tokenizer()
            self.encoder = native_runtime.Session(native_runtime.ROOT / 'portable-models/stream/encoder.mnn', threads)
            self.decoder, self.head = CudaDecoder(), CudaHead()
            self.cmvn = np.loadtxt(native_runtime.ROOT / 'decoded/asr/feature_extractor/cmvn.txt').astype(np.float32)
            self.prefix = self.embeddings[self.tokenizer.encode('<audio>')][None]
            self.suffix = self.embeddings[self.tokenizer.encode('</audio>')][None]
            torch.cuda.synchronize()
            self.load_seconds = time.perf_counter() - start

        def transcribe(self, pcm):
            self.decoder.reset()
            return super().transcribe(pcm)

    return HybridRecognizer
