import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
const source = fs.readFileSync(new URL('../src/sample-x.ts', import.meta.url), 'utf8');
const js = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText;
const { decodeSampleXWav } = await import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'));

function wav(samples) {
  const bytes = new ArrayBuffer(44 + samples.length * 2), view = new DataView(bytes);
  const tag = (offset, text) => [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  tag(0, 'RIFF'); view.setUint32(4, bytes.byteLength - 8, true); tag(8, 'WAVE'); tag(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  tag(36, 'data'); view.setUint32(40, samples.length * 2, true);
  samples.forEach((x, i) => view.setInt16(44 + 2 * i, x, true)); return bytes;
}
test('A keeps compatible WAV PCM bit-for-bit, including tails', async () => {
  const values = [-32768, -1234, 0, 1234, 32767];
  assert.deepEqual([...await decodeSampleXWav(wav(values))], values);
});
test('empty and malformed WAV are rejected before capture', async () => {
  await assert.rejects(decodeSampleXWav(wav([])), /完整/);
  await assert.rejects(decodeSampleXWav(new ArrayBuffer(8)), /标准/);
  const broken = wav([1]); new DataView(broken).setUint32(40, 4000, true);
  await assert.rejects(decodeSampleXWav(broken), /不完整/);
});
test('AudioWorklet flushes all residual PCM before flushed marker and is silent', () => {
  let Capture; const output = [];
  vm.runInNewContext(fs.readFileSync(new URL('../public/sample-x-capture.js', import.meta.url), 'utf8'), {
    AudioWorkletProcessor: class { port = { postMessage: value => output.push(value) }; },
    registerProcessor: (name, value) => { assert.equal(name, 'capture'); Capture = value; },
    Int16Array, Math,
  });
  const capture = new Capture();
  for (let i = 0; i < 13; i++) capture.process([[new Float32Array(128).fill(.25)]]);
  capture.port.onmessage({ data: 'stop' });
  assert.equal(output[0].byteLength, 3200); assert.equal(output[1].byteLength, 128); assert.equal(output[2], 'flushed');
  assert.equal(new Int16Array(output[1])[0], 8192);
  capture.process([[new Float32Array(128).fill(.5)]]); capture.port.onmessage({ data: 'stop' });
  assert.equal(output.length, 3);
});

function harness(mediaDevices) {
  const nodes = new Map(), muted = [];
  class Element {
    value = ''; textContent = ''; hidden = false; options = [];
    classList = { toggle() {} };
    setAttribute() {} addEventListener() {} pause() {} replaceChildren(...items) { this.options = items; } add(item) { this.options.push(item); }
    querySelector(selector) { if (!nodes.has(selector)) nodes.set(selector, new Element()); return nodes.get(selector); }
  }
  const output = {};
  const compiled = ts.transpileModule(source.replaceAll('import.meta.env.BASE_URL', "'/rhine/'"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  }).outputText;
  const socketType = class { static OPEN = 1; static CLOSING = 2; };
  vm.runInNewContext(compiled, { exports: output, document: { createElement: () => new Element() },
    window: { addEventListener() {} }, navigator: { mediaDevices }, WebSocket: socketType, setTimeout, clearTimeout,
    Option: class { constructor(text, value) { this.text = text; this.value = value; } }, console });
  const controls = new output.SampleXArchiveControls(value => muted.push(value), () => {});
  nodes.get('#sample-x-source').value = 'microphone';
  return { controls, nodes, muted };
}
test('cancel while awaiting browser permission closes late tracks and restores sound', async () => {
  let resolve, stopped = 0, constraints;
  const promise = new Promise(r => { resolve = r; });
  const { controls, muted } = harness({ getUserMedia: args => { constraints = args; return promise; } });
  const begin = controls.begin();
  assert.equal(constraints.audio.echoCancellation, true);
  assert.equal(constraints.audio.noiseSuppression, true);
  assert.equal(constraints.audio.autoGainControl, true);
  controls.stop(controls.active);
  resolve({ getTracks: () => [{ stop: () => stopped++ }] }); await begin;
  assert.equal(controls.active, undefined); assert.equal(stopped, 1); assert.deepEqual(muted, [true, false]);
});
test('shared screen without audio is rejected and video capture is released', async () => {
  let stopped = 0, constraints;
  const { controls, nodes } = harness({ getDisplayMedia: async args => { constraints = args; return {
    getAudioTracks: () => [], getTracks: () => [{ stop: () => stopped++ }],
  }; } });
  nodes.get('#sample-x-source').value = 'computer'; await controls.begin();
  assert.equal(constraints.systemAudio, 'include'); assert.equal(constraints.audio.suppressLocalAudioPlayback, false);
  assert.equal(controls.active, undefined); assert.equal(stopped, 1);
  assert.match(nodes.get('#sample-x-error').textContent, /共享音频/);
});
test('denied microphone permission does not leave a running session', async () => {
  const { controls, nodes, muted } = harness({ getUserMedia: async () => { throw Error('Permission denied'); } });
  await controls.begin();
  assert.equal(controls.active, undefined); assert.equal(nodes.get('#sample-x-error').hidden, false);
  assert.equal(muted.at(-1), false);
});

test('actual backend and CPU fallback are visible; selector locks during capture', () => {
  const { controls, nodes } = harness({});
  controls.connected = true;
  controls.health = {ready: true, loading: false, busy: false, backend_id:'cuda-hybrid', backend:'CUDA FP32 + MNN CPU', device:'GPU'};
  controls.render();
  assert.match(nodes.get('#sample-x-connection').textContent, /CUDA \+ CPU/);
  assert.equal(nodes.get('#sample-x-fallback').hidden, true);
  controls.health = {...controls.health, backend_id:'cpu', backend:'MNN CPU', device:'CPU', fallback_reason:'显存不足'};
  controls.active = {ready: true}; controls.render();
  assert.match(nodes.get('#sample-x-connection').textContent, /LOCAL CPU/);
  assert.match(nodes.get('#sample-x-fallback').textContent, /显存不足/);
  assert.equal(nodes.get('#sample-x-fallback').hidden, false);
  assert.equal(nodes.get('#sample-x-backend').disabled, true);
});

test('long transcript follows latest output but preserves deliberate scrollback', () => {
  const { controls, nodes } = harness({});
  const box = controls.el('transcript');
  box.scrollHeight = 600; box.clientHeight = 200; box.scrollTop = 400;
  controls.draw({ finals: new Map([[0, 'first']]), partial: 'draft' });
  assert.equal(box.scrollTop, 600);
  box.scrollHeight = 900; box.scrollTop = 50;
  controls.draw({ finals: new Map([[0, 'first'], [1, 'second']]), partial: '' });
  assert.equal(box.scrollTop, 50);
  assert.equal(nodes.get('#sample-x-final').textContent, 'first\nsecond');
});
