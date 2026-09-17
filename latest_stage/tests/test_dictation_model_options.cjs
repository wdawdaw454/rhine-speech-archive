const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../web/dictation/app.js'), 'utf8');

test('legacy selector discards retired selection and never offers Paraformer in either mode', () => {
  const select = (value) => ({value, options: [],
    replaceChildren() { this.options = []; this.value = ''; },
    append(option) { this.options.push(option); },
  });
  const elements = {mode: {value: 'streaming'}, model: select('paraformer-zh-streaming'),
    recognitionType: select('normal'), modeDescription: {}};
  const models = [
    {id: 'sensevoice-realtime', modes: ['streaming']},
    {id: 'sensevoice-small', modes: ['offline']},
    {id: 'fun-asr-nano', modes: ['streaming', 'offline']},
    {id: 'qwen3-asr', modes: ['offline']},
    {id: 'moss-transcribe-diarize', modes: ['offline'], recognition_types: ['meeting']},
  ].map(model => ({name: model.id, available: true, recognition_types: ['normal'], ...model}));
  const context = vm.createContext({elements, models, document: {createElement: () => ({})},
    isTargetMode: () => false, renderModelDescription: () => {}});
  vm.runInContext(source.slice(source.indexOf('function renderRecognitionTypes('),
    source.indexOf('function meetingSpeakerColor(')), context);
  vm.runInContext(source.slice(source.indexOf('function renderModelOptions('),
    source.indexOf('elements.mode.addEventListener(')), context);
  const render = () => vm.runInContext('renderModelOptions()', context);
  render();
  assert.equal(elements.model.value, 'sensevoice-realtime');
  assert.deepEqual(elements.model.options.map(option => option.value), ['sensevoice-realtime', 'fun-asr-nano']);
  elements.mode.value = 'offline';
  render();
  assert.deepEqual(elements.model.options.map(option => option.value),
    ['sensevoice-small', 'fun-asr-nano', 'qwen3-asr']);
  assert.equal(elements.model.value, 'sensevoice-small');
  elements.mode.value = 'streaming';
  render();
  assert.deepEqual(elements.model.options.map(option => option.value), ['sensevoice-realtime', 'fun-asr-nano']);
});
