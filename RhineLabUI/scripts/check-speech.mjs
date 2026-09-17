import assert from 'node:assert/strict';
import { test } from 'node:test';
import { speechArchives, speechColumns } from '../src/speech-catalog.ts';
import { records, columnFiles, fileLocation, archiveRowPeriod } from '../src/data.ts';
import { fileAtCell } from '../src/archive-loop.ts';

test('every model/mode capability has its own stable archive', () => {
  assert.deepEqual(speechArchives.map(r => r.id), [1, 2, 3, 5, 6, 7, 8, 9, 10].map(i => `X-${String(i).padStart(3, '0')}`));
  assert.equal(new Set(speechArchives.map(r => `${r.feature}/${r.model}`)).size, 9);
  assert.equal(speechArchives.filter(r => r.feature === 'record').length, 3);
  assert.equal(speechArchives.filter(r => r.feature === 'live').length, 3);
  assert.deepEqual(speechArchives.filter(r => r.feature === 'target').map(r => r.model), ['sensevoice-realtime']);
  assert.deepEqual(speechArchives.filter(r => r.feature === 'meeting').map(r => r.model), ['moss-transcribe-diarize']);
  assert.equal(speechArchives.filter(r => r.feature === 'voice').length, 1);
});
test('all four columns remain browsable with independent lengths', () => {
  assert.equal(speechColumns.length, 4);
  assert.deepEqual(speechColumns.map((_, lane) => columnFiles(lane).length), [3, 2, 3, 1]);
  records.forEach((record, index) => {
    const { lane, row } = fileLocation(index);
    assert.equal(fileAtCell({ lane, row }), index);
    assert.equal(fileAtCell({ lane: lane + speechColumns.length, row: row + archiveRowPeriod }), index);
    assert.equal(fileAtCell({ lane: lane - speechColumns.length, row: row - archiveRowPeriod }), index);
  });
});
test('voice enrollment shares the target speaker lane and remains an independent archive', () => {
  const lane = speechColumns.indexOf('目标说话人');
  assert.deepEqual(columnFiles(lane).map(index => records[index].id), ['X-007', 'X-009']);
  assert.equal(speechColumns.includes('声纹管理'), false);
  const targetIndex = records.findIndex(record => record.id === 'X-007');
  const voiceIndex = records.findIndex(record => record.id === 'X-009');
  const targetCell = fileLocation(targetIndex);
  assert.equal(fileAtCell({lane, row: targetCell.row + 1}), voiceIndex);
  assert.equal(fileAtCell({lane, row: targetCell.row + 2}), targetIndex);
  assert.equal(speechArchives[voiceIndex].feature, 'voice');
});
test('all archives exclude Paraformer while remaining IDs stay stable', () => {
  assert.deepEqual(speechArchives.filter(r => r.feature === 'record').map(r => r.model),
    ['sensevoice-small', 'fun-asr-nano', 'qwen3-asr']);
  assert.deepEqual(speechArchives.filter(r => /paraformer/i.test(r.model || '')), []);
});
test('long drags never point outside the feature catalog', () => {
  for (let lane = -8; lane < 9; lane++) for (let row = -40; row < 50; row++) {
    const index = fileAtCell({ lane, row });
    assert.ok(records[index], `invalid cell ${lane}/${row}`);
  }
});
