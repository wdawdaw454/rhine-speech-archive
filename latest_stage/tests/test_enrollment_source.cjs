const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../web/dictation/app.js"), "utf8");

function harness(input, state = "idle", confirm = true) {
  const requests = [], messages = [];
  const speaker = Object.fromEntries(["target-load", "speaker-name", "enroll-file", "speaker-threshold",
    "enroll-mic", "enroll-source-hint", "enroll-upload", "speaker-forget"].map(id => [id, {
      value: "目标", files: [], addEventListener(event, handler) { this.click = handler; },
    }]));
  const elements = {source: {value: input}, recognitionType: {}, recordingAudio: {
    paused: false, pause() { this.paused = true; },
  }};
  const context = vm.createContext({speaker, elements, currentState: state, pendingAction: false,
    status: {state, target_ready: true}, latestStatus: {speaker_profile: {name: "existing"}},
    active: state !== "idle", window: {confirm: () => confirm},
    toast: message => messages.push(message), speakerAction: (...args) => requests.push(args)});
  vm.runInContext(source.slice(source.indexOf('  speaker["target-load"].disabled'),
    source.indexOf('  speaker["speaker-forget"].disabled')), context);
  vm.runInContext(source.slice(source.indexOf('speaker["enroll-mic"].addEventListener'),
    source.indexOf('speaker["enroll-upload"].addEventListener')), context);
  return {speaker, elements, requests, messages};
}

for (const input of ["system", "microphone"]) {
  test(`registration submits selected ${input} source and pauses old playback`, () => {
    const h = harness(input);
    assert.equal(h.speaker["enroll-mic"].disabled, false);
    assert.match(h.speaker["enroll-source-hint"].textContent, input === "system" ? /电脑音频/ : /默认麦克风/);
    h.speaker["enroll-mic"].click();
    assert.equal(h.requests[0][0], "/api/target/enroll-recording");
    assert.deepEqual(JSON.parse(h.requests[0][1].body), {name: "目标", source: input});
    assert.equal(h.elements.recordingAudio.paused, true);
  });
}

test("WAV selection disables live registration and never falls back to microphone", () => {
  const h = harness("wav");
  assert.equal(h.speaker["enroll-mic"].disabled, true);
  assert.match(h.speaker["enroll-source-hint"].textContent, /上传注册/);
  h.speaker["enroll-mic"].click();
  assert.equal(h.requests.length, 0);
  assert.match(h.messages[0], /上传注册/);
});

test("stop enrollment uses stop endpoint, and extraction disables the button", () => {
  const h = harness("system", "enroll_recording");
  assert.equal(h.speaker["enroll-mic"].disabled, false);
  assert.equal(h.speaker["enroll-mic"].textContent, "停止并注册");
  h.speaker["enroll-mic"].click();
  assert.deepEqual(h.requests, [["/api/stop"]]);
  assert.equal(harness("system", "enrolling").speaker["enroll-mic"].disabled, true);
});

test("declining replacement does not record or alter playback", () => {
  const h = harness("system", "idle", false);
  h.speaker["enroll-mic"].click();
  assert.equal(h.requests.length, 0);
  assert.equal(h.elements.recordingAudio.paused, false);
});
