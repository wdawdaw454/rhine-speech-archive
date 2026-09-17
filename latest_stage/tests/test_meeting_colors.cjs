const assert = require("node:assert/strict");
const { test } = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../web/dictation/app.js"), "utf8");
const functions = source.slice(source.indexOf("function meetingSpeakerColor("), source.indexOf("function refreshControls("));

function element() {
  return {
    children: [], properties: {},
    style: { setProperty(key, value) { this[key] = value; } },
    append(...items) { this.children.push(...items); },
    replaceChildren() { this.children = []; },
    removeAttribute(key) { delete this[key]; },
  };
}

function harness() {
  const elements = Object.fromEntries(["meetingSegments", "committed", "meetingJson", "meetingSrt"].map(key => [key, element()]));
  const context = vm.createContext({ elements, document: { createElement: element },
    lastMeetingSegments: "", formatElapsed: String });
  vm.runInContext(functions, context);
  return { ...context, elements };
}

test("speaker IDs have stable, distinct colors, including beyond the initial palette", () => {
  const { meetingSpeakerColor: color } = harness();
  const ids = Array.from({ length: 24 }, (_, i) => `S${String(i + 1).padStart(2, "0")}`);
  assert.equal(new Set(ids.map(color)).size, ids.length);
  assert.equal(color("S01"), "#56e0c2");
  assert.equal(color("S02"), "#82b7ff");
  assert.equal(color("S1"), color("S01"));
  assert.equal(color("unexpected"), "var(--text)");
});

test("A/B/A rendering keeps the same speaker color after reorder and clear", () => {
  const { renderMeetingSegments: render, elements } = harness();
  const segments = ["S01", "S02", "S01"].map((speaker, i) => ({speaker, start: i, end: i + 1, text: "<test>"}));
  const colors = () => elements.meetingSegments.children.map(row => row.style["--speaker-color"]);
  render({ meeting_segments: segments });
  assert.deepEqual(colors(), ["#56e0c2", "#82b7ff", "#56e0c2"]);
  assert.equal(elements.meetingSegments.children[0].children[1].textContent, "<test>");
  render({ meeting_segments: [segments[1], segments[0]] });
  assert.deepEqual(colors(), ["#82b7ff", "#56e0c2"]);
  render({ meeting_segments: [] });
  assert.equal(elements.meetingSegments.hidden, true);
  render({ meeting_segments: [segments[0]] });
  assert.deepEqual(colors(), ["#56e0c2"]);
});

test("curated text colors remain readable on the dark transcript surface", () => {
  const { meetingSpeakerColor: color } = harness();
  function luminance(hex) {
    const rgb = hex.match(/[a-f\d]{2}/gi).map(v => parseInt(v, 16) / 255)
      .map(v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4);
    return rgb[0] * .2126 + rgb[1] * .7152 + rgb[2] * .0722;
  }
  for (let id = 1; id <= 8; id++) {
    assert.ok((luminance(color(`S${id}`)) + .05) / (luminance("#122332") + .05) >= 4.5);
  }
});

test("meeting category is offline only; target remains realtime only", () => {
  const elements = {mode:{value:"streaming"},recognitionType:{...element(),value:"meeting"}};
  const context = vm.createContext({elements,document:{createElement:element}});
  vm.runInContext(source.slice(source.indexOf("function renderRecognitionTypes("), source.indexOf("function meetingSpeakerColor(")),context);
  context.renderRecognitionTypes();
  assert.deepEqual(elements.recognitionType.children.map(x=>x.value),["normal","target"]);
  assert.equal(elements.recognitionType.value,"normal");
  elements.mode.value="offline";
  elements.recognitionType.value="meeting";
  context.renderRecognitionTypes();
  assert.deepEqual(elements.recognitionType.children.map(x=>x.value),["normal","meeting"]);
  assert.equal(elements.recognitionType.value,"meeting");
});
