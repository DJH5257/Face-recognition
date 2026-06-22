const fs = require("fs");
const vm = require("vm");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

const html = fs.readFileSync("backend/static/index.html", "utf8");
const appJs = fs.readFileSync("backend/static/app.js", "utf8");

assert(
  html.includes("/static/app.js?v=20260606-face-verify"),
  "index.html must load the versioned app.js to avoid stale browser cache",
);
assert(
  !appJs.includes('"/api/compare"'),
  "frontend should no longer call the legacy compare endpoint",
);

const elements = new Map();
function element(id) {
  if (!elements.has(id)) {
    elements.set(id, {
      id,
      textContent: "",
      className: "",
      disabled: false,
      files: [],
      value: "",
      src: "",
      srcObject: null,
      readyState: 2,
      videoWidth: 640,
      videoHeight: 480,
      addEventListener() {},
      removeEventListener() {},
      removeAttribute() {},
      setAttribute() {},
      append() {},
      pause() {},
      play() {
        return Promise.resolve();
      },
      parentElement: {
        classList: {
          add() {},
          remove() {},
        },
      },
      getContext() {
        return {
          fillStyle: "",
          fillRect() {},
          drawImage() {},
        };
      },
      toDataURL() {
        return "data:image/jpeg;base64,AAAA";
      },
    });
  }
  return elements.get(id);
}

const context = {
  console,
  setTimeout,
  clearTimeout,
  AbortController,
  HTMLMediaElement: { HAVE_CURRENT_DATA: 2 },
  URL: {
    createObjectURL() {
      return "blob:test";
    },
    revokeObjectURL() {},
  },
  navigator: {
    mediaDevices: {
      getUserMedia() {
        return Promise.reject(new Error("no camera in static test"));
      },
    },
  },
  document: {
    getElementById: element,
    createElement(tag) {
      return element(`${tag}-${Math.random()}`);
    },
  },
  fetch: async () => ({
    ok: true,
    text: async () => "{\"ok\":true}",
    headers: { get: () => "application/json" },
  }),
};
context.window = context;

vm.createContext(context);
vm.runInContext(appJs, context);

const constants = vm.runInContext(
  "({ CAPTURE_DURATION_MS, CAPTURE_INTERVAL_MS, ACTION_CAPTURE_INTERVAL_MS, CAPTURE_WIDTH, CAPTURE_JPEG_QUALITY, ACTION_SWITCH_PAUSE_MS, ACTION_GUIDANCE })",
  context,
);
assert(constants.CAPTURE_DURATION_MS === 2000, "capture duration must remain 2 seconds");
assert(constants.CAPTURE_INTERVAL_MS === 200, "default capture interval must remain 200ms");
assert(constants.ACTION_CAPTURE_INTERVAL_MS.blink === 80, "blink capture interval must be 80ms");
assert(constants.ACTION_CAPTURE_INTERVAL_MS.mouth_open === 120, "mouth capture interval must be 120ms");
assert(
  vm.runInContext("captureIntervalForAction('shake_head')", context) === 200,
  "head movement actions should keep the low-load 200ms capture interval",
);
assert(constants.CAPTURE_WIDTH === 360, "capture width must stay low-end friendly");
assert(constants.CAPTURE_JPEG_QUALITY === 0.7, "JPEG quality must control payload size");
assert(constants.ACTION_SWITCH_PAUSE_MS === 250, "action switch pause should stay short");
assert(
  constants.ACTION_GUIDANCE.blink.includes("再睁开"),
  "blink guidance should tell users to reopen eyes",
);
assert(appJs.includes("timing_nonce: state.timingNonce"), "frontend must echo timing nonce");
assert(appJs.includes("capture_delays_ms"), "frontend must consume server capture delays");

const formatted = vm.runInContext(
  "formatErrorDetail([{ msg: 'List should have at least 1 item' }, { msg: 'Extra inputs are not permitted' }])",
  context,
);
assert(
  formatted === "List should have at least 1 item；Extra inputs are not permitted",
  `unexpected formatted validation error: ${formatted}`,
);
assert(!formatted.includes("[object Object]"), "validation errors must be readable");

const frameCountExpressionPresent = appJs.includes(
  "Math.ceil(CAPTURE_DURATION_MS / intervalMs) + 1",
);
assert(frameCountExpressionPresent, "capture must include final frame covering the full 2 seconds");

console.log("frontend static checks passed");
