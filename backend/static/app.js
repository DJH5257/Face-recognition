const state = {
  enrollmentId: null,
  challengeId: null,
  actions: [],
  labels: {},
  stream: null,
  cameraReady: false,
  isUploading: false,
  isCameraStarting: false,
  isChallengeLoading: false,
  isCapturing: false,
  previewUrl: null,
};

const MAX_UPLOAD_BYTES = 8 * 1024 * 1024;
const MAX_UPLOAD_MB = MAX_UPLOAD_BYTES / 1024 / 1024;
const ACCEPTED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp", "image/bmp"]);
const CAPTURE_DURATION_MS = 2000;
const CAPTURE_INTERVAL_MS = 200;
const ACTION_CAPTURE_INTERVAL_MS = Object.freeze({
  blink: 80,
  mouth_open: 120,
});
const CAPTURE_WIDTH = 360;
const CAPTURE_JPEG_QUALITY = 0.7;
const ACTION_SWITCH_PAUSE_MS = 250;
const REQUEST_TIMEOUT_MS = 20000;
const VERIFY_TIMEOUT_MS = 90000;
const ACTION_GUIDANCE = Object.freeze({
  blink: "睁眼看镜头 -> 眨一下 -> 再睁开",
  smile: "嘴角明显上扬或露齿",
  nod_head: "低头再抬头",
  shake_head: "向左看 -> 向右看",
  mouth_open: "明显张嘴后闭合",
});

const $ = (id) => document.getElementById(id);

const els = {
  apiStatus: $("apiStatus"),
  photoInput: $("photoInput"),
  previewImage: $("previewImage"),
  uploadHint: $("uploadHint"),
  uploadBtn: $("uploadBtn"),
  enrollState: $("enrollState"),
  video: $("video"),
  canvas: $("captureCanvas"),
  cameraBtn: $("cameraBtn"),
  closeCameraBtn: $("closeCameraBtn"),
  challengeBtn: $("challengeBtn"),
  captureBtn: $("captureBtn"),
  cameraState: $("cameraState"),
  instruction: $("instruction"),
  actionsText: $("actionsText"),
  livenessText: $("livenessText"),
  similarityText: $("similarityText"),
  resultText: $("resultText"),
  actionDetails: $("actionDetails"),
};

async function checkApi() {
  try {
    const data = await requestJson("/api/health", { timeoutMs: 5000 });
    els.apiStatus.textContent = data.ok ? "API 正常" : "API 异常";
  } catch {
    els.apiStatus.textContent = "API 不可用";
  }
}

els.photoInput.addEventListener("change", () => {
  const file = els.photoInput.files?.[0];
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
  }
  resetEnrollment();
  if (!file) {
    els.previewImage.removeAttribute("src");
    els.previewImage.parentElement.classList.remove("hasImage");
    els.uploadHint.textContent = "选择一张 8MB 内的清晰正脸照片";
    updateActionButtons();
    return;
  }

  const error = validateImageFile(file);
  if (error) {
    els.photoInput.value = "";
    els.previewImage.removeAttribute("src");
    els.previewImage.parentElement.classList.remove("hasImage");
    els.uploadHint.textContent = "选择一张 8MB 内的清晰正脸照片";
    els.enrollState.textContent = "未上传";
    setResult(error, false);
    updateActionButtons();
    return;
  }

  state.previewUrl = URL.createObjectURL(file);
  els.previewImage.src = state.previewUrl;
  els.previewImage.parentElement.classList.add("hasImage");
  els.enrollState.textContent = "待上传";
  els.resultText.textContent = "等待验证";
  els.resultText.className = "";
  updateActionButtons();
});

els.uploadBtn.addEventListener("click", async () => {
  if (state.isUploading || state.isCapturing) return;
  const file = els.photoInput.files?.[0];
  if (!file) {
    setResult("请先选择基准人脸照片", false);
    return;
  }
  const error = validateImageFile(file);
  if (error) {
    setResult(error, false);
    return;
  }

  const form = new FormData();
  form.append("file", file);
  state.isUploading = true;
  els.enrollState.textContent = "上传中";
  updateActionButtons();
  try {
    const data = await requestJson("/api/enroll", { method: "POST", body: form, timeoutMs: 30000 });
    state.enrollmentId = data.enrollment_id;
    clearChallenge();
    els.enrollState.textContent = "已提取特征";
    setResult(data.message, true);
  } catch (err) {
    resetEnrollment();
    els.enrollState.textContent = "上传失败";
    setResult(err.message, false);
  } finally {
    state.isUploading = false;
    updateActionButtons();
  }
});

els.cameraBtn.addEventListener("click", async () => {
  if (state.isCameraStarting || state.isCapturing) return;
  if (!isCameraSupported()) {
    setResult("当前浏览器不支持摄像头采集，请使用支持 getUserMedia 的浏览器", false);
    els.cameraState.textContent = "不支持";
    updateActionButtons();
    return;
  }

  stopCamera({ keepMessage: true });
  state.isCameraStarting = true;
  state.cameraReady = false;
  els.cameraState.textContent = "检测摄像头";
  els.instruction.textContent = "正在准备摄像头";
  updateActionButtons();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 15, max: 24 },
        facingMode: "user",
      },
      audio: false,
    });
    const videoTrack = getVideoTrack();
    if (!videoTrack) {
      throw new Error("未检测到可用摄像头轨道");
    }
    const settings = videoTrack.getSettings ? videoTrack.getSettings() : {};
    if (settings.width && settings.height && (settings.width < 160 || settings.height < 120)) {
      throw new Error("摄像头分辨率过低，无法完成验证");
    }
    els.video.srcObject = state.stream;
    await waitForVideoReady();
    state.cameraReady = true;
    els.cameraState.textContent = cameraReadyText(videoTrack);
    els.instruction.textContent = "摄像头已就绪";
    setResult("摄像头已就绪，可以生成人脸验证动作", true);
  } catch (err) {
    stopCamera({ keepMessage: true });
    els.cameraState.textContent = "开启失败";
    els.instruction.textContent = "等待摄像头";
    setResult(`无法开启摄像头：${err.message}`, false);
  } finally {
    state.isCameraStarting = false;
    updateActionButtons();
  }
});

els.closeCameraBtn.addEventListener("click", () => {
  if (state.isCapturing) return;
  stopCamera();
  setResult("摄像头已关闭", true);
});

els.challengeBtn.addEventListener("click", async () => {
  if (state.isChallengeLoading || state.isCapturing) return;
  if (!state.enrollmentId) {
    setResult("请先上传基准人脸", false);
    return;
  }
  if (!state.cameraReady) {
    setResult("请先开启摄像头并等待画面就绪", false);
    return;
  }

  state.isChallengeLoading = true;
  els.instruction.textContent = "正在生成人脸验证动作";
  updateActionButtons();
  try {
    const data = await requestJson("/api/liveness/challenge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enrollment_id: state.enrollmentId }),
      timeoutMs: 15000,
    });
    state.challengeId = data.challenge_id;
    state.actions = Array.isArray(data.actions) ? data.actions : [];
    state.labels = data.labels || {};
    els.actionsText.textContent = formatActionsText();
    els.livenessText.textContent = "待检测";
    els.livenessText.className = "";
    els.similarityText.textContent = "-";
    els.resultText.textContent = "等待验证";
    els.resultText.className = "";
    els.actionDetails.innerHTML = "";
    els.instruction.textContent =
      state.actions.length > 4 ? "本次动作较多，请完成全部动作" : "准备开始人脸验证";
  } catch (err) {
    clearChallenge();
    els.instruction.textContent = "等待动作指令";
    setResult(err.message, false);
  } finally {
    state.isChallengeLoading = false;
    updateActionButtons();
  }
});

els.captureBtn.addEventListener("click", async () => {
  if (state.isCapturing) return;
  if (!state.challengeId || !state.actions.length) return;
  if (!state.cameraReady || !isVideoReady()) {
    setResult("摄像头画面未就绪，请重新开启摄像头", false);
    updateActionButtons();
    return;
  }

  state.isCapturing = true;
  els.livenessText.textContent = "采集中";
  els.livenessText.className = "";
  els.resultText.textContent = "请按提示完成动作";
  els.resultText.className = "";
  updateActionButtons();
  try {
    const frames = [];
    for (let actionIndex = 0; actionIndex < state.actions.length; actionIndex += 1) {
      const action = state.actions[actionIndex];
      const label = state.labels[action] || action;
      if (actionIndex === 0) {
        await countdown(`${label}，准备`, 1);
      }
      if (actionIndex > 0) {
        await sleep(ACTION_SWITCH_PAUSE_MS);
      }
      const captured = await captureFramesForAction(action, actionIndex, state.actions.length, label);
      frames.push(...captured);
    }
    els.instruction.textContent = "正在提交检测";
    const live = await requestJson("/api/liveness/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        challenge_id: state.challengeId,
        enrollment_id: state.enrollmentId,
        frames,
      }),
      timeoutMs: VERIFY_TIMEOUT_MS,
    });
    renderActionDetails(live.action_results);
    els.livenessText.textContent = live.message;
    els.livenessText.className = live.liveness_passed ? "ok" : "bad";
    if (typeof live.similarity === "number") {
      els.similarityText.textContent = live.similarity.toFixed(4);
    }
    if (!live.liveness_passed) {
      setResult(live.message, false);
      return;
    }
    setResult(live.message, true);
  } catch (err) {
    els.livenessText.textContent = err.message.includes("活体") ? "失败" : els.livenessText.textContent;
    setResult(err.message, false);
  } finally {
    els.instruction.textContent = "等待动作指令";
    finishChallengeCycle();
    state.isCapturing = false;
    updateActionButtons();
  }
});

async function captureFramesForAction(action, actionIndex, totalActions, label) {
  const frames = [];
  const started = performance.now();
  const intervalMs = captureIntervalForAction(action);
  const frameCount = Math.max(2, Math.ceil(CAPTURE_DURATION_MS / intervalMs) + 1);
  const guidance = ACTION_GUIDANCE[action] || label;

  for (let index = 0; index < frameCount; index += 1) {
    const scheduledAt = started + Math.min(index * intervalMs, CAPTURE_DURATION_MS);
    const waitMs = scheduledAt - performance.now();
    if (waitMs > 0) await sleep(waitMs);
    if (!isVideoReady()) {
      throw new Error("摄像头画面中断，请重新开启摄像头");
    }
    frames.push({
      action,
      index,
      timestamp: Math.round(performance.now()),
      image: captureJpegDataUrl(),
    });
    const percent = Math.min(100, Math.round((index / Math.max(frameCount - 1, 1)) * 100));
    els.instruction.textContent =
      `${label} ${actionIndex + 1}/${totalActions}：${guidance}，进度 ${percent}%`;
  }
  return frames;
}

function captureIntervalForAction(action) {
  return ACTION_CAPTURE_INTERVAL_MS[action] || CAPTURE_INTERVAL_MS;
}

function captureJpegDataUrl() {
  const video = els.video;
  const canvas = els.canvas;
  const targetWidth = CAPTURE_WIDTH;
  const ratio = video.videoHeight / Math.max(video.videoWidth, 1);
  canvas.width = targetWidth;
  canvas.height = Math.max(1, Math.round(targetWidth * ratio));
  const ctx = canvas.getContext("2d", { alpha: false });
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", CAPTURE_JPEG_QUALITY);
}

async function countdown(prefix, seconds) {
  for (let i = seconds; i > 0; i -= 1) {
    els.instruction.textContent = `${prefix} ${i}`;
    await sleep(1000);
  }
}

function renderActionDetails(results) {
  els.actionDetails.innerHTML = "";
  if (!Array.isArray(results) || !results.length) {
    els.actionDetails.textContent = "暂无动作明细";
    return;
  }
  for (const item of results) {
    const row = document.createElement("div");
    row.className = "detailItem";
    const left = document.createElement("span");
    left.textContent = `${item.label}：${item.detail}`;
    const right = document.createElement("strong");
    right.className = item.passed ? "ok" : "bad";
    right.textContent = item.passed ? "通过" : "失败";
    row.append(left, right);
    els.actionDetails.append(row);
  }
}

function updateActionButtons() {
  const locked = state.isCapturing || state.isCameraStarting || state.isChallengeLoading || state.isUploading;
  const hasFile = Boolean(els.photoInput.files?.[0]);
  const hasCamera = Boolean(state.stream);
  const canChallenge = state.cameraReady && state.enrollmentId && !locked;
  const canCapture = canChallenge && state.challengeId && state.actions.length > 0;

  els.uploadBtn.disabled = state.isUploading || state.isCapturing || state.isChallengeLoading || !hasFile;
  els.uploadBtn.textContent = state.isUploading ? "上传中..." : "上传基准人脸";

  els.cameraBtn.disabled = state.isCameraStarting || state.isCapturing;
  els.cameraBtn.textContent = state.isCameraStarting ? "准备摄像头..." : hasCamera ? "重启摄像头" : "开启摄像头";

  els.closeCameraBtn.disabled = !hasCamera || state.isCameraStarting || state.isCapturing;

  els.challengeBtn.disabled = !canChallenge;
  els.challengeBtn.textContent = state.isChallengeLoading ? "生成中..." : "生成人脸验证动作";

  els.captureBtn.disabled = !canCapture;
  els.captureBtn.textContent = state.isCapturing ? "采集中..." : "开始人脸验证";
}

function setResult(message, ok) {
  els.resultText.textContent = message;
  els.resultText.className = ok ? "ok" : "bad";
}

function validateImageFile(file) {
  if (!file) return "请先选择基准人脸照片";
  if (!ACCEPTED_IMAGE_TYPES.has(file.type)) {
    return "仅支持 JPG、PNG、WebP 或 BMP 图片";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `图片不能超过 ${MAX_UPLOAD_MB}MB`;
  }
  if (file.size <= 0) {
    return "图片文件为空，请重新选择";
  }
  return "";
}

function resetEnrollment() {
  state.enrollmentId = null;
  clearChallenge();
  els.enrollState.textContent = "未上传";
}

function clearChallenge() {
  state.challengeId = null;
  state.actions = [];
  state.labels = {};
  state.captureDelaysMs = [];
  state.timingNonce = null;
  els.actionsText.textContent = "-";
  els.livenessText.textContent = "未检测";
  els.livenessText.className = "";
  els.similarityText.textContent = "-";
  els.actionDetails.innerHTML = "";
  els.instruction.textContent = state.cameraReady ? "摄像头已就绪" : "等待动作指令";
}

function finishChallengeCycle() {
  state.challengeId = null;
  state.actions = [];
  state.labels = {};
  state.captureDelaysMs = [];
  state.timingNonce = null;
  els.actionsText.textContent = "请重新生成动作";
}

function isCameraSupported() {
  return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function getVideoTrack() {
  return state.stream?.getVideoTracks?.()[0] || null;
}

function stopCamera(options = {}) {
  const { keepMessage = false } = options;
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
  }
  state.stream = null;
  state.cameraReady = false;
  els.video.pause();
  els.video.removeAttribute("src");
  els.video.srcObject = null;
  els.cameraState.textContent = "未开启";
  if (!keepMessage) els.instruction.textContent = "等待摄像头";
  updateActionButtons();
}

async function waitForVideoReady() {
  const video = els.video;
  let metadataReady = video.videoWidth > 0 && video.videoHeight > 0;
  let canPlayReady = video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA;

  const playPromise = video.play();
  if (playPromise) await playPromise;
  if (metadataReady && canPlayReady && hasUsableVideoFrame()) return;

  await new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error("摄像头画面准备超时"));
    }, 8000);

    function cleanup() {
      window.clearTimeout(timer);
      video.removeEventListener("loadedmetadata", onLoadedMetadata);
      video.removeEventListener("canplay", onCanPlay);
      video.removeEventListener("loadeddata", onCanPlay);
      video.removeEventListener("error", onError);
    }

    function checkReady() {
      metadataReady = metadataReady || (video.videoWidth > 0 && video.videoHeight > 0);
      canPlayReady = canPlayReady || video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA;
      if (metadataReady && canPlayReady && hasUsableVideoFrame()) {
        cleanup();
        resolve();
      }
    }

    function onLoadedMetadata() {
      metadataReady = true;
      checkReady();
    }

    function onCanPlay() {
      canPlayReady = true;
      checkReady();
    }

    function onError() {
      cleanup();
      reject(new Error("摄像头画面加载失败"));
    }

    video.addEventListener("loadedmetadata", onLoadedMetadata);
    video.addEventListener("canplay", onCanPlay);
    video.addEventListener("loadeddata", onCanPlay);
    video.addEventListener("error", onError);
    checkReady();
  });
}

function isVideoReady() {
  return (
    state.stream &&
    state.cameraReady !== false &&
    hasUsableVideoFrame()
  );
}

function hasUsableVideoFrame() {
  return (
    els.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    els.video.videoWidth >= 160 &&
    els.video.videoHeight >= 120
  );
}

function cameraReadyText(track) {
  const settings = track.getSettings ? track.getSettings() : {};
  if (settings.width && settings.height) {
    return `已就绪 ${settings.width}x${settings.height}`;
  }
  return "已就绪";
}

function formatActionsText() {
  if (!state.actions.length) return "未生成";
  return state.actions.map((action) => state.labels[action] || action).join(" → ");
}

async function requestJson(url, options = {}) {
  const { timeoutMs = REQUEST_TIMEOUT_MS, ...fetchOptions } = options;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, { ...fetchOptions, signal: controller.signal });
    const text = await res.text();
    const contentType = res.headers.get("content-type") || "";
    let data = {};

    if (text) {
      if (!contentType.includes("application/json")) {
        throw new Error(`服务器返回非 JSON 响应：${res.status}`);
      }
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error("服务器返回 JSON 格式错误");
      }
    }

    if (!res.ok) {
      const message = formatErrorDetail(data.detail || data.message) || `请求失败：${res.status}`;
      throw new Error(message);
    }
    return data;
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("请求超时，请稍后重试");
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

function formatErrorDetail(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item.msg === "string") return item.msg;
        return "";
      })
      .filter(Boolean)
      .join("；");
  }
  if (detail && typeof detail.message === "string") return detail.message;
  if (detail && typeof detail.msg === "string") return detail.msg;
  return "请求参数不符合要求";
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

checkApi();
