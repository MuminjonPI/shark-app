// Shark vs Phones: runs a YOLO26n phone detector (ONNX) entirely in the browser.
// The webcam image never leaves the device; only the model file is downloaded.

const MODEL_URL =
  "https://huggingface.co/MuminjonPI/shark-phone-detector/resolve/main/best.onnx";
const SIZE = 640;        // model input size (images: [1, 3, 640, 640])
const CONF = 0.35;       // minimum phone score to count as a detection
const IOU = 0.45;        // overlap threshold for removing duplicate boxes (NMS)
const MISS_FRAMES = 10;  // frames without a phone before the shark is "hungry" again

const video = document.getElementById("cam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const scoreEl = document.getElementById("score");

// Hidden canvas used to resize each frame to 640x640 for the model
const prep = document.createElement("canvas");
prep.width = SIZE;
prep.height = SIZE;
const pctx = prep.getContext("2d", { willReadFrequently: true });

let session = null;
let busy = false;
let eaten = 0;
let phoneVisible = false;
let missCount = 0;

// 1. Frame -> model input. Letterbox (keep aspect ratio, pad with grey) like Ultralytics does.
function preprocess() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scale = Math.min(SIZE / vw, SIZE / vh);
  const nw = Math.round(vw * scale);
  const nh = Math.round(vh * scale);
  const padX = (SIZE - nw) / 2;
  const padY = (SIZE - nh) / 2;

  pctx.fillStyle = "rgb(114,114,114)";
  pctx.fillRect(0, 0, SIZE, SIZE);
  pctx.drawImage(video, padX, padY, nw, nh);
  const rgba = pctx.getImageData(0, 0, SIZE, SIZE).data;

  // RGBA pixels -> 3 colour planes (R, G, B), values 0..1
  const area = SIZE * SIZE;
  const input = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    input[i] = rgba[i * 4] / 255;
    input[i + area] = rgba[i * 4 + 1] / 255;
    input[i + 2 * area] = rgba[i * 4 + 2] / 255;
  }
  const tensor = new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]);
  return { tensor, scale, padX, padY };
}

// 2. Model output -> boxes. output0 is [1, 5, N]: rows are cx, cy, w, h, phone score.
function decode(out, n, scale, padX, padY) {
  const boxes = [];
  for (let i = 0; i < n; i++) {
    const score = out[4 * n + i];
    if (score < CONF) continue;
    const cx = out[i];
    const cy = out[n + i];
    const w = out[2 * n + i];
    const h = out[3 * n + i];
    // Undo the letterbox so the box matches the original video frame
    boxes.push({
      x1: (cx - w / 2 - padX) / scale,
      y1: (cy - h / 2 - padY) / scale,
      x2: (cx + w / 2 - padX) / scale,
      y2: (cy + h / 2 - padY) / scale,
      score,
    });
  }
  return nms(boxes);
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-9);
}

// 3. Non-maximum suppression: keep the best box, drop boxes that overlap it too much.
function nms(boxes) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) {
    if (keep.every((k) => iou(k, b) < IOU)) keep.push(b);
  }
  return keep;
}

// 4. Draw boxes and update the game state.
function draw(boxes) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 5;
  ctx.font = "bold 32px system-ui, sans-serif";

  for (const b of boxes) {
    ctx.strokeStyle = "#ff7a00";
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    ctx.fillStyle = "#ff7a00";
    ctx.fillText(`🦈 CHOMP! ${Math.round(b.score * 100)}%`, b.x1, Math.max(b.y1 - 10, 32));
  }

  if (boxes.length > 0) {
    missCount = 0;
    if (!phoneVisible) {
      eaten += 1; // count each new appearance once, not every frame
      scoreEl.textContent = `Phones eaten: ${eaten}`;
    }
    phoneVisible = true;
    statusEl.textContent = "🦈 CHOMP! Sharks don't text.";
  } else {
    missCount += 1;
    if (missCount >= MISS_FRAMES) {
      phoneVisible = false;
      statusEl.textContent = "The shark is hungry. Hold up a phone!";
    }
  }
}

async function loop() {
  if (!busy && video.readyState >= 2) {
    busy = true;
    try {
      const { tensor, scale, padX, padY } = preprocess();
      const results = await session.run({ images: tensor });
      const out = results.output0;
      draw(decode(out.data, out.dims[2], scale, padX, padY));
    } catch (e) {
      console.error(e);
    } finally {
      busy = false;
    }
  }
  requestAnimationFrame(loop);
}

async function start() {
  statusEl.textContent = "Starting camera…";
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  statusEl.textContent = "Loading the shark's brain (about 10 MB)…";
  session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });

  statusEl.textContent = "The shark is hungry. Hold up a phone!";
  loop();
}

start().catch((e) => {
  console.error(e);
  statusEl.textContent = "Error: " + e.message;
});

// Exported for testing in Node (ignored in the browser)
if (typeof module !== "undefined") module.exports = { decode, nms, iou };
