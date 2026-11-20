import { mlpInit, mlpStep, mlpPredict } from "./api.js";
import { clearCanvas, drawLineChart, decisionFieldColor } from "./canvas-utils.js";
import { buildTrack } from "./sim/track.js";
import { createCar, stepCar, checkCrash, resetCar, CAR_RADIUS } from "./sim/car.js";
import { computeLidar, RAY_COUNT } from "./sim/lidar.js";

const SAMPLE_CAP = 3000;
const RECORD_INTERVAL_MS = 80;
const PREDICT_THROTTLE_MS = 200;
const DEAD_ZONE = [0.4, 0.6];
const MAX_DT = 0.05;

const trackCanvas = document.getElementById("track-canvas");
const trackCtx = trackCanvas.getContext("2d");
const cameraCanvas = document.getElementById("camera-strip");
const cameraCtx = cameraCanvas.getContext("2d");
const lossCanvas = document.getElementById("loss-chart");
const lossCtx = lossCanvas.getContext("2d");
const accuracyCanvas = document.getElementById("accuracy-chart");
const accuracyCtx = accuracyCanvas.getContext("2d");

const btnModeManual = document.getElementById("btn-mode-manual");
const btnModeAutonomous = document.getElementById("btn-mode-autonomous");
const hiddenSizeInput = document.getElementById("hidden-size");
const activationSelect = document.getElementById("activation-select");
const btnBuild = document.getElementById("btn-build-sim");
const lrInput = document.getElementById("learning-rate");
const lrValue = document.getElementById("learning-rate-value");
const epochsPerRunInput = document.getElementById("epochs-per-run");
const btnTrain = document.getElementById("btn-train-sim");
const btnResetCar = document.getElementById("btn-reset-car");
const btnClearLog = document.getElementById("btn-clear-log");
const trainingLog = document.getElementById("training-log");

const statMode = document.getElementById("stat-mode");
const statSamples = document.getElementById("stat-samples");
const statCrashes = document.getElementById("stat-crashes");
const statPrediction = document.getElementById("stat-prediction");
const statEpoch = document.getElementById("stat-epoch");
const statLoss = document.getElementById("stat-loss");
const statAccuracy = document.getElementById("stat-accuracy");

const LOG_PLACEHOLDER =
  '<p class="info-box">Pojeźdź manualnie (strzałki / A D), a potem kliknij "Zbuduj sieć" i "Trenuj N epok".</p>';

const track = buildTrack();
const allWalls = track.outerWalls.concat(track.innerWalls);

const state = {
  mode: "manual",
  architecture: [RAY_COUNT, 12, 1],
  hiddenActivation: activationSelect.value,
  learningRate: parseFloat(lrInput.value),
  weights: null,
  biases: null,
  car: createCar(track.start),
  lidar: { distances: [], normalized: [], angles: [] },
  heldKey: null,
  lastRecordTime: 0,
  samples: { points: [], labels: [] },
  crashCount: 0,
  lastPrediction: 0.5,
  predictRequestId: 0,
  predictInFlight: false,
  lastPredictTime: 0,
  epoch: 0,
  lossHistory: [],
  accuracyHistory: [],
  trainingLoop: false,
};
state.lidar = computeLidar(state.car, allWalls);

function worldToCanvas(x, y) {
  return [x + trackCanvas.width / 2, trackCanvas.height / 2 - y];
}

function clearLog() {
  trainingLog.innerHTML = LOG_PLACEHOLDER;
}

function appendLog(text, className = "log-line") {
  const line = document.createElement("div");
  line.className = className;
  line.textContent = text;
  trainingLog.appendChild(line);
  trainingLog.scrollTop = trainingLog.scrollHeight;
}

function drawWalls(walls) {
  trackCtx.strokeStyle = "#394a6b";
  trackCtx.lineWidth = 2;
  trackCtx.beginPath();
  walls.forEach((wall) => {
    const [x1, y1] = worldToCanvas(wall.x1, wall.y1);
    const [x2, y2] = worldToCanvas(wall.x2, wall.y2);
    trackCtx.moveTo(x1, y1);
    trackCtx.lineTo(x2, y2);
  });
  trackCtx.stroke();
}

function drawLidar() {
  const [cx, cy] = worldToCanvas(state.car.x, state.car.y);
  state.lidar.angles.forEach((angle, i) => {
    const dist = state.lidar.distances[i];
    const hitX = state.car.x + Math.cos(angle) * dist;
    const hitY = state.car.y + Math.sin(angle) * dist;
    const [hx, hy] = worldToCanvas(hitX, hitY);
    trackCtx.strokeStyle = decisionFieldColor(1 - state.lidar.normalized[i]);
    trackCtx.lineWidth = 1.5;
    trackCtx.beginPath();
    trackCtx.moveTo(cx, cy);
    trackCtx.lineTo(hx, hy);
    trackCtx.stroke();
  });
}

function drawCar() {
  const [cx, cy] = worldToCanvas(state.car.x, state.car.y);
  const size = CAR_RADIUS * 1.6;
  trackCtx.save();
  trackCtx.translate(cx, cy);
  trackCtx.rotate(-state.car.heading);
  trackCtx.fillStyle = "#4f8ff7";
  trackCtx.beginPath();
  trackCtx.moveTo(size, 0);
  trackCtx.lineTo(-size * 0.6, size * 0.6);
  trackCtx.lineTo(-size * 0.6, -size * 0.6);
  trackCtx.closePath();
  trackCtx.fill();
  trackCtx.restore();
}

function drawCameraStrip() {
  clearCanvas(cameraCtx, cameraCanvas);
  const normalized = state.lidar.normalized.length ? state.lidar.normalized : new Array(RAY_COUNT).fill(1);
  const bandWidth = cameraCanvas.width / normalized.length;
  normalized.forEach((value, i) => {
    cameraCtx.fillStyle = decisionFieldColor(1 - value);
    cameraCtx.fillRect(i * bandWidth, 0, bandWidth + 0.5, cameraCanvas.height);
  });
}

function render() {
  clearCanvas(trackCtx, trackCanvas);
  drawWalls(track.outerWalls);
  drawWalls(track.innerWalls);
  drawLidar();
  drawCar();
  drawCameraStrip();

  const maxLoss = Math.max(0.2, ...(state.lossHistory.length ? state.lossHistory : [0]));
  drawLineChart(lossCtx, lossCanvas, state.lossHistory, { color: "#f76b4f", min: 0, max: maxLoss });
  drawLineChart(accuracyCtx, accuracyCanvas, state.accuracyHistory, { color: "#3ddc84", min: 0, max: 1 });

  statMode.textContent = state.mode === "manual" ? "Manualny" : "Autonomiczny";
  statSamples.textContent = state.samples.points.length;
  statCrashes.textContent = state.crashCount;
  statPrediction.textContent = state.lastPrediction.toFixed(3);
  statEpoch.textContent = state.epoch;
  if (state.lossHistory.length) {
    statLoss.textContent = state.lossHistory[state.lossHistory.length - 1].toFixed(4);
    statAccuracy.textContent = `${Math.round(state.accuracyHistory[state.accuracyHistory.length - 1] * 100)}%`;
  } else {
    statLoss.textContent = "—";
    statAccuracy.textContent = "—";
  }
}

function resolveSteerSign() {
  if (state.mode === "manual") {
    if (state.heldKey === "left") return -1;
    if (state.heldKey === "right") return 1;
    return 0;
  }
  const p = state.lastPrediction;
  if (p < DEAD_ZONE[0]) return -1;
  if (p > DEAD_ZONE[1]) return 1;
  return 0;
}

function recordSample() {
  state.samples.points.push(state.lidar.normalized.slice());
  state.samples.labels.push(state.heldKey === "right" ? 1 : 0);
  if (state.samples.points.length > SAMPLE_CAP) {
    state.samples.points.shift();
    state.samples.labels.shift();
  }
}

function requestPrediction() {
  state.predictInFlight = true;
  state.lastPredictTime = performance.now();
  const myId = ++state.predictRequestId;
  const point = state.lidar.normalized.slice();
  mlpPredict({
    architecture: state.architecture,
    hiddenActivation: state.hiddenActivation,
    weights: state.weights,
    biases: state.biases,
    points: [point],
  })
    .then(({ outputs }) => {
      if (myId === state.predictRequestId) state.lastPrediction = outputs[0];
    })
    .catch((err) => appendLog(`Błąd predykcji: ${err.message}`, "log-line log-crash"))
    .finally(() => {
      state.predictInFlight = false;
    });
}

function physicsStep(dt, ts) {
  const steerSign = resolveSteerSign();
  stepCar(state.car, steerSign, dt);
  state.lidar = computeLidar(state.car, allWalls);

  if (checkCrash(state.car, allWalls)) {
    resetCar(state.car, track.start);
    state.lidar = computeLidar(state.car, allWalls);
    state.crashCount += 1;
    appendLog("Kraksa — restart auta.", "log-line log-crash");
    return;
  }

  if (state.mode === "manual" && state.heldKey && ts - state.lastRecordTime >= RECORD_INTERVAL_MS) {
    recordSample();
    state.lastRecordTime = ts;
  }

  if (
    state.mode === "autonomous" &&
    state.weights &&
    !state.predictInFlight &&
    ts - state.lastPredictTime >= PREDICT_THROTTLE_MS
  ) {
    requestPrediction();
  }
}

let lastTs = 0;
function tick(ts) {
  const dt = lastTs ? Math.min(MAX_DT, (ts - lastTs) / 1000) : 1 / 60;
  lastTs = ts;
  physicsStep(dt, ts);
  render();
  requestAnimationFrame(tick);
}

async function buildNetwork() {
  const hiddenSize = Math.max(2, parseInt(hiddenSizeInput.value, 10) || 12);
  state.architecture = [RAY_COUNT, hiddenSize, 1];
  state.hiddenActivation = activationSelect.value;
  const net = await mlpInit({ architecture: state.architecture, hiddenActivation: state.hiddenActivation });
  state.weights = net.weights;
  state.biases = net.biases;
  state.epoch = 0;
  state.lossHistory = [];
  state.accuracyHistory = [];
  appendLog(`Zbudowano sieć [${state.architecture.join(", ")}] (backend: ${net.backend}).`);
}

async function doStep() {
  if (!state.samples.points.length || !state.weights) return null;
  const result = await mlpStep({
    architecture: state.architecture,
    hiddenActivation: state.hiddenActivation,
    weights: state.weights,
    biases: state.biases,
    dataset: { points: state.samples.points, labels: state.samples.labels },
    learningRate: state.learningRate,
  });
  state.weights = result.weights;
  state.biases = result.biases;
  state.epoch += 1;
  state.lossHistory.push(result.loss);
  state.accuracyHistory.push(result.accuracy);
  appendLog(`Epoka ${state.epoch}: strata = ${result.loss.toFixed(4)}, dokładność = ${Math.round(result.accuracy * 100)}%`);
  return result;
}

async function trainRun() {
  if (state.trainingLoop) return;
  if (!state.weights) {
    appendLog("Najpierw zbuduj sieć.", "log-line log-crash");
    return;
  }
  if (state.samples.points.length < 20) {
    appendLog('Za mało nagranych próbek (min. 20) — pojeźdź manualnie trzymając strzałki.', "log-line log-crash");
    return;
  }
  const epochs = Math.max(1, parseInt(epochsPerRunInput.value, 10) || 1);
  state.trainingLoop = true;
  btnTrain.disabled = true;
  btnBuild.disabled = true;
  for (let i = 0; i < epochs; i++) {
    await doStep();
  }
  state.trainingLoop = false;
  btnTrain.disabled = false;
  btnBuild.disabled = false;
}

function setMode(mode) {
  state.mode = mode;
  state.heldKey = null;
  btnModeManual.classList.toggle("active", mode === "manual");
  btnModeAutonomous.classList.toggle("active", mode === "autonomous");
}

btnModeManual.addEventListener("click", () => setMode("manual"));
btnModeAutonomous.addEventListener("click", () => setMode("autonomous"));

window.addEventListener("keydown", (event) => {
  if (state.mode !== "manual") return;
  if (event.key === "ArrowLeft" || event.key === "a" || event.key === "A") {
    state.heldKey = "left";
    event.preventDefault();
  } else if (event.key === "ArrowRight" || event.key === "d" || event.key === "D") {
    state.heldKey = "right";
    event.preventDefault();
  }
});

window.addEventListener("keyup", (event) => {
  if ((event.key === "ArrowLeft" || event.key === "a" || event.key === "A") && state.heldKey === "left") {
    state.heldKey = null;
  } else if ((event.key === "ArrowRight" || event.key === "d" || event.key === "D") && state.heldKey === "right") {
    state.heldKey = null;
  }
});

lrInput.addEventListener("input", () => {
  state.learningRate = parseFloat(lrInput.value);
  lrValue.textContent = state.learningRate.toFixed(2);
});
btnBuild.addEventListener("click", buildNetwork);
btnTrain.addEventListener("click", trainRun);
btnResetCar.addEventListener("click", () => {
  resetCar(state.car, track.start);
  appendLog("Zresetowano pozycję auta.");
});
btnClearLog.addEventListener("click", clearLog);

clearLog();
setMode("manual");
render();
requestAnimationFrame(tick);
