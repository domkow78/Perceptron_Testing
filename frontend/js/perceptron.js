import { fetchDataset, perceptronStep } from "./api.js";
import {
  makeTransform,
  clearCanvas,
  drawGrid,
  drawPoints,
  drawDecisionBoundary,
  drawLineChart,
} from "./canvas-utils.js";

const scatterCanvas = document.getElementById("scatter");
const scatterCtx = scatterCanvas.getContext("2d");
const chartCanvas = document.getElementById("accuracy-chart");
const chartCtx = chartCanvas.getContext("2d");
const transform = makeTransform(scatterCanvas);

const datasetSelect = document.getElementById("dataset-select");
const lrInput = document.getElementById("learning-rate");
const lrValue = document.getElementById("learning-rate-value");
const w1Input = document.getElementById("w1");
const w2Input = document.getElementById("w2");
const biasInput = document.getElementById("bias");
const w1Value = document.getElementById("w1-value");
const w2Value = document.getElementById("w2-value");
const biasValue = document.getElementById("bias-value");

const btnStep = document.getElementById("btn-step");
const btnTrain = document.getElementById("btn-train");
const btnRandom = document.getElementById("btn-random");
const btnClearLog = document.getElementById("btn-clear-log");
const trainingLog = document.getElementById("training-log");

const statEpoch = document.getElementById("stat-epoch");
const statAccuracy = document.getElementById("stat-accuracy");
const statUpdates = document.getElementById("stat-updates");
const xorNote = document.getElementById("xor-note");

const diagW1Label = document.getElementById("diag-w1-label");
const diagW2Label = document.getElementById("diag-w2-label");
const diagBiasLabel = document.getElementById("diag-bias-label");
const diagSumValue = document.getElementById("diag-sum-value");
const diagOutputValue = document.getElementById("diag-output-value");
const diagX1Value = document.getElementById("diag-x1-value");
const diagX2Value = document.getElementById("diag-x2-value");

const state = {
  dataset: null,
  weights: [0, 0],
  bias: 0,
  learningRate: parseFloat(lrInput.value),
  epoch: 0,
  accuracyHistory: [],
  highlightIndex: -1,
  trainingLoop: false,
};

function randomWeights() {
  state.weights = [
    Math.round((Math.random() * 2 - 1) * 100) / 100,
    Math.round((Math.random() * 2 - 1) * 100) / 100,
  ];
  state.bias = 0;
  state.epoch = 0;
  state.accuracyHistory = [];
  state.highlightIndex = -1;
  statUpdates.textContent = "—";
  clearLog();
  syncSlidersFromState();
}

function syncSlidersFromState() {
  w1Input.value = state.weights[0];
  w2Input.value = state.weights[1];
  biasInput.value = state.bias;
  updateValueLabels();
}

function updateValueLabels() {
  w1Value.textContent = state.weights[0].toFixed(2);
  w2Value.textContent = state.weights[1].toFixed(2);
  biasValue.textContent = state.bias.toFixed(2);
  lrValue.textContent = state.learningRate.toFixed(2);
  diagW1Label.textContent = `w1=${state.weights[0].toFixed(2)}`;
  diagW2Label.textContent = `w2=${state.weights[1].toFixed(2)}`;
  diagBiasLabel.textContent = `bias=${state.bias.toFixed(2)}`;
}

function fmtSigned(value, digits = 3) {
  const fixed = value.toFixed(digits);
  return value >= 0 ? `+${fixed}` : fixed;
}

const LOG_PLACEHOLDER =
  '<p class="info-box">Kliknij "Jeden krok (epoka)" albo "Trenuj do zbieżności", żeby zobaczyć tutaj dokładny przebieg aktualizacji wag.</p>';

function clearLog() {
  trainingLog.innerHTML = LOG_PLACEHOLDER;
}

function appendEpochLog(epochNumber, result) {
  const header = document.createElement("div");
  header.className = "log-epoch-header";
  header.textContent = `— Epoka ${epochNumber} — (${result.updates} aktualizacji, dokładność ${Math.round(
    result.accuracy * 100
  )}%)`;
  trainingLog.appendChild(header);

  if (result.trace.length === 0) {
    const line = document.createElement("div");
    line.className = "log-line log-ok";
    line.textContent = "Brak błędów na całym zbiorze — wagi bez zmian. Perceptron osiągnął zbieżność!";
    trainingLog.appendChild(line);
  } else {
    result.trace.forEach((event) => {
      const [x, y] = event.point;
      const [dw1, dw2] = event.delta_weights;
      const [w1After, w2After] = event.weights_after;
      const line = document.createElement("div");
      line.className = "log-line";
      line.textContent =
        `Punkt (${x.toFixed(2)}, ${y.toFixed(2)}): prawdziwa klasa ${event.target}, ` +
        `sieć przewidziała ${event.prediction} (net=${event.net_input.toFixed(2)}). ` +
        `Błąd = ${event.target} − ${event.prediction} = ${event.error}. ` +
        `Aktualizacja: w1 ${fmtSigned(dw1)} → ${w1After.toFixed(3)}, ` +
        `w2 ${fmtSigned(dw2)} → ${w2After.toFixed(3)}, ` +
        `bias ${fmtSigned(event.delta_bias)} → ${event.bias_after.toFixed(3)}`;
      trainingLog.appendChild(line);
    });
  }
  trainingLog.scrollTop = trainingLog.scrollHeight;
}

function render() {
  clearCanvas(scatterCtx, scatterCanvas);
  drawGrid(scatterCtx, scatterCanvas, transform);
  if (state.dataset) {
    drawPoints(
      scatterCtx,
      transform,
      state.dataset.points,
      state.dataset.labels,
      undefined,
      state.highlightIndex
    );
  }
  drawDecisionBoundary(scatterCtx, transform, state.weights, state.bias);
  drawLineChart(chartCtx, chartCanvas, state.accuracyHistory, { min: 0, max: 1 });

  statEpoch.textContent = state.epoch;
  statAccuracy.textContent = state.accuracyHistory.length
    ? `${Math.round(state.accuracyHistory[state.accuracyHistory.length - 1] * 100)}%`
    : "—";
}

async function loadDataset(name) {
  state.dataset = await fetchDataset(name);
  state.epoch = 0;
  state.accuracyHistory = [];
  state.highlightIndex = -1;
  statUpdates.textContent = "—";
  xorNote.style.display = name === "xor" ? "block" : "none";
  clearLog();
  render();
}

async function doStep() {
  if (!state.dataset) return null;
  const result = await perceptronStep({
    weights: state.weights,
    bias: state.bias,
    dataset: state.dataset,
    learningRate: state.learningRate,
  });
  state.weights = result.weights;
  state.bias = result.bias;
  state.epoch += 1;
  state.accuracyHistory.push(result.accuracy);
  statUpdates.textContent = result.updates;
  appendEpochLog(state.epoch, result);
  syncSlidersFromState();
  render();
  return result;
}

async function trainToConvergence() {
  if (state.trainingLoop) return;
  state.trainingLoop = true;
  btnTrain.disabled = true;
  btnStep.disabled = true;
  for (let i = 0; i < 200; i++) {
    const result = await doStep();
    if (!result || result.updates === 0) break;
    await new Promise((resolve) => setTimeout(resolve, 60));
  }
  state.trainingLoop = false;
  btnTrain.disabled = false;
  btnStep.disabled = false;
}

function netInputAt(point) {
  return state.weights[0] * point[0] + state.weights[1] * point[1] + state.bias;
}

scatterCanvas.addEventListener("click", (event) => {
  if (!state.dataset) return;
  const rect = scatterCanvas.getBoundingClientRect();
  const px = ((event.clientX - rect.left) / rect.width) * scatterCanvas.width;
  const py = ((event.clientY - rect.top) / rect.height) * scatterCanvas.height;

  let closestIndex = -1;
  let closestDist = Infinity;
  state.dataset.points.forEach((point, i) => {
    const [x, y] = transform.toPx(point[0], point[1]);
    const dist = Math.hypot(x - px, y - py);
    if (dist < closestDist) {
      closestDist = dist;
      closestIndex = i;
    }
  });
  if (closestDist > 20) return;

  state.highlightIndex = closestIndex;
  const point = state.dataset.points[closestIndex];
  const sum = netInputAt(point);
  const output = sum >= 0 ? 1 : 0;
  diagX1Value.textContent = point[0].toFixed(2);
  diagX2Value.textContent = point[1].toFixed(2);
  diagSumValue.textContent = sum.toFixed(2);
  diagOutputValue.textContent = output;
  render();
});

datasetSelect.addEventListener("change", () => loadDataset(datasetSelect.value));

lrInput.addEventListener("input", () => {
  state.learningRate = parseFloat(lrInput.value);
  lrValue.textContent = state.learningRate.toFixed(2);
});

[w1Input, w2Input, biasInput].forEach((input) => {
  input.addEventListener("input", () => {
    state.weights = [parseFloat(w1Input.value), parseFloat(w2Input.value)];
    state.bias = parseFloat(biasInput.value);
    updateValueLabels();
    render();
  });
});

btnStep.addEventListener("click", doStep);
btnTrain.addEventListener("click", trainToConvergence);
btnRandom.addEventListener("click", () => {
  randomWeights();
  render();
});
btnClearLog.addEventListener("click", clearLog);

randomWeights();
loadDataset(datasetSelect.value);
