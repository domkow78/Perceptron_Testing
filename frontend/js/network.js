import { fetchDataset, mlpInit, mlpStep, mlpPredict, mlpActivations } from "./api.js";
import { makeTransform, clearCanvas, drawGrid, drawPoints, drawDecisionField, drawLineChart } from "./canvas-utils.js";
import { renderNetworkDiagram } from "./network-diagram.js";
import { renderNetworkHeatmap } from "./network-heatmap.js";

const FIELD_RESOLUTION = 48;
const DIAGRAM_MAX_LAYER = 24;

const fieldCanvas = document.getElementById("decision-field");
const fieldCtx = fieldCanvas.getContext("2d");
const vizCanvas = document.getElementById("network-viz");
const vizCtx = vizCanvas.getContext("2d");
const lossCanvas = document.getElementById("loss-chart");
const lossCtx = lossCanvas.getContext("2d");
const accuracyCanvas = document.getElementById("accuracy-chart");
const accuracyCtx = accuracyCanvas.getContext("2d");
const transform = makeTransform(fieldCanvas);

const datasetSelect = document.getElementById("dataset-select");
const hiddenLayersInput = document.getElementById("hidden-layers");
const activationSelect = document.getElementById("activation-select");
const lrInput = document.getElementById("learning-rate");
const lrValue = document.getElementById("learning-rate-value");
const epochsPerRunInput = document.getElementById("epochs-per-run");

const btnBuild = document.getElementById("btn-build");
const btnStep = document.getElementById("btn-step");
const btnTrain = document.getElementById("btn-train");
const btnClearLog = document.getElementById("btn-clear-log");
const trainingLog = document.getElementById("training-log");
const pointInspector = document.getElementById("point-inspector");

const statEpoch = document.getElementById("stat-epoch");
const statLoss = document.getElementById("stat-loss");
const statAccuracy = document.getElementById("stat-accuracy");

const LOG_PLACEHOLDER = '<p class="info-box">Kliknij "Zbuduj sieć", a potem "Jeden krok" albo "Trenuj N epok".</p>';

const state = {
  architecture: [2, 10, 1],
  hiddenActivation: activationSelect.value,
  learningRate: parseFloat(lrInput.value),
  weights: null,
  biases: null,
  dataset: null,
  epoch: 0,
  lossHistory: [],
  accuracyHistory: [],
  fieldValues: null,
  highlightIndex: -1,
  highlightActivations: null,
  trainingLoop: false,
};

function parseHiddenLayers(text) {
  const parts = text
    .split(",")
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => Number.isFinite(n) && n > 0);
  return parts.length > 0 ? parts : [10];
}

function buildGridPoints(resolution, domain) {
  const [dMin, dMax] = domain;
  const cellData = (dMax - dMin) / resolution;
  const points = [];
  for (let j = 0; j < resolution; j++) {
    const y = dMin + j * cellData;
    for (let i = 0; i < resolution; i++) {
      points.push([dMin + i * cellData, y]);
    }
  }
  return points;
}

function clearLog() {
  trainingLog.innerHTML = LOG_PLACEHOLDER;
}

function appendEpochLog(epochNumber, result) {
  const line = document.createElement("div");
  line.className = "log-line";
  line.textContent = `Epoka ${epochNumber}: strata = ${result.loss.toFixed(4)}, dokładność = ${Math.round(
    result.accuracy * 100
  )}%`;
  trainingLog.appendChild(line);
  trainingLog.scrollTop = trainingLog.scrollHeight;
}

function renderViz() {
  if (!state.weights) {
    clearCanvas(vizCtx, vizCanvas);
    return;
  }
  const maxLayer = Math.max(...state.architecture);
  if (maxLayer <= DIAGRAM_MAX_LAYER) {
    renderNetworkDiagram(vizCtx, vizCanvas, state.architecture, state.weights, state.highlightActivations);
  } else {
    renderNetworkHeatmap(vizCtx, vizCanvas, state.architecture, state.weights, state.highlightActivations);
  }
}

function render() {
  clearCanvas(fieldCtx, fieldCanvas);
  if (state.fieldValues) {
    drawDecisionField(fieldCtx, fieldCanvas, transform, FIELD_RESOLUTION, state.fieldValues);
  }
  drawGrid(fieldCtx, fieldCanvas, transform);
  if (state.dataset) {
    drawPoints(fieldCtx, transform, state.dataset.points, state.dataset.labels, undefined, state.highlightIndex);
  }

  const maxLoss = Math.max(0.2, ...(state.lossHistory.length ? state.lossHistory : [0]));
  drawLineChart(lossCtx, lossCanvas, state.lossHistory, { color: "#f76b4f", min: 0, max: maxLoss });
  drawLineChart(accuracyCtx, accuracyCanvas, state.accuracyHistory, { color: "#3ddc84", min: 0, max: 1 });

  statEpoch.textContent = state.epoch;
  if (state.lossHistory.length) {
    statLoss.textContent = state.lossHistory[state.lossHistory.length - 1].toFixed(4);
    statAccuracy.textContent = `${Math.round(state.accuracyHistory[state.accuracyHistory.length - 1] * 100)}%`;
  } else {
    statLoss.textContent = "—";
    statAccuracy.textContent = "—";
  }

  renderViz();
}

async function refreshDecisionField() {
  const gridPoints = buildGridPoints(FIELD_RESOLUTION, transform.domain);
  const { outputs } = await mlpPredict({
    architecture: state.architecture,
    hiddenActivation: state.hiddenActivation,
    weights: state.weights,
    biases: state.biases,
    points: gridPoints,
  });
  state.fieldValues = outputs;
}

async function buildNetwork() {
  state.architecture = [2, ...parseHiddenLayers(hiddenLayersInput.value), 1];
  state.hiddenActivation = activationSelect.value;
  const net = await mlpInit({ architecture: state.architecture, hiddenActivation: state.hiddenActivation });
  state.weights = net.weights;
  state.biases = net.biases;
  state.epoch = 0;
  state.lossHistory = [];
  state.accuracyHistory = [];
  state.highlightIndex = -1;
  state.highlightActivations = null;
  pointInspector.textContent = "Kliknij punkt na wykresie, aby zobaczyć jego przejście przez sieć.";
  clearLog();
  await refreshDecisionField();
  render();
}

async function loadDataset(name) {
  state.dataset = await fetchDataset(name);
  state.epoch = 0;
  state.lossHistory = [];
  state.accuracyHistory = [];
  state.highlightIndex = -1;
  state.highlightActivations = null;
  clearLog();
  render();
}

async function doStep() {
  if (!state.dataset || !state.weights) return null;
  const result = await mlpStep({
    architecture: state.architecture,
    hiddenActivation: state.hiddenActivation,
    weights: state.weights,
    biases: state.biases,
    dataset: state.dataset,
    learningRate: state.learningRate,
  });
  state.weights = result.weights;
  state.biases = result.biases;
  state.epoch += 1;
  state.lossHistory.push(result.loss);
  state.accuracyHistory.push(result.accuracy);
  appendEpochLog(state.epoch, result);
  await refreshDecisionField();
  render();
  return result;
}

async function trainRun() {
  if (state.trainingLoop) return;
  const epochs = Math.max(1, parseInt(epochsPerRunInput.value, 10) || 1);
  state.trainingLoop = true;
  btnTrain.disabled = true;
  btnStep.disabled = true;
  btnBuild.disabled = true;
  for (let i = 0; i < epochs; i++) {
    await doStep();
  }
  state.trainingLoop = false;
  btnTrain.disabled = false;
  btnStep.disabled = false;
  btnBuild.disabled = false;
}

fieldCanvas.addEventListener("click", async (event) => {
  if (!state.dataset || !state.weights) return;
  const rect = fieldCanvas.getBoundingClientRect();
  const px = ((event.clientX - rect.left) / rect.width) * fieldCanvas.width;
  const py = ((event.clientY - rect.top) / rect.height) * fieldCanvas.height;

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
  const label = state.dataset.labels[closestIndex];
  const { layers } = await mlpActivations({
    architecture: state.architecture,
    hiddenActivation: state.hiddenActivation,
    weights: state.weights,
    biases: state.biases,
    point,
  });
  state.highlightActivations = layers;
  const probability = layers[layers.length - 1][0];
  const prediction = probability >= 0.5 ? 1 : 0;
  pointInspector.textContent =
    `Punkt (${point[0].toFixed(2)}, ${point[1].toFixed(2)}): prawdziwa klasa ${label}, ` +
    `sieć przewidziała ${prediction} (prawdopodobieństwo klasy 1 = ${probability.toFixed(3)}).`;
  render();
});

datasetSelect.addEventListener("change", () => loadDataset(datasetSelect.value));
lrInput.addEventListener("input", () => {
  state.learningRate = parseFloat(lrInput.value);
  lrValue.textContent = state.learningRate.toFixed(2);
});
btnBuild.addEventListener("click", buildNetwork);
btnStep.addEventListener("click", doStep);
btnTrain.addEventListener("click", trainRun);
btnClearLog.addEventListener("click", clearLog);

(async function init() {
  await loadDataset(datasetSelect.value);
  await buildNetwork();
})();
