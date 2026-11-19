import { weightColor, decisionFieldColor } from "./canvas-utils.js";

const NODE_RADIUS = 10;
const PADDING_X = 40;
const PADDING_Y = 20;

function layoutPositions(layerSizes, width, height) {
  const numLayers = layerSizes.length;
  const colGap = numLayers > 1 ? (width - 2 * PADDING_X) / (numLayers - 1) : 0;
  return layerSizes.map((size, li) => {
    const x = PADDING_X + li * colGap;
    const rowGap = size > 1 ? (height - 2 * PADDING_Y) / (size - 1) : 0;
    const startY = size > 1 ? PADDING_Y : height / 2;
    return Array.from({ length: size }, (_, ni) => ({
      x,
      y: size > 1 ? startY + ni * rowGap : startY,
    }));
  });
}

function activationColor(value) {
  const clamped = Math.max(-1, Math.min(1, value));
  return decisionFieldColor((clamped + 1) / 2);
}

/**
 * Rysuje pełny diagram węzeł-krawędź sieci. Dobry dla małych warstw
 * (poniżej ~24 neuronów) — powyżej tego progu patrz network-heatmap.js.
 */
export function renderNetworkDiagram(ctx, canvas, layerSizes, weights, activations = null) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const positions = layoutPositions(layerSizes, canvas.width, canvas.height);

  let maxAbs = 1e-6;
  weights.forEach((W) => W.forEach((row) => row.forEach((w) => (maxAbs = Math.max(maxAbs, Math.abs(w))))));

  weights.forEach((W, li) => {
    const fromPositions = positions[li];
    const toPositions = positions[li + 1];
    for (let i = 0; i < fromPositions.length; i++) {
      for (let j = 0; j < toPositions.length; j++) {
        const w = W[i][j];
        ctx.strokeStyle = weightColor(w, maxAbs);
        ctx.lineWidth = Math.max(0.5, Math.min(3, (Math.abs(w) / maxAbs) * 3));
        ctx.beginPath();
        ctx.moveTo(fromPositions[i].x, fromPositions[i].y);
        ctx.lineTo(toPositions[j].x, toPositions[j].y);
        ctx.stroke();
      }
    }
  });

  positions.forEach((layerPositions, li) => {
    layerPositions.forEach((pos, ni) => {
      const activation = activations ? activations[li][ni] : null;
      ctx.beginPath();
      ctx.fillStyle = activation != null ? activationColor(activation) : "#1e2638";
      ctx.arc(pos.x, pos.y, NODE_RADIUS, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#4f8ff7";
      ctx.stroke();
      if (activation != null) {
        ctx.fillStyle = "#0b1120";
        ctx.font = "9px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(activation.toFixed(2), pos.x, pos.y);
      }
    });
  });
}
