import { weightColor, decisionFieldColor } from "./canvas-utils.js";

const MAX_CELL = 6;
const GAP = 26;

/**
 * Rysuje każdą macierz wag jako kolorową mapę cieplną (niebieski = ujemna,
 * czerwony = dodatnia waga) oraz pasek aktywacji pod spodem. Używane dla
 * warstw zbyt dużych, żeby sensownie narysować pełny graf (patrz
 * network-diagram.js dla małych sieci).
 */
export function renderNetworkHeatmap(ctx, canvas, layerSizes, weights, activations = null) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "11px sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";

  let maxAbs = 1e-6;
  weights.forEach((W) => W.forEach((row) => row.forEach((w) => (maxAbs = Math.max(maxAbs, Math.abs(w))))));

  const topY = 26;
  const maxMatrixHeight = canvas.height - topY - (activations ? 40 : 10);
  let xCursor = 12;

  weights.forEach((W, li) => {
    const nIn = W.length;
    const nOut = W[0].length;
    const cellW = Math.max(1, Math.min(MAX_CELL, (canvas.width - xCursor - GAP) / nOut));
    const cellH = Math.max(1, Math.min(MAX_CELL, maxMatrixHeight / nIn));

    ctx.fillStyle = "#8a93a8";
    ctx.fillText(`W${li + 1} (${nIn}×${nOut})`, xCursor, topY - 8);

    for (let i = 0; i < nIn; i++) {
      for (let j = 0; j < nOut; j++) {
        ctx.fillStyle = weightColor(W[i][j], maxAbs);
        ctx.fillRect(xCursor + j * cellW, topY + i * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
    xCursor += nOut * cellW + GAP;
  });

  if (activations) {
    const barY = canvas.height - 22;
    ctx.fillStyle = "#8a93a8";
    ctx.fillText("Aktywacje warstw ukrytych (klikniętego punktu)", 12, barY - 6);

    let barX = 12;
    activations.forEach((layerActivations, li) => {
      if (li === 0 || li === activations.length - 1) return; // pomijamy wejście i wyjście
      const cellW = Math.max(1, Math.min(MAX_CELL, (canvas.width - 24) / layerActivations.length));
      layerActivations.forEach((a, ni) => {
        const clamped = Math.max(-1, Math.min(1, a));
        ctx.fillStyle = decisionFieldColor((clamped + 1) / 2);
        ctx.fillRect(barX + ni * cellW, barY, cellW + 0.5, 14);
      });
      barX += layerActivations.length * cellW + GAP;
    });
  }
}
