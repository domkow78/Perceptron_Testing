export function makeTransform(canvas, domain = [-0.3, 1.3]) {
  const [dMin, dMax] = domain;
  const size = canvas.width;
  const scale = size / (dMax - dMin);
  return {
    domain,
    scale,
    toPx(x, y) {
      return [(x - dMin) * scale, size - (y - dMin) * scale];
    },
  };
}

export function clearCanvas(ctx, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

export function drawGrid(ctx, canvas, transform, step = 0.25) {
  const [dMin, dMax] = transform.domain;
  ctx.save();
  ctx.strokeStyle = "#1c2435";
  ctx.lineWidth = 1;
  for (let v = Math.ceil(dMin / step) * step; v <= dMax; v += step) {
    const [x0] = transform.toPx(v, 0);
    ctx.beginPath();
    ctx.moveTo(x0, 0);
    ctx.lineTo(x0, canvas.height);
    ctx.stroke();

    const [, y0] = transform.toPx(0, v);
    ctx.beginPath();
    ctx.moveTo(0, y0);
    ctx.lineTo(canvas.width, y0);
    ctx.stroke();
  }

  ctx.strokeStyle = "#394a6b";
  ctx.lineWidth = 1.5;
  const [zx] = transform.toPx(0, 0);
  ctx.beginPath();
  ctx.moveTo(zx, 0);
  ctx.lineTo(zx, canvas.height);
  ctx.stroke();
  const [, zy] = transform.toPx(0, 0);
  ctx.beginPath();
  ctx.moveTo(0, zy);
  ctx.lineTo(canvas.width, zy);
  ctx.stroke();
  ctx.restore();
}

export function drawPoints(ctx, transform, points, labels, colors = ["#4f8ff7", "#f76b4f"], highlightIndex = -1) {
  points.forEach(([x, y], i) => {
    const [px, py] = transform.toPx(x, y);
    ctx.beginPath();
    ctx.fillStyle = colors[labels[i]];
    ctx.arc(px, py, i === highlightIndex ? 8 : 5, 0, Math.PI * 2);
    ctx.fill();
    if (i === highlightIndex) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
    }
  });
}

export function drawDecisionBoundary(ctx, transform, weights, bias, color = "#ffffff") {
  const [w1, w2] = weights;
  const [dMin, dMax] = transform.domain;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  if (Math.abs(w2) > 1e-6) {
    const y1 = -(w1 * dMin + bias) / w2;
    const y2 = -(w1 * dMax + bias) / w2;
    const [px1, py1] = transform.toPx(dMin, y1);
    const [px2, py2] = transform.toPx(dMax, y2);
    ctx.moveTo(px1, py1);
    ctx.lineTo(px2, py2);
  } else if (Math.abs(w1) > 1e-6) {
    const x = -bias / w1;
    const [px1, py1] = transform.toPx(x, dMin);
    const [px2, py2] = transform.toPx(x, dMax);
    ctx.moveTo(px1, py1);
    ctx.lineTo(px2, py2);
  }
  ctx.stroke();
  ctx.restore();
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpColor(c1, c2, t) {
  return [lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t)];
}

const FIELD_BLUE = [35, 70, 140];
const FIELD_NEUTRAL = [17, 23, 38];
const FIELD_RED = [150, 60, 42];

/** Koloruje wartość z [0, 1] (prawdopodobieństwo klasy 1) na niebiesko-czerwonej skali. */
export function decisionFieldColor(value) {
  const [r, g, b] =
    value < 0.5
      ? lerpColor(FIELD_BLUE, FIELD_NEUTRAL, value / 0.5)
      : lerpColor(FIELD_NEUTRAL, FIELD_RED, (value - 0.5) / 0.5);
  return `rgb(${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)})`;
}

/** Koloruje wagę w zakresie [-maxAbs, maxAbs] na tej samej niebiesko-czerwonej skali. */
export function weightColor(value, maxAbs) {
  if (maxAbs <= 1e-9) return decisionFieldColor(0.5);
  const normalized = 0.5 + 0.5 * Math.max(-1, Math.min(1, value / maxAbs));
  return decisionFieldColor(normalized);
}

/** Rysuje tło wykresu jako siatkę kolorowanych komórek wg wartości z `values` (row-major, y rosnąco). */
export function drawDecisionField(ctx, canvas, transform, resolution, values) {
  const [dMin, dMax] = transform.domain;
  const cellData = (dMax - dMin) / resolution;
  const cellPx = transform.scale * cellData;
  for (let j = 0; j < resolution; j++) {
    const yData = dMin + j * cellData;
    const [, pyBottom] = transform.toPx(dMin, yData);
    const pyTop = pyBottom - cellPx;
    for (let i = 0; i < resolution; i++) {
      const xData = dMin + i * cellData;
      const [pxLeft] = transform.toPx(xData, yData);
      ctx.fillStyle = decisionFieldColor(values[j * resolution + i]);
      ctx.fillRect(pxLeft, pyTop, cellPx + 0.5, cellPx + 0.5);
    }
  }
}

export function drawLineChart(ctx, canvas, values, { color = "#4f8ff7", min = 0, max = 1 } = {}) {
  clearCanvas(ctx, canvas);
  ctx.save();
  ctx.strokeStyle = "#1c2435";
  ctx.beginPath();
  ctx.moveTo(0, canvas.height / 2);
  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();

  if (values.length >= 2) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * canvas.width;
      const y = canvas.height - ((v - min) / (max - min)) * canvas.height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.restore();
}
