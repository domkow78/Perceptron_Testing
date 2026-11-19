export async function fetchDataset(name) {
  const res = await fetch(`/api/datasets/${name}`);
  if (!res.ok) throw new Error(`Nie udało się pobrać datasetu: ${name}`);
  return res.json();
}

export async function perceptronStep({ weights, bias, dataset, learningRate }) {
  const res = await fetch("/api/perceptron/step", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights, bias, dataset, learning_rate: learningRate }),
  });
  if (!res.ok) throw new Error("Błąd podczas kroku treningu");
  return res.json();
}

export async function perceptronPredict({ weights, bias, points }) {
  const res = await fetch("/api/perceptron/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights, bias, points }),
  });
  if (!res.ok) throw new Error("Błąd podczas predykcji");
  return res.json();
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Błąd zapytania do ${url}`);
  return res.json();
}

export function mlpInit({ architecture, hiddenActivation, seed }) {
  return postJson("/api/mlp/init", { architecture, hidden_activation: hiddenActivation, seed });
}

export function mlpStep({ architecture, hiddenActivation, weights, biases, dataset, learningRate }) {
  return postJson("/api/mlp/step", {
    architecture,
    hidden_activation: hiddenActivation,
    weights,
    biases,
    dataset,
    learning_rate: learningRate,
  });
}

export function mlpPredict({ architecture, hiddenActivation, weights, biases, points }) {
  return postJson("/api/mlp/predict", {
    architecture,
    hidden_activation: hiddenActivation,
    weights,
    biases,
    points,
  });
}

export function mlpActivations({ architecture, hiddenActivation, weights, biases, point }) {
  return postJson("/api/mlp/activations", {
    architecture,
    hidden_activation: hiddenActivation,
    weights,
    biases,
    point,
  });
}
