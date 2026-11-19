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
