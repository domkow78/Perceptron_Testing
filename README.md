# Perceptron_Testing

Interaktywna nauka sieci neuronowych: od pojedynczego perceptronu, przez sieci
o rosnącej skali (10 / 100 / 1000+ neuronów), aż po symulację auta RC (kamera +
lidar) sterowanego siecią neuronową.

Projekt jest budowany etapami — aktualny stan: **Etap 3 — sieć 1000+ neuronów (PyTorch)**.

## Stos technologiczny

- Backend: Python + Flask, sieci neuronowe od zera w NumPy + PyTorch dla
  większej skali (bezstanowe API, backend dobierany automatycznie wg
  rozmiaru sieci — `backend/nn/backend_select.py`).
- Frontend: czysty JavaScript (ES modules) + Canvas, bez frameworka i bez build stepu.

## Uruchomienie

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Otwórz `http://localhost:5000` w przeglądarce.

## Testy

```powershell
cd backend
pytest
```

## Etapy

1. **Perceptron** — klasyczna reguła uczenia Rosenblatta, wizualizacja granicy
   decyzyjnej, ręczne suwaki wag, pokazanie ograniczenia na przykładzie XOR.
2. **Sieć wielowarstwowa (10 / 100 neuronów)** — konfigurowalna architektura
   (dowolna liczba warstw ukrytych i neuronów, wybór funkcji aktywacji),
   propagacja wsteczna od zera w NumPy, pole decyzyjne zamiast jednej linii,
   diagram sieci dla małych architektur i mapa cieplna wag dla większych.
3. **Sieć 1000+ neuronów** — ten sam interfejs (`network.html`), ta sama strona
   co w Etapie 2. Dla dużych architektur (`max(layer_sizes) >= 256`, zobacz
   `TORCH_BACKEND_THRESHOLD` w `backend/nn/backend_select.py`) serwer
   automatycznie przełącza się z ręcznego NumPy na `TorchMLP`
   (`nn.Linear` + autograd, ten sam pełnopartyjny spadek gradientu) —
   pokazuje, że ten sam algorytm da się policzyć biblioteką zamiast ręcznie
   wyprowadzonym backpropem. Frontend o tym nie wie — wybór backendu jest
   szczegółem serwera, nie opcją w UI.

   Znane ograniczenie: wagi i biasy są nadal przesyłane w całości jako JSON
   przy każdym kroku treningu (zgodnie z bezstanową architekturą API z Etapu
   1/2) — dla bardzo dużych/wielowarstwowych sieci (np. `[2,1000,1000,1]`)
   oznacza to spory ruch sieciowy na każdy krok. To świadomy kompromis, nie
   błąd — naprawa wymagałaby trzymania wag po stronie serwera (sesje), co
   jest poza zakresem tego etapu.
4. Symulacja 2D auta RC: lidar (raycasting), uproszczona kamera, sterowanie
   siecią neuronową wybranego rozmiaru.
