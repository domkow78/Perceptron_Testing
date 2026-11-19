# Perceptron_Testing

Interaktywna nauka sieci neuronowych: od pojedynczego perceptronu, przez sieci
o rosnącej skali (10 / 100 / 1000+ neuronów), aż po symulację auta RC (kamera +
lidar) sterowanego siecią neuronową.

Projekt jest budowany etapami — aktualny stan: **Etap 2 — sieć wielowarstwowa (10 / 100 neuronów)**.

## Stos technologiczny

- Backend: Python + Flask, sieci neuronowe od zera w NumPy (bezstanowe API).
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
3. Sieć 1000+ neuronów (ten sam interfejs, backend na PyTorch).
4. Symulacja 2D auta RC: lidar (raycasting), uproszczona kamera, sterowanie
   siecią neuronową wybranego rozmiaru.
