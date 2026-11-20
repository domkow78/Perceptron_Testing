# Perceptron_Testing

Interaktywna nauka sieci neuronowych: od pojedynczego perceptronu, przez sieci
o rosnącej skali (10 / 100 / 1000+ neuronów), aż po symulację auta RC (kamera +
lidar) sterowanego siecią neuronową.

Projekt jest budowany etapami — aktualny stan: **Etap 4 — symulacja auta RC (lidar + klonowanie zachowania)**.

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
4. **Symulacja 2D auta RC** — fizyka auta, lidar (raycasting) i tor liczone
   w całości po stronie frontendu (`frontend/js/sim/*.js`, pętla
   `requestAnimationFrame`); backend wywoływany tylko do budowy/treningu/
   predykcji sieci. Tor to owal z dwiema szykanami (S-zakrętami wpisanymi
   w proste odcinki — bez nich zwykły owal wymagałby skrętu tylko w jedną
   stronę przez całe okrążenie, co dałoby sieci stale jednostronne dane),
   zapisany jako dwie zamknięte pętle ścian-odcinków
   (`frontend/js/sim/track.js`, generowany metodą "żółwia": pół okrążenia
   o łącznym skręcie 180° przejechane dwa razy automatycznie zamyka się
   w pętlę). Auto porusza się modelem unicycle (stała
   prędkość, jedyny kanał sterowania to skręt — `frontend/js/sim/car.js`).
   Lidar to 7 promieni w rozpiętości 180°, znormalizowane do [0,1]
   (`frontend/js/sim/lidar.js`); ta sama tablica odczytów napędza pasek
   "kamery" (kolory z już istniejącej `decisionFieldColor`).

   Sterowanie jest **binarne** (skręt lewo=0 / prawo=1), bo `MLP`/`TorchMLP`
   mają na stałe jeden neuron sigmoidalny + binarną entropię krzyżową —
   dzięki temu **Etap 4 nie zmienia ani linijki w backendzie**: wejście
   sieci to po prostu 7 odczytów lidaru zamiast 2D punktu, a
   `/api/mlp/init|step|predict` działają bez modyfikacji. Dane treningowe
   (klonowanie zachowania) powstają przez jazdę manualną (strzałki/A D) —
   próbka jest nagrywana tylko wtedy, gdy trzymany jest klawisz skrętu
   (jazda na wprost nie jest nagrywana, żeby uniknąć niejednoznacznej
   trzeciej klasy dla klasyfikatora binarnego).

   Znane ograniczenia: sterowanie włącz/wyłącz (bez siły skrętu) daje jazdę
   "na przełącznik", nie płynną linię wyścigową — złagodzone strefą
   martwą wokół 0.5 w trybie autonomicznym, ale nie wyeliminowane; jakość
   wytrenowanego kierowcy zależy od jakości jazdy manualnej użytkownika
   (akceptowalne jako demo edukacyjne, nie produkcyjna robotyka).
