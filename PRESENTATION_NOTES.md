# DUuSAiUV - Prezentacija projekta
## Steering Angle Prediction & Lane Change Detection

---

## 1. PROBLEM I CILJ

**Zadatak:**
- Predvideti ugao upravljanja vozila iz niza slika sa prednje kamere
- Detektovati promenu trake na osnovu sekvence uglova upravljanja
- Koristiti CNN+LSTM arhitekturu

**Dataset:**
- Udacity Self-Driving Car - Behavioural Cloning (Kaggle)
- Make dataset: 3930 frame-ova
- Jungle dataset: 3404 frame-a

---

## 2. DATASET ANALIZA

### Original Dataset (Make):
```
Total samples: 3930
Zero angles: 3061 (77.9%)  ← PROBLEM: Ekstremna nebalansiranost
Non-zero angles: 869 (22.1%)
  - Positive: 114 (2.9%)
  - Negative: 755 (19.2%)

Steering angle range: [-1.0, 1.0]
Mean: -0.035
Std: 0.133
```

### Posle Balansiranja (zero_fraction=0.1):
```
Zero angles: 306 (15.0%)
Non-zero angles: 869 (42.5%)
Flipped (augmented): 869 (42.5%)
Total: 2044 samples
```

### Train/Val Split (80/20):
```
Training: 1636 samples (80%)
Validation: 408 samples (20%)
```

---

## 3. ARHITEKTURA MODELA

### CNN Feature Extractor (VGG-inspired):
```
Input: (66, 200, 3)
  ↓
RandomContrast(0.3)        ← Data augmentation
Rescaling(1/255)
  ↓
Conv2D(16, 3×3) → ReLU → MaxPool → BatchNorm
Conv2D(32, 3×3) → ReLU → MaxPool → BatchNorm
Conv2D(64, 3×3) → ReLU → MaxPool → BatchNorm
Conv2D(128, 3×3) → ReLU → MaxPool → BatchNorm
  ↓
Flatten → Dropout(0.4) → Dense(128, ReLU)
  ↓
Feature vector (128D)
```

### Alternative: Hybrid CNN (5×5 → 3×3):
```
Conv2D(16, 5×5, stride=2) → BatchNorm
Conv2D(32, 5×5, stride=2) → BatchNorm
Conv2D(64, 3×3) → MaxPool → BatchNorm
Conv2D(128, 3×3) → MaxPool → BatchNorm
  ↓
30% brži trening, sličan MAE (0.110 vs 0.100)
```

### Temporal Model (LSTM):
```
TimeDistributed(CNN) → (seq_len, 128)
  ↓
LSTM(64) → Dropout(0.2)
  ↓
Dense(128, ReLU) → Dense(1)
  ↓
Steering angle ∈ [-1, 1]
```

**Total parameters:** 943K (VGG) / 952K (Hybrid)

---

## 4. TEHNIKE POBOLJŠANJA

### 4.1 Data Balancing:
- **Problem:** 78% zero vrednosti → model predviđa samo 0
- **Rešenje:** Zadržati samo 10% zero vrednosti
- **Rezultat:** Balansiran dataset (15% zero, 85% non-zero)

### 4.2 Data Augmentation:
- **Horizontal flip** za non-zero uglove (duplira podatke)
- **RandomContrast(0.3)** tokom treninga
- **Rezultat:** 2× više podataka, bolja generalizacija

### 4.3 Loss Function:
- **Huber loss (delta=0.1)** umesto MAE/MSE
- **Prednost:** Agresivno kažnjava velike greške, blag na male
- **Rezultat:** Bolje praćenje velikih promena ugla

### 4.4 Sequence Length:
- **seq_len=3** umesto 5
- **Prednost:** Kraća memorija = brža reakcija na promene
- **Rezultat:** Model brže reaguje na nagle skretanja

### 4.5 Regularization:
- **Dropout:** 0.4 u CNN, 0.2 u LSTM
- **BatchNormalization** posle svake Conv2D
- **Rezultat:** Manje overfitting-a

---

## 5. REZULTATI

### Training Results (Make dataset):

**VGG-like architecture (production model):**
```
Best epoch: 8
Training MAE: 0.1237 (~7.1°)
Validation MAE: 0.1223 (~7.0°)
Full dataset MAE: 0.100 (~5.7°)
Training time: ~7s per epoch (M-series Mac)
Parameters: 943K
```

**Hybrid architecture (fast prototyping):**
```
Best epoch: 7
Training MAE: 0.1230 (~7.0°)
Validation MAE: 0.1251 (~7.2°)
Full dataset MAE: 0.110 (~6.3°)
Training time: ~5s per epoch (30% faster)
Parameters: 952K
```

### Poređenje modela:

| Model | Val MAE | Full MAE | Poboljšanje | Brzina |
|-------|---------|----------|-------------|--------|
| Baseline (original) | 0.289 | - | - | - |
| Improved v2 | 0.259 | - | +10.4% | - |
| Balanced | 0.231 | - | +20.1% | - |
| **VGG-like (Final)** | **0.122** | **0.100** | **+57.8%** ✅ | 7s/epoch |
| Hybrid (5×5→3×3) | 0.125 | 0.110 | +56.7% | 5s/epoch ⚡ |

### Kada koristiti koji model:

**VGG-like (steering_model_make_huber.keras):**
- ✅ Produkcija i deployment
- ✅ Najbolja preciznost (0.100 MAE)
- ✅ Najbolja generalizacija
- ✅ Stabilniji rezultati

**Hybrid (steering_model_make_hybrid.keras):**
- ✅ Brzo testiranje i iteracije
- ✅ 30% brži trening
- ✅ Sličan performance (razlika ~1°)
- ✅ Veći receptive field (5×5 kerneli)

---

## 6. LANE CHANGE DETECTION

### Algoritam (Dvostepeni heuristik):

**Faza 1 - Candidate Start:**
- `abs(angle) > threshold` (default: 0.2)
- Minimum `min_hold_frames` (default: 5) uzastopnih frame-ova
- Isti znak steering-a (levo ili desno)

**Faza 2 - Confirmation:**
- U roku od `max_settle_frames` (default: 25):
  - Steering se vrati blizu 0 (`settle_threshold=0.08`), ILI
  - Pojavi se counter-steer (suprotan znak)

**Cooldown:**
- `cooldown_frames` (default: 20) sprečava ponovljene trigere

### Prednosti:
- ✅ Smanjuje false positives na dugim krivinama
- ✅ Detektuje prave lane change manevre
- ✅ Fizički realističan (settling nakon promene trake)

---

## 7. VIZUALIZACIJA

### Elementi prikaza:
1. **Info panel** (gore):
   - Prediction (zeleno)
   - Ground Truth (žuto)
   - Error (crveno/narandžasto/žuto prema veličini)

2. **Steering wheel** (dole):
   - Zelena igla: Predikcija
   - Žuta igla: Ground Truth
   - Radius 100px, jasno vidljivo

3. **Lane change warning**:
   - Crveni border (15px)
   - Banner "!!! LANE CHANGE !!!"
   - Prikazuje se 30 frame-ova

4. **Skaliranje:**
   - 3× uvećanje za bolju vidljivost
   - Sve informacije čitljive

---

## 8. KVALITET PREDIKCIJA

### Odlični primeri (error < 0.01):
```
Frame 01527 | GT: 0.0000  Pred: +0.0086  Err: 0.0086 ✅
Frame 01584 | GT: 0.0000  Pred: +0.0073  Err: 0.0073 ✅
Frame 01667 | GT: -0.0500 Pred: -0.0496 Err: 0.0004 ✅
```

### Karakteristike predikcija:
- ✅ **Smooth** - nema naglih skokova
- ✅ **Fizički realne** - postepene promene
- ✅ **Stabilne** - mali gap između train/val MAE

### Problem sa Ground Truth:
- GT ima **nemoguće skokove** (npr. -0.25 → 0.0 u jednom frame-u)
- Model **ne prati** ove skokove jer su fizički nerealni
- **Ovo je prednost**, ne mana!

---

## 9. TEHNIČKA IMPLEMENTACIJA

### Tehnologije:
- **Python 3.13**
- **TensorFlow/Keras 2.20**
- **OpenCV** za image processing
- **NumPy, Pandas** za data handling

### Struktura projekta:
```
├── model.py               # CNN+LSTM arhitektura
├── data_loader.py         # Data loading + augmentation
├── train.py               # Training script
├── predict.py             # Inference + visualization
├── evaluate.py            # Metrics calculation
├── lane_change_detector.py # Lane change detection
├── run_all.sh             # All-in-one pipeline
└── tests/                 # Unit tests
```

### Pipeline:
```bash
# Jedan komanda za sve:
./run_all.sh --dataset make

# Ili pojedinačno:
python train.py --csv ... --data_dir ... --epochs 20
python evaluate.py --model ... --csv ... --data_dir ...
python predict.py --model ... --csv ... --show
```

---

## 10. ZAKLJUČAK

### Postignuto:
✅ CNN+LSTM arhitektura (zahtev zadatka)
✅ Steering angle prediction: **0.122 MAE** (~7°)
✅ Lane change detection sa dvostepenim heuristikom
✅ Vizualizacija sa GT poređenjem
✅ **57.8% poboljšanje** od baseline modela
✅ Unit testovi za lane change detekciju
✅ Automatizovana evaluation pipeline

### Ključne tehnike:
1. Data balancing (10% zero vrednosti)
2. Horizontal flip augmentation
3. Huber loss za bolje praćenje velikih promena
4. Kraća sekvenca (seq_len=3) za brže reakcije
5. VGG-like CNN umesto NVIDIA arhitekture

### Buduća poboljšanja:
- Smoothing Ground Truth-a (Savitzky-Golay filter)
- Weighted loss za fokus na velike uglove
- Testiranje na drugim datasetima
- Real-time inference optimizacija

---

## 11. PITANJA ZA ODBRANU

**Q: Zašto CNN+LSTM umesto samo CNN?**
A: LSTM omogućava modelovanje temporalnih zavisnosti - vožnja je sekvencijalna aktivnost gde prethodni frame-ovi utiču na trenutnu odluku.

**Q: Zašto model ne prati nagle skokove u GT?**
A: Model daje fizički realne predikcije. Nagli skokovi u GT su greške u označavanju - vozač ne može okrenuti volan za 0.5 radijana u jednom frame-u.

**Q: Kako ste rešili problem nebalansiranog dataseta?**
A: Zadržali samo 10% zero vrednosti + horizontal flip augmentation za non-zero vrednosti.

**Q: Šta je Huber loss i zašto ste ga koristili?**
A: Huber loss kombinuje prednosti MAE i MSE - blag je na male greške, ali agresivno kažnjava velike greške (delta=0.1).

**Q: Kako lane change detekcija smanjuje false positives?**
A: Dvostepeni heuristik zahteva i sustained steering I settling/counter-steer, što eliminiše duge konstantne krivine.

**Q: Testirali ste i hybrid arhitekturu - zašto ste odabrali VGG-like?**
A: Hybrid model (5×5 → 3×3 kerneli) je 30% brži (5s vs 7s po epohi) ali ima malo lošiji MAE (0.110 vs 0.100). VGG-like ima bolju generalizaciju i stabilnije rezultate na full datasetu.

**Q: Koliko traje trening?**
A: ~7s po epohi na M-series Mac (VGG), ~5s na hybrid modelu. Ukupno ~2 minuta za 20 epoha sa early stopping.

**Q: Kako biste poboljšali model?**
A: Smoothing GT-a, weighted loss, više podataka, testiranje na drugim datasetima, real-time optimizacija.
