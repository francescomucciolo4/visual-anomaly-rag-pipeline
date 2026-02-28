# Anomaly Detection con Autoencoder Convoluzionale sul MVTec Anomaly Detection dataset - Categoria "bottle"
In questo progetto viene mostrata l'implementazione di sistema di rilevamento anomalie a livello immagine utilizzando un Convolutional Autoencoder (CAE) syl dataset MVTec Anomaly Detection dataset, in particolare sulla categoria "Bottle".
L'obiettivo è identificare difetti su bottiglie confrontando l'immagine originale e quella generata dal modello a partire dall'originale, calcolando un punteggio di anomalia basato sull'errore di ricostruzione.

## Dataset
Il progetto utilizza la categoria "Bottle" del dataset MVTec Anomaly Detection, che include:
- 209 immagini di bottiglie senza difetti per il training
- 83 immagini di bottiglie con e senza difetti per la fase di inferenza:
  - 20 immagini di bottiglie senza difetti - Categoria "Good"
  - 20 immagini di bottiglie con difetti evidenti - Categoria "Broken large"
  - 22 immagini di bottiglie con difetti piccoli - Categoria "Broken small"
  - 21 immagini di bottiglie con contaminazioni - Categoria "Contamination"

Il modello viene esclusivamente addestrato sulle immagini senza difetti del training set, in particolare 180 sono state usate per l'addestramento e le restanti 29 per la validazione.
La fase di testing avviene sulle 83 immagini descritte in precedenza, contenenti sia esempi normali che con anomalie.


## Architettura del Modello
- Input: immagini RGB 256×256
- Encoder convoluzionale (4 blocchi Conv + InstanceNorm + LeakyReLU)
- Bottleneck convoluzionale 1×1 (compressione dei canali 256 → 32 → 256)
- Decoder simmetrico con Upsampling bilineare
- Attivazione finale: Sigmoid

  
## Metodologia
1. Addestramento del modello solo su immagini senza difetti
2. Ricostruzione delle immagini di test
3. Calcolo dell’anomaly score per ogni immagine
4. Generazione della ROC curve
5. Calcolo automatico della soglia ottimale tramite Youden’s J statistic (massimizzazione di TPR − FPR)
6. Valutazione tramite confusion matrix


## Metriche di Valutazione
- ROC AUC
- Confusion Matrix
- Precision
- Recall
- F1-score
- Analisi della distribuzione degli anomaly score


## Tecnologie Utilizzate
- Python
- PyTorch
- NumPy
- Scikit-learn
- Pandas
- Matplotlib
- Pandas


## Risultati
IMAGE-LEVEL DETECTION RESULTS
============================================================

Model: best_autoencoder.pth
Dataset: 20 good + 63 anomalies = 83 total

ROC AUC:        0.8992
Threshold:      0.0740

Accuracy Good:    100.0%
Accuracy Anomaly: 69.8%
Mean Accuracy:    84.9%

Precision: 100.0%
Recall:    69.8%
F1 Score:  0.8224
