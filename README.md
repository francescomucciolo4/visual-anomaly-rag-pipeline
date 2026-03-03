# Anomaly Detection con Autoencoder Convoluzionale sul MVTec Anomaly Detection Dataset - Categoria "Bottle"

In questo progetto viene mostrata l'implementazione di un sistema di rilevamento anomalie a livello immagine utilizzando un **Convolutional Autoencoder (CAE)** sul dataset **MVTec Anomaly Detection**, in particolare sulla categoria **"Bottle"**.  

L'obiettivo è identificare difetti sulle bottiglie confrontando l'immagine originale con quella generata dal modello a partire dall'originale, calcolando un **punteggio di anomalia** basato sull'errore di ricostruzione.

---

## Dataset

Il progetto utilizza la categoria **"Bottle"** del dataset **MVTec Anomaly Detection**:

- **Training set:** 209 immagini senza difetti  
  - 180 immagini per addestramento  
  - 29 immagini per validazione
- **Test set:** 83 immagini con e senza difetti
  - 20 immagini "Good" (senza difetti)  
  - 20 immagini "Broken large" (difetti evidenti)  
  - 22 immagini "Broken small" (difetti piccoli)  
  - 21 immagini "Contamination" (contaminazioni)

>  Il modello è addestrato **solo** su immagini senza difetti.  

---

## Architettura del Modello

- **Input:** immagini RGB 256×256  
- **Encoder convoluzionale:** 4 blocchi Conv + InstanceNorm + LeakyReLU  
- **Bottleneck convoluzionale 1×1:** compressione canali 256 → 32 → 256  
- **Decoder simmetrico:** Upsampling bilineare  
- **Attivazione finale:** Sigmoid  

---

## Metodologia

1. Addestramento del modello solo su immagini senza difetti  
2. Ricostruzione delle immagini del test set  
3. Calcolo dell’**anomaly score** per ogni immagine  
4. Generazione della **ROC curve**  
5. Calcolo automatico della **soglia ottimale** tramite Youden’s J statistic (massimizzazione di TPR − FPR)  
6. Valutazione tramite **confusion matrix**  

---

## Metriche di Valutazione

- **ROC AUC**  
- **Confusion Matrix**  
- **Precision**  
- **Recall**  
- **F1-score**  
- Analisi della distribuzione degli anomaly score  

---

## Tecnologie Utilizzate

- Python  
- PyTorch  
- NumPy  
- Scikit-learn  
- Pandas  
- Matplotlib  

---

## Risultati

| Metrica                  | Valore       |
|--------------------------|-------------|
| ROC AUC                  | 0.8992      |
| Threshold                | 0.0740      |
| Accuracy Good            | 100.0%      |
| Accuracy Anomaly         | 69.8%       |
| Mean Accuracy            | 84.9%       |
| Precision                | 100.0%      |
| Recall                   | 69.8%       |
| F1 Score                 | 0.8224      |

---

## Note

> Questo repository ha finalità esclusivamente illustrative e di portfolio personale
> Parte del codice è stato generato utilizzando Claude 
