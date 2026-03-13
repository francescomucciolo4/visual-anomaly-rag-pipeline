# Industrial Anomaly Detection Pipeline con CAE, VLM e RAG

<p align="center">
  <img src="https://github.com/user-attachments/assets/4e66f3b3-afa7-4be7-9ebf-f0b93bc7378d" width="900" />
  <br><br>
  <em>Risultati Anomaly Detection</em>
</p>

Pipeline end-to-end per il rilevamento e l'analisi di difetti su bottiglie di vetro, combinando un **Convolutional Autoencoder (CAE)** per il rilevamento delle anomalie, un **Vision Language Model (LLaVA)** per la descrizione visiva dei difetti e un sistema **RAG (Retrieval-Augmented Generation)** per la generazione automatica di report di qualità.

---

## Architettura del Sistema
```
                  Immagine
                      ↓
        [CAE] → Anomaly Score
            ↓ (se anomalia)
        [LLaVA] → Descrizione testuale del difetto
                      ↓
        ┌─────────────────────────────┐
        │         RAG SYSTEM          │
        │                             │
        │  Query: descrizione VLM     │
        │         ↓                   │
        │  [ChromaDB] similarity      │
        │  search su Knowledge Base   │
        │         ↓                   │
        │  Chunk rilevanti retrieval  │
        │         ↓                   │
        │  [Llama 3] genera report    │
        └─────────────────────────────┘
                      ↓
  Report: causa probabile + azioni correttive + urgenza
```

---

## Dataset

Il progetto utilizza la categoria **"Bottle"** del dataset **MVTec Anomaly Detection**:

- **Training set:** 209 immagini senza difetti
  - 180 immagini per addestramento
  - 29 immagini per validazione
- **Test set:** 83 immagini
  - 20 immagini "Good"
  - 20 immagini "Broken large"
  - 22 immagini "Broken small"
  - 21 immagini "Contamination"

> Il modello è addestrato **solo** su immagini senza difetti.

---

## Componenti

### 1. Convolutional Autoencoder
- **Input:** immagini RGB 256×256
- **Encoder:** 4 blocchi Conv + InstanceNorm + LeakyReLU
- **Bottleneck:** compressione canali 256 → 32 → 256
- **Decoder:** Upsampling bilineare + Sigmoid
- **Anomaly score:** errore di ricostruzione (CombinedLoss)
- **Threshold:** calcolato automaticamente con Youden's J statistic

### 2. Vision Language Model
- **Modello:** LLaVA-LLaMA3 via Ollama
- **Input:** immagini classificate come anomale dal CAE + descrizione testuale con classe del difetto come contesto
- **Output:** descrizione testuale in linguaggio naturale del difetto visibile

### 3. RAG Agent
- **Vector store:** ChromaDB
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **LLM:** Llama 3 via Ollama
- **Input:** descrizioni testuali generate dal VLM
- **Knowledge base:** catalogo difetti + azioni correttive (indicizzata in ChromaDB)
- **Retrieval:** similarity search tra la descrizione VLM e i chunk della knowledge base
- **Output:** report per ogni anomalia con causa probabile e azioni correttive

---

## Risultati Anomaly Detection

| Metrica          | Valore  |
|------------------|---------|
| ROC AUC          | 0.8992  |
| Threshold        | 0.0740  |
| Accuracy Good    | 100.0%  |
| Accuracy Anomaly | 69.8%   |
| Mean Accuracy    | 84.9%   |
| Precision        | 100.0%  |
| Recall           | 69.8%   |
| F1 Score         | 0.8224  |

---

## Esempio di Report generato
```
Image: test\broken_large\000.png
Class: broken_large
Anomaly Score: 0.0893
VLM Description: The top of the bottle has a large crack in it.
```

**Quality Control Report**

**Defect Summary:**
The top of the glass bottle has a large crack, classified as "broken_large". 
This defect compromises the structural integrity of the bottle and may pose 
a risk to product safety and quality.

**Probable Causes:**
1. Improper handling or transportation causing stress and fatigue on the glass
2. Defective glass manufacturing process or material
3. Inadequate quality control checks during manufacturing

**Corrective Actions:**
1. Implement additional quality control checks during manufacturing
2. Improve handling and transportation procedures
3. Conduct a thorough investigation into the manufacturing process

**Urgency Level:** High

The presence of a large crack on the bottle's top opening poses a significant risk to product safety and quality. Immediate corrective action is required to prevent further defects and ensure the integrity of the product.

---

## Tecnologie Utilizzate

- Python, PyTorch, NumPy, Scikit-learn, Pandas, Matplotlib
- Ollama (LLaVA-LLaMA3 , Llama 3)
- LangChain, ChromaDB, sentence-transformers

---

## Note
> Questo repository ha finalità illustrative e di portfolio personale
>
> Parte del codice è stato generato con il supporto di Claude
