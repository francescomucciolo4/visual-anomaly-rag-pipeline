"""
=============================================================================
ANOMALY DETECTION CON SSIM - VERSIONE DIDATTICA
=============================================================================

COSA FA QUESTO SCRIPT:
1. Carica un'immagine e la passa attraverso l'autoencoder
2. Confronta l'originale con la ricostruzione usando SSIM
3. Identifica le anomalie dove SSIM è basso (differenza strutturale)
4. Applica una soglia per creare una maschera binaria
5. Confronta con la ground truth

PERCHÉ SSIM E NON MSE:
- MSE confronta pixel per pixel (troppo sensibile a piccole variazioni)
- SSIM confronta REGIONI (finestre 11x11) considerando:
  * Luminanza (brightness)
  * Contrasto (contrast)
  * Struttura (pattern)
- SSIM è più robusto al rumore e cattura meglio anomalie strutturali
=============================================================================
"""

import torch
from cnn_anomaly_detection import CAE256_Latent100
from dataset_anomaly_detection import test_transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from skimage.morphology import remove_small_objects

# =====================================================================
# CONFIGURAZIONE
# =====================================================================

MODEL_PATH = r"C:\Users\Francesco\Desktop\Progetto personale\best_autoencoder.pth"
ANOMALY_ROOT = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\test"
GT_ROOT = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\ground_truth"

CLASS_NAME = "broken_large"
FILENAME = "000.png"

OUTPUT_DIR = "inference_ssim_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# CARICAMENTO MODELLO E DATI
# =====================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CAE256_Latent100().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()

img_path = os.path.join(ANOMALY_ROOT, CLASS_NAME, FILENAME)
gt_path = os.path.join(GT_ROOT, CLASS_NAME, FILENAME.replace(".png", "_mask.png"))

img = Image.open(img_path).convert("RGB")
gt = Image.open(gt_path).convert("L")

img_tensor = test_transforms(img).unsqueeze(0).to(device)
gt_tensor = test_transforms(gt).squeeze().cpu().numpy()


# =====================================================================
# PASSO 1: CALCOLO SSIM
# =====================================================================

print("\n" + "="*70)
print("PASSO 1: CALCOLO STRUCTURAL SIMILARITY INDEX (SSIM)")
print("="*70)

def compute_ssim_anomaly_map(original, reconstructed):
    """
    Calcola una mappa di anomalie usando SSIM.
    
    LOGICA:
    1. SSIM confronta due immagini usando finestre locali (11x11 pixel)
    2. Per ogni finestra, calcola un valore tra -1 e 1:
       - 1 = immagini identiche (NESSUNA anomalia)
       - 0 = completamente diverse (ALTA anomalia)
    3. Restituisce una mappa dove ogni pixel ha il suo SSIM score
    
    Args:
        original: immagine originale (Tensor C×H×W)
        reconstructed: immagine ricostruita (Tensor C×H×W)
    
    Returns:
        anomaly_map: mappa 2D dove valori ALTI = anomalie
        ssim_global: SSIM medio dell'intera immagine
    """
    # Converti da (C, H, W) a (H, W, C) per skimage
    orig_np = original.permute(1, 2, 0).cpu().numpy()
    recon_np = reconstructed.permute(1, 2, 0).cpu().numpy()
    
    print(f"[DEBUG] Original image: min={orig_np.min():.4f}, max={orig_np.max():.4f}")
    print(f"[DEBUG] Reconstructed image: min={recon_np.min():.4f}, max={recon_np.max():.4f}")
    
    print("\n[1.1] Calcolo SSIM con finestre 11x11...")
    
    # Calcola SSIM
    # - win_size=11: usa finestre di 11×11 pixel
    # - channel_axis=2: i canali RGB sono sull'asse 2
    # - data_range=1.0: i valori sono normalizzati tra 0 e 1
    # - full=True: restituisce sia lo score globale che la mappa pixel-wise
    ssim_global, ssim_map = ssim(
        orig_np, 
        recon_np, 
        win_size=11,
        channel_axis=2,
        data_range=1.0,
        full=True
    )
    
    print(f"      SSIM globale: {ssim_global:.4f}")
    print(f"      (1.0 = perfetto, 0.0 = completamente diverso)")
    
    # Se abbiamo 3 canali, fai la media per ottenere una mappa 2D
    if len(ssim_map.shape) == 3:
        ssim_map = ssim_map.mean(axis=2)
        print(f"      Mappa SSIM: shape {ssim_map.shape} (media sui 3 canali RGB)")
    
    print("\n[1.2] Conversione in mappa di anomalie...")
    # SSIM alto (vicino a 1) = simile = NESSUNA anomalia
    # Quindi invertiamo: anomaly = 1 - SSIM
    # Risultato: valori ALTI nella mappa = ANOMALIE
    anomaly_map = 1.0 - ssim_map
    
    print(f"      Anomaly score min: {anomaly_map.min():.4f}")
    print(f"      Anomaly score max: {anomaly_map.max():.4f}")
    print(f"      Anomaly score medio: {anomaly_map.mean():.4f}")
    
    return anomaly_map, ssim_global


# Esegui inference
with torch.no_grad():
    output = model(img_tensor)

original = img_tensor[0]
reconstructed = output[0]

# Calcola mappa SSIM
anomaly_map, ssim_score = compute_ssim_anomaly_map(original, reconstructed)

# Stampa SSIM globale
print("SSIM globale:", ssim_score)

# Stampa mappa delle anomalie
print("Anomaly map shape:", anomaly_map.shape)
print("Anomaly map valori min/max/mean:", anomaly_map.min(), anomaly_map.max(), anomaly_map.mean())

# Se vuoi stampare tutta la matrice (attenzione, è grande!)
print("Anomaly map completa:\n", anomaly_map)


# =====================================================================
# PASSO 2: NORMALIZZAZIONE
# =====================================================================

print("\n" + "="*70)
print("PASSO 2: NORMALIZZAZIONE DELLA MAPPA")
print("="*70)

print("\n[2.1] Normalizzazione tra 0 e 1...")
# Porta tutti i valori tra 0 e 1 per facilitare il thresholding
anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

print(f"      Nuovo range: [{anomaly_map_norm.min():.4f}, {anomaly_map_norm.max():.4f}]")


# =====================================================================
# PASSO 3: THRESHOLDING
# =====================================================================

print("\n" + "="*70)
print("PASSO 3: APPLICAZIONE SOGLIA (THRESHOLD)")
print("="*70)

print("""
PERCHÉ USIAMO UNA SOGLIA:
La mappa di anomalie è continua (valori tra 0 e 1), ma noi vogliamo
una risposta binaria: ANOMALO o NORMALE.

METODO PERCENTILE:
- Ordina tutti i pixel dal meno al più anomalo
- Prende il 95° percentile come soglia
- Solo il TOP 5% dei pixel più anomali viene marcato
- Questo riduce i falsi positivi (pixel normali che sembrano leggermente diversi)
""")

def apply_threshold(anomaly_map, percentile=95):
    """
    Applica una soglia percentile alla mappa di anomalie.
    
    Args:
        anomaly_map: mappa continua di anomalie
        percentile: quale percentile usare (95 = top 5%)
    
    Returns:
        threshold: valore della soglia
        binary_mask: maschera binaria (1=anomalo, 0=normale)
    """
    threshold = np.percentile(anomaly_map, percentile)
    binary_mask = (anomaly_map > threshold).astype(np.uint8)
    
    num_anomalous = binary_mask.sum()
    total_pixels = binary_mask.size
    percentage = (num_anomalous / total_pixels) * 100
    
    print(f"\n[3.1] Soglia calcolata: {threshold:.4f}")
    print(f"      Pixel anomali: {num_anomalous} / {total_pixels} ({percentage:.2f}%)")
    
    return threshold, binary_mask


threshold, binary_mask = apply_threshold(anomaly_map_norm, percentile=95)


# =====================================================================
# PASSO 4: POST-PROCESSING
# =====================================================================

print("\n" + "="*70)
print("PASSO 4: POST-PROCESSING (PULIZIA DELLA MASCHERA)")
print("="*70)

print("""
PROBLEMA: La maschera binaria può avere "rumore" (pixel isolati).
SOLUZIONE: Morphological operations

1. BINARY CLOSING: Connette regioni vicine
   - Usa un kernel 5×5
   - "Chiude" piccoli buchi
   - Connette pixel anomali vicini

2. REMOVE SMALL OBJECTS: Rimuove componenti connessi troppo piccoli
   - Solo gruppi di almeno 50 pixel vengono mantenuti
   - Elimina rumore isolato
""")

def post_process_mask(mask, min_area=50, kernel_size=5):
    """
    Pulisce la maschera binaria.
    
    Args:
        mask: maschera binaria grezza
        min_area: dimensione minima di un'anomalia valida (in pixel)
        kernel_size: dimensione del kernel per closing
    
    Returns:
        cleaned_mask: maschera pulita
    """
    
    mask_bool = mask.astype(bool)
    
    # Closing morfologico
    print(f"\n[4.1] Binary closing con kernel {kernel_size}×{kernel_size}...")
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    mask_closed = ndimage.binary_closing(mask_bool, structure=kernel)
    
    before_pixels = mask_bool.sum()
    after_closing = mask_closed.sum()
    print(f"      Pixel anomali prima: {before_pixels}")
    print(f"      Pixel anomali dopo closing: {after_closing}")
    
    # Rimozione oggetti piccoli
    print(f"\n[4.2] Rimozione componenti < {min_area} pixel...")
    try:
        mask_cleaned = remove_small_objects(mask_closed, min_size=min_area)
        after_cleaning = mask_cleaned.sum()
        removed = after_closing - after_cleaning
        print(f"      Pixel rimossi (rumore): {removed}")
        print(f"      Pixel finali: {after_cleaning}")
    except:
        mask_cleaned = mask_closed
        print(f"      (Fallback: nessuna rimozione)")
    
    return mask_cleaned.astype(np.uint8)


binary_mask_clean = post_process_mask(binary_mask, min_area=50, kernel_size=5)


# =====================================================================
# PASSO 5: VALUTAZIONE
# =====================================================================

print("\n" + "="*70)
print("PASSO 5: CONFRONTO CON GROUND TRUTH")
print("="*70)

print("""
METRICHE UTILIZZATE:

1. IoU (Intersection over Union):
   - Quanto le nostre predizioni si sovrappongono alla GT
   - IoU = Area(Predizione ∩ GT) / Area(Predizione ∪ GT)
   - Range: 0 (pessimo) a 1 (perfetto)

2. Precision (Precisione):
   - Dei pixel che diciamo anomali, quanti lo sono davvero?
   - Precision = True Positives / (True Positives + False Positives)

3. Recall (Sensibilità):
   - Delle anomalie reali, quante ne troviamo?
   - Recall = True Positives / (True Positives + False Negatives)

4. F1 Score:
   - Media armonica di Precision e Recall
   - Bilancia i due obiettivi
""")

def compute_metrics(pred_mask, gt_mask):
    """Calcola metriche di valutazione."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    tp = np.logical_and(pred, gt).sum()  # True Positives
    fp = np.logical_and(pred, ~gt).sum() # False Positives
    fn = np.logical_and(~pred, gt).sum() # False Negatives
    tn = np.logical_and(~pred, ~gt).sum()# True Negatives
    
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    
    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    }


gt_binary = (gt_tensor > 0).astype(np.uint8)
metrics = compute_metrics(binary_mask_clean, gt_binary)

print("\n[5.1] Risultati:")
print(f"      IoU (Intersection over Union): {metrics['IoU']:.4f}")
print(f"      Precision (Precisione):        {metrics['Precision']:.4f}")
print(f"      Recall (Sensibilità):          {metrics['Recall']:.4f}")
print(f"      F1 Score (Bilanciamento):      {metrics['F1']:.4f}")

print(f"\n[5.2] Conteggio pixel:")
print(f"      True Positives  (TP): {metrics['TP']:6d} ✓ (anomalie trovate correttamente)")
print(f"      False Positives (FP): {metrics['FP']:6d} ✗ (falsi allarmi)")
print(f"      False Negatives (FN): {metrics['FN']:6d} ✗ (anomalie mancate)")
print(f"      True Negatives  (TN): {metrics['TN']:6d} ✓ (pixel normali corretti)")


# =====================================================================
# PASSO 6: VISUALIZZAZIONE
# =====================================================================

print("\n" + "="*70)
print("PASSO 6: CREAZIONE VISUALIZZAZIONI")
print("="*70)

def create_overlay(pred_mask, gt_mask):
    """
    Crea un overlay colorato per visualizzare errori:
    - Verde: True Positives (anomalie trovate correttamente)
    - Rosso: False Positives (falsi allarmi)
    - Blu: False Negatives (anomalie mancate)
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)
    
    overlay = np.zeros((gt.shape[0], gt.shape[1], 3))
    overlay[tp] = [0, 1, 0]  # Verde
    overlay[fp] = [1, 0, 0]  # Rosso
    overlay[fn] = [0, 0, 1]  # Blu
    
    return overlay


overlay = create_overlay(binary_mask_clean, gt_binary)

# ---- Visualizzazione completa ----
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('SSIM-Based Anomaly Detection - Step by Step', fontsize=16, fontweight='bold')

# Riga 1: Input e processing
axs[0, 0].imshow(original.permute(1, 2, 0).cpu().numpy())
axs[0, 0].set_title("1. Original Image", fontsize=12, fontweight='bold')
axs[0, 0].axis('off')

axs[0, 1].imshow(reconstructed.permute(1, 2, 0).cpu().numpy())
axs[0, 1].set_title("2. Autoencoder\nReconstruction", fontsize=12, fontweight='bold')
axs[0, 1].axis('off')

im1 = axs[0, 2].imshow(anomaly_map_norm, cmap='hot', vmin=0, vmax=1)
axs[0, 2].set_title(f"3. SSIM Anomaly Map\n(Global SSIM={ssim_score:.3f})", fontsize=12, fontweight='bold')
axs[0, 2].axis('off')
plt.colorbar(im1, ax=axs[0, 2], fraction=0.046, pad=0.04)

axs[0, 3].imshow(binary_mask, cmap='gray')
axs[0, 3].set_title(f"4. Binary Mask\n(Threshold={threshold:.3f})", fontsize=12, fontweight='bold')
axs[0, 3].axis('off')

# Riga 2: Risultati
axs[1, 0].imshow(binary_mask_clean, cmap='gray')
axs[1, 0].set_title("5. Cleaned Mask\n(After Post-processing)", fontsize=12, fontweight='bold')
axs[1, 0].axis('off')

axs[1, 1].imshow(gt_binary, cmap='gray')
axs[1, 1].set_title("6. Ground Truth", fontsize=12, fontweight='bold')
axs[1, 1].axis('off')

axs[1, 2].imshow(overlay)
axs[1, 2].set_title("7. Error Analysis\nGreen=TP, Red=FP, Blue=FN", fontsize=12, fontweight='bold')
axs[1, 2].axis('off')

# Metrics display
axs[1, 3].axis('off')
metrics_text = f"""
FINAL METRICS:

IoU:       {metrics['IoU']:.4f}
Precision: {metrics['Precision']:.4f}
Recall:    {metrics['Recall']:.4f}
F1 Score:  {metrics['F1']:.4f}

PIXEL COUNT:
TP: {metrics['TP']:,}
FP: {metrics['FP']:,}
FN: {metrics['FN']:,}
TN: {metrics['TN']:,}
"""
axs[1, 3].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ssim_pipeline_complete.png"), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[6.1] Visualizzazione salvata: ssim_pipeline_complete.png")

# ---- Visualizzazione semplificata ----
fig2, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(original.permute(1, 2, 0).cpu().numpy())
axs[0].set_title("Original", fontsize=14, fontweight='bold')
axs[0].axis('off')

axs[1].imshow(gt_binary, cmap='gray')
axs[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
axs[1].axis('off')

axs[2].imshow(binary_mask_clean, cmap='gray')
axs[2].set_title("SSIM Prediction", fontsize=14, fontweight='bold')
axs[2].axis('off')

axs[3].imshow(overlay)
axs[3].set_title(f"Results\nF1={metrics['F1']:.4f} | IoU={metrics['IoU']:.4f}", 
                 fontsize=14, fontweight='bold', color='darkgreen')
axs[3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ssim_results_simple.png"), dpi=150, bbox_inches='tight')
plt.close()

print(f"[6.2] Visualizzazione salvata: ssim_results_simple.png")


# =====================================================================
# RIEPILOGO FINALE
# =====================================================================

print("\n" + "="*70)
print("RIEPILOGO FINALE")
print("="*70)

print(f"""
PIPELINE SSIM COMPLETA:

1. AUTOENCODER
   ↓ L'autoencoder tenta di ricostruire l'immagine
   ↓ Immagini con difetti vengono ricostruite male
   
2. SSIM COMPARISON
   ↓ SSIM confronta originale vs ricostruita usando finestre 11×11
   ↓ Genera mappa di anomalie (1-SSIM)
   ↓ Global SSIM: {ssim_score:.4f}
   
3. THRESHOLDING
   ↓ Soglia al 95° percentile: {threshold:.4f}
   ↓ {binary_mask.sum()} pixel marcati come anomali
   
4. POST-PROCESSING
   ↓ Binary closing (kernel 5×5)
   ↓ Rimozione oggetti < 50 pixel
   ↓ {binary_mask_clean.sum()} pixel finali
   
5. EVALUATION
   ↓ Confronto con Ground Truth
   ✓ F1 Score: {metrics['F1']:.4f}
   ✓ IoU: {metrics['IoU']:.4f}

RISULTATI SALVATI IN: {OUTPUT_DIR}
""")

print("="*70 + "\n")