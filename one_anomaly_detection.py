import torch
import torch.nn.functional as F
from cnn_anomaly_detection import CAE256_Latent100
from dataset_anomaly_detection import test_transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ---- Config ----
MODEL_PATH = r"C:\Users\Francesco\Desktop\Progetto personale\best_autoencoder.pth"

ANOMALY_ROOT = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\test"
GT_ROOT = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\ground_truth"

CLASS_NAME = "broken_large"      # scegli la classe
FILENAME = "000.png"             # scegli l'immagine

OUTPUT_DIR = "inference_ssim_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Modello ----
model = CAE256_Latent100().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---- Caricamento immagine e GT ----
img_path = os.path.join(ANOMALY_ROOT, CLASS_NAME, FILENAME)
gt_path = os.path.join(
    GT_ROOT,
    CLASS_NAME,
    FILENAME.replace(".png", "_mask.png")
)

img = Image.open(img_path).convert("RGB")
gt = Image.open(gt_path).convert("L")

img_tensor = test_transforms(img).unsqueeze(0).to(device)
gt_tensor = test_transforms(gt).squeeze().cpu().numpy()  # per visualizzazione


def compute_ssim_heatmap(original, reconstructed, window_size=11):
    """
    Calcola la heatmap SSIM tra l'immagine originale e quella ricostruita.
    
    Args:
        original: Tensor (C, H, W) - immagine originale
        reconstructed: Tensor (C, H, W) - immagine ricostruita
        window_size: dimensione della finestra per SSIM (default 11x11)
    
    Returns:
        ssim_map: numpy array (H, W) - mappa SSIM (valori alti = simili, bassi = anomalie)
        stats: dizionario con statistiche
    """
    # Converti in numpy e trasforma in (H, W, C)
    orig_np = original.permute(1, 2, 0).cpu().numpy()
    recon_np = reconstructed.permute(1, 2, 0).cpu().numpy()
    
    # Calcola SSIM con mappa completa
    # full=True restituisce sia lo score globale che la mappa pixel-wise
    ssim_score, ssim_map = ssim(
        orig_np, 
        recon_np, 
        win_size=window_size,
        channel_axis=2,  # specifica che i canali sono sull'ultimo asse
        data_range=1.0,  # assumiamo che le immagini siano normalizzate [0, 1]
        full=True
    )
    
    # SSIM restituisce una mappa (H, W, C) quando ci sono più canali
    # Prendiamo la media sui canali per ottenere una mappa 2D
    if len(ssim_map.shape) == 3:
        ssim_map = ssim_map.mean(axis=2)
    
    # SSIM restituisce valori tra -1 e 1, dove 1 = identico
    # Per anomalie vogliamo valori alti dove c'è differenza
    # Quindi invertiamo: anomaly_map = 1 - SSIM
    anomaly_map = 1.0 - ssim_map
    
    # Statistiche
    stats = {
        "ssim_global": ssim_score,
        "anomaly_mean": np.mean(anomaly_map),
        "anomaly_std": np.std(anomaly_map),
        "anomaly_max": np.max(anomaly_map),
        "anomaly_min": np.min(anomaly_map)
    }
    
    return anomaly_map, stats


def compute_mse_heatmap(original, reconstructed):
    """
    Calcola la heatmap MSE classica per confronto.
    """
    error_pixel = original - reconstructed
    error_squared = error_pixel ** 2
    heatmap = error_squared.mean(dim=0).cpu().numpy()
    
    stats = {
        "mse_total": error_squared.mean().item(),
        "mse_max": error_squared.max().item(),
        "mse_min": error_squared.min().item()
    }
    
    return heatmap, stats


def adaptive_threshold(heatmap, method='percentile', percentile=95, factor=3.0):
    """
    Applica una soglia adattiva alla heatmap.
    
    Args:
        heatmap: numpy array - mappa delle anomalie (deve essere 2D)
        method: 'percentile', 'otsu', 'mean_std'
        percentile: percentile da usare se method='percentile'
        factor: fattore moltiplicativo per mean+std se method='mean_std'
    
    Returns:
        threshold: valore della soglia
        binary_mask: maschera binaria
    """
    # Assicurati che sia 2D
    if len(heatmap.shape) > 2:
        heatmap = heatmap.mean(axis=-1)
    
    if method == 'percentile':
        threshold = np.percentile(heatmap, percentile)
    
    elif method == 'mean_std':
        mean = np.mean(heatmap)
        std = np.std(heatmap)
        threshold = mean + factor * std
    
    elif method == 'otsu':
        # Implementazione semplice di Otsu
        from skimage.filters import threshold_otsu
        # Otsu richiede un'immagine 2D
        threshold = threshold_otsu(heatmap)
    
    else:
        raise ValueError(f"Unknown thresholding method: {method}")
    
    binary_mask = (heatmap > threshold).astype(np.uint8)
    
    return threshold, binary_mask


def post_process_mask(mask, min_area=50):
    """
    Post-processing della maschera per rimuovere rumore.
    
    Args:
        mask: maschera binaria (H, W)
        min_area: area minima per mantenere un componente connesso
    
    Returns:
        cleaned_mask: maschera pulita
    """
    from skimage.morphology import remove_small_objects
    from scipy import ndimage
    
    # Assicurati che sia 2D
    if len(mask.shape) > 2:
        mask = mask.mean(axis=-1)
    
    # Converti in bool
    mask_bool = mask.astype(bool)
    
    # Chiusura morfologica usando scipy (più robusto)
    # Usa un kernel 5x5
    kernel = np.ones((5, 5), dtype=bool)
    mask_closed = ndimage.binary_closing(mask_bool, structure=kernel, iterations=1)
    
    # Rimuovi piccoli oggetti (rumore)
    try:
        mask_cleaned = remove_small_objects(mask_closed, min_size=min_area)
    except:
        # Se fallisce, ritorna la maschera chiusa senza rimuovere piccoli oggetti
        mask_cleaned = mask_closed
    
    return mask_cleaned.astype(np.uint8)


def compute_metrics(pred_mask, gt_mask):
    """
    Calcola metriche di valutazione.
    """
    # Assicuriamoci che siano binarie
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    # True/False Positives/Negatives
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    
    # Metriche
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    }


def create_overlay(pred_mask, gt_mask):
    """
    Crea overlay colorato: Verde=TP, Rosso=FP, Blu=FN
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)
    
    overlay = np.zeros((gt.shape[0], gt.shape[1], 3))
    overlay[tp] = [0, 1, 0]   # Verde
    overlay[fp] = [1, 0, 0]   # Rosso
    overlay[fn] = [0, 0, 1]   # Blu
    
    return overlay


# =====================================================================
# INFERENCE
# =====================================================================

print("=" * 70)
print("SSIM-BASED ANOMALY DETECTION")
print("=" * 70)

with torch.no_grad():
    output = model(img_tensor)

orig = img_tensor[0]
recon = output[0]

# ---- 1. Calcola heatmap SSIM ----
print("\n[1] Calculating SSIM heatmap...")
ssim_heatmap, ssim_stats = compute_ssim_heatmap(orig, recon, window_size=11)

print(f"  Global SSIM: {ssim_stats['ssim_global']:.4f}")
print(f"  Anomaly score (mean): {ssim_stats['anomaly_mean']:.4f}")
print(f"  Anomaly score (max): {ssim_stats['anomaly_max']:.4f}")

# ---- 2. Calcola heatmap MSE (per confronto) ----
print("\n[2] Calculating MSE heatmap (for comparison)...")
mse_heatmap, mse_stats = compute_mse_heatmap(orig, recon)
mse_heatmap_norm = (mse_heatmap - mse_heatmap.min()) / (mse_heatmap.max() - mse_heatmap.min() + 1e-8)

print(f"  Global MSE: {mse_stats['mse_total']:.6f}")

# ---- 3. Normalizza SSIM heatmap ----
ssim_heatmap_norm = (ssim_heatmap - ssim_heatmap.min()) / (ssim_heatmap.max() - ssim_heatmap.min() + 1e-8)

# ---- 4. Applica threshold adattivo ----
print("\n[3] Applying adaptive threshold...")

# Prova diversi metodi di thresholding
threshold_ssim_95, mask_ssim_95 = adaptive_threshold(ssim_heatmap_norm, method='percentile', percentile=95)
threshold_ssim_otsu, mask_ssim_otsu = adaptive_threshold(ssim_heatmap_norm, method='otsu')
threshold_mse_95, mask_mse_95 = adaptive_threshold(mse_heatmap_norm, method='percentile', percentile=95)

print(f"  SSIM threshold (95th percentile): {threshold_ssim_95:.4f}")
print(f"  SSIM threshold (Otsu): {threshold_ssim_otsu:.4f}")
print(f"  MSE threshold (95th percentile): {threshold_mse_95:.4f}")

# ---- 5. Post-processing ----
print("\n[4] Post-processing masks...")
mask_ssim_95_clean = post_process_mask(mask_ssim_95, min_area=50)
mask_ssim_otsu_clean = post_process_mask(mask_ssim_otsu, min_area=50)
mask_mse_95_clean = post_process_mask(mask_mse_95, min_area=50)

# ---- 6. Ground Truth ----
gt_binary = (gt_tensor > 0).astype(np.uint8)

# ---- 7. Calcola metriche ----
print("\n[5] Computing metrics...")
print("\n--- SSIM (95th percentile) ---")
metrics_ssim_95 = compute_metrics(mask_ssim_95_clean, gt_binary)
for k, v in metrics_ssim_95.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

print("\n--- SSIM (Otsu threshold) ---")
metrics_ssim_otsu = compute_metrics(mask_ssim_otsu_clean, gt_binary)
for k, v in metrics_ssim_otsu.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

print("\n--- MSE (95th percentile) ---")
metrics_mse_95 = compute_metrics(mask_mse_95_clean, gt_binary)
for k, v in metrics_mse_95.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# ---- 8. Crea overlays ----
overlay_ssim_95 = create_overlay(mask_ssim_95_clean, gt_binary)
overlay_ssim_otsu = create_overlay(mask_ssim_otsu_clean, gt_binary)
overlay_mse_95 = create_overlay(mask_mse_95_clean, gt_binary)

# =====================================================================
# VISUALIZZAZIONE
# =====================================================================

print("\n[6] Creating visualizations...")

# ---- Plot 1: Confronto heatmap ----
fig1, axs = plt.subplots(2, 4, figsize=(20, 10))

axs[0, 0].imshow(orig.permute(1, 2, 0).cpu().numpy())
axs[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
axs[0, 0].axis('off')

axs[0, 1].imshow(recon.permute(1, 2, 0).cpu().numpy())
axs[0, 1].set_title("Reconstructed", fontsize=12, fontweight='bold')
axs[0, 1].axis('off')

im1 = axs[0, 2].imshow(ssim_heatmap_norm, cmap='hot', vmin=0, vmax=1)
axs[0, 2].set_title(f"SSIM Heatmap\n(SSIM={ssim_stats['ssim_global']:.3f})", fontsize=12, fontweight='bold')
axs[0, 2].axis('off')
plt.colorbar(im1, ax=axs[0, 2], fraction=0.046, pad=0.04)

im2 = axs[0, 3].imshow(mse_heatmap_norm, cmap='hot', vmin=0, vmax=1)
axs[0, 3].set_title(f"MSE Heatmap\n(MSE={mse_stats['mse_total']:.4f})", fontsize=12, fontweight='bold')
axs[0, 3].axis('off')
plt.colorbar(im2, ax=axs[0, 3], fraction=0.046, pad=0.04)

axs[1, 0].imshow(gt_binary, cmap='gray')
axs[1, 0].set_title("Ground Truth", fontsize=12, fontweight='bold')
axs[1, 0].axis('off')

axs[1, 1].imshow(mask_ssim_95_clean, cmap='gray')
axs[1, 1].set_title(f"SSIM Mask (p95)\nIoU={metrics_ssim_95['IoU']:.3f}", fontsize=12, fontweight='bold')
axs[1, 1].axis('off')

axs[1, 2].imshow(mask_ssim_otsu_clean, cmap='gray')
axs[1, 2].set_title(f"SSIM Mask (Otsu)\nIoU={metrics_ssim_otsu['IoU']:.3f}", fontsize=12, fontweight='bold')
axs[1, 2].axis('off')

axs[1, 3].imshow(mask_mse_95_clean, cmap='gray')
axs[1, 3].set_title(f"MSE Mask (p95)\nIoU={metrics_mse_95['IoU']:.3f}", fontsize=12, fontweight='bold')
axs[1, 3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_heatmaps.png"), dpi=150, bbox_inches='tight')
plt.close()

# ---- Plot 2: Overlays dettagliati ----
fig2, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(gt_binary, cmap='gray')
axs[0].set_title("Ground Truth", fontsize=14, fontweight='bold')
axs[0].axis('off')

axs[1].imshow(overlay_ssim_95)
axs[1].set_title(f"SSIM (p95) - F1={metrics_ssim_95['F1']:.3f}\nGreen=TP, Red=FP, Blue=FN", fontsize=12, fontweight='bold')
axs[1].axis('off')

axs[2].imshow(overlay_ssim_otsu)
axs[2].set_title(f"SSIM (Otsu) - F1={metrics_ssim_otsu['F1']:.3f}\nGreen=TP, Red=FP, Blue=FN", fontsize=12, fontweight='bold')
axs[2].axis('off')

axs[3].imshow(overlay_mse_95)
axs[3].set_title(f"MSE (p95) - F1={metrics_mse_95['F1']:.3f}\nGreen=TP, Red=FP, Blue=FN", fontsize=12, fontweight='bold')
axs[3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_overlays.png"), dpi=150, bbox_inches='tight')
plt.close()

# ---- Plot 3: Side-by-side migliore risultato ----
# Scegli il metodo migliore (in base a F1)
best_method = max(
    [("SSIM-p95", metrics_ssim_95, mask_ssim_95_clean, overlay_ssim_95),
     ("SSIM-Otsu", metrics_ssim_otsu, mask_ssim_otsu_clean, overlay_ssim_otsu),
     ("MSE-p95", metrics_mse_95, mask_mse_95_clean, overlay_mse_95)],
    key=lambda x: x[1]['F1']
)

best_name, best_metrics, best_mask, best_overlay = best_method

fig3, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(orig.permute(1, 2, 0).cpu().numpy())
axs[0].set_title("Original", fontsize=14, fontweight='bold')
axs[0].axis('off')

axs[1].imshow(gt_binary, cmap='gray')
axs[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
axs[1].axis('off')

axs[2].imshow(best_mask, cmap='gray')
axs[2].set_title(f"Prediction ({best_name})", fontsize=14, fontweight='bold')
axs[2].axis('off')

axs[3].imshow(best_overlay)
axs[3].set_title(f"Overlay ({best_name})\nF1={best_metrics['F1']:.3f} | IoU={best_metrics['IoU']:.3f}", 
                 fontsize=14, fontweight='bold')
axs[3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "best_result.png"), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*70}")
print(f"✓ Inference completata!")
print(f"✓ Risultati salvati in '{OUTPUT_DIR}'")
print(f"✓ Metodo migliore: {best_name} (F1={best_metrics['F1']:.3f})")
print(f"{'='*70}\n")
