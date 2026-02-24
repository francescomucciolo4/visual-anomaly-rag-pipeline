"""
SSIM ANOMALY DETECTION - VERSIONE MIGLIORATA
Risolve il problema del Recall basso abbassando il percentile
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
from skimage.filters import threshold_otsu

# =====================================================================
# CONFIGURAZIONE
# =====================================================================

MODEL_PATH = r"C:\Users\Francesco\Desktop\Progetto personale\best_autoencoder.pth"
ANOMALY_ROOT = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\test"
GT_ROOT = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\ground_truth"

CLASS_NAME = "broken_large"
FILENAME = "000.png"

OUTPUT_DIR = "inference_ssim_improved"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# COMPUTE SSIM
# =====================================================================

print("="*70)
print("SSIM ANOMALY DETECTION - IMPROVED VERSION")
print("="*70)

with torch.no_grad():
    output = model(img_tensor)

original = img_tensor[0]
reconstructed = output[0]

# Calcola SSIM
orig_np = original.permute(1, 2, 0).cpu().numpy()
recon_np = reconstructed.permute(1, 2, 0).cpu().numpy()

ssim_global, ssim_map = ssim(
    orig_np, recon_np,
    win_size=11,
    channel_axis=2,
    data_range=1.0,
    full=True
)

if len(ssim_map.shape) == 3:
    ssim_map = ssim_map.mean(axis=2)

anomaly_map = 1.0 - ssim_map
anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

print(f"\nGlobal SSIM: {ssim_global:.4f}")
print(f"Anomaly map range: [{anomaly_map_norm.min():.4f}, {anomaly_map_norm.max():.4f}]")

# =====================================================================
# FUNZIONI HELPER
# =====================================================================

def post_process_mask(mask, min_area=50, kernel_size=5):
    """Post-processing standard."""
    mask_bool = mask.astype(bool)
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    mask_closed = ndimage.binary_closing(mask_bool, structure=kernel)
    
    try:
        mask_cleaned = remove_small_objects(mask_closed, min_size=min_area)
    except:
        mask_cleaned = mask_closed
    
    return mask_cleaned.astype(np.uint8)


def compute_metrics(pred_mask, gt_mask):
    """Calcola metriche."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    
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
        "FN": int(fn)
    }


def create_overlay(pred_mask, gt_mask):
    """Crea overlay colorato."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)
    
    overlay = np.zeros((gt.shape[0], gt.shape[1], 3))
    overlay[tp] = [0, 1, 0]
    overlay[fp] = [1, 0, 0]
    overlay[fn] = [0, 0, 1]
    
    return overlay


# =====================================================================
# TEST MULTIPLI PERCENTILI
# =====================================================================

print("\n" + "="*70)
print("TESTING DIFFERENT THRESHOLDING METHODS")
print("="*70)

gt_binary = (gt_tensor > 0).astype(np.uint8)
gt_coverage = gt_binary.sum() / gt_binary.size

print(f"\nGround Truth Analysis:")
print(f"  Anomaly pixels: {gt_binary.sum():,} / {gt_binary.size:,}")
print(f"  Coverage: {gt_coverage*100:.2f}%")
print(f"  Ideal percentile: {(1-gt_coverage)*100:.1f}")

# Test diversi percentili
percentiles_to_test = [85, 88, 90, 92, 95]
results = {}

for p in percentiles_to_test:
    threshold = np.percentile(anomaly_map_norm, p)
    mask = (anomaly_map_norm > threshold).astype(np.uint8)
    mask_clean = post_process_mask(mask, min_area=50, kernel_size=5)
    
    metrics = compute_metrics(mask_clean, gt_binary)
    results[f"p{p}"] = {
        "metrics": metrics,
        "mask": mask_clean,
        "threshold": threshold
    }

# Test Otsu
threshold_otsu_val = threshold_otsu(anomaly_map_norm)
mask_otsu = (anomaly_map_norm > threshold_otsu_val).astype(np.uint8)
mask_otsu_clean = post_process_mask(mask_otsu, min_area=50, kernel_size=5)
metrics_otsu = compute_metrics(mask_otsu_clean, gt_binary)

results["Otsu"] = {
    "metrics": metrics_otsu,
    "mask": mask_otsu_clean,
    "threshold": threshold_otsu_val
}

# =====================================================================
# DISPLAY RESULTS
# =====================================================================

print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)

print(f"\n{'Method':<15} {'Threshold':>10} {'F1':>8} {'IoU':>8} {'Prec':>8} {'Rec':>8}")
print("-" * 70)

for method_name, result in results.items():
    m = result['metrics']
    t = result['threshold']
    print(f"{method_name:<15} {t:>10.4f} {m['F1']:>8.4f} {m['IoU']:>8.4f} "
          f"{m['Precision']:>8.4f} {m['Recall']:>8.4f}")

# Find best method
best_method_name = max(results.keys(), key=lambda k: results[k]['metrics']['F1'])
best_result = results[best_method_name]
best_metrics = best_result['metrics']

print(f"\n{'='*70}")
print(f"🏆 BEST METHOD: {best_method_name}")
print(f"{'='*70}")
print(f"  F1 Score:  {best_metrics['F1']:.4f}")
print(f"  IoU:       {best_metrics['IoU']:.4f}")
print(f"  Precision: {best_metrics['Precision']:.4f}")
print(f"  Recall:    {best_metrics['Recall']:.4f}")
print(f"  Threshold: {best_result['threshold']:.4f}")

# Compare with original p95
original_p95_metrics = results['p95']['metrics']
improvement_f1 = best_metrics['F1'] - original_p95_metrics['F1']
improvement_recall = best_metrics['Recall'] - original_p95_metrics['Recall']

print(f"\nIMPROVEMENT vs p95:")
print(f"  F1:     {improvement_f1:+.4f} ({improvement_f1/original_p95_metrics['F1']*100:+.1f}%)")
print(f"  Recall: {improvement_recall:+.4f} ({improvement_recall/original_p95_metrics['Recall']*100:+.1f}%)")

# =====================================================================
# VISUALIZATION
# =====================================================================

print("\n[VISUALIZATION] Creating comparison plots...")

# Plot 1: Confronto tutti i metodi
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Threshold Comparison: Finding the Best Method', fontsize=16, fontweight='bold')

methods_to_show = ['p85', 'p88', 'p90', 'p92', 'p95', 'Otsu']
positions = [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1)]

for (method, pos) in zip(methods_to_show, positions):
    if method in results:
        mask = results[method]['mask']
        metrics = results[method]['metrics']
        
        axs[pos].imshow(mask, cmap='gray')
        
        color = 'darkgreen' if method == best_method_name else 'black'
        title = f"{method}\nF1={metrics['F1']:.3f} | Rec={metrics['Recall']:.3f}"
        
        if method == best_method_name:
            title = f"⭐ {title} ⭐"
        
        axs[pos].set_title(title, fontsize=12, fontweight='bold', color=color)
        axs[pos].axis('off')
        
        if method == best_method_name:
            for spine in axs[pos].spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(4)

# GT
axs[1, 2].imshow(gt_binary, cmap='gray')
axs[1, 2].set_title("Ground Truth", fontsize=12, fontweight='bold')
axs[1, 2].axis('off')

# Heatmap
im = axs[1, 3].imshow(anomaly_map_norm, cmap='hot', vmin=0, vmax=1)
axs[1, 3].set_title("Anomaly Map", fontsize=12, fontweight='bold')
axs[1, 3].axis('off')
plt.colorbar(im, ax=axs[1, 3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_threshold_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Best result overlay
overlay_best = create_overlay(best_result['mask'], gt_binary)
overlay_p95 = create_overlay(results['p95']['mask'], gt_binary)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(original.permute(1, 2, 0).cpu().numpy())
axs[0].set_title("Original", fontsize=14, fontweight='bold')
axs[0].axis('off')

axs[1].imshow(gt_binary, cmap='gray')
axs[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
axs[1].axis('off')

axs[2].imshow(overlay_p95)
axs[2].set_title(f"OLD (p95)\nF1={original_p95_metrics['F1']:.3f} | Rec={original_p95_metrics['Recall']:.3f}", 
                fontsize=14, fontweight='bold')
axs[2].axis('off')

axs[3].imshow(overlay_best)
axs[3].set_title(f"BEST ({best_method_name})\nF1={best_metrics['F1']:.3f} | Rec={best_metrics['Recall']:.3f}", 
                fontsize=14, fontweight='bold', color='darkgreen')
axs[3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_best_result.png"), dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: F1 vs Percentile curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

percentile_values = [85, 88, 90, 92, 95]
f1_values = [results[f"p{p}"]['metrics']['F1'] for p in percentile_values]
recall_values = [results[f"p{p}"]['metrics']['Recall'] for p in percentile_values]
precision_values = [results[f"p{p}"]['metrics']['Precision'] for p in percentile_values]

ax1.plot(percentile_values, f1_values, 'o-', linewidth=2, markersize=10, label='F1', color='blue')
ax1.axvline(x=float(best_method_name[1:]) if best_method_name.startswith('p') else 95, 
            color='green', linestyle='--', linewidth=2, label=f'Best ({best_method_name})')
ax1.set_xlabel('Percentile Threshold', fontsize=12, fontweight='bold')
ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax1.set_title('F1 Score vs Percentile', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim([0, 1])

ax2.plot(percentile_values, precision_values, 'o-', linewidth=2, markersize=8, label='Precision', color='green')
ax2.plot(percentile_values, recall_values, 's-', linewidth=2, markersize=8, label='Recall', color='red')
ax2.set_xlabel('Percentile Threshold', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_optimization_curves.png"), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*70}")
print(f"✓ Analysis completed!")
print(f"✓ Results saved in '{OUTPUT_DIR}'")
print(f"{'='*70}\n")