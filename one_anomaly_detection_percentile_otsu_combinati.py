"""
HYPERPARAMETER OPTIMIZATION: Post-Processing Parameters
Trova i valori ottimali di:
- min_area: dimensione minima degli oggetti da mantenere
- kernel_size: dimensione del kernel per binary closing
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
from itertools import product
import pandas as pd
import seaborn as sns

# =====================================================================
# CONFIGURAZIONE
# =====================================================================

MODEL_PATH = r"C:\Users\Francesco\Desktop\Progetto personale\best_autoencoder.pth"
ANOMALY_ROOT = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\test"
GT_ROOT = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\ground_truth"

CLASS_NAME = "broken_large"
FILENAME = "000.png"

OUTPUT_DIR = "inference_hyperopt_postprocess"
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
print("HYPERPARAMETER OPTIMIZATION FOR POST-PROCESSING")
print("="*70)

with torch.no_grad():
    output = model(img_tensor)

original = img_tensor[0]
reconstructed = output[0]

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

gt_binary = (gt_tensor > 0).astype(np.uint8)

# Soglie per Method 3 (Weighted Union)
threshold_high = np.percentile(anomaly_map_norm, 97)
threshold_low = threshold_otsu(anomaly_map_norm)

print(f"Thresholds: High={threshold_high:.4f}, Low={threshold_low:.4f}")

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def weighted_union_threshold(anomaly_map, high_thresh, low_thresh, distance_thresh=5):
    """Method 3: Weighted Union con distance threshold configurabile."""
    from scipy.ndimage import distance_transform_edt
    
    seeds = (anomaly_map > high_thresh).astype(np.uint8)
    candidates = (anomaly_map > low_thresh).astype(np.uint8)
    
    distance_from_seeds = distance_transform_edt(~seeds.astype(bool))
    
    nearby_candidates = np.logical_and(
        candidates.astype(bool),
        distance_from_seeds <= distance_thresh
    )
    
    final_mask = np.logical_or(seeds, nearby_candidates).astype(np.uint8)
    
    return final_mask


def post_process_mask(mask, min_area=50, kernel_size=5, iterations=1):
    """
    Post-processing configurabile.
    
    Args:
        mask: maschera binaria
        min_area: dimensione minima oggetti (in pixel)
        kernel_size: dimensione kernel per closing (deve essere dispari)
        iterations: numero di iterazioni di closing
    """
    mask_bool = mask.astype(bool)
    
    # Binary closing
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), dtype=bool)
        mask_closed = ndimage.binary_closing(mask_bool, structure=kernel, iterations=iterations)
    else:
        mask_closed = mask_bool
    
    # Remove small objects
    if min_area > 0:
        try:
            mask_cleaned = remove_small_objects(mask_closed, min_size=min_area)
        except:
            mask_cleaned = mask_closed
    else:
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
        "F1": f1
    }


# =====================================================================
# HYPERPARAMETER GRID SEARCH
# =====================================================================

print("\n" + "="*70)
print("GRID SEARCH: Post-Processing Hyperparameters")
print("="*70)

# Genera raw mask (prima del post-processing)
raw_mask = weighted_union_threshold(anomaly_map_norm, threshold_high, threshold_low, distance_thresh=5)

# Baseline (senza post-processing)
metrics_no_postproc = compute_metrics(raw_mask, gt_binary)
print(f"\nBaseline (NO post-processing):")
print(f"  F1={metrics_no_postproc['F1']:.4f} | IoU={metrics_no_postproc['IoU']:.4f} | "
      f"Prec={metrics_no_postproc['Precision']:.4f} | Rec={metrics_no_postproc['Recall']:.4f}")

# Definisci range di iperparametri da testare
print("\n[1] Testing Post-Processing Hyperparameters...")

hyperparams = {
    "min_area": [0, 10, 20, 30, 40, 50, 75, 100, 150, 200],
    "kernel_size": [0, 3, 5, 7, 9, 11],
    "iterations": [1, 2]
}

print(f"\nParameter ranges:")
print(f"  min_area: {hyperparams['min_area']}")
print(f"  kernel_size: {hyperparams['kernel_size']}")
print(f"  iterations: {hyperparams['iterations']}")

total_combinations = len(hyperparams['min_area']) * len(hyperparams['kernel_size']) * len(hyperparams['iterations'])
print(f"\nTotal combinations to test: {total_combinations}")

# Grid search
results = []
test_count = 0

for min_area, kernel_size, iterations in product(
    hyperparams['min_area'],
    hyperparams['kernel_size'],
    hyperparams['iterations']
):
    test_count += 1
    
    # Post-process
    mask_postproc = post_process_mask(raw_mask, min_area=min_area, 
                                     kernel_size=kernel_size, 
                                     iterations=iterations)
    
    # Evaluate
    metrics = compute_metrics(mask_postproc, gt_binary)
    
    results.append({
        "min_area": min_area,
        "kernel_size": kernel_size,
        "iterations": iterations,
        "F1": metrics['F1'],
        "IoU": metrics['IoU'],
        "Precision": metrics['Precision'],
        "Recall": metrics['Recall']
    })
    
    # Progress
    if test_count % 20 == 0:
        print(f"  Progress: {test_count}/{total_combinations} tested...")

print(f"  ✓ Completed: {total_combinations} combinations tested")

# Converti in DataFrame
df_results = pd.DataFrame(results)

# =====================================================================
# ANALYSIS & BEST CONFIGURATION
# =====================================================================

print("\n" + "="*70)
print("RESULTS ANALYSIS")
print("="*70)

# Top 10 configurazioni per F1
print("\n🏆 TOP 10 CONFIGURATIONS (by F1 Score):")
print("-" * 70)

top_10 = df_results.nlargest(10, 'F1')
for idx, row in top_10.iterrows():
    print(f"  Rank {top_10.index.get_loc(idx) + 1}: "
          f"min_area={int(row['min_area']):3d}, kernel_size={int(row['kernel_size']):2d}, "
          f"iterations={int(row['iterations']):1d} → "
          f"F1={row['F1']:.4f}, IoU={row['IoU']:.4f}, "
          f"Prec={row['Precision']:.4f}, Rec={row['Recall']:.4f}")

# Best configuration
best_idx = df_results['F1'].idxmax()
best_config = df_results.loc[best_idx]

print(f"\n{'='*70}")
print(f"🥇 BEST CONFIGURATION:")
print(f"{'='*70}")
print(f"  min_area:    {int(best_config['min_area'])}")
print(f"  kernel_size: {int(best_config['kernel_size'])}")
print(f"  iterations:  {int(best_config['iterations'])}")
print(f"\n  F1 Score:    {best_config['F1']:.4f}")
print(f"  IoU:         {best_config['IoU']:.4f}")
print(f"  Precision:   {best_config['Precision']:.4f}")
print(f"  Recall:      {best_config['Recall']:.4f}")

# Improvement vs no post-processing
improvement = best_config['F1'] - metrics_no_postproc['F1']
print(f"\n  Improvement vs NO post-proc: {improvement:+.4f} ({improvement/metrics_no_postproc['F1']*100:+.1f}%)")

# =====================================================================
# ANALYSIS: Impact of Each Parameter
# =====================================================================

print("\n" + "="*70)
print("PARAMETER IMPACT ANALYSIS")
print("="*70)

# 1. Min Area Impact (fixing other params)
print("\n[1] MIN_AREA Impact (kernel_size=5, iterations=1):")
subset = df_results[(df_results['kernel_size'] == 5) & (df_results['iterations'] == 1)]
subset_sorted = subset.sort_values('min_area')
print("\n  min_area  |   F1   |  IoU   |  Prec  |  Rec")
print("  " + "-" * 50)
for _, row in subset_sorted.iterrows():
    marker = " ⭐" if row['F1'] == subset['F1'].max() else ""
    print(f"  {int(row['min_area']):5d}     | {row['F1']:.4f} | {row['IoU']:.4f} | "
          f"{row['Precision']:.4f} | {row['Recall']:.4f}{marker}")

# 2. Kernel Size Impact
print("\n[2] KERNEL_SIZE Impact (min_area=50, iterations=1):")
subset = df_results[(df_results['min_area'] == 50) & (df_results['iterations'] == 1)]
subset_sorted = subset.sort_values('kernel_size')
print("\n  kernel  |   F1   |  IoU   |  Prec  |  Rec")
print("  " + "-" * 50)
for _, row in subset_sorted.iterrows():
    marker = " ⭐" if row['F1'] == subset['F1'].max() else ""
    print(f"  {int(row['kernel_size']):5d}   | {row['F1']:.4f} | {row['IoU']:.4f} | "
          f"{row['Precision']:.4f} | {row['Recall']:.4f}{marker}")

# 3. Iterations Impact
print("\n[3] ITERATIONS Impact (min_area=50, kernel_size=5):")
subset = df_results[(df_results['min_area'] == 50) & (df_results['kernel_size'] == 5)]
subset_sorted = subset.sort_values('iterations')
print("\n  iters  |   F1   |  IoU   |  Prec  |  Rec")
print("  " + "-" * 50)
for _, row in subset_sorted.iterrows():
    marker = " ⭐" if row['F1'] == subset['F1'].max() else ""
    print(f"  {int(row['iterations']):5d}  | {row['F1']:.4f} | {row['IoU']:.4f} | "
          f"{row['Precision']:.4f} | {row['Recall']:.4f}{marker}")

# =====================================================================
# ADDITIONAL OPTIMIZATION: Distance Threshold
# =====================================================================

print("\n" + "="*70)
print("BONUS: Distance Threshold Optimization")
print("="*70)

print("\n[4] Testing distance_thresh parameter (for Weighted Union)...")

distance_thresholds = [3, 4, 5, 6, 7, 8, 10]
distance_results = []

for dist_thresh in distance_thresholds:
    # Generate mask with this distance threshold
    mask_raw = weighted_union_threshold(anomaly_map_norm, threshold_high, threshold_low, 
                                       distance_thresh=dist_thresh)
    
    # Apply best post-processing found
    mask_postproc = post_process_mask(mask_raw, 
                                     min_area=int(best_config['min_area']),
                                     kernel_size=int(best_config['kernel_size']),
                                     iterations=int(best_config['iterations']))
    
    metrics = compute_metrics(mask_postproc, gt_binary)
    
    distance_results.append({
        "distance_thresh": dist_thresh,
        "F1": metrics['F1'],
        "IoU": metrics['IoU'],
        "Precision": metrics['Precision'],
        "Recall": metrics['Recall']
    })

df_distance = pd.DataFrame(distance_results)

print("\n  distance |   F1   |  IoU   |  Prec  |  Rec")
print("  " + "-" * 50)
for _, row in df_distance.iterrows():
    marker = " ⭐" if row['F1'] == df_distance['F1'].max() else ""
    print(f"  {int(row['distance_thresh']):5d}    | {row['F1']:.4f} | {row['IoU']:.4f} | "
          f"{row['Precision']:.4f} | {row['Recall']:.4f}{marker}")

best_distance = df_distance.loc[df_distance['F1'].idxmax()]

print(f"\n🎯 OPTIMAL DISTANCE THRESHOLD: {int(best_distance['distance_thresh'])} pixels")
print(f"   F1: {best_distance['F1']:.4f}")

# =====================================================================
# FINAL OPTIMAL CONFIGURATION
# =====================================================================

print("\n" + "="*70)
print("🏆 FINAL OPTIMAL CONFIGURATION")
print("="*70)

print(f"""
POST-PROCESSING PARAMETERS:
  min_area:        {int(best_config['min_area'])} pixels
  kernel_size:     {int(best_config['kernel_size'])}×{int(best_config['kernel_size'])}
  iterations:      {int(best_config['iterations'])}

WEIGHTED UNION PARAMETER:
  distance_thresh: {int(best_distance['distance_thresh'])} pixels

THRESHOLDING:
  high_threshold:  {threshold_high:.4f} (p97)
  low_threshold:   {threshold_low:.4f} (Otsu)

FINAL PERFORMANCE:
  F1 Score:        {best_distance['F1']:.4f}
  IoU:             {best_distance['IoU']:.4f}
  Precision:       {best_distance['Precision']:.4f}
  Recall:          {best_distance['Recall']:.4f}

IMPROVEMENT vs BASELINE (no post-proc):
  F1:     {best_distance['F1'] - metrics_no_postproc['F1']:+.4f} ({(best_distance['F1'] / metrics_no_postproc['F1'] - 1)*100:+.1f}%)
  Prec:   {best_distance['Precision'] - metrics_no_postproc['Precision']:+.4f}
  Rec:    {best_distance['Recall'] - metrics_no_postproc['Recall']:+.4f}
""")

# =====================================================================
# VISUALIZATION
# =====================================================================

print("\n[VISUALIZATION] Creating heatmaps and plots...")

# Plot 1: Heatmap F1 vs min_area and kernel_size (iterations=1)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for iter_val, ax in zip([1, 2], axes):
    subset = df_results[df_results['iterations'] == iter_val]
    pivot = subset.pivot(index='min_area', columns='kernel_size', values='F1')
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.80, vmax=0.90,
                cbar_kws={'label': 'F1 Score'}, ax=ax, linewidths=0.5)
    
    ax.set_title(f'F1 Score Heatmap (iterations={iter_val})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Kernel Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Min Area', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_hyperparameter_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Line plots per parameter
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Min Area
subset = df_results[(df_results['kernel_size'] == 5) & (df_results['iterations'] == 1)]
axes[0].plot(subset['min_area'], subset['F1'], 'o-', linewidth=2, markersize=8, color='blue')
axes[0].axvline(x=best_config['min_area'], color='red', linestyle='--', linewidth=2, 
                label=f"Best: {int(best_config['min_area'])}")
axes[0].set_xlabel('Min Area (pixels)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
axes[0].set_title('Min Area Impact', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Kernel Size
subset = df_results[(df_results['min_area'] == 50) & (df_results['iterations'] == 1)]
axes[1].plot(subset['kernel_size'], subset['F1'], 'o-', linewidth=2, markersize=8, color='green')
axes[1].axvline(x=best_config['kernel_size'], color='red', linestyle='--', linewidth=2,
                label=f"Best: {int(best_config['kernel_size'])}")
axes[1].set_xlabel('Kernel Size', fontsize=12, fontweight='bold')
axes[1].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
axes[1].set_title('Kernel Size Impact', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Distance Threshold
axes[2].plot(df_distance['distance_thresh'], df_distance['F1'], 'o-', linewidth=2, markersize=8, color='purple')
axes[2].axvline(x=best_distance['distance_thresh'], color='red', linestyle='--', linewidth=2,
                label=f"Best: {int(best_distance['distance_thresh'])}")
axes[2].set_xlabel('Distance Threshold (pixels)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
axes[2].set_title('Distance Threshold Impact', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_parameter_impact.png"), dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Comparison raw vs best post-processing
best_mask_raw = weighted_union_threshold(anomaly_map_norm, threshold_high, threshold_low, 
                                        distance_thresh=int(best_distance['distance_thresh']))
best_mask_postproc = post_process_mask(best_mask_raw,
                                      min_area=int(best_config['min_area']),
                                      kernel_size=int(best_config['kernel_size']),
                                      iterations=int(best_config['iterations']))

def create_overlay(pred_mask, gt_mask):
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

overlay_raw = create_overlay(best_mask_raw, gt_binary)
overlay_postproc = create_overlay(best_mask_postproc, gt_binary)
metrics_best_raw = compute_metrics(best_mask_raw, gt_binary)
metrics_best_postproc = compute_metrics(best_mask_postproc, gt_binary)

fig, axs = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Optimal Configuration: Before vs After Post-Processing', fontsize=16, fontweight='bold')

# Row 1: Raw
axs[0, 0].imshow(original.permute(1, 2, 0).cpu().numpy())
axs[0, 0].set_title("Original", fontsize=12, fontweight='bold')
axs[0, 0].axis('off')

axs[0, 1].imshow(best_mask_raw, cmap='gray')
axs[0, 1].set_title(f"Raw Mask (NO post-proc)\nF1={metrics_best_raw['F1']:.4f}", 
                   fontsize=12, fontweight='bold')
axs[0, 1].axis('off')

axs[0, 2].imshow(overlay_raw)
axs[0, 2].set_title(f"Raw Overlay\nPrec={metrics_best_raw['Precision']:.3f} | Rec={metrics_best_raw['Recall']:.3f}",
                   fontsize=12, fontweight='bold')
axs[0, 2].axis('off')

# Row 2: Post-processed
axs[1, 0].imshow(gt_binary, cmap='gray')
axs[1, 0].set_title("Ground Truth", fontsize=12, fontweight='bold')
axs[1, 0].axis('off')

axs[1, 1].imshow(best_mask_postproc, cmap='gray')
axs[1, 1].set_title(f"Optimized Post-proc\nF1={metrics_best_postproc['F1']:.4f} ({metrics_best_postproc['F1']-metrics_best_raw['F1']:+.4f})",
                   fontsize=12, fontweight='bold', color='darkgreen')
axs[1, 1].axis('off')

axs[1, 2].imshow(overlay_postproc)
axs[1, 2].set_title(f"Optimized Overlay\nPrec={metrics_best_postproc['Precision']:.3f} | Rec={metrics_best_postproc['Recall']:.3f}",
                   fontsize=12, fontweight='bold', color='darkgreen')
axs[1, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_optimal_configuration.png"), dpi=150, bbox_inches='tight')
plt.close()

# Save results to CSV
df_results.to_csv(os.path.join(OUTPUT_DIR, "hyperparameter_results.csv"), index=False)
df_distance.to_csv(os.path.join(OUTPUT_DIR, "distance_threshold_results.csv"), index=False)

print(f"\n{'='*70}")
print(f"✓ Hyperparameter optimization completed!")
print(f"✓ Results saved in '{OUTPUT_DIR}'")
print(f"✓ CSV files saved for further analysis")
print(f"{'='*70}\n")