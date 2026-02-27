import torch
import torch.nn.functional as F
from cnn_anomaly_detection import CAE256_FC_Latent32
from dataset_anomaly_detection import val_transforms
from training_anomaly_detection import SSIMLoss
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from scipy import ndimage
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu
from itertools import product
import pandas as pd


# =====================================================================
# SSIM PIXEL-LEVEL
# =====================================================================

class SSIMLoss_PixelLevel(SSIMLoss):
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
            self.window = self.window.type_as(img1)
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean(dim=1)


# =====================================================================
# CONFIGURAZIONE
# =====================================================================

MODEL_PATH_BASE = r"C:\Users\Francesco\Desktop\Progetto personale"
ANOMALY_ROOT = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\test"
GT_ROOT = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\ground_truth"

# Test su tutte le categorie
TEST_CASES = [
    {"class": "broken_large", "filename": "000.png"},
    {"class": "broken_small", "filename": "000.png"},
    {"class": "contamination", "filename": "000.png"}
]

ANOMALY_METHOD = 'combined'  # 'mse', 'ssim', 'combined'

if ANOMALY_METHOD == 'mse':
    MODEL_PATH = os.path.join(MODEL_PATH_BASE, "best_autoencoder_mse.pth")
elif ANOMALY_METHOD == 'ssim':
    MODEL_PATH = os.path.join(MODEL_PATH_BASE, "best_autoencoder_ssim.pth")
elif ANOMALY_METHOD == 'combined':
    MODEL_PATH = os.path.join(MODEL_PATH_BASE, "best_autoencoder_combined.pth")
else:
    raise ValueError(f"Unknown ANOMALY_METHOD: {ANOMALY_METHOD}")

print(f"\n📂 Loading model: {MODEL_PATH}")
print(f"🔍 Anomaly detection method: {ANOMALY_METHOD.upper()}\n")

OUTPUT_DIR = f"inference_hyperopt_fullgrid_{ANOMALY_METHOD}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CAE256_FC_Latent32().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def compute_anomaly_map(img_tensor, model, method):
    with torch.no_grad():
        output = model(img_tensor)
    
    if method == 'mse':
        mse_map = F.mse_loss(output, img_tensor, reduction='none')
        anomaly_map = mse_map.mean(dim=1)[0].cpu().numpy()
        
    elif method == 'ssim':
        ssim_calculator = SSIMLoss_PixelLevel().to(device)
        ssim_map_tensor = ssim_calculator(output, img_tensor)
        anomaly_map = ssim_map_tensor[0].cpu().numpy()
        
    elif method == 'combined':
        mse_map = F.mse_loss(output, img_tensor, reduction='none').mean(dim=1)[0].cpu().numpy()
        mse_norm = (mse_map - mse_map.min()) / (mse_map.max() - mse_map.min() + 1e-8)
        
        ssim_calculator = SSIMLoss_PixelLevel().to(device)
        ssim_map_tensor = ssim_calculator(output, img_tensor)
        ssim_map = ssim_map_tensor[0].cpu().numpy()
        ssim_norm = (ssim_map - ssim_map.min()) / (ssim_map.max() - ssim_map.min() + 1e-8)
        
        anomaly_map = 0.4 * mse_norm + 0.6 * ssim_norm
    
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    return anomaly_map_norm


def weighted_union_threshold(anomaly_map, high_thresh, low_thresh, distance_thresh=5):
    from scipy.ndimage import distance_transform_edt
    seeds = (anomaly_map > high_thresh).astype(np.uint8)
    candidates = (anomaly_map > low_thresh).astype(np.uint8)
    distance_from_seeds = distance_transform_edt(~seeds.astype(bool))
    nearby_candidates = np.logical_and(candidates.astype(bool), distance_from_seeds <= distance_thresh)
    final_mask = np.logical_or(seeds, nearby_candidates).astype(np.uint8)
    return final_mask


def post_process_mask(mask, min_area=50, kernel_size=5, iterations=1):
    mask_bool = mask.astype(bool)
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), dtype=bool)
        mask_closed = ndimage.binary_closing(mask_bool, structure=kernel, iterations=iterations)
    else:
        mask_closed = mask_bool
    if min_area > 0:
        try:
            mask_cleaned = remove_small_objects(mask_closed, min_size=min_area)
        except:
            mask_cleaned = mask_closed
    else:
        mask_cleaned = mask_closed
    return mask_cleaned.astype(np.uint8)


def compute_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return {"IoU": iou, "Precision": precision, "Recall": recall, "F1": f1}


# =====================================================================
# FULL GRID SEARCH
# =====================================================================

print("="*80)
print(f"FULL GRID SEARCH OPTIMIZATION - METHOD: {ANOMALY_METHOD.upper()}")
print("="*80)

# Definisci tutti gli iperparametri da testare
hyperparams = {
    "percentile": [90, 92, 94, 95, 96, 97, 98, 99],
    "min_area": [0, 10, 20, 30, 40, 50, 75, 100],
    "kernel_size": [0, 3, 5, 7, 9],
    "iterations": [1, 2],
    "distance_thresh": [3, 4, 5, 6, 7, 8, 10]
}

all_class_results = []

for test_case in TEST_CASES:
    class_name = test_case["class"]
    filename = test_case["filename"]
    
    print(f"\n{'='*80}")
    print(f"CLASS: {class_name.upper()} | FILE: {filename}")
    print(f"{'='*80}")
    
    # Load image and GT
    img_path = os.path.join(ANOMALY_ROOT, class_name, filename)
    gt_path = os.path.join(GT_ROOT, class_name, filename.replace(".png", "_mask.png"))
    
    img = Image.open(img_path).convert("RGB")
    gt = Image.open(gt_path).convert("L")
    
    img_tensor = val_transforms(img).unsqueeze(0).to(device)
    gt_tensor = val_transforms(gt).squeeze().cpu().numpy()
    gt_binary = (gt_tensor > 0).astype(np.uint8)
    
    # Compute anomaly map
    anomaly_map_norm = compute_anomaly_map(img_tensor, model, ANOMALY_METHOD)
    
    print(f"\nAnomaly map range: [{anomaly_map_norm.min():.4f}, {anomaly_map_norm.max():.4f}]")
    
    # =====================================================================
    # FULL GRID SEARCH
    # =====================================================================
    
    threshold_low = threshold_otsu(anomaly_map_norm)
    
    total_combinations = (len(hyperparams['percentile']) * 
                         len(hyperparams['min_area']) * 
                         len(hyperparams['kernel_size']) * 
                         len(hyperparams['iterations']) * 
                         len(hyperparams['distance_thresh']))
    
    print(f"\n[FULL GRID SEARCH] Testing {total_combinations} combinations...")
    print(f"  Percentiles: {len(hyperparams['percentile'])}")
    print(f"  Min areas: {len(hyperparams['min_area'])}")
    print(f"  Kernel sizes: {len(hyperparams['kernel_size'])}")
    print(f"  Iterations: {len(hyperparams['iterations'])}")
    print(f"  Distance thresholds: {len(hyperparams['distance_thresh'])}")
    
    results = []
    test_count = 0
    
    for percentile, min_area, kernel_size, iterations, distance_thresh in product(
        hyperparams['percentile'],
        hyperparams['min_area'],
        hyperparams['kernel_size'],
        hyperparams['iterations'],
        hyperparams['distance_thresh']
    ):
        test_count += 1
        
        # Compute threshold_high based on percentile
        threshold_high = np.percentile(anomaly_map_norm, percentile)
        
        # Generate mask
        raw_mask = weighted_union_threshold(anomaly_map_norm, threshold_high, threshold_low, 
                                           distance_thresh=distance_thresh)
        
        # Post-process
        mask_final = post_process_mask(raw_mask, min_area=min_area, 
                                       kernel_size=kernel_size, 
                                       iterations=iterations)
        
        # Evaluate
        metrics = compute_metrics(mask_final, gt_binary)
        
        results.append({
            "percentile": percentile,
            "threshold_high": threshold_high,
            "min_area": min_area,
            "kernel_size": kernel_size,
            "iterations": iterations,
            "distance_thresh": distance_thresh,
            "F1": metrics['F1'],
            "IoU": metrics['IoU'],
            "Precision": metrics['Precision'],
            "Recall": metrics['Recall']
        })
        
        # Progress update
        if test_count % 100 == 0:
            print(f"  Progress: {test_count}/{total_combinations} ({test_count/total_combinations*100:.1f}%)")
    
    print(f"  ✓ Completed: {total_combinations} combinations tested")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # =====================================================================
    # ANALYSIS
    # =====================================================================
    
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    # Top 10 configurations
    top_10 = df_results.nlargest(10, 'F1')
    print("\n🏆 TOP 10 CONFIGURATIONS (by F1 Score):")
    print("-" * 100)
    for idx, row in top_10.iterrows():
        rank = top_10.index.get_loc(idx) + 1
        print(f"  Rank {rank:2d}: "
              f"percentile={int(row['percentile']):2d}, "
              f"min_area={int(row['min_area']):3d}, "
              f"kernel={int(row['kernel_size']):2d}, "
              f"iter={int(row['iterations']):1d}, "
              f"dist={int(row['distance_thresh']):2d} → "
              f"F1={row['F1']:.4f}, IoU={row['IoU']:.4f}, "
              f"Prec={row['Precision']:.4f}, Rec={row['Recall']:.4f}")
    
    # Best configuration
    best_idx = df_results['F1'].idxmax()
    best_config = df_results.loc[best_idx]
    
    print(f"\n{'='*80}")
    print(f"🥇 BEST CONFIGURATION FOR {class_name.upper()}:")
    print(f"{'='*80}")
    print(f"  Percentile:      {int(best_config['percentile'])}%")
    print(f"  Threshold High:  {best_config['threshold_high']:.4f}")
    print(f"  Threshold Low:   {threshold_low:.4f} (Otsu)")
    print(f"  Min Area:        {int(best_config['min_area'])} pixels")
    print(f"  Kernel Size:     {int(best_config['kernel_size'])}×{int(best_config['kernel_size'])}")
    print(f"  Iterations:      {int(best_config['iterations'])}")
    print(f"  Distance Thresh: {int(best_config['distance_thresh'])} pixels")
    print(f"\n  F1 Score:        {best_config['F1']:.4f}")
    print(f"  IoU:             {best_config['IoU']:.4f}")
    print(f"  Precision:       {best_config['Precision']:.4f}")
    print(f"  Recall:          {best_config['Recall']:.4f}")
    
    # Save to summary
    all_class_results.append({
        "class": class_name,
        "percentile": int(best_config['percentile']),
        "threshold_high": best_config['threshold_high'],
        "threshold_low": threshold_low,
        "min_area": int(best_config['min_area']),
        "kernel_size": int(best_config['kernel_size']),
        "iterations": int(best_config['iterations']),
        "distance_thresh": int(best_config['distance_thresh']),
        "F1": best_config['F1'],
        "IoU": best_config['IoU'],
        "Precision": best_config['Precision'],
        "Recall": best_config['Recall']
    })
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    
    class_output_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    # Save full results
    df_results.to_csv(os.path.join(class_output_dir, "full_grid_results.csv"), index=False)
    
    # Save top 50 for analysis
    df_results.nlargest(50, 'F1').to_csv(os.path.join(class_output_dir, "top50_configs.csv"), index=False)
    
    # =====================================================================
    # PARAMETER IMPACT ANALYSIS
    # =====================================================================
    
    print("\n" + "-"*80)
    print("PARAMETER IMPACT ANALYSIS")
    print("-"*80)
    
    # Group by each parameter
    print("\n[1] PERCENTILE Impact (averaged over other params):")
    percentile_impact = df_results.groupby('percentile')['F1'].agg(['mean', 'std', 'max'])
    print(percentile_impact.to_string())
    
    print("\n[2] MIN_AREA Impact:")
    min_area_impact = df_results.groupby('min_area')['F1'].agg(['mean', 'std', 'max'])
    print(min_area_impact.to_string())
    
    print("\n[3] KERNEL_SIZE Impact:")
    kernel_impact = df_results.groupby('kernel_size')['F1'].agg(['mean', 'std', 'max'])
    print(kernel_impact.to_string())
    
    print("\n[4] DISTANCE_THRESH Impact:")
    distance_impact = df_results.groupby('distance_thresh')['F1'].agg(['mean', 'std', 'max'])
    print(distance_impact.to_string())
    
    # =====================================================================
    # VISUALIZATION
    # =====================================================================
    
    print("\n[VISUALIZATION] Creating plots...")
    
    # 1. Parameter impact plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].plot(percentile_impact.index, percentile_impact['mean'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].fill_between(percentile_impact.index, 
                            percentile_impact['mean'] - percentile_impact['std'],
                            percentile_impact['mean'] + percentile_impact['std'], 
                            alpha=0.3)
    axes[0, 0].axvline(x=best_config['percentile'], color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Percentile', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('F1 Score (mean)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title(f'{class_name.upper()} - Percentile Impact', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(min_area_impact.index, min_area_impact['mean'], 'o-', linewidth=2, markersize=8)
    axes[0, 1].fill_between(min_area_impact.index, 
                            min_area_impact['mean'] - min_area_impact['std'],
                            min_area_impact['mean'] + min_area_impact['std'], 
                            alpha=0.3)
    axes[0, 1].axvline(x=best_config['min_area'], color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Min Area (pixels)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('F1 Score (mean)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'{class_name.upper()} - Min Area Impact', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(kernel_impact.index, kernel_impact['mean'], 'o-', linewidth=2, markersize=8)
    axes[1, 0].fill_between(kernel_impact.index, 
                            kernel_impact['mean'] - kernel_impact['std'],
                            kernel_impact['mean'] + kernel_impact['std'], 
                            alpha=0.3)
    axes[1, 0].axvline(x=best_config['kernel_size'], color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Kernel Size', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('F1 Score (mean)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'{class_name.upper()} - Kernel Size Impact', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(distance_impact.index, distance_impact['mean'], 'o-', linewidth=2, markersize=8)
    axes[1, 1].fill_between(distance_impact.index, 
                            distance_impact['mean'] - distance_impact['std'],
                            distance_impact['mean'] + distance_impact['std'], 
                            alpha=0.3)
    axes[1, 1].axvline(x=best_config['distance_thresh'], color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Distance Threshold (pixels)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('F1 Score (mean)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title(f'{class_name.upper()} - Distance Impact', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(class_output_dir, "parameter_impact.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Best configuration visualization
    threshold_high = best_config['threshold_high']
    
    best_mask_raw = weighted_union_threshold(anomaly_map_norm, threshold_high, threshold_low, 
                                            distance_thresh=int(best_config['distance_thresh']))
    best_mask_final = post_process_mask(best_mask_raw,
                                       min_area=int(best_config['min_area']),
                                       kernel_size=int(best_config['kernel_size']),
                                       iterations=int(best_config['iterations']))
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{class_name.upper()} - Optimal Configuration ({ANOMALY_METHOD.upper()})', 
                 fontsize=16, fontweight='bold')
    
    axs[0, 0].imshow(img_tensor[0].permute(1, 2, 0).cpu().numpy())
    axs[0, 0].set_title("Original", fontsize=12, fontweight='bold')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(anomaly_map_norm, cmap='hot')
    axs[0, 1].set_title(f"Anomaly Map ({ANOMALY_METHOD})", fontsize=12, fontweight='bold')
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(best_mask_raw, cmap='gray')
    axs[0, 2].set_title("Raw Mask", fontsize=12, fontweight='bold')
    axs[0, 2].axis('off')
    
    axs[1, 0].imshow(gt_binary, cmap='gray')
    axs[1, 0].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(best_mask_final, cmap='gray')
    axs[1, 1].set_title(f"Optimized Mask\nF1={best_config['F1']:.4f}", 
                       fontsize=12, fontweight='bold', color='darkgreen')
    axs[1, 1].axis('off')
    
    def create_overlay(pred_mask, gt_mask):
        pred = pred_mask.astype(bool)
        gt = gt_mask.astype(bool)
        overlay = np.zeros((gt.shape[0], gt.shape[1], 3))
        overlay[np.logical_and(pred, gt)] = [0, 1, 0]
        overlay[np.logical_and(pred, ~gt)] = [1, 0, 0]
        overlay[np.logical_and(~pred, gt)] = [0, 0, 1]
        return overlay
    
    overlay = create_overlay(best_mask_final, gt_binary)
    axs[1, 2].imshow(overlay)
    axs[1, 2].set_title(f"Overlay (TP=green, FP=red, FN=blue)\nPrec={best_config['Precision']:.3f} | Rec={best_config['Recall']:.3f}", 
                       fontsize=12, fontweight='bold')
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(class_output_dir, "optimal_config_visual.png"), dpi=150, bbox_inches='tight')
    plt.close()

# =====================================================================
# SUMMARY ACROSS ALL CLASSES
# =====================================================================

print("\n" + "="*80)
print("SUMMARY: OPTIMAL CONFIGURATIONS PER CLASS")
print("="*80)

df_summary = pd.DataFrame(all_class_results)

print("\n" + df_summary.to_string(index=False))

df_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_all_classes.csv"), index=False)

# Comparison visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].bar(df_summary['class'], df_summary['F1'], color=['#2ecc71', '#3498db', '#e74c3c'])
axes[0, 0].set_title('F1 Score per Class', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('F1 Score', fontsize=12)
axes[0, 0].grid(True, alpha=0.3, axis='y')

axes[0, 1].bar(df_summary['class'], df_summary['IoU'], color=['#2ecc71', '#3498db', '#e74c3c'])
axes[0, 1].set_title('IoU per Class', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('IoU', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, axis='y')

axes[0, 2].bar(df_summary['class'], df_summary['percentile'], color=['#2ecc71', '#3498db', '#e74c3c'])
axes[0, 2].set_title('Best Percentile per Class', fontsize=14, fontweight='bold')
axes[0, 2].set_ylabel('Percentile', fontsize=12)
axes[0, 2].grid(True, alpha=0.3, axis='y')

axes[1, 0].bar(df_summary['class'], df_summary['min_area'], color=['#2ecc71', '#3498db', '#e74c3c'])
axes[1, 0].set_title('Best Min Area per Class', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Min Area (pixels)', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='y')

axes[1, 1].bar(df_summary['class'], df_summary['kernel_size'], color=['#2ecc71', '#3498db', '#e74c3c'])
axes[1, 1].set_title('Best Kernel Size per Class', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Kernel Size', fontsize=12)
axes[1, 1].grid(True, alpha=0.3, axis='y')

axes[1, 2].bar(df_summary['class'], df_summary['distance_thresh'], color=['#2ecc71', '#3498db', '#e74c3c'])
axes[1, 2].set_title('Best Distance Threshold per Class', fontsize=14, fontweight='bold')
axes[1, 2].set_ylabel('Distance (pixels)', fontsize=12)
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_all_classes.png"), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*80}")
print(f"✓ Full grid search optimization completed!")
print(f"✓ Total combinations tested per class: {total_combinations}")
print(f"✓ Results saved in '{OUTPUT_DIR}'")
print(f"{'='*80}\n")