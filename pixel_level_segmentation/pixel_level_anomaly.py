import torch
import torch.nn.functional as F
from cae_anomaly_detection import CAE256_FC_Latent32
from dataset_anomaly_detection import val_transforms
from training_anomaly_detection import SSIMLoss
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from scipy import ndimage
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
from tqdm import tqdm
import random


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

OUTPUT_DIR = f"inference_full_dataset_{ANOMALY_METHOD}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create subdirectory for visualizations
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CAE256_FC_Latent32().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


GLOBAL_CONFIG = {
    'percentile': 96,
    'min_area': 50,
    'kernel_size': 7,
    'iterations': 2,
    'distance_thresh': 5
}

print(f"\n📋 GLOBAL CONFIGURATION:")
print(f"  Percentile:      {GLOBAL_CONFIG['percentile']}%")
print(f"  Min Area:        {GLOBAL_CONFIG['min_area']} pixels")
print(f"  Kernel Size:     {GLOBAL_CONFIG['kernel_size']}×{GLOBAL_CONFIG['kernel_size']}")
print(f"  Iterations:      {GLOBAL_CONFIG['iterations']}")
print(f"  Distance Thresh: {GLOBAL_CONFIG['distance_thresh']} pixels")

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


def compute_per_region_overlap(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def create_overlay(pred_mask, gt_mask):
    """Create overlay visualization: TP=green, FP=red, FN=blue"""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    overlay = np.zeros((gt.shape[0], gt.shape[1], 3))
    overlay[np.logical_and(pred, gt)] = [0, 1, 0]    # TP green
    overlay[np.logical_and(pred, ~gt)] = [1, 0, 0]   # FP red
    overlay[np.logical_and(~pred, gt)] = [0, 0, 1]   # FN blue
    return overlay


def save_visualization(img_tensor, anomaly_map, raw_mask, final_mask, gt_binary, 
                       class_name, filename, overlap, save_dir):
    """Save detailed visualization for an image"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    fig.suptitle(f'{class_name.upper()} - {filename}\n'
                 f'Overlap: {overlap:.4f} | Method: {ANOMALY_METHOD.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # Original
    axes[0, 0].imshow(img_tensor[0].permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Anomaly Map
    im = axes[0, 1].imshow(anomaly_map, cmap='hot')
    axes[0, 1].set_title(f"Anomaly Map ({ANOMALY_METHOD})", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Raw Mask
    axes[0, 2].imshow(raw_mask, cmap='gray')
    axes[0, 2].set_title("Raw Mask", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Ground Truth
    axes[1, 0].imshow(gt_binary, cmap='gray')
    axes[1, 0].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Final Mask
    axes[1, 1].imshow(final_mask, cmap='gray')
    axes[1, 1].set_title(f"Final Mask\nOverlap={overlap:.4f}", 
                        fontsize=12, fontweight='bold', color='darkgreen')
    axes[1, 1].axis('off')
    
    # Overlay
    overlay = create_overlay(final_mask, gt_binary)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Overlay (TP=green, FP=red, FN=blue)", 
                        fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(save_dir, f"{class_name}_{filename}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =====================================================================
# FULL DATASET EVALUATION
# =====================================================================

print("\n" + "="*80)
print(f"FULL DATASET EVALUATION - METHOD: {ANOMALY_METHOD.upper()}")
print("="*80)

# Storage per risultati
all_image_results = []
image_level_data = []
pixel_level_data = []

# Storage for visualizations
visualizations_to_save = []

# =====================================================================
# PROCESS GOOD IMAGES (label=0)
# =====================================================================

print("\n" + "="*80)
print("Processing: GOOD (no anomalies)")
print("="*80)

good_dir = os.path.join(ANOMALY_ROOT, 'good')
good_files = sorted([f for f in os.listdir(good_dir) if f.endswith('.png')])

print(f"Found {len(good_files)} good images")

for img_file in tqdm(good_files, desc="Good images"):
    img_path = os.path.join(good_dir, img_file)
    img = Image.open(img_path).convert("RGB")
    img_tensor = val_transforms(img).unsqueeze(0).to(device)
    
    # Compute anomaly map
    anomaly_map_norm = compute_anomaly_map(img_tensor, model, ANOMALY_METHOD)
    
    # Image-level score
    image_score = anomaly_map_norm.mean()
    
    image_level_data.append({
        'filename': img_file,
        'label': 0,  
        'score': image_score
    })
    
    gt_binary = np.zeros_like(anomaly_map_norm, dtype=np.uint8)
    
    pixel_level_data.append({
        'filename': img_file,
        'gt_pixels': gt_binary.flatten(),
        'score_pixels': anomaly_map_norm.flatten()
    })

# =====================================================================
# PROCESS ANOMALY IMAGES (label=1)
# =====================================================================

anomaly_classes = ['broken_large', 'broken_small', 'contamination']

# Per salvare alcune immagini rappresentative per classe
samples_per_class = 3  # Salva 3 immagini per classe

for class_name in anomaly_classes:
    print(f"\n{'='*80}")
    print(f"Processing: {class_name.upper()}")
    print(f"{'='*80}")
    
    class_dir = os.path.join(ANOMALY_ROOT, class_name)
    gt_class_dir = os.path.join(GT_ROOT, class_name)
    
    image_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
    
    print(f"Found {len(image_files)} images")
    
    class_overlaps = []
    class_visualizations = []
    
    for img_file in tqdm(image_files, desc=class_name):
        img_path = os.path.join(class_dir, img_file)
        gt_path = os.path.join(gt_class_dir, img_file.replace('.png', '_mask.png'))
        
        img = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")
        
        img_tensor = val_transforms(img).unsqueeze(0).to(device)
        gt_tensor = val_transforms(gt).squeeze().cpu().numpy()
        gt_binary = (gt_tensor > 0).astype(np.uint8)
        
        # Compute anomaly map
        anomaly_map_norm = compute_anomaly_map(img_tensor, model, ANOMALY_METHOD)
        
        # Image-level score
        image_score = anomaly_map_norm.mean()
        
        image_level_data.append({
            'filename': img_file,
            'label': 1,  # Anomaly
            'score': image_score
        })
        
        # Pixel-level data
        pixel_level_data.append({
            'filename': img_file,
            'gt_pixels': gt_binary.flatten(),
            'score_pixels': anomaly_map_norm.flatten()
        })
        
        # Generate segmentation mask
        threshold_high = np.percentile(anomaly_map_norm, GLOBAL_CONFIG['percentile'])
        threshold_low = threshold_otsu(anomaly_map_norm)
        
        raw_mask = weighted_union_threshold(anomaly_map_norm, threshold_high, threshold_low, 
                                           distance_thresh=GLOBAL_CONFIG['distance_thresh'])
        
        final_mask = post_process_mask(raw_mask,
                                       min_area=GLOBAL_CONFIG['min_area'],
                                       kernel_size=GLOBAL_CONFIG['kernel_size'],
                                       iterations=GLOBAL_CONFIG['iterations'])
        
        # Compute per-region overlap
        overlap = compute_per_region_overlap(final_mask, gt_binary)
        class_overlaps.append(overlap)
        
        all_image_results.append({
            'class': class_name,
            'filename': img_file,
            'overlap': overlap,
            'image_score': image_score
        })
        
        # Store for visualization
        class_visualizations.append({
            'img_tensor': img_tensor,
            'anomaly_map': anomaly_map_norm,
            'raw_mask': raw_mask,
            'final_mask': final_mask,
            'gt_binary': gt_binary,
            'class_name': class_name,
            'filename': img_file,
            'overlap': overlap
        })
    
    # Class summary
    mean_overlap = np.mean(class_overlaps)
    print(f"  Mean Per-Region Overlap: {mean_overlap:.4f}")
    
    # Select representative samples to visualize
    # Best, median, worst overlap
    sorted_vis = sorted(class_visualizations, key=lambda x: x['overlap'], reverse=True)
    
    if len(sorted_vis) >= 3:
        # Best
        visualizations_to_save.append(sorted_vis[0])
        # Median
        visualizations_to_save.append(sorted_vis[len(sorted_vis)//2])
        # Worst
        visualizations_to_save.append(sorted_vis[-1])
    else:
        visualizations_to_save.extend(sorted_vis)

# =====================================================================
# SAVE VISUALIZATIONS
# =====================================================================

print("\n" + "="*80)
print("SAVING VISUALIZATIONS")
print("="*80)

for vis_data in tqdm(visualizations_to_save, desc="Saving visualizations"):
    save_visualization(
        vis_data['img_tensor'],
        vis_data['anomaly_map'],
        vis_data['raw_mask'],
        vis_data['final_mask'],
        vis_data['gt_binary'],
        vis_data['class_name'],
        vis_data['filename'],
        vis_data['overlap'],
        VIS_DIR
    )

print(f"✓ Saved {len(visualizations_to_save)} visualizations to {VIS_DIR}")

# =====================================================================
# COMPUTE METRICS
# =====================================================================

print("\n" + "="*80)
print("COMPUTING PAPER METRICS")
print("="*80)

# Image-level ROC AUC
df_image = pd.DataFrame(image_level_data)
image_roc_auc = roc_auc_score(df_image['label'], df_image['score'])

print(f"\n[1] Image-Level ROC AUC: {image_roc_auc:.4f}")

# Pixel-level ROC AUC
all_gt_pixels = []
all_score_pixels = []

for data in pixel_level_data:
    all_gt_pixels.extend(data['gt_pixels'])
    all_score_pixels.extend(data['score_pixels'])

pixel_roc_auc = roc_auc_score(all_gt_pixels, all_score_pixels)

print(f"[2] Pixel-Level ROC AUC: {pixel_roc_auc:.4f}")

# Per-Region Overlap per classe
df_results = pd.DataFrame(all_image_results)

overlap_by_class = df_results.groupby('class')['overlap'].mean()

print(f"\n[3] Per-Region Overlap (by class):")
for class_name, overlap in overlap_by_class.items():
    print(f"  {class_name:15s}: {overlap:.4f}")

# Overall Per-Region Overlap
overall_overlap = df_results['overlap'].mean()

print(f"\n[4] Overall Per-Region Overlap: {overall_overlap:.4f}")

# =====================================================================
# PAPER FORMAT RESULTS
# =====================================================================

print("\n" + "="*80)
print("RESULTS IN PAPER FORMAT")
print("="*80)

print(f"""
Category: Bottle
Method:   {ANOMALY_METHOD.upper()} Autoencoder

Per-Region Overlap:  {overall_overlap:.2f}
ROC AUC:            {image_roc_auc:.2f}

Breakdown by anomaly type:
  broken_large:     {overlap_by_class.get('broken_large', 0):.2f}
  broken_small:     {overlap_by_class.get('broken_small', 0):.2f}
  contamination:    {overlap_by_class.get('contamination', 0):.2f}

Pixel-Level ROC AUC: {pixel_roc_auc:.2f}
""")

print("\n" + "="*80)
print("COMPARISON WITH PAPER BASELINE")
print("="*80)

paper_overlap = 0.15
paper_roc_auc = 0.93

print(f"""
                      Paper    This Work   Difference
Per-Region Overlap:   {paper_overlap:.2f}     {overall_overlap:.2f}        {overall_overlap - paper_overlap:+.2f} ({(overall_overlap/paper_overlap - 1)*100:+.0f}%)
Image ROC AUC:        {paper_roc_auc:.2f}     {image_roc_auc:.2f}        {image_roc_auc - paper_roc_auc:+.2f} ({(image_roc_auc/paper_roc_auc - 1)*100:+.0f}%)

✓ Per-Region Overlap: {'BETTER' if overall_overlap > paper_overlap else 'WORSE'} than paper
✓ ROC AUC:            {'BETTER' if image_roc_auc > paper_roc_auc else 'WORSE'} than paper
""")

# =====================================================================
# SAVE RESULTS
# =====================================================================

# Save detailed results
df_image.to_csv(os.path.join(OUTPUT_DIR, "image_level_results.csv"), index=False)
df_results.to_csv(os.path.join(OUTPUT_DIR, "segmentation_results.csv"), index=False)

# Save summary
summary_dict = {
    'Method': ANOMALY_METHOD.upper(),
    'Per_Region_Overlap': overall_overlap,
    'Image_ROC_AUC': image_roc_auc,
    'Pixel_ROC_AUC': pixel_roc_auc,
    'broken_large_overlap': overlap_by_class.get('broken_large', 0),
    'broken_small_overlap': overlap_by_class.get('broken_small', 0),
    'contamination_overlap': overlap_by_class.get('contamination', 0),
    'num_good': len([x for x in image_level_data if x['label'] == 0]),
    'num_anomalies': len([x for x in image_level_data if x['label'] == 1])
}

df_summary = pd.DataFrame([summary_dict])
df_summary.to_csv(os.path.join(OUTPUT_DIR, "paper_format_results.csv"), index=False)

# Save paper format text file
with open(os.path.join(OUTPUT_DIR, "RESULTS_PAPER_FORMAT.txt"), 'w') as f:
    f.write("="*80 + "\n")
    f.write("RESULTS IN PAPER FORMAT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Category: Bottle\n")
    f.write(f"Method:   {ANOMALY_METHOD.upper()} Autoencoder\n\n")
    f.write(f"Per-Region Overlap:  {overall_overlap:.2f}\n")
    f.write(f"ROC AUC:            {image_roc_auc:.2f}\n\n")
    f.write(f"Breakdown by anomaly type:\n")
    f.write(f"  broken_large:     {overlap_by_class.get('broken_large', 0):.2f}\n")
    f.write(f"  broken_small:     {overlap_by_class.get('broken_small', 0):.2f}\n")
    f.write(f"  contamination:    {overlap_by_class.get('contamination', 0):.2f}\n\n")
    f.write(f"Pixel-Level ROC AUC: {pixel_roc_auc:.2f}\n\n")
    f.write("="*80 + "\n")
    f.write("COMPARISON WITH PAPER\n")
    f.write("="*80 + "\n\n")
    f.write(f"                      Paper    This Work   Difference\n")
    f.write(f"Per-Region Overlap:   {paper_overlap:.2f}     {overall_overlap:.2f}        {overall_overlap - paper_overlap:+.2f} ({(overall_overlap/paper_overlap - 1)*100:+.0f}%)\n")
    f.write(f"Image ROC AUC:        {paper_roc_auc:.2f}     {image_roc_auc:.2f}        {image_roc_auc - paper_roc_auc:+.2f} ({(image_roc_auc/paper_roc_auc - 1)*100:+.0f}%)\n")

print(f"\n{'='*80}")
print("SUMMARY SAVED")
print(f"{'='*80}")
print(f"  image_level_results.csv       - All image scores")
print(f"  segmentation_results.csv      - Per-image overlaps")
print(f"  paper_format_results.csv      - Paper metrics summary")
print(f"  RESULTS_PAPER_FORMAT.txt      - Human-readable results")
print(f"  visualizations/               - {len(visualizations_to_save)} sample images")

# =====================================================================
# VISUALIZATION: SUMMARY PLOTS
# =====================================================================

print("\n[VISUALIZATION] Creating summary plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Image-level ROC curve
fpr, tpr, thresholds = roc_curve(df_image['label'], df_image['score'])

axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {image_roc_auc:.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[0, 0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Image-Level ROC Curve', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Per-Region Overlap by class
classes = list(overlap_by_class.index)
overlaps = list(overlap_by_class.values)

axes[0, 1].bar(classes, overlaps, color=['#2ecc71', '#3498db', '#e74c3c'])
axes[0, 1].axhline(y=overall_overlap, color='black', linestyle='--', linewidth=2, 
                   label=f'Overall: {overall_overlap:.3f}')
axes[0, 1].axhline(y=paper_overlap, color='red', linestyle='--', linewidth=2, 
                   label=f'Paper: {paper_overlap:.3f}')
axes[0, 1].set_ylabel('Per-Region Overlap', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Per-Region Overlap by Class', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].set_ylim(0, max(overlaps) * 1.2)

# Plot 3: Score distribution
good_scores = df_image[df_image['label'] == 0]['score']
anomaly_scores = df_image[df_image['label'] == 1]['score']

axes[1, 0].hist(good_scores, bins=30, alpha=0.6, label='Good', color='green')
axes[1, 0].hist(anomaly_scores, bins=30, alpha=0.6, label='Anomaly', color='red')
axes[1, 0].set_xlabel('Anomaly Score', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Score Distribution', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Comparison table
comparison_data = [
    ['Per-Region Overlap', f'{paper_overlap:.3f}', f'{overall_overlap:.3f}', 
     f'{overall_overlap - paper_overlap:+.3f}'],
    ['Image ROC AUC', f'{paper_roc_auc:.3f}', f'{image_roc_auc:.3f}', 
     f'{image_roc_auc - paper_roc_auc:+.3f}']
]

axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=comparison_data,
                         colLabels=['Metric', 'Paper', 'This Work', 'Diff'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

for i in range(4):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

if overall_overlap > paper_overlap:
    table[(1, 3)].set_facecolor('#2ecc71')
else:
    table[(1, 3)].set_facecolor('#e74c3c')

if image_roc_auc > paper_roc_auc:
    table[(2, 3)].set_facecolor('#2ecc71')
else:
    table[(2, 3)].set_facecolor('#e74c3c')

axes[1, 1].set_title('Comparison with Paper', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "paper_comparison_summary.png"), dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Summary visualization saved")

print(f"\n{'='*80}")
print(f"✓ Full dataset evaluation completed!")
print(f"✓ Results saved in '{OUTPUT_DIR}'")
print(f"✓ Visualizations saved in '{VIS_DIR}'")
print(f"{'='*80}\n")