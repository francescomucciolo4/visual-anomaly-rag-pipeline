import torch
import torch.nn as nn
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from cae import CAE256_FC_Latent32
from training import CombinedLoss


MODEL_PATH = "best_autoencoder.pth"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_ROOT = os.path.join(BASE_DIR, "bottle", "test")
OUTPUT_DIR = "detection_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_images(root):
    """Load all test images (good + anomalies)"""
    images = []
    
    # Good images
    good_dir = os.path.join(root, "good")
    if os.path.exists(good_dir):
        for fname in sorted(os.listdir(good_dir)):
            if fname.endswith('.png'):
                images.append({
                    'path': os.path.join(good_dir, fname),
                    'label': 0,
                    'class': 'good'
                })
    
    # Anomaly images
    for class_name in ['broken_large', 'broken_small', 'contamination']:
        class_dir = os.path.join(root, class_name)
        if os.path.exists(class_dir):
            for fname in sorted(os.listdir(class_dir)):
                if fname.endswith('.png'):
                    images.append({
                        'path': os.path.join(class_dir, fname),
                        'label': 1,
                        'class': class_name
                    })
    
    return images


def preprocess_image(img_path):
    """Load and preprocess image"""
    from dataset_anomaly_detection import val_transforms
    img = Image.open(img_path).convert('RGB')
    return val_transforms(img).unsqueeze(0)


def visualize_samples(results, model, output_dir, n=3):
    """Save visualization of best/worst examples"""
    
    # Separate by category
    tp = [r for r in results if r['label']==1 and r['pred']==1]
    fp = [r for r in results if r['label']==0 and r['pred']==1]
    fn = [r for r in results if r['label']==1 and r['pred']==0]
    tn = [r for r in results if r['label']==0 and r['pred']==0]
    
    # Sort by score
    tp = sorted(tp, key=lambda x: x['score'], reverse=True)[:n]
    fp = sorted(fp, key=lambda x: x['score'], reverse=True)[:n] if fp else []
    fn = sorted(fn, key=lambda x: x['score'])[:n]
    tn = sorted(tn, key=lambda x: x['score'])[:n]
    
    # Decide number of rows (skip FP if empty)
    categories = []
    if tp:
        categories.append((tp, 'True Positives (Anomalies Detected)', 'green'))
    if fp:
        categories.append((fp, 'False Positives (False Alarms)', 'orange'))
    if fn:
        categories.append((fn, 'False Negatives (Missed Anomalies)', 'red'))
    if tn:
        categories.append((tn, 'True Negatives (Good Classified)', 'blue'))
    
    n_rows = len(categories)
    if n_rows == 0:
        print("⚠️ No samples to visualize")
        return
    
    # Create figure
    fig, axes = plt.subplots(n_rows, 3*n, figsize=(5*n, 4*n_rows))
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Detection Examples (Original | Reconstruction | Error Map)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    for row, (samples, label, color) in enumerate(categories):
        # Row label
        if n_rows > 1:
            axes[row, 0].text(-0.3, 0.5, label, 
                             transform=axes[row, 0].transAxes,
                             fontsize=11, fontweight='bold', color=color,
                             rotation=90, va='center', ha='right')
        else:
            axes[0].text(-0.3, 0.5, label, 
                        transform=axes[0].transAxes,
                        fontsize=11, fontweight='bold', color=color,
                        rotation=90, va='center', ha='right')
        
        for col, sample in enumerate(samples):
            # Load image
            img = preprocess_image(sample['path']).to(DEVICE)
            
            with torch.no_grad():
                recon = model(img)
                error = torch.abs(img - recon).mean(dim=1)[0].cpu().numpy()
            
            img_np = img[0].permute(1,2,0).cpu().numpy()
            recon_np = recon[0].permute(1,2,0).cpu().numpy()
            
            # Get axes
            if n_rows > 1:
                ax_orig = axes[row, col*3]
                ax_recon = axes[row, col*3+1]
                ax_error = axes[row, col*3+2]
            else:
                ax_orig = axes[col*3]
                ax_recon = axes[col*3+1]
                ax_error = axes[col*3+2]
            
            # Original
            ax_orig.imshow(img_np)
            ax_orig.axis('off')
            if row == 0:
                ax_orig.set_title('Original', fontsize=10, fontweight='bold')
            
            # Reconstruction
            ax_recon.imshow(recon_np)
            ax_recon.axis('off')
            if row == 0:
                ax_recon.set_title('Reconstruction', fontsize=10, fontweight='bold')
            
            # Error map
            ax_error.imshow(error, cmap='hot')
            ax_error.axis('off')
            if row == 0:
                ax_error.set_title('Error Map', fontsize=10, fontweight='bold')
            
            # Score label
            ax_recon.text(0.5, -0.05, f'Score: {sample["score"]:.4f}',
                         transform=ax_recon.transAxes,
                         ha='center', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_examples.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved detection_examples.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = CAE256_FC_Latent32()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    
    criterion = CombinedLoss(alpha=0.4, beta=0.6)
    
    # Load images
    images = load_images(TEST_ROOT)
    n_good = sum(1 for x in images if x['label']==0)
    n_anom = sum(1 for x in images if x['label']==1)
    print(f"Loaded {len(images)} images ({n_good} good, {n_anom} anomalies)")
    
    # Compute anomaly scores
    print("\nComputing anomaly scores...")
    results = []
    for img_info in tqdm(images):
        img = preprocess_image(img_info['path']).to(DEVICE)
        with torch.no_grad():
            recon = model(img)
            score = criterion(recon, img).item()
        results.append({
            'path': img_info['path'],
            'label': img_info['label'],
            'class': img_info['class'],
            'score': score
        })
    
    # Calculate metrics
    y_true = np.array([r['label'] for r in results])
    y_score = np.array([r['score'] for r in results])
    
    # ROC AUC
    auroc = roc_auc_score(y_true, y_score)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    
    # Predictions
    y_pred = (y_score >= threshold).astype(int)
    for i, r in enumerate(results):
        r['pred'] = y_pred[i]
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Print results
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    print(f"\nImage-Level ROC AUC:     {auroc:.4f}")
    print(f"Optimal Threshold:       {threshold:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Good    Anomaly")
    print(f"  Actual Good     {tn:3d}       {fp:3d}")
    print(f"         Anomaly  {fn:3d}       {tp:3d}")
    print(f"\nAccuracy Good:           {tn/(tn+fp):.1%}  ({tn}/{tn+fp})")
    print(f"Accuracy Anomaly:        {tp/(tp+fn):.1%}  ({tp}/{tp+fn})")
    print(f"Mean Accuracy:           {(tn/(tn+fp)+tp/(tp+fn))/2:.1%}")
    print(f"\nPrecision:               {tp/(tp+fp) if (tp+fp)>0 else 0:.1%}")
    print(f"Recall:                  {tp/(tp+fn) if (tp+fn)>0 else 0:.1%}")
    print(f"F1 Score:                {2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0:.4f}")
    print("="*60 + "\n")
    
    # Save results CSV
    df = pd.DataFrame([{
        'class': r['class'],
        'label': r['label'],
        'score': r['score'],
        'prediction': r['pred']
    } for r in results])
    df.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)
    print(f"✓ Saved results.csv")
    
    # Save summary
    with open(os.path.join(OUTPUT_DIR, 'SUMMARY.txt'), 'w') as f:
        f.write("IMAGE-LEVEL DETECTION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Dataset: {n_good} good + {n_anom} anomalies = {len(images)} total\n\n")
        f.write(f"ROC AUC:        {auroc:.4f}\n")
        f.write(f"Threshold:      {threshold:.4f}\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"                Predicted\n")
        f.write(f"             Good  Anomaly\n")
        f.write(f"Actual Good   {tn:3d}      {fp:3d}\n")
        f.write(f"       Anom   {fn:3d}      {tp:3d}\n\n")
        f.write(f"Accuracy Good:    {tn/(tn+fp):.1%}\n")
        f.write(f"Accuracy Anomaly: {tp/(tp+fn):.1%}\n")
        f.write(f"Mean Accuracy:    {(tn/(tn+fp)+tp/(tp+fn))/2:.1%}\n\n")
        f.write(f"Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.1%}\n")
        f.write(f"Recall:    {tp/(tp+fn) if (tp+fn)>0 else 0:.1%}\n")
        f.write(f"F1 Score:  {2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0:.4f}\n")
    print(f"✓ Saved SUMMARY.txt")
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={auroc:.4f})')
    plt.plot([0,1], [0,1], 'k--', linewidth=1, alpha=0.5)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], 
                s=100, c='red', marker='o', zorder=5,
                label=f'Optimal Threshold={threshold:.4f}')
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - Image-Level Detection', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=150)
    plt.close()
    print(f"✓ Saved roc_curve.png")
    
    # Score distribution
    good_scores = [r['score'] for r in results if r['label']==0]
    anom_scores = [r['score'] for r in results if r['label']==1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(good_scores, bins=25, alpha=0.7, color='green', 
             label=f'Good (n={len(good_scores)})', edgecolor='black')
    plt.hist(anom_scores, bins=25, alpha=0.7, color='red', 
             label=f'Anomaly (n={len(anom_scores)})', edgecolor='black')
    plt.axvline(threshold, color='blue', linestyle='--', linewidth=2,
                label=f'Threshold={threshold:.4f}')
    plt.xlabel('Anomaly Score (Combined Loss)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Score Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'score_distribution.png'), dpi=150)
    plt.close()
    print(f"✓ Saved score_distribution.png")
    
    # Visualize examples
    print("\nCreating detection examples...")
    visualize_samples(results, model, OUTPUT_DIR, n=3)
    
    print(f"\n{'='*60}")
    print(f"All results saved in '{OUTPUT_DIR}/'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
