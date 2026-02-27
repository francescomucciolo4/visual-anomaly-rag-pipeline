import torch
import torch.nn as nn
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from aecnn import CAE256_FC_Latent32
from training import CombinedLoss


MODEL_PATH = "best_autoencoder_combined.pth"
TEST_ROOT = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\test"
OUTPUT_DIR = "detection_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METHOD = "combined"
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def load_images(root):
    images = []
    good_dir = os.path.join(root, "good")
    if os.path.exists(good_dir):
        for fname in os.listdir(good_dir):
            if fname.lower().endswith(IMG_EXTENSIONS):
                images.append({'path': os.path.join(good_dir, fname),
                               'label': 0, 'filename': fname, 'class': 'good'})
    
    for class_name in os.listdir(root):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir) or class_name == 'good':
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(IMG_EXTENSIONS):
                images.append({'path': os.path.join(class_dir, fname),
                               'label': 1, 'filename': fname, 'class': class_name})
    return images


def preprocess_image(img_path):
    from dataset_anomaly_detection import val_transforms
    img = Image.open(img_path).convert('RGB')
    img = val_transforms(img)
    img = img.unsqueeze(0)
    return img


def save_sample_visualizations(results_list, model, criterion, output_dir, n_samples=5):
    """
    Salva visualizzazioni di campioni:
    - True Positives (anomalie correttamente rilevate)
    - False Positives (good classificate come anomalie)
    - False Negatives (anomalie non rilevate)
    - True Negatives (good correttamente classificate)
    """
    
    vis_dir = os.path.join(output_dir, "sample_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Separa per categoria
    tp_samples = [r for r in results_list if r['label'] == 1 and r['prediction'] == 1]
    fp_samples = [r for r in results_list if r['label'] == 0 and r['prediction'] == 1]
    fn_samples = [r for r in results_list if r['label'] == 1 and r['prediction'] == 0]
    tn_samples = [r for r in results_list if r['label'] == 0 and r['prediction'] == 0]
    
    # Ordina per score
    tp_samples = sorted(tp_samples, key=lambda x: x['anomaly_score'], reverse=True)
    fp_samples = sorted(fp_samples, key=lambda x: x['anomaly_score'], reverse=True)
    fn_samples = sorted(fn_samples, key=lambda x: x['anomaly_score'])
    tn_samples = sorted(tn_samples, key=lambda x: x['anomaly_score'])
    
    categories = [
        ('TP', tp_samples[:n_samples], 'True Positives (Correctly Detected Anomalies)', 'green'),
        ('FP', fp_samples[:n_samples], 'False Positives (Good classified as Anomaly)', 'orange'),
        ('FN', fn_samples[:n_samples], 'False Negatives (Missed Anomalies)', 'red'),
        ('TN', tn_samples[:n_samples], 'True Negatives (Correctly Classified Good)', 'blue')
    ]
    
    for cat_name, samples, title, color in categories:
        if not samples:
            continue
            
        n = len(samples)
        if n == 0:
            continue
            
        fig, axes = plt.subplots(n, 3, figsize=(15, 5*n))
        if n == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', color=color)
        
        for idx, sample in enumerate(samples):
            # Load and process image
            img = preprocess_image(sample['path']).to(DEVICE)
            
            with torch.no_grad():
                recon = model(img)
                
                # Compute reconstruction error map
                error_map = torch.abs(img - recon).mean(dim=1)[0].cpu().numpy()
            
            # Original image
            img_np = img[0].permute(1, 2, 0).cpu().numpy()
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title(f"Original\n{sample['class']}: {sample['filename']}", 
                                   fontsize=10, fontweight='bold')
            axes[idx, 0].axis('off')
            
            # Reconstruction
            recon_np = recon[0].permute(1, 2, 0).cpu().numpy()
            axes[idx, 1].imshow(recon_np)
            axes[idx, 1].set_title(f"Reconstruction\nScore: {sample['anomaly_score']:.4f}", 
                                   fontsize=10, fontweight='bold')
            axes[idx, 1].axis('off')
            
            # Error map
            im = axes[idx, 2].imshow(error_map, cmap='hot')
            axes[idx, 2].set_title(f"Error Map\nLabel: {sample['label']} | Pred: {sample['prediction']}", 
                                   fontsize=10, fontweight='bold')
            axes[idx, 2].axis('off')
            plt.colorbar(im, ax=axes[idx, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{cat_name}_samples.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved {cat_name} visualizations ({n} samples)")
    
    # Summary grid: best and worst from each category
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    fig.suptitle('Detection Summary: Best Examples from Each Category', 
                 fontsize=16, fontweight='bold')
    
    all_categories = [
        (tp_samples[0] if tp_samples else None, 'TP (Best Detection)', 'green'),
        (fp_samples[0] if fp_samples else None, 'FP (Highest False Alarm)', 'orange'),
        (fn_samples[0] if fn_samples else None, 'FN (Worst Miss)', 'red'),
        (tn_samples[0] if tn_samples else None, 'TN (Best Normal)', 'blue')
    ]
    
    for row_idx, (sample, label, color) in enumerate(all_categories):
        if sample is None:
            for col in range(3):
                axes[row_idx, col].axis('off')
            continue
            
        img = preprocess_image(sample['path']).to(DEVICE)
        
        with torch.no_grad():
            recon = model(img)
            error_map = torch.abs(img - recon).mean(dim=1)[0].cpu().numpy()
        
        # Original
        img_np = img[0].permute(1, 2, 0).cpu().numpy()
        axes[row_idx, 0].imshow(img_np)
        if row_idx == 0:
            axes[row_idx, 0].set_title("Original", fontsize=12, fontweight='bold')
        axes[row_idx, 0].set_ylabel(label, fontsize=11, fontweight='bold', color=color)
        axes[row_idx, 0].axis('off')
        
        # Reconstruction
        recon_np = recon[0].permute(1, 2, 0).cpu().numpy()
        axes[row_idx, 1].imshow(recon_np)
        if row_idx == 0:
            axes[row_idx, 1].set_title("Reconstruction", fontsize=12, fontweight='bold')
        axes[row_idx, 1].axis('off')
        
        # Error map
        im = axes[row_idx, 2].imshow(error_map, cmap='hot')
        if row_idx == 0:
            axes[row_idx, 2].set_title("Error Map", fontsize=12, fontweight='bold')
        axes[row_idx, 2].axis('off')
        plt.colorbar(im, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)
        
        # Add text info
        info_text = (f"Class: {sample['class']}\n"
                    f"Score: {sample['anomaly_score']:.4f}\n"
                    f"Label: {sample['label']} | Pred: {sample['prediction']}")
        axes[row_idx, 0].text(0.5, -0.1, info_text, 
                             transform=axes[row_idx, 0].transAxes,
                             ha='center', va='top', fontsize=9,
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "detection_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved detection summary grid")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model = CAE256_FC_Latent32()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()
    
    criterion = CombinedLoss(alpha=0.4, beta=0.6)
    
    images = load_images(TEST_ROOT)
    print(f"Loaded {len(images)} images ({sum(1 for x in images if x['label']==0)} good, "
          f"{sum(1 for x in images if x['label']==1)} anomalous)")
    
    results = []
    for img_info in tqdm(images, desc="Detecting"):
        img = preprocess_image(img_info['path']).to(DEVICE)
        with torch.no_grad():
            recon = model(img)
            score = criterion(recon, img).item()
        results.append({
            'filename': img_info['filename'],
            'path': img_info['path'],
            'label': img_info['label'],
            'class': img_info['class'],
            'anomaly_score': score
        })
    
    y_true = np.array([r['label'] for r in results])
    y_score = np.array([r['anomaly_score'] for r in results])
    auroc = roc_auc_score(y_true, y_score)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_score >= optimal_threshold).astype(int)
    
    # Add predictions to results
    for i, r in enumerate(results):
        r['prediction'] = y_pred[i]
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc_good = tn / (tn + fp)
    acc_anom = tp / (tp + fn)
    
    print(f"\n{'='*80}")
    print("DETECTION RESULTS")
    print(f"{'='*80}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Accuracy Good: {acc_good:.2%}")
    print(f"Accuracy Anomaly: {acc_anom:.2%}")
    print(f"Mean Accuracy: {(acc_good+acc_anom)/2:.2%}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    print(f"{'='*80}\n")
    
    # Save CSV
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'path'} for r in results])
    df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUROC={auroc:.4f}")
    plt.plot([0,1],[0,1],'k--', linewidth=1)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], c='red', s=100, 
                label=f'Optimal (TPR={tpr[optimal_idx]:.2f}, FPR={fpr[optimal_idx]:.2f})', 
                zorder=5)
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - Image-Level Detection', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"roc_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Score Distribution
    good_scores = [r['anomaly_score'] for r in results if r['label']==0]
    anom_scores = [r['anomaly_score'] for r in results if r['label']==1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(good_scores, bins=30, alpha=0.6, label=f'Good (n={len(good_scores)})', color='green')
    plt.hist(anom_scores, bins=30, alpha=0.6, label=f'Anomalous (n={len(anom_scores)})', color='red')
    plt.axvline(optimal_threshold, color='blue', linestyle='--', linewidth=2, 
                label=f'Threshold={optimal_threshold:.4f}')
    plt.xlabel('Anomaly Score', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Score Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"score_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save sample visualizations
    print("\n[VISUALIZATION] Creating sample visualizations...")
    save_sample_visualizations(results, model, criterion, OUTPUT_DIR, n_samples=5)
    
    # Save summary text
    with open(os.path.join(OUTPUT_DIR, "DETECTION_SUMMARY.txt"), 'w') as f:
        f.write("="*80 + "\n")
        f.write("IMAGE-LEVEL DETECTION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Method: {METHOD}\n")
        f.write(f"Total Images: {len(images)}\n")
        f.write(f"  Good: {sum(1 for x in images if x['label']==0)}\n")
        f.write(f"  Anomalous: {sum(1 for x in images if x['label']==1)}\n\n")
        f.write(f"AUROC: {auroc:.4f}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"               Predicted\n")
        f.write(f"             Good  Anomaly\n")
        f.write(f"Actual Good   {tn:3d}    {fp:3d}\n")
        f.write(f"       Anom   {fn:3d}    {tp:3d}\n\n")
        f.write(f"Accuracy Good: {acc_good:.2%}\n")
        f.write(f"Accuracy Anomaly: {acc_anom:.2%}\n")
        f.write(f"Mean Accuracy: {(acc_good+acc_anom)/2:.2%}\n\n")
        f.write(f"Precision: {tp/(tp+fp):.2%}\n")
        f.write(f"Recall: {tp/(tp+fn):.2%}\n")
        f.write(f"F1 Score: {2*tp/(2*tp+fp+fn):.4f}\n")
    
    print(f"\n{'='*80}")
    print("FILES SAVED")
    print(f"{'='*80}")
    print(f"  results.csv               - All image scores and predictions")
    print(f"  DETECTION_SUMMARY.txt     - Human-readable summary")
    print(f"  roc_curve.png             - ROC curve")
    print(f"  score_distribution.png    - Score histogram")
    print(f"  sample_visualizations/    - Example detections")
    print(f"    ├── TP_samples.png      - True Positives")
    print(f"    ├── FP_samples.png      - False Positives")
    print(f"    ├── FN_samples.png      - False Negatives")
    print(f"    ├── TN_samples.png      - True Negatives")
    print(f"    └── detection_summary.png - Best examples grid")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()