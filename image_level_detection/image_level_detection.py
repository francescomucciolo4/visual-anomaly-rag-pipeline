import torch
import torch.nn as nn
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from image_level_detection.aecnn import CAE256_FC_Latent32
from training import CombinedLoss


MODEL_PATH = "best_autoencoder.pth"
TEST_ROOT = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\test"
OUTPUT_DIR = "detection_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METHOD = "ssim"
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def load_images(root):
    images = []
    good_dir = os.path.join(root, "good")
    if os.path.exists(good_dir):
        for fname in os.listdir(good_dir):
            if fname.lower().endswith(IMG_EXTENSIONS):
                images.append({'path': os.path.join(good_dir, fname),
                               'label': 0, 'filename': fname})
    
    for class_name in os.listdir(root):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir) or class_name == 'good':
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(IMG_EXTENSIONS):
                images.append({'path': os.path.join(class_dir, fname),
                               'label': 1, 'filename': fname})
    return images


def preprocess_image(img_path):
    from dataset_anomaly_detection import val_transforms
    img = Image.open(img_path).convert('RGB')
    img = val_transforms(img)
    img = img.unsqueeze(0)
    return img


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
        results.append({'filename': img_info['filename'],
                        'label': img_info['label'],
                        'anomaly_score': score})
    
    y_true = np.array([r['label'] for r in results])
    y_score = np.array([r['anomaly_score'] for r in results])
    auroc = roc_auc_score(y_true, y_score)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_score >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc_good = tn / (tn + fp)
    acc_anom = tp / (tp + fn)
    
    print(f"\nAUROC: {auroc:.4f}")
    print(f"Threshold: {optimal_threshold:.4f}")
    print(f"Acc Good: {acc_good:.2f}, Acc Anom: {acc_anom:.2f}, Mean: {(acc_good+acc_anom)/2:.2f}")
    print(f"CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={auroc:.4f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR,"roc_curve.png"), dpi=150)
    plt.close()
    
    good_scores = [r['anomaly_score'] for r in results if r['label']==0]
    anom_scores = [r['anomaly_score'] for r in results if r['label']==1]
    plt.figure()
    plt.hist(good_scores, bins=20, alpha=0.6, label='Good', color='green')
    plt.hist(anom_scores, bins=20, alpha=0.6, label='Anomalous', color='red')
    plt.axvline(optimal_threshold, color='blue', linestyle='--', label=f'Threshold={optimal_threshold:.4f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR,"score_distribution.png"), dpi=150)
    plt.close()
    
    print(f"\n✓ Results saved in '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()