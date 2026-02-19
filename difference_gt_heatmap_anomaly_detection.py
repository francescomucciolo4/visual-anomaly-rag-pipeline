import torch
from cnn_anomaly_detection import Autoencoder
from dataset_anomaly_detection import anomaly_loader, gt_loader
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_auc_score

# ---- Config ----
MODEL_PATH = r"C:\Users\Francesco\Desktop\Progetto personale\best_autoencoder.pth"
OUTPUT_DIR = "inference_results_anomalies"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modello
model = Autoencoder().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def create_heatmap(original, reconstructed):
    error_pixel = original - reconstructed
    error_squared = error_pixel ** 2
    heatmap = error_squared.mean(dim=0).cpu().numpy()

    mse_image = error_squared.mean().item()
    mse_per_channel = error_squared.view(3, -1).mean(dim=1).tolist()
    max_error = error_squared.max().item()
    min_error = error_squared.min().item()

    stats = {
        "mse_total": mse_image,
        "mse_per_channel": mse_per_channel,
        "max_error": max_error,
        "min_error": min_error
    }

    return heatmap, stats


# -----------------------
# Accumulatori globali
# -----------------------
all_pixel_gt = []
all_pixel_scores = []

total_iou = 0
total_precision = 0
total_recall = 0
num_images = 0


# ---- Inference ----
with torch.no_grad():
    for idx, (anomaly_batch, gt_batch) in enumerate(zip(anomaly_loader, gt_loader)):

        anomaly_batch = anomaly_batch.to(device)
        gt_batch = gt_batch.to(device)

        outputs = model(anomaly_batch)

        for i in range(anomaly_batch.size(0)):

            orig = anomaly_batch[i]
            recon = outputs[i]
            gt = gt_batch[i]

            heatmap, stats = create_heatmap(orig, recon)

            # Ground Truth binaria
            gt_mask = gt.squeeze().cpu().numpy()          # (3, 128, 128)
            gt_mask = (gt_mask > 0).astype(np.uint8)      # 0/1
            print(f"----------------------------------------------------------- {gt_mask.shape}")

            # Salviamo per ROC
            all_pixel_gt.extend(gt_mask.flatten())
            all_pixel_scores.extend(heatmap.flatten())

            # Threshold temporanea
            threshold = 0.01
            pred_mask = (heatmap > threshold).astype(np.uint8)

            # ---- Metriche ----
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = intersection / (union + 1e-8)

            tp = intersection
            fp = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
            fn = np.logical_and(pred_mask == 0, gt_mask == 1).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            total_iou += iou
            total_precision += precision
            total_recall += recall
            num_images += 1

            print(
                f"Image {idx}_{i} | "
                f"MSE={stats['mse_total']:.6f} | "
                f"IoU={iou:.4f} | "
                f"Precision={precision:.4f} | "
                f"Recall={recall:.4f}"
            )

            # ---- Visualizzazione ----
            fig, axs = plt.subplots(1, 5, figsize=(18, 6))

            axs[0].imshow(orig.permute(1,2,0).cpu().numpy())
            axs[0].set_title("Original")

            axs[1].imshow(recon.permute(1,2,0).cpu().numpy())
            axs[1].set_title("Reconstructed")

            axs[2].imshow(heatmap, cmap='hot')
            axs[2].set_title("Heatmap")
            
            gt_mask_2d = (gt_mask.sum(axis=0) > 0).astype(np.uint8)  # riduce a (128,128)
            axs[3].imshow(gt_mask_2d, cmap='gray')
            axs[3].set_title("Ground Truth")

            axs[4].imshow(pred_mask, cmap='gray')
            axs[4].set_title("Predicted Mask")

            for ax in axs:
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"result_{idx}_{i}.png"))
            plt.close()


# ================================
# RISULTATI GLOBALI
# ================================

mean_iou = total_iou / num_images
mean_precision = total_precision / num_images
mean_recall = total_recall / num_images

print("\n===== RISULTATI GLOBALI =====")
print(f"Mean IoU: {mean_iou:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")

# ---- ROC AUC pixel-wise ----
roc_auc = roc_auc_score(all_pixel_gt, all_pixel_scores)
print(f"Pixel-wise ROC AUC: {roc_auc:.4f}")

# ================================
# RICERCA SOGLIA OTTIMALE
# ================================

print("\n===== Ricerca soglia ottimale =====")

best_threshold = 0
best_iou = 0

all_pixel_gt = np.array(all_pixel_gt)
all_pixel_scores = np.array(all_pixel_scores)

thresholds = np.linspace(0, np.max(all_pixel_scores), 50)

for t in thresholds:
    preds = (all_pixel_scores > t).astype(np.uint8)

    intersection = np.logical_and(preds, all_pixel_gt).sum()
    union = np.logical_or(preds, all_pixel_gt).sum()
    iou = intersection / (union + 1e-8)

    if iou > best_iou:
        best_iou = iou
        best_threshold = t

print(f"Best Threshold: {best_threshold:.6f}")
print(f"Best IoU: {best_iou:.4f}")

print(f"\nInference completata! Risultati salvati in '{OUTPUT_DIR}'")
