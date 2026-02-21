import torch
from cnn_anomaly_detection import CAE256_Latent100
from dataset_anomaly_detection import test_transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# ---- Config ----
MODEL_PATH = r"C:\Users\Francesco\Desktop\Progetto personale\best_autoencoder.pth"

ANOMALY_ROOT = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\test"
GT_ROOT = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\ground_truth"

CLASS_NAME = "broken_large"      # scegli la classe
FILENAME = "000.png"             # scegli l'immagine

OUTPUT_DIR = "inference_single_image"
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


# ---- Inference singola immagine ----
with torch.no_grad():
    output = model(img_tensor)

orig = img_tensor[0]
recon = output[0]

heatmap, stats = create_heatmap(orig, recon)

# ---- Normalizzazione heatmap (importante) ----
heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

# ---- Threshold temporanea ----
threshold = np.percentile(heatmap_norm, 90)  # top 5% dei pixel più “errati”
pred_mask = (heatmap_norm > threshold).astype(np.uint8)

# Assicuriamoci che la GT sia binaria
gt_binary = (gt_tensor > 0).astype(np.uint8)

# True Positive (sovrapposizione)
tp = np.logical_and(pred_mask == 1, gt_binary == 1)

# False Positive
fp = np.logical_and(pred_mask == 1, gt_binary == 0)

# False Negative
fn = np.logical_and(pred_mask == 0, gt_binary == 1)

overlay = np.zeros((gt_binary.shape[0], gt_binary.shape[1], 3))

# Verde = True Positive
overlay[tp] = [0, 1, 0]

# Rosso = False Positive
overlay[fp] = [1, 0, 0]

# Blu = False Negative
overlay[fn] = [0, 0, 1]

iou = np.sum(tp) / (np.sum(tp) + np.sum(fp) + np.sum(fn) + 1e-8)
f1 = 2 * np.sum(tp) / (2 * np.sum(tp) + np.sum(fp) + np.sum(fn) + 1e-8)


print(f"Iou={iou}")
print(f"f1={f1}")
print(f"MSE totale={stats['mse_total']:.6f}")
print(f"MSE per canale={stats['mse_per_channel']}")
print(f"Max={stats['max_error']:.6f}, Min={stats['min_error']:.6f}")



# ---- Visualizzazione ----
fig, axs = plt.subplots(1, 6, figsize=(24, 6))

axs[0].imshow(orig.permute(1, 2, 0).cpu().numpy())
axs[0].set_title("Original")
axs[0].axis('off')

axs[1].imshow(recon.permute(1, 2, 0).cpu().numpy())
axs[1].set_title("Reconstructed")
axs[1].axis('off')

axs[2].imshow(heatmap_norm, cmap='hot')
axs[2].set_title("Heatmap")
axs[2].axis('off')

axs[3].imshow(gt_binary, cmap='gray')
axs[3].set_title("Ground Truth")
axs[3].axis('off')

axs[4].imshow(pred_mask, cmap='gray')
axs[4].set_title("Predicted Mask")
axs[4].axis('off')

axs[5].imshow(overlay)
axs[5].set_title("Overlay (TP/FP/FN)")
axs[5].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "single_result.png"))
plt.close()


print(f"Inference completata! Risultato salvato in '{OUTPUT_DIR}'")