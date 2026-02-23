import torch
from cnn_anomaly_detection import CAE256_NoLinear
from dataset_anomaly_detection import test_loader
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- Config ----
MODEL_PATH = r"C:\Users\Francesco\Desktop\Progetto personale\best_autoencoder.pth"
OUTPUT_DIR = "inference_results_good"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modello
model = CAE256_NoLinear().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def create_heatmap(original, reconstructed):
    """
    Crea una heatmap delle anomalie e calcola statistiche numeriche.
    
    Args:
        original (torch.Tensor): immagine originale [3, H, W]
        reconstructed (torch.Tensor): immagine ricostruita [3, H, W]
    
    Returns:
        heatmap (np.ndarray): heatmap degli errori [H, W]
        stats (dict): dizionario con errori numerici
    """
    # Errore per pixel
    error_pixel = original - reconstructed
    
    # Errore al quadrato (MSE per pixel)
    error_squared = error_pixel ** 2
    
    # Heatmap: media sui canali
    heatmap = error_squared.mean(dim=0).cpu().numpy()
    
    # Statistiche numeriche
    mse_image = error_squared.mean().item()               # MSE totale immagine
    mse_per_channel = error_squared.view(3, -1).mean(dim=1).tolist()  # MSE per canale R,G,B
    max_error = error_squared.max().item()               # massimo errore al quadrato
    min_error = error_squared.min().item()               # minimo errore al quadrato

    stats = {
        "mse_total": mse_image,
        "mse_per_channel": mse_per_channel,
        "max_error": max_error,
        "min_error": min_error
    }

    return heatmap, stats


# ---- Inference ----
with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        batch = batch.to(device)
        outputs = model(batch)

        for i in range(batch.size(0)):
            orig = batch[i]
            recon = outputs[i]

            # Heatmap
            heatmap, stats = create_heatmap(orig, recon)
            
            print(f"Image {idx}: MSE totale={stats['mse_total']:.6f}, " f"MSE per canale={stats['mse_per_channel']}, "
                                f"Max={stats['max_error']:.6f}, Min={stats['min_error']:.6f}")

            # Visualizzazione
            fig, axs = plt.subplots(1, 3, figsize=(12, 8))
            axs[0].imshow(orig.permute(1, 2, 0).cpu().numpy())
            axs[0].set_title("Original")
            axs[0].axis('off')

            axs[1].imshow(recon.permute(1, 2, 0).cpu().numpy())
            axs[1].set_title("Reconstructed")
            axs[1].axis('off')

            axs[2].imshow(heatmap, cmap='hot')
            axs[2].set_title("Anomaly Heatmap")
            axs[2].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"result_{idx}_{i}.png"))
            plt.close()

print(f"Inference completata! Risultati salvati in '{OUTPUT_DIR}'")
