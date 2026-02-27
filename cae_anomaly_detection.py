import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class CAE256_FC_Latent32(nn.Module):
    def __init__(self):
        super(CAE256_FC_Latent32, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 256, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),

            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

'''BatchNorm usa le statistiche globali apprese dalle immagini buone e tende a “normalizzare”
anche le anomalie in fase di test, riducendo il contrasto tra normale e difettoso, mentre InstanceNorm lavora 
per singola immagine e preserva meglio le deviazioni locali utili per rilevare anomalie.


Con ReLU, i valori negativi vengono azzerati, quindi il gradiente è nullo per x < 0 
e il neurone può smettere di aggiornarsi (dead neuron).
Con LeakyReLU, invece, i valori negativi vengono moltiplicati per un piccolo coefficiente (es. 0.2)
e quindi il gradiente rimane diverso da zero, preservando l’informazione.

ConvTranspose aumenta la dimensione inserendo zeri e applicando una convoluzione, ma la sovrapposizione
non uniforme del kernel può creare squilibri (checkerboard artifacts); l’upsampling invece interpola prima i 
in modo uniforme e solo dopo applica una convoluzione standard, producendo ricostruzioni più stabili.

Per image-level detection basta un bottleneck linear perché serve solo un vettore globale (perde dettagli), mentre fully convolutional 
è più adatto se vuoi anche heatmap o dettagli locali.

Nel decoder si usa ReLU perché vogliamo produrre valori positivi stabili per la ricostruzione, mentre LeakyReLU 
servirebbe solo a preservare gradienti negativi che qui non sono utili.

'''