# conda activate anomaly_env

import torchvision.transforms as T

from torch.utils.data import Dataset
from PIL import Image
import os

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset



train_transforms = T.Compose([
    T.Resize((256, 256)),  # ridimensiona le immagini da 900x900 a 256x256
    T.ToTensor(),           # converte in tensore PyTorch [0,1]
])



class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) 
                            for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


train_dataset = MVTecDataset(r"C:\Users\Francesco\Desktop\Progetto personale\bottle\train\good", transform=train_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,      
    shuffle=True,       
    num_workers=2       
)

# Test transformations (stesse del train)
test_transforms = train_transforms  # o puoi aggiungere altre augmentations se vuoi

# Dataset test
test_dataset = MVTecDataset(r"C:/Users/Francesco/Desktop/Progetto personale/bottle/test/good", transform=test_transforms
)

# DataLoader test
test_loader = DataLoader(
    test_dataset, 
    batch_size=1,   # meglio 1 per analizzare ogni immagine singolarmente
    shuffle=False
)

# Cartella con tutte le anomalie di test
anomaly_root = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\test"

anomaly_dataset = ConcatDataset([
    MVTecDataset(os.path.join(anomaly_root, "broken_large"), transform=test_transforms),
    MVTecDataset(os.path.join(anomaly_root, "broken_small"), transform=test_transforms),
    MVTecDataset(os.path.join(anomaly_root, "contamination"), transform=test_transforms)
])

anomaly_loader = DataLoader(anomaly_dataset, batch_size=4, shuffle=False)

ground_truth_root = r"C:\Users\Francesco\Desktop\Progetto personale\bottle\ground_truth"

gt_dataset = ConcatDataset([
    MVTecDataset(os.path.join(ground_truth_root, "broken_large"), transform=test_transforms),
    MVTecDataset(os.path.join(ground_truth_root, "broken_small"), transform=test_transforms),
    MVTecDataset(os.path.join(ground_truth_root, "contamination"), transform=test_transforms)
])
gt_loader = DataLoader(gt_dataset, batch_size=4, shuffle=False)