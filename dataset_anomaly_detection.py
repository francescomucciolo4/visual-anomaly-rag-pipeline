import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import torch


train_transforms = T.Compose([
    T.Resize((256, 256)),  
    T.RandomRotation(degrees=10),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    T.ToTensor(),
])

val_transforms = T.Compose([
    T.Resize((256, 256)),  
    T.ToTensor()
])


class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted([
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


class TransformableSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx):
        img_path = self.subset.dataset.image_paths[self.subset.indices[idx]]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.subset)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "data", "bottle", "train", "good")

full_dataset = MVTecDataset(root_dir=ROOT_DIR)
train_size = len(full_dataset) - 29
val_size = 29

torch.manual_seed(42)
train_subset, val_subset = random_split(
    full_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_dataset = TransformableSubset(train_subset, transform=train_transforms)
val_dataset = TransformableSubset(val_subset, transform=val_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


if __name__ == "__main__":
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    print(f"Train: {len(train_dataset)} images, batch shape: {train_batch.shape}")
    print(f"Val: {len(val_dataset)} images, batch shape: {val_batch.shape}")
