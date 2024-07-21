from typing import Dict

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.stop_flag = False

    def __call__(self, train_loss, val_loss):
        if (val_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.stop_flag = True
        else:
            self.counter = 0


def get_dataloaders(data_dir: str, batch_size: int = 32, val_pct: float = 0.1) -> Dict[str, DataLoader]:
    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(data_dir, transform=img_transforms)
    val_size = int(val_pct * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size, shuffle=True, num_workers=4, pin_memory=False)
    return {'train': train_dl, 'val': val_dl}

