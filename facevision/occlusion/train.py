import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

from facevision.occlusion.classifier import OcclusionClassifier


def train_classifier(data_dir: str, batch_size: int = 32, val_pct: float = 0.1):

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
    dataloaders = {'train': train_dl, 'val': val_dl}

    classifier = OcclusionClassifier(num_classes=4, pretrained=False)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    classifier.fit(dataloaders, criterion, optimizer, exp_lr_scheduler)
    classifier.plot_history()


if __name__ == '__main__':
    DATASET_DIR = r"C:\Users\j.mokhtari\Downloads\datasets\glasses-and-coverings\occluded_faces"
    train_classifier(DATASET_DIR)
